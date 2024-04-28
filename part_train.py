from tqdm import tqdm
import numpy as np
import tensorflow as tf
import concurrent.futures
import os
import re

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


mirrored_strategy = tf.distribute.MirroredStrategy()

EACH_CHAIR_PARTS_COUNT_PTH = "./each_chair_parts_count.npy"
DATASET_DIR_PTH = "./chair_voxel_data"

LOAD_OBJS_COUNT = None
VOXEL_MAP_SHAPE = (128, 128, 128)

BATCH_SIZE_PER_REPLICA = 32
GLOBAL_BATCH_SIZE = BATCH_SIZE_PER_REPLICA * mirrored_strategy.num_replicas_in_sync


def sorted_alphanumeric(data):
    def convert(text): return int(text) if text.isdigit() else text.lower()
    def alphanum_key(key): return [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(data, key=alphanum_key)


@tf.function
def tf_data_to_dense(voxel_data, voxel_map_shape=(128, 128, 128)):
    dense_voxel_maps = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)

    for voxel_coords_tensor in voxel_data:
        value_map = tf.ones(tf.shape(voxel_coords_tensor)[:-1], dtype=tf.float32)

        voxel_map = tf.zeros(voxel_map_shape, dtype=tf.float32)
        voxel_map = tf.tensor_scatter_nd_add(voxel_map, voxel_coords_tensor, value_map)

        dense_voxel_maps = dense_voxel_maps.write(dense_voxel_maps.size(), voxel_map)

    return tf.expand_dims(dense_voxel_maps.stack(), axis=-1)


class DataGenerator:
    def __init__(self, dataset_dir_pth, each_chair_parts_count_pth, objs_count=None, batch_size=4):
        self.dataset_dir_pth = dataset_dir_pth

        self.each_chair_parts_count = np.load(each_chair_parts_count_pth)[:objs_count]
        self.num_objts = len(self.each_chair_parts_count)

        self.data_names = np.array(sorted_alphanumeric(os.listdir(self.dataset_dir_pth)), dtype=str)[:self._get_total_parts_size()]
        self.num_parts = len(self.data_names)

        self.batch_szie = batch_size

        self.part_voxels_coords = self._load_voxel_data()

    def _get_total_parts_size(self):
        count = 0

        if self.num_objts == None:
            return None

        for i in range(self.num_objts):
            count += self.each_chair_parts_count[i]

        return count

    def _load_voxel_data(self):
        print('Trying to load {} objects with total {} parts.'.format(self.num_objts, self.num_parts))

        def load_data(data_pth):
            return np.load(data_pth)

        with concurrent.futures.ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
            data_paths = [os.path.join(self.dataset_dir_pth, data_name) for data_name in self.data_names]
            part_voxels_coords = list(tqdm(executor.map(load_data, data_paths), total=len(data_paths)))

        return part_voxels_coords

    def get_tf_dataset(self):
        print('Generating TF dataset.')

        part_indices = tf.range(self.num_parts)
        
        part_voxels_coords = tf.RaggedTensor.from_row_lengths(tf.concat(self.part_voxels_coords, axis=0), row_lengths=[a.shape[0] for a in self.part_voxels_coords])
        
        dataset = tf.data.Dataset.from_tensor_slices((part_indices, part_voxels_coords)).batch(self.batch_szie, drop_remainder=True)
        return dataset

    def reset_index(self):
        self.curr_index = 0


class PartNetwork:
    def __init__(self, hparam):
        self.latent_code_dim = hparam['model_latent_code_dim']

        self.fc_channels = hparam['model_fc_channels']

        self.conv_size = hparam['model_conv_size']

        self.num_latent_codes_parts = hparam['model_num_latent_codes_parts']

        self.learning_rate_network = hparam['model_learning_rate_network']
        self.learning_rate_codes = hparam['model_learning_rate_codes']

        self.model_voxel_map_shape = hparam['model_voxel_map_shape']

        self.checkpoint_dir = hparam['model_checkpoint_dir']

        self.trained_epoch = tf.Variable(0)

        self._init_model()
        self._init_optimizer()
        self._init_losser()
        self._init_checkpoint()

    def _init_model(self):
        self.part_generator = self._get_generator()

        init_latent_code_parts = tf.random.normal((self.num_latent_codes_parts, self.latent_code_dim))
        self.latent_code_vars_parts = tf.Variable(init_latent_code_parts, trainable=True)

        self.part_generator_trainable_variables = self.part_generator.trainable_variables

    def _get_generator(self):
        with mirrored_strategy.scope():

            with tf.name_scope('Network/'):

                latent_code = tf.keras.layers.Input(shape=(self.latent_code_dim,))

                with tf.name_scope('FC_layers'):

                    fc0 = tf.keras.layers.Dense(self.fc_channels, activation='relu')(latent_code)

                    fc1 = tf.keras.layers.Dense(self.fc_channels, activation='relu')(fc0)

                    fc2 = tf.keras.layers.Dense(self.fc_channels, activation='relu')(fc1)

                    fc2_as_volume = tf.keras.layers.Reshape((1, 1, 1, self.fc_channels))(fc2)

                with tf.name_scope('GLO_VoxelDecoder'):

                    decoder_1 = self._conv_t_block_3d(fc2_as_volume, num_filters=32, size=self.conv_size, strides=2)

                    decoder_2 = self._conv_t_block_3d(decoder_1, num_filters=32, size=self.conv_size, strides=2)

                    decoder_3 = self._conv_t_block_3d(decoder_2, num_filters=32, size=self.conv_size, strides=2)

                    decoder_4 = self._conv_t_block_3d(decoder_3, num_filters=16, size=self.conv_size, strides=2)

                    decoder_5 = self._conv_t_block_3d(decoder_4, num_filters=8, size=self.conv_size, strides=2)

                    decoder_6 = self._conv_t_block_3d(decoder_5, num_filters=4, size=self.conv_size, strides=2)

                    volume_out = self._conv_t_block_3d(decoder_6, num_filters=1, size=self.conv_size, strides=2, output_mode=True)

            model = tf.keras.Model(inputs=[latent_code], outputs=[volume_out])

        return model

    def _conv_t_block_3d(self, tensor, num_filters, size, strides, alpha_lrelu=0.2, output_mode=False):
        conv_3D_transpose = tf.keras.layers.Conv3DTranspose(
            filters=num_filters,
            kernel_size=size,
            strides=strides,
            padding='same',
            kernel_initializer=tf.keras.initializers.glorot_normal(),
            use_bias=False
        )

        tensor = conv_3D_transpose(tensor)

        if output_mode:
            return tensor

        tensor = tf.keras.layers.BatchNormalization()(tensor)

        tensor = tf.keras.layers.LeakyReLU(alpha=alpha_lrelu)(tensor)

        return tensor

    def _init_optimizer(self):
        with mirrored_strategy.scope():
            self.optimizer_part_generator = tf.keras.optimizers.Adam(learning_rate=self.learning_rate_network)
            self.optimizer_latent_for_parts = tf.keras.optimizers.Adam(learning_rate=self.learning_rate_codes)

    def _init_losser(self):
        self.losser_bce = tf.keras.losses.BinaryCrossentropy(reduction=tf.keras.losses.Reduction.NONE)

    def _init_checkpoint(self):
        self.checkpoint = tf.train.Checkpoint(
            part_generator=self.part_generator,
            latent_code_vars_parts=self.latent_code_vars_parts,
            optimizer_part_generator=self.optimizer_part_generator,
            optimizer_latent_for_parts=self.optimizer_latent_for_parts,
            trained_epoch=self.trained_epoch
        )

        self.manager = tf.train.CheckpointManager(checkpoint=self.checkpoint,
                                                  directory=self.checkpoint_dir,
                                                  max_to_keep=1)

        self._load_checkpoint()

    def _load_checkpoint(self):
        latest_checkpoint = self.manager.latest_checkpoint

        if latest_checkpoint is not None:
            print('Checkpoint {} restored'.format(latest_checkpoint))
        else:
            print('No checkpoint was restored.')

        self.checkpoint.restore(latest_checkpoint).expect_partial()

    @tf.function
    def train_step_parts(self, latent_code_vars, true_voxels_part):
        with tf.GradientTape() as tape:
            pred_logits_voxels = self.part_generator(latent_code_vars)

            pred_voxels_part = tf.sigmoid(pred_logits_voxels)

            loss = self.losser_bce(true_voxels_part, pred_voxels_part)
            loss = tf.nn.compute_average_loss(loss)

            model_losses = self.part_generator.losses
            if model_losses:
                loss = loss + tf.nn.scale_regularization_loss(tf.add_n(model_losses))

        network_vars = self.part_generator_trainable_variables
        gradients = tape.gradient(loss, network_vars + [latent_code_vars])

        self.optimizer_part_generator.apply_gradients(zip(gradients[:len(network_vars)], network_vars))
        self.optimizer_latent_for_parts.apply_gradients(zip(gradients[len(network_vars):], [latent_code_vars]))

        return loss

    @tf.function
    def distributed_train_step(self, train_func, args):
        per_replica_losses = mirrored_strategy.run(train_func, args=args)
        return mirrored_strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses, axis=None)

    def get_latent_code_vars_by_indices(self, latent_code_vars, latent_code_indices):
        return tf.Variable(tf.gather(latent_code_vars, latent_code_indices), trainable=True)

    def update_latent_code_vars(self, indices, latent_code_vars):
        self.latent_code_vars_parts.assign(tf.tensor_scatter_nd_update(self.latent_code_vars_parts, tf.expand_dims(indices, 1), latent_code_vars))

    def save_models(self):
        self.manager.save(checkpoint_number=self.trained_epoch.numpy())


if __name__ == '__main__':
    data_generator = DataGenerator(dataset_dir_pth=DATASET_DIR_PTH,
                                   each_chair_parts_count_pth=EACH_CHAIR_PARTS_COUNT_PTH,
                                   objs_count=LOAD_OBJS_COUNT,
                                   batch_size=GLOBAL_BATCH_SIZE)

    dataset = data_generator.get_tf_dataset()

    dataset_size = tf.data.experimental.cardinality(dataset).numpy()

    dist_dataset = mirrored_strategy.experimental_distribute_dataset(dataset)

    model_hparam = {
        'model_latent_code_dim': 256,
        'model_fc_channels': 512,
        'model_conv_size': 4,
        'model_num_latent_codes_parts': data_generator.num_parts,
        'model_learning_rate_network': 5e-4,
        'model_learning_rate_codes': 1e-3,
        'model_voxel_map_shape': VOXEL_MAP_SHAPE,
        'model_checkpoint_dir': './ckpt_dist'
    }

    part_network = PartNetwork(model_hparam)

    max_no_improvement = 3
    min_loss_change = 10

    prev_avg_loss = None

    no_improvement_count = 0

    while True:
        total_loss = []

        part_network.trained_epoch.assign(part_network.trained_epoch+1)

        pbar = tqdm(dist_dataset, desc='[Epoch {}]'.format(part_network.trained_epoch.numpy()), total=dataset_size)

        for latent_code_indices, true_voxels_coords in pbar:
            true_voxels_maps = mirrored_strategy.run(tf_data_to_dense, args=(true_voxels_coords,))
            latent_code_vars = mirrored_strategy.run(part_network.get_latent_code_vars_by_indices, args=(part_network.latent_code_vars_parts, latent_code_indices,))

            loss = part_network.distributed_train_step(part_network.train_step_parts, (latent_code_vars, true_voxels_maps))

            total_loss.append(loss)

            mirrored_strategy.run(part_network.update_latent_code_vars, args=(latent_code_indices, latent_code_vars,))

            avg_loss = sum(total_loss) / len(total_loss)

            pbar.set_postfix({'Avg Loss': '{:.9f}'.format(avg_loss)})

        if prev_avg_loss is not None:
            curr_change = prev_avg_loss - avg_loss

            if curr_change < min_loss_change:
                no_improvement_count += 1
                print('No improvement count increased to {}.'.format(no_improvement_count))
            else:
                no_improvement_count = 0
                print('Reset No improvement count.')

        prev_avg_loss = avg_loss

        if no_improvement_count >= max_no_improvement:
            print('Early stopping as loss change has been less than {} for {} consecutive epochs.'.format(min_loss_change, max_no_improvement))
            break

        part_network.save_models()
