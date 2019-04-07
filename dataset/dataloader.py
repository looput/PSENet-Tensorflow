import os

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

# import sys
# sys.path.append('/workspace/lupu/PSENet_curve')
import util
from dataset.dataset_factory import datasets_map
from dataset.preprocess import process_data_np,process_data_tf
from configuration import TRAIN_CONFIG

config = TRAIN_CONFIG
NUM_POINT=4


class DataLoader(object):
    def __init__(self, data_name, batch_size, train=True):
        self.batch_size = int(batch_size)
        self.train = train
        self.data_size = datasets_map[data_name].split_sizes
        file_pattern = datasets_map[data_name].file_pattern
        dataset_dir = datasets_map[data_name].dataset_dir

        split_name = 'train' if train == True else 'test'
        if util.str.contains(file_pattern, '%'):
            filename = util.io.join_path(
                dataset_dir, file_pattern % split_name)
        else:
            filename = util.io.join_path(dataset_dir, file_pattern)
        self.build_dataset([filename], batch_size=self.batch_size)

        print('load file: ', filename, '>> %d images' %
              self.data_size[split_name])

    def build_dataset(self, filenames, batch_size=1):
        data_config = config['data_config']
        # Creates a dataset that reads all of the examples from filenames.
        # filenames = ["ctw_train.tfrecord"]
        dataset = tf.data.TFRecordDataset(
            filenames, num_parallel_reads=data_config['read_num_p'])

        # dataset = tf.contrib.data.parallel_interleave(dataset, cycle_length=data_config['read_num_p'])
        # for version 1.5 and above use tf.data.TFRecordDataset

        # example proto decode
        def _parse_function(example_proto):
            keys_to_features = {'image': tf.FixedLenFeature([], tf.string),
                                'shape': tf.FixedLenFeature([3], tf.int64),
                                'xs': tf.VarLenFeature(tf.float32),
                                'ys': tf.VarLenFeature(tf.float32),
                                'num_points': tf.VarLenFeature(tf.int64),
                                'bboxes': tf.VarLenFeature(tf.float32),
                                'words': tf.VarLenFeature(tf.string),
                                'labels': tf.VarLenFeature(tf.int64),
                                }
            parsed_features = tf.parse_single_example(
                example_proto, keys_to_features)

            # some image is decode to 1 chanenl, so we need set the number of channel be 3
            image = tf.image.decode_image(parsed_features['image'], channels=3)

            image = tf.reshape(image, parsed_features['shape'])

            # convert to dense tensor, the num_points means the point's number of each bbox [n1,n2,n2]
            num_points = tf.sparse_tensor_to_dense(
                parsed_features['num_points'], default_value=0)
            # BBOX data is actually dense, convert it to dense tensor
            bboxes = tf.sparse_tensor_to_dense(
                parsed_features['bboxes'], default_value=0)
            # Since information about shape is lost reshape it
            bboxes = tf.reshape(bboxes, [-1, 4])

            x = tf.sparse_tensor_to_dense(
                parsed_features['xs'], default_value=0)
            y = tf.sparse_tensor_to_dense(
                parsed_features['ys'], default_value=0)
            # the shape inormation lost in store, so we restore it
            xs = tf.reshape(x, shape=[-1, NUM_POINT, 1])
            ys = tf.reshape(y, shape=[-1, NUM_POINT, 1])

            polys = tf.concat((xs, ys), -1)
            polys = tf.reshape(polys, [-1, NUM_POINT*2])
            label = tf.sparse_tensor_to_dense(
                parsed_features['labels'], default_value=0)
            return image, label, polys, num_points, bboxes

        def _process_data(image, label, polys, num_points, bboxes):
            return process_data_tf(image, label, polys, num_points, bboxes)

        # Parse the record into tensors.
        dataset = dataset.map(
            _parse_function, num_parallel_calls=data_config['para_num_p'])
        if self.train == True:
            dataset = dataset.map(
                _process_data, num_parallel_calls=data_config['pro_num_p'])
            # # Shuffle the dataset
            
            # # Repeat the input indefinitly
            dataset = dataset.repeat()
            dataset = dataset.shuffle(buffer_size=data_config['buffer_size'])
            # dataset=dataset.apply(tf.contrib.data.shuffle_and_repeat(data_config['buffer_size']))

            # Generate batches
            dataset = dataset.batch(batch_size)
            dataset = dataset.prefetch(buffer_size=data_config['prefetch'])
        else:
            dataset = dataset.batch(1)
        # Create a one-shot iterator
        self.iterator = dataset.make_one_shot_iterator()

    def load_data(self):
        # Get batch X and y
        return self.iterator.get_next()


if __name__ == '__main__':
    import crash_on_ipy
    import os
    import tqdm
    os.environ['CUDA_VISIBLE_DEVICES'] = '3'

    dataset = DataLoader('ctw1500', 8, train=True)

    data_get = dataset.load_data()

    tfconfig = tf.ConfigProto(allow_soft_placement=True)
    tfconfig.gpu_options.allow_growth = True
    with tf.Session(config=tfconfig) as sess:

        pbar = tqdm.tqdm(total=2000)
        for i in range(2000):
            pbar.update(1)
            image, gt, gt_kernel, training_mask = sess.run(data_get)

        #     plt.figure(0)
        #     plt.imshow(image[0,:, :, :])
        #     plt.figure(1)
        #     print(gt_kernel.shape)
        #     plt.imshow(gt_kernel[0,0, :,:])
        #     plt.pause(0.1)
        pbar.close()
