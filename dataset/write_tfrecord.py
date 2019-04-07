import tensorflow as tf
import numpy as np
import util
import matplotlib.pyplot as plt 
import math
import argparse

import sys
sys.path.append('../')
from configuration import TRAIN_CONFIG
config=TRAIN_CONFIG

def ic15_cvt_to_tfrecords(output_file, data_path, gt_path):
    # write records to a tfrecords file
    writer = tf.python_io.TFRecordWriter(output_file)

    image_names = util.io.ls(data_path, '.jpg')  # [0:10];
    print("%d images found in %s" % (len(image_names), data_path))
    for idx, image_name in enumerate(image_names):
        oriented_bboxes = []
        axs, ays = [], []
        num_list = []
        bboxes = []
        labels = []
        labels_text = []
        path = util.io.join_path(data_path, image_name)
        print("\tconverting image: %d/%d %s" %
              (idx, len(image_names), image_name))
        # NOTE: In python3, when read byte data, must use 'rb'
        image_data = tf.gfile.FastGFile(path, 'rb').read()

        image = util.img.imread(path, rgb=True)
        shape = image.shape
        h, w = shape[0:2]
        h *= 1.0
        w *= 1.0
        image_name = util.str.split(image_name, '.')[0]
        gt_name = 'gt_' + image_name + '.txt'
        gt_filepath = util.io.join_path(gt_path, gt_name)
        lines = util.io.read_lines(gt_filepath)

        for line in lines:
            line = util.str.remove_all(line, '\xef\xbb\xbf')
            line = util.str.remove_all(line, '\ufeff')

            gt = util.str.split(line, ',')
            oriented_box = [int(gt[i]) for i in range(8)]

            oriented_box = np.asarray(oriented_box) / ([w, h] * 4)
            oriented_bboxes.append(oriented_box)

            xs = oriented_box.reshape(4, 2)[:, 0]
            ys = oriented_box.reshape(4, 2)[:, 1]
            xmin = xs.min()
            xmax = xs.max()
            ymin = ys.min()
            ymax = ys.max()
            bboxes.append(np.array([xmin, ymin, xmax, ymax]))
            axs.append(xs)
            ays.append(ys)
            num_list.append(4)

            # might be wrong here, but it doesn't matter because the label is not going to be used in detection
            # NOTE: change str to bytes, for tf store data
            labels_text.append(str.encode(gt[-1]))
            ignored = util.str.contains(gt[-1], '###')
            if ignored:
                # labels.append(config.ignore_label);
                labels.append(0)
            else:
                # labels.append(config.text_label)
                labels.append(1)

        # write the serialized objec to the disk
        serialized = convert_to_example(
            [image_data],list(shape), np.asanyarray(axs), np.asarray(ays), num_list, np.asarray(bboxes), labels_text, labels)
        writer.write(serialized)

    writer.close()

# Each line has 32 values, representing xmin, ymin, xmax, ymax (of the circumsribed rectangle), pw1, ph1, ..., pw14, ph14.
def ctw_cvt_to_tfrecords(output_file, data_path, gt_path):
    # write records to a tfrecords file
    writer = tf.python_io.TFRecordWriter(output_file)

    image_names = util.io.ls(data_path, '.jpg')  # [0:10];
    print("%d images found in %s" % (len(image_names), data_path))
    for idx, image_name in enumerate(image_names):
        oriented_bboxes = []
        axs, ays = [], []
        num_list = []
        bboxes = []
        labels = []
        labels_text = []
        path = util.io.join_path(data_path, image_name)
        print("\tconverting image: %d/%d %s" %
              (idx, len(image_names), image_name))
        # NOTE: In python3, when read byte data, must use 'rb'
        image_data = tf.gfile.FastGFile(path, 'rb').read()

        image = util.img.imread(path, rgb=True)
        shape = image.shape
        h, w = shape[0:2]
        h *= 1.0
        w *= 1.0
        image_name = util.str.split(image_name, '.')[0]
        gt_name = image_name + '.txt'
        gt_filepath = util.io.join_path(gt_path, gt_name)
        lines = util.io.read_lines(gt_filepath)

        for line in lines:
            line = util.str.remove_all(line, '\xef\xbb\xbf')
            line = util.str.remove_all(line, '\ufeff')

            gt = util.str.split(line, ',')
            rect_box=[int(gt[i]) for i in range(0,4)]
            oriented_box = [int(gt[i]) for i in range(4,32)]
            oriented_box = np.asarray(oriented_box)
            rect_box=np.asarray(rect_box)

            xs = (oriented_box.reshape(14, 2)[:, 0]+rect_box[0])/w
            ys = (oriented_box.reshape(14, 2)[:, 1]+rect_box[1])/h

            bboxes.append(rect_box/([w,h]*2))
            axs.append(xs)
            ays.append(ys)
            num_list.append(28)

            # might be wrong here, but it doesn't matter because the label is not going to be used in detection
            # NOTE: change str to bytes, for tf store data
            labels_text.append(str.encode(''))
            labels.append(1)

        # write the serialized objec to the disk
        serialized = convert_to_example(
            [image_data],list(shape), np.asanyarray(axs), np.asarray(ays), num_list, np.asarray(bboxes), labels_text, labels)
        writer.write(serialized)

    writer.close()


def GetPts_td(lst, idx):
    pts_lst = lst[idx].split()
    hard = int(pts_lst[1])
    x = int(pts_lst[2])
    y = int(pts_lst[3])
    w = int(pts_lst[4])
    h = int(pts_lst[5])
    theta = float(pts_lst[6])
    x1 = math.cos(theta) * (-0.5 * w) - math.sin(theta) * (-0.5 * h) + x + 0.5 * w
    y1 = math.sin(theta) * (-0.5 * w) + math.cos(theta) * (-0.5 * h) + y + 0.5 * h
    x2 = math.cos(theta) * (0.5 * w) - math.sin(theta) * (-0.5 * h) + x + 0.5 * w
    y2 = math.sin(theta) * (0.5 * w) + math.cos(theta) * (-0.5 * h) + y + 0.5 * h
    x3 = math.cos(theta) * (0.5 * w) - math.sin(theta) * (0.5 * h) + x + 0.5 * w
    y3 = math.sin(theta) * (0.5 * w) + math.cos(theta) * (0.5 * h) + y + 0.5 * h
    x4 = math.cos(theta) * (-0.5 * w) - math.sin(theta) * (0.5 * h) + x + 0.5 * w
    y4 = math.sin(theta) * (-0.5 * w) + math.cos(theta) * (0.5 * h) + y + 0.5 * h
    pts = np.array([[x1, y1], [x2, y2], [x3, y3], [x4, y4]], dtype=np.int32)
    return pts, hard

def td_to_tfrecords(output_file,data_path,gt_path):
    writer = tf.python_io.TFRecordWriter(output_file)

    image_names = util.io.ls(data_path, '.JPG')  # [0:10];
    print("%d images found in %s" % (len(image_names), data_path))
    for idx, image_name in enumerate(image_names):
        oriented_bboxes = []
        bboxes = []
        axs,ays,num_list=[],[],[]
        labels = []
        labels_text = []
        path = util.io.join_path(data_path, image_name)
        print("\tconverting image: %d/%d %s" %
                (idx, len(image_names), image_name))
        # NOTE: In python3, when read byte data, must use 'rb'
        image_data = tf.gfile.FastGFile(path, 'rb').read()

        image = util.img.imread(path, rgb=True)
        shape = image.shape
        h, w = shape[0:2]
        h *= 1.0
        w *= 1.0
        image_name = util.str.split(image_name, '.')[0]
        gt_name = image_name + '.gt'
        gt_filepath = util.io.join_path(gt_path, gt_name)
        lines = util.io.read_lines(gt_filepath)
        lst = [x.strip() for x in lines]

        for idx in range(len(lines)):
            oriented_box, hard = GetPts_td(lst, idx)

            oriented_box=np.reshape(oriented_box,(-1))
            oriented_box = oriented_box/ ([w, h] * 4)
            oriented_bboxes.append(oriented_box)

            xs = oriented_box.reshape(4, 2)[:, 0]
            ys = oriented_box.reshape(4, 2)[:, 1]
            xmin = xs.min()
            xmax = xs.max()
            ymin = ys.min()
            ymax = ys.max()
            bboxes.append([xmin, ymin, xmax, ymax])
            axs.append(xs)
            ays.append(ys)
            num_list.append(4)

            # might be wrong here, but it doesn't matter because the label is not going to be used in detection
            # NOTE: change str to bytes, for tf store data
            labels_text.append(str.encode('NA'))
            if hard==1:
                labels.append(0)
            else:
                labels.append(1)

        # import pudb; pudb.set_trace()
        example = convert_to_example(
            [image_data],list(shape),np.asanyarray(axs), np.asarray(ays), num_list,  np.asarray(bboxes), labels_text,labels)
        writer.write(example)
    
    writer.close()

def convert_to_example(image_data, shape,x, y, num, bboxes, word, label):
    # Feature contains a map of string to feature proto objects
    feature = {}
    feature['image'] = tf.train.Feature(
        bytes_list=tf.train.BytesList(value=image_data))
    feature['shape'] = tf.train.Feature(
        int64_list=tf.train.Int64List(value=shape))
    feature['xs'] = tf.train.Feature(
        float_list=tf.train.FloatList(value=x.flatten()))
    feature['ys'] = tf.train.Feature(
        float_list=tf.train.FloatList(value=y.flatten()))
    feature['num_points'] = tf.train.Feature(
        int64_list=tf.train.Int64List(value=num))
    feature['bboxes'] = tf.train.Feature(
        float_list=tf.train.FloatList(value=bboxes.flatten()))
    feature['words'] = tf.train.Feature(
        bytes_list=tf.train.BytesList(value=word))
    feature['labels'] = tf.train.Feature(
        int64_list=tf.train.Int64List(value=label))

    # Construct the Example proto object
    example = tf.train.Example(features=tf.train.Features(feature=feature))

    # Serialize the example to a string
    serialized = example.SerializeToString()
    return serialized


if __name__ =='__main__':
    parser = argparse.ArgumentParser(description="convert data to tfrecord")
    parser.add_argument(
        "--data-folder",
        metavar="FILE",
        help="path to data",
    )
    args = parser.parse_args()

    # root_dir = util.io.get_absolute_path('/workspace/lupu/icdar2015')
    root_dir = util.io.get_absolute_path(args.data_folder)

    training_data_dir = util.io.join_path(root_dir, 'train_images')
    training_gt_dir = util.io.join_path(root_dir, 'train_gts')

    ic15_cvt_to_tfrecords('ic15_train.tfrecord',data_path=training_data_dir,gt_path=training_gt_dir)
    # ctw_cvt_to_tfrecords('ctw_train.tfrecord',data_path=training_data_dir,gt_path=training_gt_dir)
    # td_to_tfrecords('TD500_train.tfrecord',data_path='/workspace/datasets/MSRA-TD500/train',gt_path='/workspace/datasets/MSRA-TD500/train')
