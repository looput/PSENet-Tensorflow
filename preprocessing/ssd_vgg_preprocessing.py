# Copyright 2015 Paul Balanca. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Pre-processing images for SSD-type networks.
"""
from enum import Enum, IntEnum
import numpy as np
import random

import tensorflow as tf
import tf_extended as tfe

from tensorflow.python.ops import control_flow_ops
import cv2
import util
from preprocessing import tf_image

slim = tf.contrib.slim

# Resizing strategies.
Resize = IntEnum('Resize', ('NONE',                # Nothing!
                            # Crop (and pad if necessary).
                            'CENTRAL_CROP',
                            # Pad, and resize to output shape.
                            'PAD_AND_RESIZE',
                            'WARP_RESIZE'))        # Warp resize.

import config
# VGG mean parameters.
_R_MEAN = config.r_mean
_G_MEAN = config.g_mean
_B_MEAN = config.b_mean

# Some training pre-processing parameters.

MAX_EXPAND_SCALE = config.max_expand_scale
# Minimum overlap to keep a bbox after cropping.
BBOX_CROP_OVERLAP = config.bbox_crop_overlap
MIN_OBJECT_COVERED = config.min_object_covered
# Distortion ratio during cropping.
CROP_ASPECT_RATIO_RANGE = config.crop_aspect_ratio_range
AREA_RANGE = config.area_range

LABEL_IGNORE = config.ignore_label
USING_SHORTER_SIDE_FILTERING = config.using_shorter_side_filtering

MIN_SHORTER_SIDE = config.min_shorter_side
MAX_SHORTER_SIDE = config.max_shorter_side

USE_ROTATION = config.use_rotation
FLIP = config.flip
SCALE=config.scale
ROTATE = config.rotate
ROTATE_90=config.rotate_90
USE_NM_CROP=True

def tf_image_whitened(image, means=[_R_MEAN, _G_MEAN, _B_MEAN]):
    """Subtracts the given means from each image channel.

    Returns:
        the centered image.
    """
    if image.get_shape().ndims != 3:
        raise ValueError('Input must be of size [height, width, C>0]')
    num_channels = image.get_shape().as_list()[-1]
    if len(means) != num_channels:
        raise ValueError('len(means) must match the number of channels')

    mean = tf.constant(means, dtype=image.dtype)
    image = image - mean
    return image


def tf_image_unwhitened(image, means=[_R_MEAN, _G_MEAN, _B_MEAN], to_int=True):
    """Re-convert to original image distribution, and convert to int if
    necessary.

    Returns:
      Centered image.
    """
    mean = tf.constant(means, dtype=image.dtype)
    image = image + mean
    if to_int:
        image = tf.cast(image, tf.int32)
    return image


def np_image_unwhitened(image, means=[_R_MEAN, _G_MEAN, _B_MEAN], to_int=True):
    """Re-convert to original image distribution, and convert to int if
    necessary. Numpy version.

    Returns:
      Centered image.
    """
    img = np.copy(image)
    img += np.array(means, dtype=img.dtype)
    if to_int:
        img = img.astype(np.uint8)
    return img


def tf_summary_image(image, bboxes, name='image', unwhitened=False):
    """Add image with bounding boxes to summary.
    """
    if unwhitened:
        image = tf_image_unwhitened(image)
    image = tf.expand_dims(image, 0)
    bboxes = tf.expand_dims(bboxes, 0)
    image_with_box = tf.image.draw_bounding_boxes(image, bboxes)
    tf.summary.image(name, image_with_box)


def apply_with_random_selector(x, func, num_cases):
    """Computes func(x, sel), with sel sampled from [0...num_cases-1].

    Args:
        x: input Tensor.
        func: Python function to apply.
        num_cases: Python int32, number of cases to sample sel from.

    Returns:
        The result of func(x, sel), where func receives the value of the
        selector as a python integer, but sel is sampled dynamically.
    """
    sel = tf.random_uniform([], maxval=num_cases, dtype=tf.int32)
    # Pass the real x only to one of the func calls.
    return control_flow_ops.merge([
        func(control_flow_ops.switch(x, tf.equal(sel, case))[1], case)
        for case in range(num_cases)])[0]


def distort_color(image, color_ordering=0, fast_mode=True, scope=None):
    """Distort the color of a Tensor image.

    Each color distortion is non-commutative and thus ordering of the color ops
    matters. Ideally we would randomly permute the ordering of the color ops.
    Rather then adding that level of complication, we select a distinct ordering
    of color ops for each preprocessing thread.

    Args:
        image: 3-D Tensor containing single image in [0, 1].
        color_ordering: Python int, a type of distortion (valid values: 0-3).
        fast_mode: Avoids slower ops (random_hue and random_contrast)
        scope: Optional scope for name_scope.
    Returns:
        3-D Tensor color-distorted image on range [0, 1]
    Raises:
        ValueError: if color_ordering not in [0, 3]
    """
    with tf.name_scope(scope, 'distort_color', [image]):
        if fast_mode:
            if color_ordering == 0:
                image = tf.image.random_brightness(image, max_delta=32. / 255.)
                image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
            else:
                image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
                image = tf.image.random_brightness(image, max_delta=32. / 255.)
        else:
            if color_ordering == 0:
                image = tf.image.random_brightness(image, max_delta=32. / 255.)
                image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
                image = tf.image.random_hue(image, max_delta=0.2)
                image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
            elif color_ordering == 1:
                image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
                image = tf.image.random_brightness(image, max_delta=32. / 255.)
                image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
                image = tf.image.random_hue(image, max_delta=0.2)
            elif color_ordering == 2:
                image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
                image = tf.image.random_hue(image, max_delta=0.2)
                image = tf.image.random_brightness(image, max_delta=32. / 255.)
                image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
            elif color_ordering == 3:
                image = tf.image.random_hue(image, max_delta=0.2)
                image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
                image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
                image = tf.image.random_brightness(image, max_delta=32. / 255.)
            else:
                raise ValueError('color_ordering must be in [0, 3]')
        # The random_* ops do not necessarily clamp.
        return tf.clip_by_value(image, 0.0, 1.0)


def distorted_bounding_box_crop(image,
                                labels,
                                bboxes,
                                xs, ys,
                                min_object_covered,
                                aspect_ratio_range,
                                area_range,
                                max_attempts=200,
                                scope=None):
    """Generates cropped_image using a one of the bboxes randomly distorted.

    See `tf.image.sample_distorted_bounding_box` for more documentation.

    Args:
        image: 3-D Tensor of image (it will be converted to floats in [0, 1]).
        bbox: 2-D float Tensor of bounding boxes arranged [num_boxes, coords]
            where each coordinate is [0, 1) and the coordinates are arranged
            as [ymin, xmin, ymax, xmax]. If num_boxes is 0 then it would use the whole
            image.
        min_object_covered: An optional `float`. Defaults to `0.1`. The cropped
            area of the image must contain at least this fraction of any bounding box
            supplied.
        aspect_ratio_range: An optional list of `floats`. The cropped area of the
            image must have an aspect ratio = width / height within this range.
        area_range: An optional list of `floats`. The cropped area of the image
            must contain a fraction of the supplied image within in this range.
        max_attempts: An optional `int`. Number of attempts at generating a cropped
            region of the image of the specified constraints. After `max_attempts`
            failures, return the entire image.
        scope: Optional scope for name_scope.
    Returns:
        A tuple, a 3-D Tensor cropped_image and the distorted bbox
    """
    with tf.name_scope(scope, 'distorted_bounding_box_crop', [image, bboxes, xs, ys]):
        # Each bounding box has shape [1, num_boxes, box coords] and
        # the coordinates are ordered [ymin, xmin, ymax, xmax].
        num_bboxes = tf.shape(bboxes)[0]

        def has_bboxes():
            return bboxes, labels, xs, ys

        def no_bboxes():
            xmin = tf.random_uniform((1, 1), minval=0, maxval=0.9)
            ymin = tf.random_uniform((1, 1), minval=0, maxval=0.9)
            w = tf.constant(0.1, dtype=tf.float32)
            h = w
            xmax = xmin + w
            ymax = ymin + h
            rnd_bboxes = tf.concat([ymin, xmin, ymax, xmax], axis=1)
            rnd_labels = tf.constant([config.background_label], dtype=tf.int64)
            rnd_xs = tf.concat([xmin, xmax, xmax, xmin], axis=1)
            rnd_ys = tf.concat([ymin, ymin, ymax, ymax], axis=1)

            return rnd_bboxes, rnd_labels, rnd_xs, rnd_ys

        bboxes, labels, xs, ys = tf.cond(num_bboxes > 0, has_bboxes, no_bboxes)
        bbox_begin, bbox_size, distort_bbox = tf.image.sample_distorted_bounding_box(
            tf.shape(image),
            bounding_boxes=tf.expand_dims(bboxes, 0),
            min_object_covered=min_object_covered,
            aspect_ratio_range=aspect_ratio_range,
            area_range=area_range,
            max_attempts=max_attempts,
            use_image_if_no_bounding_boxes=True)

        # Draw the bounding box in an image summary.
        # image_with_box = tf.image.draw_bounding_boxes(tf.expand_dims(image, 0),
        #                                               distort_bbox)
        # tf.summary.image('images_with_box', image_with_box)

        distort_bbox = distort_bbox[0, 0]

        # Crop the image to the specified bounding box.
        cropped_image = tf.slice(image, bbox_begin, bbox_size)
        # Restore the shape since the dynamic slice loses 3rd dimension.
        cropped_image.set_shape([None, None, 3])

        # Update bounding boxes: resize and filter out.
        bboxes, xs, ys = tfe.bboxes_resize(distort_bbox, bboxes, xs, ys)
        labels, bboxes, xs, ys = tfe.bboxes_filter_overlap(labels, bboxes, xs, ys,
                                                           threshold=BBOX_CROP_OVERLAP, assign_value=LABEL_IGNORE)
        return cropped_image, labels, bboxes, xs, ys, distort_bbox


def tf_rotate_image(image, xs, ys):
    image, bboxes, xs, ys = tf.py_func(rotate_image, [image, xs, ys], [
                                       tf.uint8, tf.float32, tf.float32, tf.float32])
    image.set_shape([None, None, 3])
    bboxes.set_shape([None, 4])
    xs.set_shape([None, 4])
    ys.set_shape([None, 4])
    return image, bboxes, xs, ys

def tf_resize_image(image,bboxes,xs,ys,scale):
    image=tf.expand_dims(image,0)
    image_shape=tf.to_float(tf.shape(image))

    image=tf.image.resize_bilinear(
        image, size=[tf.to_int32(image_shape[1]*scale), tf.to_int32(image_shape[2]*scale)])
    image=tf.squeeze(image,0)

    # NOTE there is no need to scale the bbbox, because this 已经归一化
    # bboxes=bboxes*scale
    # xs,ys=xs*scale,ys*scale

    return image,bboxes,xs,ys


def tf_scale_image(image, bboxes, xs, ys, min_size):
    
    h, w = tf.shape(image)[0],tf.shape(image)[1]

    def scale_op():
        scale = 1280./tf.cast(tf.maximum(h, w),tf.float32)
        return tf_resize_image(image, bboxes, xs, ys, scale)

    image=tf.to_float(image)
    image, bboxes, xs, ys = tf.cond(tf.greater_equal(
        tf.maximum(h, w), 1280), scale_op, lambda: (image, bboxes, xs, ys))
    h, w = tf.shape(image)[0],tf.shape(image)[1]
    h=tf.cast(h,tf.float32)
    w=tf.cast(w,tf.float32)

    # NOTE: https://stackoverflow.com/questions/41123879/numpy-random-choice-in-tensorflow
    elems = tf.convert_to_tensor([0.5, 1.0, 2.0, 3.0])
    samples = tf.multinomial(
        tf.log([[0.25, 0.25, 0.25, 0.25]]), 1)  # note log-prob
    scale = elems[tf.cast(samples[0][0], tf.int32)]

    scale = tf.cond(tf.less_equal(tf.minimum(h, w)*scale, min_size),
                    lambda: (min_size+10)*1.0/tf.minimum(h, w), lambda: scale)
    image,bboxes,xs,ys=tf_resize_image(image,bboxes,xs,ys,scale)
    return image, bboxes, xs, ys

def random_select(image_shape,bboxes):
    h, w,_ = image_shape
    th, tw = (640,640)
    if w == tw and h == th:
        print('return origin image')
        return np.array([0.,0.,1.,1.],np.float32)
    
    mask=np.zeros((int(h),int(w)))
    for bbox in bboxes:
        mask[int(bbox[0]*h):int(bbox[0]*h+bbox[2]*h),int(bbox[1]*w):int(bbox[1]*w+bbox[3]*w)]=1
    if random.random() > 3.0 / 8.0 and np.max(mask)>0:
        tl = np.min(np.where(mask > 0), axis = 1) - (640,640)
        tl[tl < 0] = 0
        br = np.max(np.where(mask > 0), axis = 1) - (640,640)
        br[br < 0] = 0
        br[0] = min(br[0], h - th)
        br[1] = min(br[1], w - tw)
        
        i = random.randint(tl[0], br[0])
        j = random.randint(tl[1], br[1])
    else:
        i = random.randint(0, h - th)
        j = random.randint(0, w - tw)
    
    # print([i/float(h),j/float(w),i/float(h)+th/float(h),j/float(w)+tw/float(w)])
    return np.array([i/float(h),j/float(w),i/float(h)+th/float(h),j/float(w)+tw/float(w)],np.float32)

def rotate_image(image, xs, ys):
    rotation_angle = np.random.randint(low=-10, high=10)
    scale = np.random.uniform(low=1., high=1.)
#     scale = 1.0
    h, w = image.shape[0:2]
    # rotate image
    image, M = util.img.rotate_about_center(image, rotation_angle, scale=scale)
    nh, nw = image.shape[0:2]

    # rotate bboxes
    xs = xs * w
    ys = ys * h

    def rotate_xys(xs, ys):
        xs = np.reshape(xs, -1)
        ys = np.reshape(ys, -1)
        xs, ys = np.dot(M, np.transpose([xs, ys, 1]))
        xs = np.reshape(xs, (-1, 4))
        ys = np.reshape(ys, (-1, 4))
        return xs, ys
    xs, ys = rotate_xys(xs, ys)
    xs = xs * 1.0 / nw
    ys = ys * 1.0 / nh
    xmin = np.min(xs, axis=1)
    xmin[np.where(xmin < 0)] = 0

    xmax = np.max(xs, axis=1)
    xmax[np.where(xmax > 1)] = 1

    ymin = np.min(ys, axis=1)
    ymin[np.where(ymin < 0)] = 0

    ymax = np.max(ys, axis=1)
    ymax[np.where(ymax > 1)] = 1

    bboxes = np.transpose(np.asarray([ymin, xmin, ymax, xmax]))
    image = np.asarray(image, np.uint8)
    return image, bboxes, xs, ys

def generate_sample(image_shape,bboxes):
    '''there are bug, may choose none text region
    '''
    min_obj_crop=MIN_OBJECT_COVERED
    image_h,image_w,_=image_shape.astype(np.int32)
    scale_list=[0.5,1.,2.,3.]
    max_attemp=200
    scale_choose=np.random.choice(scale_list,max_attemp,p=[0.25,0.25,0.25,0.25])
    # TODO origin ver is scale the image to 1280, then choose scale
    if bboxes.shape[0]>0:
        line_h=np.zeros(int(image_h),dtype=np.int32)
        line_w=np.zeros(int(image_w),dtype=np.int32)

        for bbox in bboxes:
            line_h[int(bbox[0]*image_h):int((bbox[0]+bbox[2])*image_h)]=1
            line_w[int(bbox[1]*image_w):int((bbox[1]+bbox[3])*image_w)]=1

            # for left crop boundary, can't crop, add 2, use 3 sign it
            line_h[int(bbox[0]*image_h):int((bbox[0]+bbox[2]*min_obj_crop)*image_h)]+=2
            line_w[int(bbox[1]*image_w):int((bbox[1]+bbox[3]*min_obj_crop)*image_w)]+=2

            #for right side boundary, add 1, use 2 to sign
            line_h[int((bbox[0]+bbox[2]*(1-min_obj_crop))*image_h):int((bbox[0]+bbox[2])*image_h)]+=1  
            line_w[int((bbox[1]+bbox[3]*(1-min_obj_crop))*image_w):int((bbox[1]+bbox[3])*image_w)]+=1  
            # for both is 4
        for i in range(max_attemp):
            scale=scale_choose[i]
            scale=640./min(640/scale,min(image_h,image_w))
            target_h=int(640./scale)
            target_w=int(640./scale)
            bbox_begin_h_max = np.maximum(image_h-target_h, 0)
            bbox_begin_w_max = np.maximum(image_w-target_w, 0)

            bbox_begin_h = int(np.random.uniform(0, int(bbox_begin_h_max)))
            bbox_begin_w = int(np.random.uniform(0, int(bbox_begin_w_max)))

            y1,x1=bbox_begin_h,bbox_begin_w
            y2,x2=bbox_begin_h+target_h,bbox_begin_w+target_w

            if line_h[y1]!= 3 and line_h[y1] != 4 and line_h[min(y2,image_h-1)] != 2 and line_h[min(y2,image_h-1)] != 4:
                if line_w[x1] != 3 and line_w[x1] != 4 and line_w[min(x2,image_w-1)] != 2 and line_w[min(x2,image_w-1)] != 4:
                    # FIXME maynot contain text region
                    if np.sum(line_h[y1:min(y2,image_h)])>0 and np.sum(line_w[x1:min(x2,image_w)])>0:
                        return np.array([y1/float(image_h),x1/float(image_w),y2/float(image_h),x2/float(image_w)],dtype=np.float32)
        # if don't find box, random choose a text as center
        print('don\'t find text!!!,choose center')
        index=np.random.choice(bboxes.shape[0])
        bbox=bboxes[index]
        center=[(bbox[0]+bbox[1])/2.,(bbox[1]+bbox[3])/2.]
        scale=scale_choose[i]
        scale=640./min(640/scale,min(image_h,image_w))
        target_h,target_w=(640./scale),(640./scale)
        d_y,d_x=target_h/image_h/2.,target_w/image_w/2.
        return np.array([center[0]-d_y,center[1]-d_x,center[0]+d_y,center[1]+d_x],dtype=np.float32)
    else:
        # print('didn\'t contain text!')
        return np.array([0,0,1,1],np.float32)


def preprocess_for_train(image, labels, bboxes, xs, ys,
                         out_shape, data_format='NHWC',
                         scope='ssd_preprocessing_train'):
    """Preprocesses the given image for training.

    Note that the actual resizing scale is sampled from
        [`resize_size_min`, `resize_size_max`].

    Args:
        image: A `Tensor` representing an image of arbitrary size.
        output_height: The height of the image after preprocessing.
        output_width: The width of the image after preprocessing.
        resize_side_min: The lower bound for the smallest side of the image for
            aspect-preserving resizing.
        resize_side_max: The upper bound for the smallest side of the image for
            aspect-preserving resizing.

    Returns:
        A preprocessed image.
    """
    fast_mode = False
    with tf.name_scope(scope, 'ssd_preprocessing_train', [image, labels, bboxes]):
        if image.get_shape().ndims != 3:
            raise ValueError('Input must be of size [height, width, C>0]')

        # Randomly flip the image horizontally.
        if FLIP:
            image, bboxes, xs, ys = tf_image.random_flip_left_right_bbox(
                image, bboxes, xs, ys)
        if ROTATE:
            # random rotate the image [-10, 10]
            image, bboxes, xs, ys = tf_rotate_image(image, xs, ys)

        # samples = tf.multinomial(tf.log([[0.25, 0.25, 0.25, 0.25]]), 1) # note log-prob
        # scale=elems[tf.cast(samples[0][0], tf.int32)]
        # if SCALE:
        #     image,bboxes,xs,ys=tf_scale_image(image,bboxes,xs,ys,640)

        image_shape = tf.cast(tf.shape(image), dtype=tf.float32)
        image_h, image_w = image_shape[0], image_shape[1]

        if USE_NM_CROP:
            mask=tf.greater_equal(labels,1)
            valid_bboxes=tf.boolean_mask(bboxes,mask)
            # FIXME bboxes may is empty
            # NOTE tf_func must return value must be numpy, or you will madding!!!!!!
            crop_bbox=tf.py_func(generate_sample,[image_shape,valid_bboxes],tf.float32)
        else:
            scales=tf.random_shuffle([0.5,1.])
            scales=tf.Print(scales,[crop])
            target_h = tf.cast(640/scales[0], dtype=tf.float32)
            target_w = tf.cast(640/scales[0], dtype=tf.float32)
            bbox_begin_h_max = tf.maximum(image_h-target_h, 0)
            bbox_begin_w_max = tf.maximum(image_w-target_w, 0)
            bbox_begin_h = tf.random_uniform(
                [], minval=0, maxval=bbox_begin_h_max, dtype=tf.float32)
            bbox_begin_w = tf.random_uniform(
                [], minval=0, maxval=bbox_begin_w_max, dtype=tf.float32)

            crop_bbox = [bbox_begin_h/image_h, bbox_begin_w/image_w, \
                (bbox_begin_h+target_h)/image_h, (bbox_begin_w+target_w)/image_w]

        image=tf.image.crop_and_resize(tf.expand_dims(image,0),[crop_bbox],[0],(640,640),extrapolation_value=128)
        image=tf.squeeze(image,0)
        bboxes, xs, ys = tfe.bboxes_resize(crop_bbox, bboxes, xs, ys)
        labels, bboxes, xs, ys = tfe.bboxes_filter_overlap(labels, bboxes, xs, ys,
                                                        threshold=BBOX_CROP_OVERLAP, assign_value=LABEL_IGNORE)

        if ROTATE_90:
            rnd = tf.random_uniform((), minval=0, maxval=1)
            image,bboxes,xs,ys=tf.cond(tf.less(rnd,0.2),lambda:tf_image.random_rotate90(image,bboxes,xs,ys),lambda:(image,bboxes,xs,ys))

        # tf_summary_image(tf.to_float(image), bboxes, 'crop_image')

        # what is the enpand's meanoing?
        # expand image
        if MAX_EXPAND_SCALE > 1:
            rnd2 = tf.random_uniform((), minval=0, maxval=1)

            def expand():
                scale = tf.random_uniform([], minval=1.0,
                                          maxval=MAX_EXPAND_SCALE, dtype=tf.float32)
                image_shape = tf.cast(tf.shape(image), dtype=tf.float32)
                image_h, image_w = image_shape[0], image_shape[1]
                target_h = tf.cast(image_h * scale, dtype=tf.int32)
                target_w = tf.cast(image_w * scale, dtype=tf.int32)
                tf.logging.info('expanded')
                return tf_image.resize_image_bboxes_with_crop_or_pad(
                    image, bboxes, xs, ys, target_h, target_w)

            def no_expand():
                return image, bboxes, xs, ys

            image, bboxes, xs, ys = tf.cond(
                tf.less(rnd2, config.expand_prob), expand, no_expand)

        # Convert to float scaled [0, 1].
        # if image.dtype != tf.float32:
            # image = tf.image.convert_image_dtype(image, dtype=tf.float32)
            # tf_summary_image(image, bboxes, 'image_with_bboxes')

        # Distort image and bounding boxes.
        dst_image = image
        # use tf.image.sample_distorted_bounding_box() random crop train patch, but can't control the scale 
        if False:    
            dst_image, labels, bboxes, xs, ys, distort_bbox = \
                distorted_bounding_box_crop(image, labels, bboxes, xs, ys,
                                            min_object_covered=MIN_OBJECT_COVERED,
                                            aspect_ratio_range=CROP_ASPECT_RATIO_RANGE,
                                            area_range=AREA_RANGE)
            # Resize image to output size.
            dst_image = tf_image.resize_image(dst_image, out_shape,
                                            method=tf.image.ResizeMethod.BILINEAR,
                                            align_corners=False)

        # Filter bboxes using the length of shorter sides
        if USING_SHORTER_SIDE_FILTERING:
            xs = xs * out_shape[1]
            ys = ys * out_shape[0]
            labels, bboxes, xs, ys = tfe.bboxes_filter_by_shorter_side(labels,
                                                                       bboxes, xs, ys,
                                                                       min_height=MIN_SHORTER_SIDE, max_height=MAX_SHORTER_SIDE,
                                                                       assign_value=LABEL_IGNORE)
            xs = xs / out_shape[1]
            ys = ys / out_shape[0]

        # Randomly distort the colors. There are 4 ways to do it.
        dst_image = apply_with_random_selector(
                dst_image/255.0,
                lambda x, ordering: distort_color(x, ordering, fast_mode),
                num_cases=4)
        dst_image=dst_image*255.
        # tf_summary_image(dst_image, bboxes, 'image_color_distorted')

        # FIXME: change the input value
        # NOTE: resnet v1 use VGG data process, so we use the same way
        image = tf_image_whitened(image, [_R_MEAN, _G_MEAN, _B_MEAN])
        # Image data format.
        if data_format == 'NCHW':
            image = tf.transpose(image, perm=(2, 0, 1)) 
        return image, labels, bboxes, xs, ys


def preprocess_for_eval(image, labels, bboxes, xs, ys,
                        scale=1.0,out_shape=None, data_format='NHWC',
                        resize=Resize.WARP_RESIZE,
                        do_resize=True,
                        scope='ssd_preprocessing_train'):
    """Preprocess an image for evaluation.

    Args:
        image: A `Tensor` representing an image of arbitrary size.
        out_shape: Output shape after pre-processing (if resize != None)
        resize: Resize strategy.

    Returns:
        A preprocessed image.
    """
    with tf.name_scope(scope):
        if image.get_shape().ndims != 3:
            raise ValueError('Input must be of size [height, width, C>0]')

        image = tf.to_float(image)
        image = tf_image_whitened(image, [_R_MEAN, _G_MEAN, _B_MEAN])

        if do_resize:
            if resize == Resize.NONE:
                pass
            else:
                if out_shape is None:
                    i_shape=tf.to_float(tf.shape(image))
                    shape=[tf.cast(i_shape[0]*scale,tf.int32),tf.cast(i_shape[1]*scale,tf.int32)]
                    image = tf_image.resize_image(image, shape,
                                                method=tf.image.ResizeMethod.BILINEAR,
                                                align_corners=False)
                    image_shape=tf.shape(image)
                    image_h,image_w=image_shape[0],image_shape[1]
                    image_h=tf.cast(tf.rint(image_h/32)*32,tf.int32)
                    image_w=tf.cast(tf.rint(image_w/32)*32,tf.int32)
                    image = tf_image.resize_image(
                        image, [image_h, image_w], method=tf.image.ResizeMethod.BILINEAR, align_corners=False)
                else:
                    image = tf_image.resize_image(image, out_shape,
                                                method=tf.image.ResizeMethod.BILINEAR,
                                                align_corners=False)

        # Image data format.
        if data_format == 'NCHW':
            image = tf.transpose(image, perm=(2, 0, 1))
        return image, labels, bboxes, xs, ys


def preprocess_image(image,
                     labels=None,
                     bboxes=None,
                     xs=None, ys=None,
                     scale=1.0,
                     out_shape=None,
                     data_format='NHWC',
                     is_training=False,
                     **kwargs):
    """Pre-process an given image.

    Args:
      image: A `Tensor` representing an image of arbitrary size.
      output_height: The height of the image after preprocessing.
      output_width: The width of the image after preprocessing.
      is_training: `True` if we're preprocessing the image for training and
        `False` otherwise.
      resize_side_min: The lower bound for the smallest side of the image for
        aspect-preserving resizing. If `is_training` is `False`, then this value
        is used for rescaling.
      resize_side_max: The upper bound for the smallest side of the image for
        aspect-preserving resizing. If `is_training` is `False`, this value is
         ignored. Otherwise, the resize side is sampled from
         [resize_size_min, resize_size_max].

    Returns:
      A preprocessed image.
    """
    if is_training:
        return preprocess_for_train(image, labels, bboxes, xs, ys,
                                    out_shape=out_shape,
                                    data_format=data_format)
    else:
        return preprocess_for_eval(image, labels, bboxes, xs, ys,
                                   scale=scale,
                                   out_shape=out_shape,
                                   data_format=data_format,
                                   **kwargs)
