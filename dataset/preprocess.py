# dataloader add 3.0 scale
# dataloader add filer text
import numpy as np
from PIL import Image

import util
import cv2
import random
import torchvision.transforms as transforms
import torch
import pyclipper
import Polygon as plg
import tensorflow as tf
from preprocessing.ssd_vgg_preprocessing import tf_image_whitened

from configuration import TRAIN_CONFIG
config=TRAIN_CONFIG
random.seed(123456)



def random_horizontal_flip(imgs):
    if random.random() < 0.5:
        for i in range(len(imgs)):
            imgs[i] = np.flip(imgs[i], axis=1).copy()
    return imgs

def random_rotate(imgs):
    max_angle = 10
    angle = random.random() * 2 * max_angle - max_angle
    for i in range(len(imgs)):
        img = imgs[i]
        w, h = img.shape[:2]
        rotation_matrix = cv2.getRotationMatrix2D((h / 2, w / 2), angle, 1)
        img_rotation = cv2.warpAffine(img, rotation_matrix, (h, w))
        imgs[i] = img_rotation
    return imgs

def scale(img, long_size=2240):
    h, w = img.shape[0:2]
    scale = long_size * 1.0 / max(h, w)
    img = cv2.resize(img, dsize=None, fx=scale, fy=scale)
    return img

def random_scale(img, min_size,ran_scale=[0.5, 1.0, 2.0, 3.0]):
    h, w = img.shape[0:2]
    if max(h, w) > 1280:
        scale = 1280.0 / max(h, w)
        img = cv2.resize(img, dsize=None, fx=scale, fy=scale)

    h, w = img.shape[0:2]
    random_scale = np.array(ran_scale)
    scale = np.random.choice(random_scale)
    if min(h, w) * scale <= min_size:
        scale = (min_size + 10) * 1.0 / min(h, w)
    img = cv2.resize(img, dsize=None, fx=scale, fy=scale)
    return img

def random_crop(imgs, img_size):
    h, w = imgs[0].shape[0:2]
    th, tw = img_size
    # padding the image
    if h<th or w<tw:
        for idx in range(len(imgs)):
            image = imgs[idx]
            color =[123., 117., 104.] if len(image.shape)==3 else [0]
            top=(th-h)//2 if th-h>0 else 0
            bottom=th-top-h if th-h>0 else 0
            left=(tw-w)//2 if tw-w>0 else 0
            right=tw-left-w if tw-w>0 else 0

            imgs[idx] = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT,value=color)

    h, w = imgs[0].shape[0:2]
    if w == tw and h == th:
        return imgs

    if random.random() > 3.0 / 8.0 and np.max(imgs[1]) > 0:
        tl = np.min(np.where(imgs[1] > 0), axis = 1) - img_size
        tl[tl < 0] = 0
        br = np.max(np.where(imgs[1] > 0), axis = 1) - img_size
        br[br < 0] = 0
        br[0] = min(br[0], h - th)
        br[1] = min(br[1], w - tw)
        
        i = random.randint(tl[0], br[0])
        j = random.randint(tl[1], br[1])
    else:
        i = random.randint(0, h - th)
        j = random.randint(0, w - tw)
    
    # return i, j, th, tw
    for idx in range(len(imgs)):
        if len(imgs[idx].shape) == 3:
            imgs[idx] = imgs[idx][i:i + th, j:j + tw, :]
        else:
            imgs[idx] = imgs[idx][i:i + th, j:j + tw]
    return imgs

def dist(a, b):
    return np.sqrt(np.sum((a - b) ** 2))

def perimeter(bbox):
    peri = 0.0
    for i in range(bbox.shape[0]):
        peri += dist(bbox[i], bbox[(i + 1) % bbox.shape[0]])
    return peri

def shrink(bboxes, rate, max_shr=20):
    rate = rate * rate
    shrinked_bboxes = []
    for bbox in bboxes:
        area = plg.Polygon(bbox).area()
        peri = perimeter(bbox)

        pco = pyclipper.PyclipperOffset()
        pco.AddPath(bbox, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
        offset = min((int)(area * (1 - rate) / (peri + 0.001) + 0.5), max_shr)
        
        shrinked_bbox = pco.Execute(-offset)
        if len(shrinked_bbox) == 0:
            shrinked_bboxes.append(bbox)
            continue
        
        shrinked_bbox = np.array(shrinked_bbox[0])
        if shrinked_bbox.shape[0] <= 2:
            shrinked_bboxes.append(bbox)
            continue
        
        shrinked_bboxes.append(shrinked_bbox)
    
    return np.array(shrinked_bboxes)


def process_data_np(image, label, bboxes):
    # FIXME the mine size ??
    img = random_scale(image, config['min_size'],config['ran_scale'])

    gt_text = np.zeros(img.shape[0:2], dtype='uint8')
    training_mask = np.ones(img.shape[0:2], dtype='uint8')
    if bboxes.shape[0] > 0:
        bboxes = np.reshape(bboxes * ([img.shape[1], img.shape[0]] * 4), (bboxes.shape[0], int(bboxes.shape[1] / 2), 2)).astype(np.int32)
        # print(bboxes)
        for i in range(bboxes.shape[0]):
            cv2.drawContours(gt_text, [bboxes[i]], -1, i + 1, -1)
            if not label[i]:
                cv2.drawContours(training_mask, [bboxes[i]], -1, 0, -1)

    gt_kernals = []
    for rate in config['rate']:
        gt_kernal = np.zeros(img.shape[0:2], dtype='uint8')
        kernal_bboxes = shrink(bboxes, rate)
        for i in range(bboxes.shape[0]):
            cv2.drawContours(gt_kernal, [kernal_bboxes[i]], -1, 1, -1)
        gt_kernals.append(gt_kernal)


    imgs = [img, gt_text, training_mask]
    imgs.extend(gt_kernals)

    imgs = random_horizontal_flip(imgs)
    imgs = random_rotate(imgs)
    imgs = random_crop(imgs, (640,640))

    img, gt_text, training_mask, gt_kernals = imgs[0], imgs[1], imgs[2], imgs[3:]
    
    gt_text[gt_text > 0] = 1
    gt_kernals = np.array(gt_kernals)

    img = Image.fromarray(img)
    # img = img.convert('RGB')
    img = transforms.ColorJitter(brightness = 32.0 / 255, saturation = 0.5)(img)
    img=np.asarray(img)


    return img,gt_text,gt_kernals,training_mask

def process_data_tf(image, label, polys, num_points, bboxes):
    # TODO: the images are normalized using the channel means and standard deviations
    image = tf.identity(image, 'input_image')

    img, gt_text, gt_kernals, training_mask = tf.py_func(process_data_np, [image, label, polys], [
        tf.uint8, tf.uint8, tf.uint8, tf.uint8])

    # gt_kernals.set_shape([640,640,6])
    # training_mask.set_shape([640,640,1])
    img.set_shape([640,640,3])
    gt_text.set_shape([640,640])
    gt_kernals.set_shape([len(config['rate']),640,640])
    training_mask.set_shape([640,640])

    img = tf.to_float(img)
    gt_text = tf.to_float(gt_text)
    gt_kernals = tf.to_float(gt_kernals)
    training_mask = tf.to_float(training_mask)

    img = tf_image_whitened(img, [123., 117., 104.])

    return img, gt_text, gt_kernals, training_mask

def process_td_np(image, label,bboxes):
    # generate mask
    height = image.shape[0]
    width = image.shape[1]
    patch_size=config['train_image_shape'][0]

    mask = np.zeros(image.shape[0:2], dtype='uint8')
    training_mask = np.ones(image.shape[0:2], dtype='uint8')
    if bboxes.shape[0] > 0:
        bboxes = np.reshape(bboxes * ([image.shape[1], image.shape[0]] * 4), (bboxes.shape[0], int(bboxes.shape[1] / 2), 2)).astype(np.int32)
        for i in range(bboxes.shape[0]):
            cv2.drawContours(mask, [bboxes[i]], -1, i + 1, -1)
            if not label[i]:
                cv2.drawContours(training_mask, [bboxes[i]], -1, 0, -1)

    # random crop parameters
    loopTimes = 0
    MaxTimes = 100
    while True:
        #random parameters
        scale_h = np.random.uniform(0.05, 1)
        scale_w = np.random.uniform(0.05, 1)
        aspect_ratio = float(height)/width*scale_h/scale_w
        if aspect_ratio<0.3 or aspect_ratio>3: continue
        patch_h = int(height*scale_h)
        patch_w = int(width*scale_w)
        patch_h0 = np.random.randint(0, height-patch_h+1)
        patch_w0 = np.random.randint(0, width-patch_w+1)
        # compute overlap
        overlap_text = mask[patch_h0:patch_h0+patch_h, patch_w0:patch_w0+patch_w] > 0
        overlap_text_count = np.sum(overlap_text)
        min_overlap_ratio = [0.01, 0.03, 0.05, 0.07]
        random_ratio = np.random.randint(0, 4)
        if overlap_text_count > patch_h*patch_w*min_overlap_ratio[random_ratio]: break
        loopTimes += 1
        if loopTimes >= MaxTimes:
            patch_h = height
            patch_w = width
            patch_h0 = 0
            patch_w0 = 0
            break

    # random crop & resize
    image = image.astype(np.float32)
    image = image[patch_h0:patch_h0+patch_h, patch_w0:patch_w0+patch_w, :]
    image = cv2.resize(image, (patch_size, patch_size), interpolation=cv2.INTER_LINEAR)

    mask = mask[patch_h0:patch_h0+patch_h, patch_w0:patch_w0+patch_w]
    mask = cv2.resize(mask, (patch_size, patch_size), interpolation=cv2.INTER_NEAREST)

    training_mask = training_mask[patch_h0:patch_h0+patch_h, patch_w0:patch_w0+patch_w]
    training_mask = cv2.resize(training_mask, (patch_size, patch_size), interpolation=cv2.INTER_NEAREST)

    # random rotate
    prob = np.random.uniform(0,1)
    if prob <= 0.2:
        rtimes = 1
    elif prob >= 0.8:
        rtimes = 3
    else:
        rtimes = 0
    for rcount in range(rtimes):
        image = np.rot90(image)
        mask = np.rot90(mask)
        training_mask = np.rot90(training_mask)
    # cv2.imwrite('train_input/{}'.format(idx),image)

    # normalization
    # image = image.transpose((2,0,1))

    return image,mask,training_mask

def process_td_tf(image, label, polys, num_points, bboxes):
     # TODO: the images are normalized using the channel means and standard deviations
    image = tf.identity(image, 'input_image')

    img, gt_text, training_mask = tf.py_func(process_td_np, [image, label, polys], [
        tf.float32, tf.uint8, tf.uint8])

    # gt_kernals.set_shape([640,640,6])
    # training_mask.set_shape([640,640,1])
    img.set_shape([640,640,3])
    gt_text.set_shape([640,640])
    training_mask.set_shape([640,640])

    # img = tf.to_float(img)
    gt_text = tf.to_float(gt_text)
    training_mask = tf.to_float(training_mask)

    img = tf_image_whitened(img, [123., 117., 104.])

    return img, gt_text, training_mask

# def preprocess_for_eval(image, labels, bboxes, xs, ys,
#                         scale=1.0,out_shape=None, data_format='NHWC',
#                         resize=Resize.WARP_RESIZE,
#                         do_resize=True,
#                         scope='ssd_preprocessing_train'):
#     """Preprocess an image for evaluation.

#     Args:
#         image: A `Tensor` representing an image of arbitrary size.
#         out_shape: Output shape after pre-processing (if resize != None)
#         resize: Resize strategy.

#     Returns:
#         A preprocessed image.
#     """
#     with tf.name_scope(scope):
#         if image.get_shape().ndims != 3:
#             raise ValueError('Input must be of size [height, width, C>0]')

#         image = tf.to_float(image)
#         image = tf_image_whitened(image, [_R_MEAN, _G_MEAN, _B_MEAN])

#         if do_resize:
#             if resize == Resize.NONE:
#                 pass
#             else:
#                 if out_shape is None:
#                     i_shape=tf.to_float(tf.shape(image))
#                     shape=[tf.cast(i_shape[0]*scale,tf.int32),tf.cast(i_shape[1]*scale,tf.int32)]
#                     image = tf_image.resize_image(image, shape,
#                                                 method=tf.image.ResizeMethod.BILINEAR,
#                                                 align_corners=False)
#                     image_shape=tf.shape(image)
#                     image_h,image_w=image_shape[0],image_shape[1]
#                     image_h=tf.cast(tf.rint(image_h/32)*32,tf.int32)
#                     image_w=tf.cast(tf.rint(image_w/32)*32,tf.int32)
#                     image = tf_image.resize_image(
#                         image, [image_h, image_w], method=tf.image.ResizeMethod.BILINEAR, align_corners=False)
#                 else:
#                     image = tf_image.resize_image(image, out_shape,
#                                                 method=tf.image.ResizeMethod.BILINEAR,
#                                                 align_corners=False)

#         # Image data format.
#         if data_format == 'NCHW':
#             image = tf.transpose(image, perm=(2, 0, 1))
#         return image, labels, bboxes, xs, ys