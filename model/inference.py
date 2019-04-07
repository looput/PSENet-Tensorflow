#! /usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright © 2018 loop     Huazhong University of Science and Technology
import json
import os
import queue
import re
import time

import numpy as np
import tensorflow as tf
from tensorflow.python.client import timeline
from tensorflow.python.tools import inspect_checkpoint as chkp
from tensorboard.plugins.beholder import Beholder
from tensorflow.python import debug as tf_debug

import cv2
import matplotlib.pyplot as plt
import tqdm
import util
from dataset.dataloader import DataLoader
from .model_v2 import model,model_deconv
from PIL import Image
from preprocessing import ssd_vgg_preprocessing
from PSE_C import mylib
from skimage.measure import label, regionprops

slim = tf.contrib.slim


def time_it(func):
    def newFunc(*args, **args2):
        t0 = time.time()
        # print("@%s, {%s} start" % (time.strftime("%X", time.localtime()), func.__name__))
        back = func(*args, **args2)
        # print("@%s, {%s} end" % (time.strftime("%X", time.localtime()), func.__name__))
        print("@%.3fs taken for function: [%s]" %
              (time.time() - t0, func.__name__))
        return back
    return newFunc


def expansion_p(CC, Si):
    Q = queue.Queue()
    T = set()
    P = set()
    h, w = CC.shape
    for y in range(h):
        for x in range(w):
            if CC[y, x] > 0:
                T.add(((y, x), CC[y, x]))
                P.add((y, x))
                Q.put(((y, x), CC[y, x]))

    while Q.empty() == False:
        p, label = Q.get()
        for y in range(p[0]-1, p[0]+2):
            for x in range(p[1]-1, p[1]+2):
                if y >= 0 and y < h and x >= 0 and x < w:
                    if (y, x) not in P and Si[y, x] == 1:
                        P.add((y, x))
                        T.add(((y, x), label))
                        Q.put(((y, x), label))
                        CC[y, x] = label
    # now all element in Si are give label value
    return CC

# NOTE implement by C++, so the divese will be fast
# now it took 17ms
# @time_it
def expansion(CC, Si):
    def check(arr):
        if arr.shape[-1] == 1:
            arr = np.squeeze(arr, -1)
            return arr.astype(np.int32)
        else:
            return arr.astype(np.int32)
    CC = check(CC)
    Si = check(Si)

    CC_out = CC.copy(order='C')
    ps = mylib.PyExpand()
    ps.expansion(CC_out, Si.copy(order='C'))
    return CC_out


def rect_to_xys(rect, image_shape):
    """Convert rect to xys, i.e., eight points
    The `image_shape` is used to to make sure all points return are valid, i.e., within image area
    """
    h, w = image_shape[0:2]

    def get_valid_x(x):
        if x < 0:
            return 0
        if x >= w:
            return w - 1
        return x

    def get_valid_y(y):
        if y < 0:
            return 0
        if y >= h:
            return h - 1
        return y

    # rect = ((rect[0], rect[1]), (rect[2], rect[3]), rect[4])

    points = cv2.boxPoints(rect)
    points = np.int0(points)
    for i_xy, (x, y) in enumerate(points):
        x = get_valid_x(x)
        y = get_valid_y(y)
        points[i_xy, :] = [x, y]
    points = np.reshape(points, -1)
    return points

# @time_it
def process_map(segment_map, threshold_k=0.55, thershold=0.55):
    segment_map = [np.squeeze(seg, 0) for seg in segment_map]
    # First, binary the segment map, choose OTSU or other argrithem
    # TODO 分割图使用相同排列顺序
    S1 = (segment_map[-1]) > threshold_k  # (640,640)
    # get cc and label them with different number
    CC = label(S1, connectivity=2)

    expand_cc=CC
    # TODO 分割图使用相同排列顺序
    for i in range(len(segment_map)-2,-1,-1):
        S_i = segment_map[i] > thershold
        expand_cc = expansion(expand_cc, S_i)
    return expand_cc


def region_to_bbox(mask, image_size, min_height=10, min_area=300):
    h, w = image_size
    score_map = (mask*255).astype(np.uint8)

    # NOTE TypeError: Layout of the output array incompatible with cv::Mat
    _, contours, _ = cv2.findContours(
        score_map.copy(), mode=cv2.RETR_CCOMP, method=cv2.CHAIN_APPROX_SIMPLE)
    # for bbox_contours in contours:
    rect = cv2.minAreaRect(contours[0])
    rect = list(rect)
    sw, sh = rect[1][:]
    # if min(sw, sh) < min_height:
    #     return None
    # if sw*sh < min_area:
    #     return None
    # if max(sw, sh) * 1.0 / min(sw, sh) < 2:
    #     return None
    rect = tuple(rect)
    xys = rect_to_xys(rect, [h, w])
    # boxes.append(np.append(xys,1))
    return xys

def region_to_poly(mask,image_size):
    h, w = image_size
    score_map = (mask*255).astype(np.uint8)

    # NOTE TypeError: Layout of the output array incompatible with cv::Mat
    _, contours, _ = cv2.findContours(
        score_map.copy(), mode=cv2.RETR_CCOMP, method=cv2.CHAIN_APPROX_SIMPLE)

    # for bbox_contours in contours:
    rect = cv2.minAreaRect(contours[0])
    rect = list(rect)
    sw, sh = rect[1][:]
    # if min(sw, sh) < min_height:
    #     return None
    # if sw*sh < min_area:
    #     return None
    # if max(sw, sh) * 1.0 / min(sw, sh) < 2:
    #     return None

    cnt=contours[0]
    # define main island contour approx. and hull
    perimeter = cv2.arcLength(cnt,True)
    epsilon = 0.01*cv2.arcLength(cnt,True)
    approx = cv2.approxPolyDP(cnt,epsilon,True)
    # aprox [n_p,1,2]
    xys=np.reshape(approx,[-1])
    return xys

def map_to_polys(segment_maps, result_map, image_size, aver_score=0.9):
    cc_num = result_map.max()  # this mean number of cc
    polys = []
    scores=[]
    for i in range(1, cc_num+1):
        # get each cc bounding
        mask = (result_map == i)
        region_score = np.sum(
            mask*np.squeeze(segment_maps[0], (0, -1)))/np.sum(mask)
        if region_score > aver_score:
            poly = region_to_poly(
                mask, (image_size['h'], image_size['w']))
        else:
            poly = None
        if poly is not None:
            polys.append(poly.tolist())
            scores.append(region_score)
    return polys,scores

def map_to_bboxes(segment_maps, result_map, image_size, aver_score=0.9):
    cc_num = result_map.max()  # this mean number of cc
    bboxes = np.empty((0, 8))
    scores=np.empty((0,1))
    for i in range(1, cc_num+1):
        # get each cc bounding
        mask = (result_map == i)
        region_score = np.sum(
            mask*np.squeeze(segment_maps[0], (0, -1)))/np.sum(mask)
        if region_score > aver_score:
            bbox = region_to_bbox(
                mask, (image_size['h'], image_size['w']))
        else:
            bbox = None
        if bbox is not None:
            bboxes = np.concatenate(
                (bboxes, bbox[np.newaxis, :]), axis=0)
            scores=np.concatenate((scores,np.array([[region_score]])),0)
    return bboxes,scores


def write_to_file(bboxes, image_name, output_dir,scores=None):
    ''' the socres is the average score of boundingbox region
    '''
    if os.path.isdir(output_dir) == False:
        os.makedirs(output_dir)
    if isinstance(bboxes,list)==False:
        bboxes = bboxes.tolist()
    # save to file
    filename = os.path.join(output_dir, 'res_%s.txt' % (image_name))
    # filename = os.path.join(output_dir, '%s.txt' % (image_name))
    lines = []
    for b_idx, bbox in enumerate(bboxes):
        values = [int(v) for v in bbox]
        line=''
        if scores is None:
            for i in range(len(values)-1):
                line+="%d,"%values[i]
            line+="%d\r\n"%values[-1]
        else:
            values.append(scores[b_idx])
            for i in range(len(values)-1):
                line+="%d,"%values[i]
            line+="%f\r\n"%values[-1]
        lines.append(line)
    util.io.write_lines(filename, lines)
    # print( 'result has been written to:', filename)

    # cv2.polylines(im[:, :, ::-1], [box.astype(np.int32).reshape((-1, 1, 2))], True, color=(255, 255, 0), thickness=3)


def log_to_file(imgs_path,file_name,image_arr,bboxes,segment_maps):
    for i,seg_map in enumerate(segment_maps):
        # check file exits
        if os.path.isdir(imgs_path) == False:
            os.makedirs(imgs_path)
        img_path = os.path.join(imgs_path, os.path.basename(file_name))
        seg_map_3c=np.repeat(seg_map[0,:,:,:],3,2)*255
        att_im = cv2.addWeighted(seg_map_3c.astype(np.uint8), 0.5, image_arr, 0.5, 0.0)
        save_img=att_im if i==0 else np.concatenate((save_img,att_im),1)

    for poly in bboxes:
        poly=np.asarray([poly])
        image_arr=cv2.polylines(image_arr,poly.astype(np.int).reshape([1,-1,2]),True,(0,255,0),thickness=2)
    save_img=np.concatenate((save_img,image_arr),1)

    cv2.imwrite(img_path.replace('.jpg','_{}.png'.format(i)),save_img.astype(np.uint8))

def eval_model(config, FLAGS,para_list=None,is_log=False):
    image_size = config['image_size']
    with tf.Graph().as_default():
        # image_ph = tf.placeholder(dtype=tf.uint8, shape=[
        #                           None, None, 3], name='input')
        path_ph = tf.placeholder(dtype=tf.string)

        raw_data = tf.read_file(path_ph)
        image = tf.image.decode_jpeg(raw_data,channels=3)
        image.set_shape((None, None, 3))
        out_shape=(image_size['h'], image_size['w']) if image_size['fixed_size']==True else None
        scale=1.0 if image_size['fixed_size']==True else image_size['scale']
        image_process, _, _, _, _ = ssd_vgg_preprocessing.preprocess_image(
            image,scale=scale,out_shape=out_shape, is_training=False)

        image_process = tf.expand_dims(image_process, 0)
        seg_maps, _ = model(image_process, is_training=False)

        # rescale seg_maps to origin size
        seg_map_list = []
        for i in range(config['n']):
            seg_map_list.append(tf.image.resize_images(seg_maps[:, :, :, i:i+1], [
                tf.shape(image)[0],  tf.shape(image)[1]]))

        # choose the complete map as mask, apply to shrink map
        mask = tf.greater_equal(seg_map_list[0], config['threshold'])
        mask = tf.to_float(mask)
        seg_map_list = [seg_map*mask for seg_map in seg_map_list]

        chp_name = util.io.get_filename(config['ckpt'])

        dump_path = util.io.join_path(
            config['log_dir'], 'test', FLAGS.train_name)
        # os.system('rm -r {}'.format(dump_path))

        global_step = tf.train.get_or_create_global_step()
        # global_step = tf.Variable(0,trainable=False)
        # Variables to restore: moving avg. or normal weights.
        
        if FLAGS.using_moving_average:
            variable_averages = tf.train.ExponentialMovingAverage(0.9999)
            variables_to_restore = variable_averages.variables_to_restore()
            variables_to_restore[global_step.op.name] = global_step

            filter_variable={}
            for var in variables_to_restore:
                if var.find('deformable/Variable')==-1:
                    filter_variable[var]=variables_to_restore[var]
        else:
            # variables_to_restore = slim.get_variables_to_restore()
            variables_to_restore = tf.get_collection(
                tf.GraphKeys.GLOBAL_VARIABLES)
            filter_variable=[]
            for var in variables_to_restore:
                if var.name.find('deformable/Variable')==-1:
                    filter_variable.append(var)

        saver = tf.train.Saver(filter_variable)

        with tf.name_scope('debug'):
            # tf.summary.image('input',tf.expand_dims(image,0))
            tf.summary.image('process', image_process)
            for i in range(config['n']):
                tf.summary.image(
                    ('%d_' % i+seg_map_list[i].op.name), seg_map_list[i])

        summary = tf.summary.merge_all()

        tfconfig = tf.ConfigProto(allow_soft_placement=True)
        tfconfig.gpu_options.allow_growth = True
        # config.gpu_options.per_process_gpu_memory_fraction = 0.4
        with tf.Session(config=tfconfig) as sess:
            trans_var = tf.get_collection('transform')
            init_transform = tf.variables_initializer(trans_var)
            sess.run(init_transform)

            # ckpt='/workspace/lupu/PSENet/Logs/train/run_bat-b/model.ckpt-21579'
            # for var in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES):
            #     print(var)
                # if var.name.find('BatchNorm')!=-1:
                #   chkp.print_tensors_in_checkpoint_file(config['ckpt'], tensor_name=var.name.split(':')[0], all_tensors=False)

            saver.restore(sess, config['ckpt'])
            print('restore model from: ', config['ckpt'])

            # coord = tf.train.Coordinator()
            # threads = tf.train.start_queue_runners(coord=coord)
            sum_writer = tf.summary.FileWriter(dump_path, graph=sess.graph)

            files = tf.gfile.Glob(os.path.join(config['test_dir'], '*.jpg'))
            try:
                files_sorted = sorted(files, key=lambda path: int(
                    path.split('/')[-1].split('.')[0]))
            except:
                files_sorted = sorted(files, key=lambda path: int(
                    path.split('_')[-1].split('.')[0]))
            pbar = tqdm.tqdm(total=len(files_sorted))
            for iter, file_name in enumerate(files_sorted):
                pbar.update(1)
                # image_data = util.img.imread(
                #     util.io.join_path(config['test_dir'], image_name), rgb=True)
                image_name = os.path.basename(file_name)
                image_name = image_name.split('.')[0]

                if is_log:
                    # NOTE: useful func, analysis tf graph run time
                    run_options = tf.RunOptions(
                        trace_level=tf.RunOptions.FULL_TRACE)
                    run_metadata = tf.RunMetadata()
                    # the input size must be  times 32(stride?)
                    image_arr, segment_maps, summ = sess.run(
                        [image, seg_map_list, summary], feed_dict={path_ph: file_name}, options=run_options, run_metadata=run_metadata)

                    # segment_maps=(segment_maps>0.5).astype(np.float32)

                    sum_writer.add_summary(summ,global_step=iter)
                    # add mermory and time info in graph
                    sum_writer.add_run_metadata(run_metadata, 'step%d' % iter)
                else:
                    # segment_maps is list
                    segment_maps = sess.run(
                        seg_map_list, feed_dict={path_ph: file_name})

                para_list=[(config['threshold_kernel'],config['threshold'],config['aver_score'])] if para_list==None else para_list
                
                pbar_para = tqdm.tqdm(total=len(para_list))
                for index,para in enumerate(para_list):
                    pbar_para.update(1)
                    config['threshold_kernel'],config['threshold'],config['aver_score']=para[0],para[1],para[2]
                    config['id']=0 if len(para_list)==1 else index+1
                    infer_path = util.io.join_path(
                        dump_path, chp_name+'_'+str(config['id']))

                    txt_path = util.io.join_path(infer_path, 'txt_result')
                    zip_path = util.io.join_path(infer_path, 'detect.zip')
                    imgs_path = util.io.join_path(infer_path, 'image_log')

                    result_map = process_map(
                        segment_maps, config['threshold_kernel'], config['threshold'])

                    bboxes,scores = map_to_bboxes(
                        segment_maps, result_map, image_size, aver_score=config['aver_score'])

                    if is_log:
                        log_to_file(imgs_path,file_name,image_arr,bboxes,segment_maps)
                    
                    write_to_file(bboxes, image_name, txt_path)
                pbar_para.close()
                # plt.show()
            pbar.close()

    # ====================================
    # Logging .....

    for index,para in enumerate(para_list):
        config['threshold_kernel'],config['threshold'],config['aver_score']=para[0],para[1],para[2]
        config['id']=config['id'] if len(para_list)==1 else index+1
        infer_path = util.io.join_path(
            dump_path, chp_name+'_'+str(config['id']))

        txt_path = util.io.join_path(infer_path, 'txt_result')
        zip_path = util.io.join_path(infer_path, 'detect.zip')

        flags_log = tf.app.flags.FLAGS.flag_values_dict()
        with open(os.path.join(infer_path, 'flags.json'), 'w') as f:
            json.dump(flags_log, f, indent=2)
        with open(os.path.join(infer_path, 'config.json'), 'w') as f:
            json.dump(config, f, indent=2)

        cmd = 'cd %s;zip -j %s %s/*' % (infer_path,
                                        os.path.basename(zip_path), 'txt_result')
        zip_path=util.io.get_absolute_path(zip_path)
        infer_path=util.io.get_absolute_path(infer_path)
        # print(cmd)
        util.cmd.cmd(cmd)
        # print("zip file created: ", zip_path)

        os.chdir('./metric')
        para = {'g': 'gt.zip',
                's': zip_path,
                'o': infer_path}
        import script
        func_name = 'script.eval(para)'
        try:
            res = eval(func_name)
            os.chdir('../')
        except:
            print('eval error!')
            os.chdir('../')
        with open(os.path.join(infer_path, 'result.json'), 'w') as f:
            json.dump(res, f, indent=2)
