#! /usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright © 2018 loop     Huazhong University of Science and Technology

import os

import tensorflow as tf
import numpy as np

import tqdm
from model import inference
from configuration import TEST_CONFIG

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

tf.app.flags.DEFINE_string('train_name', '', '')
tf.app.flags.DEFINE_string('gpus', '', '')
tf.app.flags.DEFINE_bool('using_moving_average', short_name='ma',default=True,help='')
tf.app.flags.DEFINE_bool('loging', short_name='lg',default=False,help='')

# TEST_DIR='/workspace/lupu/icdar2015/ch4_training_images'

FLAGS = tf.app.flags.FLAGS

os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpus

def _set_shape(config,size=(768,1280)):
    config['image_size']={
        'h':size[0],
        'w':size[1]
    }

def _set_para(config,kn=0.55,th=0.6,avs=0.9):
    config['threshold_kernel']=kn
    config['threshold']=th
    config['aver_score']=avs

def main(argv=None):
    import crash_on_ipy 
    config = TEST_CONFIG

    checkpoint_path = os.path.join(
        TEST_CONFIG['log_dir'], 'train', FLAGS.train_name)
    ckpts = [tf.train.latest_checkpoint(checkpoint_path)]
    # ckpts = tf.train.get_checkpoint_state(
    #     checkpoint_path).all_model_checkpoint_paths
    print(ckpts)
    # ckpts=[os.path.join(checkpoint_path,'model.ckpt-32499')]

    para_list=[]
    for score in (np.linspace(0.8,0.9,num=1+1)):
        for th_kn in (np.linspace(0.5,0.6,num=10+1)):
            for th in (np.linspace(0.5,0.6,num=10+1)):
                para_list.append((th_kn,th,score))

    for iter,ckpt in enumerate(ckpts):
        print('==========run progrss {} / {}============'.format(iter,len(ckpts)))
        config['ckpt'] = ckpt
        config['id'] = '-1'

        inference.eval_model(config, FLAGS,para_list=None,is_log=FLAGS.lg)


if __name__ == '__main__':
    tf.app.run()

'''r the test log dir struct
    Logs/test
        /--{train_name}
            /--model_{step}_{p}
                /--txt_reslut
                    image_001.txt
                    ...
                detect.zip
                result.zip
                test_PSENET.py
                para.json
            /--model_{step}_{p}
                ...
        /--{train_name}
            /--...
'''
