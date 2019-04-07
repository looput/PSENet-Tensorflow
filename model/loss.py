#! /usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright © 2018 loop     Huazhong University of Science and Technology

import tensorflow as tf


from net import resnet_v1
import configuration

config=configuration.TRAIN_CONFIG

            
def cal_dice_loss(pred,gt):
    union=tf.reduce_sum(tf.multiply(pred,gt),[1,2])
    pred_square=tf.reduce_sum(tf.square(pred),[1,2])
    gt_square=tf.reduce_sum(tf.square(gt),[1,2])

    dice_loss=1.-(2*union+1e-5)/(pred_square+gt_square+1e-5)

    # dice_loss=tf.Print(dice_loss,[gt_square],message='gt_square: ',summarize=5)
    return dice_loss

def loss(pred_seg_maps,gt_map,kernels,training_mask):
    '''
    L = λLc + (1 − λ)Ls
    where Lc and Ls represent the losses for the complete text instances and the shrunk ones respec- tively, 
    and λ balances the importance between Lc and Ls
    It is common that the text instances usually occupy only an extremely small region in natural images,
    which makes the predictions of network bias to the non-text region, 
    when binary cross entropy [2] is used. Inspired by [20], 
    we adopt dice coefficient in our experiment. 
    The dice coefficient D(Si, Gi) is formulated as in Eqn
    '''
    with tf.name_scope('Loss'):
        
        n=len(config['rate'])+1
        # for complete loss
        pred_text_map=pred_seg_maps[:,0,:,:]

        # NOTE: the mask is pred_map, may try gt_map?
        mask=tf.to_float(tf.greater(pred_text_map*training_mask,0.5))
        pred_text_map=pred_text_map*training_mask
        gt_map=gt_map*training_mask

        def online_hard_min(maps):
            pred_map,gt_map=maps

            # NOTE: OHM 3    
            pos_mask = tf.cast(tf.equal(gt_map, 1.),dtype=tf.float32)     # [h,w,1]
            neg_mask = tf.cast(tf.equal(gt_map, 0.),dtype=tf.float32)
            n_pos=tf.reduce_sum((pos_mask),[0,1])

            neg_val_all = tf.boolean_mask(pred_map, neg_mask)       # [N]
            n_neg=tf.minimum(tf.shape(neg_val_all)[-1],tf.cast(n_pos*3,tf.int32))
            n_neg=tf.cond(tf.greater(n_pos,0),lambda:n_neg,lambda:tf.shape(neg_val_all)[-1])
            neg_hard, neg_idxs = tf.nn.top_k(neg_val_all, k=n_neg)       #[batch_size,k][batch_size, k]
            # TODO ERROR  slice index -1 of dimension 0 out of bounds.
            neg_min=tf.cond(tf.greater(tf.shape(neg_hard)[-1],0),lambda:neg_hard[-1],lambda:1.)      # [k]

            neg_hard_mask=tf.cast(tf.greater_equal(pred_map,neg_min),dtype=tf.float32)
            pred_ohm=pos_mask*pred_map+neg_hard_mask*neg_mask*pred_map
            return pred_ohm,gt_map

        
        if config['OHM']:
            pred_maps,gt_maps=tf.map_fn(online_hard_min,(pred_text_map,gt_map))  
        else:
            pred_maps,gt_maps=pred_text_map,gt_map
        ohm_dice_loss=cal_dice_loss(pred_maps,gt_maps)

        dice_loss=tf.reduce_mean(ohm_dice_loss)
        tf.add_to_collection('losses', 0.7*dice_loss)

        for i,_ in enumerate(config['rate']):
            # for shrink loss
            pred_map=pred_seg_maps[:,i+1,:,:]
            gt_map=kernels[:,i,:,:]
            
            pred_map=pred_map*mask
            gt_map=gt_map*mask

            dice_loss=cal_dice_loss(pred_map,gt_map)
            dice_loss=tf.reduce_mean(dice_loss)
            # NOTE the paper is divide Ls by (n-1), I don't divide this for long time
            tf.add_to_collection('losses', (1-0.7)*dice_loss/(n-1))
        
    # The total loss is defined as the cross entropy loss plus all of the weight
    # decay terms (L2 loss).
    return tf.add_n(tf.get_collection('losses'), name='total_loss')


# TODO: how to test this FPN net work prople
if __name__ == '__main__':
    # test this unit
    import numpy as np

    test_input = tf.Variable(initial_value=tf.ones(
        (5, 224, 224, 3), tf.float32))
    output, f = model(test_input)

    init_op = tf.global_variables_initializer()

    restore = slim.assign_from_checkpoint_fn(
        "resnet_v1_50.ckpt", slim.get_trainable_variables(), ignore_missing_vars=True)
    with tf.Session() as sess:
        sess.run(init_op)
        restore(sess)
        out, f_res = sess.run([output, f])
        print(np.sum(f_res[0]))
