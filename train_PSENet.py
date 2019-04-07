# TODO: use class to ornagize code

import json
import os
import re
import time
import datetime

import numpy as np
import tensorflow as tf
from tensorflow.python.client import timeline

import util
from configuration import TRAIN_CONFIG
from dataset.dataloader import DataLoader
from model.loss import loss
from model.model_v2 import model,model_deconv

slim = tf.contrib.slim

tf.app.flags.DEFINE_boolean(name='use_pretrain', short_name='up',default=False, help='')
tf.app.flags.DEFINE_boolean(name='restore',short_name='rs',default=False, help='')
tf.app.flags.DEFINE_string(name='run_name',short_name='rn', default='', help='')
tf.app.flags.DEFINE_string(name='gpus', short_name='g',default='', help='')

tf.app.flags.DEFINE_integer(name='summary_step',short_name='ss', default=200, help='')
tf.app.flags.DEFINE_integer(name='save_epo',short_name='se', default=20, help='')

tf.app.flags.DEFINE_string('about','','')


FLAGS = tf.app.flags.FLAGS
SUMMARY_STEP=FLAGS.ss
SAVE_EPO=FLAGS.se

os.environ['TF_ENABLE_WINOGRAD_NONFUSED'] = '1' # For speed up convolution
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpus
num_gpu=len(FLAGS.gpus.split(','))
checkpoint_path=os.path.join(TRAIN_CONFIG['log_dir'],'train',FLAGS.run_name)

def tower_loss(scope, images, labels,kernals,training_mask):
    # Build inference Graph.
    pred_gts,_ = model(images)

    # Build the portion of the Graph calculating the losses. Note that we will
    # assemble the total_loss using a custom function below.
    _ = loss(pred_gts, labels,kernals,training_mask)

    # Assemble all of the losses for the current tower only.
    losses = tf.get_collection('losses', scope)

    # Calculate the total loss for the current tower.
    total_loss = tf.add_n(losses, name='total_loss')

    # Remove 'tower_[0-9]/' from the name in case this is a multi-GPU training
    # session. This helps the clarity of presentation on tensorboard.
    loss_name = re.sub('%s_[0-9]*/' % 'TOWER', '', total_loss.op.name)
    tf.summary.scalar(loss_name, total_loss)

    tf.summary.image(re.sub('%s_[0-9]*/' % 'TOWER','',pred_gts.op.name),tf.expand_dims(pred_gts[:,0,:,:],-1))
    tf.summary.image(re.sub('%s_[0-9]*/' % 'TOWER','',labels.op.name), tf.expand_dims(labels[:,:,:],-1))
    for i in range(len(TRAIN_CONFIG['rate'])):    
        tf.summary.image(re.sub('%s_[0-9]*/' % 'TOWER','%d_'%i,pred_gts.op.name),tf.expand_dims(pred_gts[:,i+1,:,:],-1))
        tf.summary.image(re.sub('%s_[0-9]*/' % 'TOWER','%d_'%i,labels.op.name), tf.expand_dims(kernals[:,i,:,:],-1))
    tf.summary.image(re.sub('%s_[0-9]*/' % 'TOWER','',images.op.name), images)
    tf.summary.image(re.sub('%s_[0-9]*/' % 'TOWER','',training_mask.op.name), tf.expand_dims(training_mask,-1))
    return total_loss


def average_gradients(tower_grads):
    """Calculate the average gradient for each shared variable across all towers.
    Note that this function provides a synchronization point across all towers.
    Args:
      tower_grads: List of lists of (gradient, variable) tuples. The outer list
        is over individual gradients. The inner list is over the gradient
        calculation for each tower.
    Returns:
       List of pairs of (gradient, variable) where the gradient has been averaged
       across all towers.
    """
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        # Note that each grad_and_vars looks like the following:
        #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
        grads = []
        for g, _ in grad_and_vars:
            # Add 0 dimension to the gradients to represent the tower.
            expanded_g = tf.expand_dims(g, 0)

            # Append on a 'tower' dimension which we will average over below.
            grads.append(expanded_g)

        # Average over the 'tower' dimension.
        grad = tf.concat(axis=0, values=grads)
        grad = tf.reduce_mean(grad, 0)

        # Keep in mind that the Variables are redundant because they are shared
        # across towers. So .. we will just return the first tower's pointer to
        # the Variable.
        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)
    return average_grads


def main(argv=None):
    config=TRAIN_CONFIG

    with tf.device('/cpu:0'):
        dataloader = DataLoader(config['data_name'],config['batch_size']/num_gpu)
        data_size=dataloader.data_size

    # NOTE: global_step is a special variable, can't create by below
    # global_step = tf.Variable(0,trainable=False)
    global_step=tf.train.create_global_step()
    # NOTE: change lr accodring to epoch
    lr_config=TRAIN_CONFIG['lr_config']
    num_batches_per_epoch = \
        int(data_size['train'] / TRAIN_CONFIG['batch_size'])
    lr_boundaries = [int(e * num_batches_per_epoch) for e in lr_config['lr_boundaries']]
    lr=tf.train.piecewise_constant(global_step,lr_boundaries,lr_config['lr_values'])
    with tf.name_scope('optimter'):
        opt = tf.train.MomentumOptimizer(
            learning_rate=lr, momentum=0.99)
    tf.summary.scalar('learning_rate',lr)
    # multi GPUs reference from tf/model/cifer10
    # Calculate the gradients for each model tower.
    tower_grads = []
    reuse_variables=None
    # FIXME when use multi GPU, crashed with segment fault
    with tf.variable_scope(tf.get_variable_scope()):
        for i in range(num_gpu):
            with tf.device('/gpu:%d' % i):
                print('/gpu:%d' % int(FLAGS.gpus.split(',')[i]))
                with tf.name_scope('%s_%d' % ('TOWER', i)) as scope:
                    # Dequeues one batch for the GPU
                    image,gt_text,gt_kernals,training_mask = dataloader.load_data()

                    loss = tower_loss(scope, image, gt_text,gt_kernals,training_mask)
                    # Reuse variables for the next tower.
                    tf.get_variable_scope().reuse_variables()

                    # Retain the summaries from the final tower.
                    summaries = tf.get_collection(tf.GraphKeys.SUMMARIES, scope)

                    # just for summary
                    loss_sum=loss if i==0 else (loss_sum+loss)

                    # gather regularization loss and add to tower_0 only
                    if i == 0:
                        # regularization_loss = tf.add_n(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
                        # If no loss_filter_fn is passed, assume we want the default behavior,
                        # which is that batch_normalization variables are excluded from loss.
                        def exclude_batch_norm(name):
                            return 'batch_normalization' not in name

                        weight_decay=1e-5
                        # Add weight decay to the loss.
                        l2_loss = weight_decay * tf.add_n(
                            # loss is computed using fp32 for numerical stability.
                            [tf.nn.l2_loss(tf.cast(v, tf.float32)) for v in tf.trainable_variables()
                             if exclude_batch_norm(v.name)])

                        loss = loss + l2_loss
                        tf.summary.scalar('regularztion_loss',loss)

                    # Calculate the gradients for the batch of data on this CIFAR tower.
                    grads = opt.compute_gradients(loss)

                    # Keep track of the gradients across all towers.
                    tower_grads.append(grads)

    tf.summary.scalar('toatal_loss',loss_sum/int(num_gpu))
    # We must calculate the mean of each gradient. Note that this is the
    # synchronization point across all towers.
    grads = average_gradients(tower_grads)

    # Add histograms for gradients.
    for grad, var in grads:
        if grad is not None:
            summaries.append(tf.summary.histogram(
                var.op.name + '/gradients', grad))
    
    apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

    # tf.logging.info('using moving average in training, \
    #     with decay = %f'%(FLAGS.moving_average_decay))
    ema = tf.train.ExponentialMovingAverage(0.9999,global_step)
    ema_op = ema.apply(tf.trainable_variables())
    ema_up_op=tf.group(ema_op)

    # TODO what if multi GPU used ??
    update_ops=tf.get_collection(tf.GraphKeys.UPDATE_OPS)   
    batch_norm_updates_op = tf.group(*update_ops)
    with tf.control_dependencies([ema_up_op,apply_gradient_op, batch_norm_updates_op]):
        train_op = tf.no_op(name='train_op')

    if FLAGS.restore==False and FLAGS.use_pretrain == True:
        restore_fn = slim.assign_from_checkpoint_fn(
            "Logs/model/resnet_imagenet/model.ckpt-225207", slim.get_trainable_variables(), ignore_missing_vars=True)

    saver = tf.train.Saver(max_to_keep=30,keep_checkpoint_every_n_hours=2.5)
    summary_op = tf.summary.merge_all()
    init_op = tf.global_variables_initializer()

    tfconfig = tf.ConfigProto(allow_soft_placement=True)
    # tfconfig.gpu_options.allow_growth = True
    # config.gpu_options.per_process_gpu_memory_fraction = 0.4

    
    with tf.Session(config=tfconfig) as sess:
        # for key, value in tf.flags.FLAGS.__flags.items():
        flags_log=tf.app.flags.FLAGS.flag_values_dict()
        trans_var = tf.get_collection('transform')
        init_transform = tf.variables_initializer(trans_var)
        sess.run(init_transform)
        if FLAGS.restore:

            ckpt = tf.train.latest_checkpoint(checkpoint_path)
            saver.restore(sess,ckpt)
            start_step = global_step.eval()+1
            print('continue training from previous checkpoint, start step: %d'%start_step)

            with open(os.path.join(checkpoint_path,'%s-config.json'%datetime.datetime.now().strftime('%m_%d-%H_%M')), 'w') as f:
                log_json={"flags":flags_log,"config":config}
                json.dump(log_json,f, indent=2)
        else:
            # delete the run log dir
            if tf.gfile.Exists(checkpoint_path):
                tf.gfile.DeleteRecursively(os.path.abspath(checkpoint_path))
            tf.gfile.MkDir(checkpoint_path)

            with open(os.path.join(checkpoint_path, '%s-config.json'%datetime.datetime.now().strftime('%m_%d-%H_%M')), 'w') as f:
                log_json={"flags":flags_log,"config":config}
                json.dump(log_json,f, indent=2)

            sess.run(init_op)
            start_step = 0
            if FLAGS.use_pretrain == True:
                restore_fn(sess)

        # NOTE: if not start the queue, no data will be generate!
        # REF: https://programtalk.com/python-examples/tensorflow.contrib.slim.prefetch_queue.prefetch_queue/
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        sum_writer = tf.summary.FileWriter(
            checkpoint_path, graph=None)

        run_opts = tf.RunOptions(report_tensor_allocations_upon_oom = True)
        start = time.time()
        for step in range(start_step, int(1200*data_size['train']/TRAIN_CONFIG['batch_size'])):
            # for every 20 epoch and 5000 step to save model
            if step%(num_batches_per_epoch*SAVE_EPO)==num_batches_per_epoch*SAVE_EPO-1:
                print("save the model at step:%d!"%step)
                _ = sess.run(train_op)
                saver.save(sess, os.path.join(checkpoint_path, 'model.ckpt'),
                            global_step=step)

            elif step % SUMMARY_STEP == SUMMARY_STEP-1:
                avg_time_per_step = (time.time() - start)/SUMMARY_STEP
                avg_examples_per_second = (SUMMARY_STEP * config['batch_size'])/(time.time() - start)
                start = time.time()
                
                if TRAIN_CONFIG['profile']==True:
                    print('Profile the net ......')
                    # NOTE: useful func, analysis tf graph run time
                    run_options = tf.RunOptions(
                        trace_level=tf.RunOptions.FULL_TRACE)
                    run_metadata = tf.RunMetadata()

                    _,loss_s,summary = sess.run(
                        [train_op,loss_sum,summary_op], options=run_options, run_metadata=run_metadata)

                    # for time analysis
                    trace = timeline.Timeline(step_stats=run_metadata.step_stats)
                    with open('timeline-placeholder.ctf.json', 'w') as trace_file:
                        trace_file.write(trace.generate_chrome_trace_format())
                    # add mermory and time info in graph
                    sum_writer.add_run_metadata(run_metadata, 'step%d' % step)
                else:
                    _,loss_s,summary = sess.run(
                        [train_op,loss_sum,summary_op])

                sum_writer.add_summary(summary, global_step=step)
                print('step:{} epcho: {}, loss value: {}, {:.2f} seconds/step, {:.2f} examples/second'.format(
                    step, step/num_batches_per_epoch, loss_s/num_gpu,avg_time_per_step,avg_examples_per_second))
            
            elif step%10==9:
                _,loss_s = sess.run(
                        [train_op,loss_sum],options=run_opts)
                print('step:{} epcho: {}, loss value: {}'.format(
                    step, step/num_batches_per_epoch, loss_s/num_gpu))
            else:
                _ = sess.run(train_op,options=run_opts)


if __name__ == '__main__':
    tf.app.run()
