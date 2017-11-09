# Copyright 2017 Yahoo Inc.
# Licensed under the terms of the Apache 2.0 license.
# Please see LICENSE file in the project root for terms.

# Distributed MNIST on grid based on TensorFlow MNIST example

from __future__ import absolute_import
from __future__ import division
from __future__ import nested_scopes
from __future__ import print_function

def print_log(worker_num, arg):
  print("{0}: {1}".format(worker_num, arg))

def map_fun(args, ctx):
  from tensorflowonspark import TFNode
  from datetime import datetime
  import math
  import numpy
  import tensorflow as tf
  import time
  import re

  worker_num = ctx.worker_num
  job_name = ctx.job_name
  task_index = ctx.task_index
  cluster_spec = ctx.cluster_spec
  NUM_CLASSES = 100
  IMAGE_PIXELS=32
  NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 50000
  NUM_EPOCHS_PER_DECAY = 350.0      # Epochs after which learning rate decays.
  LEARNING_RATE_DECAY_FACTOR = 0.1  # Learning rate decay factor.
  INITIAL_LEARNING_RATE = 0.1       # Initial learning rate.
  TOWER_NAME = 'tower'

  # Delay PS nodes a bit, since workers seem to reserve GPUs more quickly/reliably (w/o conflict)
  if job_name == "ps":
    time.sleep((worker_num + 1) * 5)

  # Parameters
  hidden_units = 128
  batch_size   = args.batch_size

  # Get TF cluster and server instances
  cluster, server = TFNode.start_cluster_server(ctx, 1, args.rdma)
  

  def feed_dict(batch):
    # Convert from [(images, labels)] to two numpy arrays of the proper type
    images = []
    labels = []
    for item in batch:
      images.append(item[0])
      labels.append(item[1])
    xs = numpy.array(images)
    xs = xs.astype(numpy.float32)
    xs = xs/255.0
    ys = numpy.array(labels)
    ys = ys.astype(numpy.uint8)
    return (xs, ys)

  if job_name == "ps":
    server.join()
  elif job_name == "worker":

    # Assigns ops to the local worker by default.
    with tf.device(tf.train.replica_device_setter(
        worker_device="/job:worker/task:%d" % task_index,
        cluster=cluster)):

      print("In a TFCluster.")
#      global_step = tf.train.get_or_create_global_step()
      # Input placeholders
      with tf.name_scope('input'):
        x = tf.placeholder(tf.float32, [None, IMAGE_PIXELS*IMAGE_PIXELS*3], name='x-input')
        y_ = tf.placeholder(tf.float32, [None, 100], name='y-input') 
        images = tf.reshape(x, [-1, IMAGE_PIXELS, IMAGE_PIXELS, 3])
        print (images.shape)
        tf.summary.image('input', images, 10)
        
      def _activation_summary(x):
        """Helper to create summaries for activations.
        Creates a summary that provides a histogram of activations.
        Creates a summary that measures the sparsity of activations.
        Args:
          x: Tensor
        Returns:
          nothing
        """
        # Remove 'tower_[0-9]/' from the name in case this is a multi-GPU training
        # session. This helps the clarity of presentation on tensorboard.
        tensor_name = re.sub('%s_[0-9]*/' % TOWER_NAME, '', x.op.name)
        tf.summary.histogram(tensor_name + '/activations', x)
        tf.summary.scalar(tensor_name + '/sparsity',
                                           tf.nn.zero_fraction(x))
      
      def _variable_on_cpu(name, shape, initializer):
        """Helper to create a Variable stored on CPU memory.
        Args:
          name: name of the variable
          shape: list of ints
          initializer: initializer for Variable
        Returns:
          Variable Tensor
        """
        with tf.device('/cpu:0'):
          dtype = tf.float32
          var = tf.get_variable(name, shape, initializer=initializer, dtype=dtype)
        return var

      def _variable_with_weight_decay(name, shape, stddev, wd):
        """Helper to create an initialized Variable with weight decay.
        Note that the Variable is initialized with a truncated normal distribution.
        A weight decay is added only if one is specified.
        Args:
          name: name of the variable
          shape: list of ints
          stddev: standard deviation of a truncated Gaussian
          wd: add L2Loss weight decay multiplied by this float. If None, weight
                decay is not added for this Variable.
        Returns:
          Variable Tensor
        """
        dtype = tf.float32
        var = _variable_on_cpu(
              name,
              shape,
              tf.truncated_normal_initializer(stddev=stddev, dtype=dtype))
        if wd is not None:
          weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
          tf.add_to_collection('losses', weight_decay)
        return var
        
      with tf.variable_scope('conv1') as scope:
        kernel = _variable_with_weight_decay('weights',
                                             shape=[5, 5, 3, 256],
                                             stddev=5e-2,
                                             wd=0.0)
        conv = tf.nn.conv2d(images, kernel, [1, 1, 1, 1], padding='SAME')
        biases = _variable_on_cpu('biases', [256], tf.constant_initializer(0.0))
        pre_activation = tf.nn.bias_add(conv, biases)
        conv1 = tf.nn.relu(pre_activation, name=scope.name)
        _activation_summary(conv1)

      # pool1
      pool1 = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
                             padding='SAME', name='pool1')
      # norm1
      norm1 = tf.nn.lrn(pool1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
                        name='norm1')

      # conv2
      with tf.variable_scope('conv2') as scope:
        kernel = _variable_with_weight_decay('weights',
                                             shape=[5, 5, 256, 128],
                                             stddev=5e-2,
                                             wd=0.0)
        conv = tf.nn.conv2d(norm1, kernel, [1, 1, 1, 1], padding='SAME')
        biases = _variable_on_cpu('biases', [128], tf.constant_initializer(0.1))
        pre_activation = tf.nn.bias_add(conv, biases)
        conv2 = tf.nn.relu(pre_activation, name=scope.name)
        _activation_summary(conv2)

      # norm2
      norm2 = tf.nn.lrn(conv2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
                        name='norm2')
      # pool2
      pool2 = tf.nn.max_pool(norm2, ksize=[1, 3, 3, 1],
                             strides=[1, 2, 2, 1], padding='SAME', name='pool2')

      # local3
      with tf.variable_scope('local3') as scope:
        # Move everything into depth so we can perform a single matrix multiply.
        reshape = tf.contrib.layers.flatten(pool2)
        dim = reshape.get_shape()[1].value
        weights = _variable_with_weight_decay('weights', shape=[dim, 1024],
                                              stddev=0.04, wd=0.004)
        biases = _variable_on_cpu('biases', [1024], tf.constant_initializer(0.1))
        local3 = tf.nn.relu(tf.matmul(reshape, weights) + biases, name=scope.name)
        _activation_summary(local3)

      # local4
      with tf.variable_scope('local4') as scope:
        weights = _variable_with_weight_decay('weights', shape=[1024, 256],
                                              stddev=0.04, wd=0.004)
        biases = _variable_on_cpu('biases', [256], tf.constant_initializer(0.1))
        local4 = tf.nn.relu(tf.matmul(local3, weights) + biases, name=scope.name)
        _activation_summary(local4)

      # linear layer(WX + b),
      # We don't apply softmax here because
      # tf.nn.sparse_softmax_cross_entropy_with_logits accepts the unscaled logits
      # and performs the softmax internally for efficiency.
      with tf.variable_scope('softmax_linear') as scope:
        weights = _variable_with_weight_decay('weights', [256, NUM_CLASSES],
                                              stddev=1/256.0, wd=0.0)
        biases = _variable_on_cpu('biases', [NUM_CLASSES],
                                  tf.constant_initializer(0.0))
        softmax_linear = tf.add(tf.matmul(local4, weights), biases, name=scope.name)
        _activation_summary(softmax_linear)
        
      logits = softmax_linear
          
      # Calculate the average cross entropy loss across the batch.
#      labels = tf.reshape(y_, [100, 10])
      print (y_.shape)
      print (logits.shape)
      labels = tf.cast(y_, tf.int64)
      cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
          labels=labels, logits=logits, name='cross_entropy_per_example')
      cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
      tf.add_to_collection('losses', cross_entropy_mean)

      # The total loss is defined as the cross entropy loss plus all of the weight
      # decay terms (L2 loss).
      total_loss = tf.add_n(tf.get_collection('losses'), name='total_loss')
      global_step = tf.Variable(0)
      inc = tf.assign_add(global_step, 1, name='increment')
#      num_batches_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN / batch_size
#      decay_steps = int(num_batches_per_epoch * NUM_EPOCHS_PER_DECAY)

      # Decay the learning rate exponentially based on the number of steps.
#      lr = tf.train.exponential_decay(INITIAL_LEARNING_RATE,
#                                  global_step,
#                                  decay_steps,
#                                  LEARNING_RATE_DECAY_FACTOR,
#                                  staircase=True)
#      tf.summary.scalar('learning_rate', lr)
      
      train_step = tf.train.AdamOptimizer(1e-4).minimize(total_loss)
      correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(y_, 1))
      accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
      label = tf.argmax(y_, 1, name="label")
      prediction = tf.argmax(logits, 1,name="prediction")  
      
          


##########################################################


      # Merge all the summaries and write them out to
      # /tmp/tensorflow/mnist/logs/mnist_with_summaries (by default)
      merged = tf.summary.merge_all()
      
#      saver = tf.train.Saver()
      init_op = tf.global_variables_initializer()

    # Create a "supervisor", which oversees the training process and stores model state into HDFS
#    logdir = TFNode.hdfs_path(ctx, args.model)
    logdir = "/tmp/" + args.model
    print("tensorflow model path: {0}".format(logdir))
    summary_writer = tf.summary.FileWriter("tensorboard_%d" %(worker_num), graph=tf.get_default_graph())

    if args.mode == "train":
      sv = tf.train.Supervisor(is_chief=(task_index == 0),
                               logdir=logdir,
                               init_op=init_op,
                               summary_op=None,
                               summary_writer=summary_writer,
                               global_step=global_step,
                               stop_grace_secs=300,
                               saver = None
#                               save_model_secs=10
                               )
    else:
      sv = tf.train.Supervisor(is_chief=(task_index == 0),
                               logdir=logdir,
                               summary_op=None,
                               saver=saver,
                               global_step=global_step,
                               stop_grace_secs=300,
                               save_model_secs=0)

    # The supervisor takes care of session initialization, restoring from
    # a checkpoint, and closing when done or an error occurs.
    with sv.managed_session(server.target) as sess:
      print("{0} session ready".format(datetime.now().isoformat()))
      # Loop until the supervisor shuts down or 1000000 steps have completed.
      step = -1
      tf_feed = TFNode.DataFeed(ctx.mgr, args.mode == "train")
      tf_feed_test = TFNode.DataFeed(ctx.mgr, args.mode != "train")
      while step < args.steps:
        # Run a training step asynchronously.
        # See `tf.train.SyncReplicasOptimizer` for additional details on how to
        # perform *synchronous* training.
#        print (args.steps)
#        print (sv.should_stop())
#        print (tf_feed.should_stop())
        step = step + 1
#        print (step)
        temp = sess.run(global_step)
#        print (temp)
        # using feed_dict
        batch_xs, batch_ys = feed_dict(tf_feed.next_batch(batch_size))
        test_xs, test_ys = feed_dict(tf_feed_test.next_batch(batch_size))
        feed = {x: batch_xs, y_: batch_ys}

        print (len(batch_xs) > 0)
        if len(batch_xs) > 0:
          if args.mode == "train":
            summary, _,_ = sess.run([merged, train_step, inc], feed_dict=feed)
            # print accuracy and save model checkpoint to HDFS every 100 steps
            if (step % 100 == 0):
              labels, preds, acc = sess.run([label, prediction, accuracy], feed_dict={x: test_xs, y_: test_ys})
              for l,p in zip(labels,preds):
                print("{0} step: {1} accuracy: {2}, Label: {3}, Prediction: {4}".format(datetime.now().isoformat(), temp, acc, l, p))
              
#              results = ["{0} Label: {1}, Prediction: {2}".format(datetime.now().isoformat(), l, p) for l,p in zip(labels,preds)]
#              tf_feed.batch_results(results)

            if sv.is_chief:
              summary_writer.add_summary(summary, step)
          else: # args.mode == "inference"
            labels, preds, acc = sess.run([label, prediction, accuracy], feed_dict=feed)

            results = ["{0} Label: {1}, Prediction: {2}".format(datetime.now().isoformat(), l, p) for l,p in zip(labels,preds)]
            tf_feed.batch_results(results)
            print("acc: {0}".format(acc))

      if sv.should_stop() or step >= args.steps:
        tf_feed.terminate()

    # Ask for all the services to stop.
    print("{0} stopping supervisor".format(datetime.now().isoformat()))
    sv.stop()