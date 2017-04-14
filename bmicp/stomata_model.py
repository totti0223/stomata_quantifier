#!/usr/bin/env python
# -*- coding: utf-8 -*-

import tensorflow as tf
from math import sqrt
MOVING_AVERAGE_DECAY = 0.9999


def tf_inference(images, BATCH_SIZE, image_size, NUM_CLASSES):
    def put_kernels_on_grid (kernel, pad = 1):
        #https://gist.github.com/kukuruza/03731dc494603ceab0c5
        '''Visualize conv. features as an image (mostly for the 1st layer).
        Place kernel into a grid, with some paddings between adjacent filters.
        Args:
          kernel:            tensor of shape [Y, X, NumChannels, NumKernels]
          (grid_Y, grid_X):  shape of the grid. Require: NumKernels == grid_Y * grid_X
                               User is responsible of how to break into two multiples.
          pad:               number of black pixels around each filter (between them)
        Return:
          Tensor of shape [(Y+2*pad)*grid_Y, (X+2*pad)*grid_X, NumChannels, 1].
        '''
        # get shape of the grid. NumKernels == grid_Y * grid_X
        def factorization(n):
            for i in range(int(sqrt(float(n))), 0, -1):
                if n % i == 0:
                    if i == 1: print('Who would enter a prime number of filters')
                    return (i, int(n / i))
        (grid_Y, grid_X) = factorization (kernel.get_shape()[3].value)
        #print ('grid: %d = (%d, %d)' % (kernel.get_shape()[3].value, grid_Y, grid_X))

        x_min = tf.reduce_min(kernel)
        x_max = tf.reduce_max(kernel)

        kernel1 = (kernel - x_min) / (x_max - x_min)

        # pad X and Y
        x1 = tf.pad(kernel1, tf.constant( [[pad,pad],[pad, pad],[0,0],[0,0]] ), mode = 'CONSTANT')

        # X and Y dimensions, w.r.t. padding
        Y = kernel1.get_shape()[0] + 2 * pad
        X = kernel1.get_shape()[1] + 2 * pad

        channels = kernel1.get_shape()[2]

        # put NumKernels to the 1st dimension
        x2 = tf.transpose(x1, (3, 0, 1, 2))
        # organize grid on Y axis
        x3 = tf.reshape(x2, tf.pack([grid_X, Y * grid_Y, X, channels]))

        # switch X and Y axes
        x4 = tf.transpose(x3, (0, 2, 1, 3))
        # organize grid on X axis
        x5 = tf.reshape(x4, tf.pack([1, X * grid_X, Y * grid_Y, channels]))

        # back to normal order (not combining with the next step for clarity)
        x6 = tf.transpose(x5, (2, 1, 3, 0))

        # to tf.image_summary order [batch_size, height, width, channels],
        #   where in this case batch_size == 1
        x7 = tf.transpose(x6, (3, 0, 1, 2))

        # scaling to [0, 255] is not necessary for tensorboard
        return x7

    def _variable_with_weight_decay(name, shape, stddev, wd):
        var = tf.get_variable(name, shape=shape, initializer=tf.truncated_normal_initializer(stddev=stddev))
        if wd:
            weight_decay = tf.mul(tf.nn.l2_loss(var), wd, name='weight_loss')
            tf.add_to_collection('losses', weight_decay)
        return var

    def _activation_summary(x):
        tensor_name = x.op.name
        tf.scalar_summary(tensor_name + '/sparsity', tf.nn.zero_fraction(x))

    with tf.variable_scope('conv1') as scope:
        kernel = tf.get_variable('weights', shape=[3, 3, 3, 32], initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv = tf.nn.conv2d(images, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.get_variable('biases', shape=[32], initializer=tf.constant_initializer(0.0))
        bias = tf.nn.bias_add(conv, biases)
        conv1 = tf.nn.relu(bias, name=scope.name)
        _activation_summary(conv1)
        grid = put_kernels_on_grid (kernel)
        tf.image_summary('conv1/features', grid, max_images=1)
    pool1 = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool1')

    with tf.variable_scope('conv2') as scope:
        kernel = tf.get_variable('weights', shape=[3, 3, 32, 64], initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv = tf.nn.conv2d(pool1, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.get_variable('biases', shape=[64], initializer=tf.constant_initializer(0.0))
        bias = tf.nn.bias_add(conv, biases)
        conv2 = tf.nn.relu(bias, name=scope.name)
        _activation_summary(conv2)
    pool2 = tf.nn.max_pool(conv2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool2')

    with tf.variable_scope('conv3') as scope:
        kernel = tf.get_variable('weights', shape=[3, 3, 64, 128], initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv = tf.nn.conv2d(pool2, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.get_variable('biases', shape=[128], initializer=tf.constant_initializer(0.0))
        bias = tf.nn.bias_add(conv, biases)
        conv3 = tf.nn.relu(bias, name=scope.name)
        _activation_summary(conv3)
    pool3 = tf.nn.max_pool(conv3, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool3')

    with tf.variable_scope('conv4') as scope:
        kernel = tf.get_variable('weights', shape=[3, 3, 128, 256], initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv = tf.nn.conv2d(pool3, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.get_variable('biases', shape=[256], initializer=tf.constant_initializer(0.0))
        bias = tf.nn.bias_add(conv, biases)
        conv4 = tf.nn.relu(bias, name=scope.name)
        _activation_summary(conv4)
    pool4 = tf.nn.max_pool(conv4, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool4')

    with tf.variable_scope('fc5') as scope:
        dim = 1
        for d in pool4.get_shape()[1:].as_list():
            dim *= d
        reshape = tf.reshape(pool4, [BATCH_SIZE, dim])
        weights = _variable_with_weight_decay('weights', shape=[dim, 1024], stddev=0.02, wd=0.005)
        biases = tf.get_variable('biases', shape=[1024], initializer=tf.constant_initializer(0.0))
        fc5 = tf.nn.relu(tf.nn.bias_add(tf.matmul(reshape, weights), biases), name=scope.name)
        _activation_summary(fc5)

    with tf.variable_scope('fc6') as scope:
        weights = _variable_with_weight_decay('weights', shape=[1024, 256], stddev=0.02, wd=0.005)
        biases = tf.get_variable('biases', shape=[256], initializer=tf.constant_initializer(0.0))
        fc6 = tf.nn.relu(tf.nn.bias_add(tf.matmul(fc5, weights), biases), name=scope.name)
        _activation_summary(fc6)

    with tf.variable_scope('fc7') as scope:
        weights = tf.get_variable('weights', shape=[256, NUM_CLASSES], initializer=tf.truncated_normal_initializer(stddev=0.02))
        biases = tf.get_variable('biases', shape=[NUM_CLASSES], initializer=tf.constant_initializer(0.0))
        fc7 = tf.nn.bias_add(tf.matmul(fc6, weights), biases, name=scope.name)
        _activation_summary(fc7)

    return fc7

def inference_deep(images_placeholder, keep_prob, image_size, num_classes):

    x_image = tf.reshape(images_placeholder, [-1, image_size, image_size, 3])
    #print x_image

    with tf.name_scope('conv1') as scope:
        W_conv1 = weight_variable([3, 3, 3, 32])
        b_conv1 = bias_variable([32])
        h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
        #print h_conv1

    with tf.name_scope('pool1') as scope:
        h_pool1 = max_pool_2x2(h_conv1)
        #print h_pool1

    with tf.name_scope('conv2') as scope:
        W_conv2 = weight_variable([3, 3, 32, 64])
        b_conv2 = bias_variable([64])
        h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
        #print h_conv2

    with tf.name_scope('pool2') as scope:
        h_pool2 = max_pool_2x2(h_conv2)
        #print h_pool2

    with tf.name_scope('conv3') as scope:
        W_conv3 = weight_variable([3, 3, 64, 128])
        b_conv3 = bias_variable([128])
        h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv3) + b_conv3)
        #print h_conv3

    with tf.name_scope('pool3') as scope:
        h_pool3 = max_pool_2x2(h_conv3)
        #print h_pool3

    with tf.name_scope('fc1') as scope:
        w = int(image_size / pow(2,3))
        W_fc1 = weight_variable([w*w*128, 1024])
        b_fc1 = bias_variable([1024])
        h_pool3_flat = tf.reshape(h_pool3, [-1, w*w*128])
        #print h_pool3_flat
        h_fc1 = tf.matmul(h_pool3_flat, W_fc1) + b_fc1
        h_fc1_drop = tf.nn.dropout(tf.nn.relu(h_fc1), keep_prob)
        #print h_fc1_drop

    with tf.name_scope('fc2') as scope:
        W_fc2 = weight_variable([1024, num_classes])
        b_fc2 = bias_variable([num_classes])
        h_fc2 = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
        #print h_fc2

    with tf.name_scope('softmax') as scope:
        y_conv=tf.nn.softmax(h_fc2)
        #print y_conv

    return y_conv#, h_conv1, h_pool1, h_conv2, h_pool2, h_conv3, h_pool3, h_pool3_flat, h_fc1_drop, h_fc2


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def tf_print(tensor, name):
    #print name, tensor
    return tensor

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

def loss(logits, labels):
    cross_entropy = -tf.reduce_sum(labels*tf.log(tf.clip_by_value(logits,1e-10,1.0)))
    tf.scalar_summary("cross_entropy", cross_entropy)
    return cross_entropy    

def tf_loss(logits, labels,BATCH_SIZE, NUM_CLASSES):
    sparse_labels = tf.reshape(labels, [BATCH_SIZE, 1])
    indices = tf.reshape(tf.range(BATCH_SIZE), [BATCH_SIZE, 1])
    concated = tf.concat(1, [indices, sparse_labels])
    dense_labels = tf.sparse_to_dense(concated, [BATCH_SIZE, NUM_CLASSES], 1.0, 0.0)

    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits, dense_labels)
    mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
    tf.add_to_collection('losses', mean)
    return tf.add_n(tf.get_collection('losses'), name='total_loss')
def tf_train(total_loss, global_step):
    loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
    losses = tf.get_collection('losses')
    loss_averages_op = loss_averages.apply(losses + [total_loss])

    for l in losses + [total_loss]:
        tf.scalar_summary(l.op.name + ' (raw)', l)

    # Apply gradients, and add histograms
    with tf.control_dependencies([loss_averages_op]):
        opt = tf.train.AdamOptimizer()
        grads = opt.compute_gradients(total_loss)
    apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)
    for var in tf.trainable_variables():
        tf.histogram_summary(var.op.name, var)
    for grad, var in grads:
        tf.histogram_summary(var.op.name + '/gradients', grad)

    # Track the moving averages of all trainable variables
    variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
    variables_averages_op = variable_averages.apply(tf.trainable_variables())

    with tf.control_dependencies([apply_gradient_op, variables_averages_op]):
        train_op = tf.no_op(name='train')
    return train_op
def training(loss, learning_rate):
    train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss)
    return train_step
def training2(total_loss, global_step):
    loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
    losses = tf.get_collection('losses')
    loss_averages_op = loss_averages.apply(losses + [total_loss])

    for l in losses + [total_loss]:
        tf.scalar_summary(l.op.name + ' (raw)', l)

    # Apply gradients, and add histograms
    with tf.control_dependencies([loss_averages_op]):
        opt = tf.train.AdamOptimizer()
        grads = opt.compute_gradients(total_loss)
    apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)
    for var in tf.trainable_variables():
        tf.histogram_summary(var.op.name, var)
    #for grad, var in grads:
     #   if grad:
      #      tf.histogram_summary(var.op.name + '/gradients', grad)

    # Track the moving averages of all trainable variables
    variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
    variables_averages_op = variable_averages.apply(tf.trainable_variables())

    with tf.control_dependencies([apply_gradient_op, variables_averages_op]):
        train_op = tf.no_op(name='train')
    return train_op
def accuracy(logits, labels):
    labels = tf.cast(labels, tf.int64)
    correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    tf.scalar_summary("accuracy", accuracy)

    return accuracy

#http://walkingmask.hatenablog.com/entry/2016/08/16/02365
def draw_input(images,IMAGE_SIZE):
    '''
    visualize input images
    (?, 28, 28, 1)
    '''
    input_imgs = tf.slice(images, [0,0,0,0], [50,-1,-1,-1])
    input_imgs = tf.split(0, 50, input_imgs)

    for i in range(50):
      input_imgs[i] = tf.reshape(input_imgs[i], [IMAGE_SIZE, IMAGE_SIZE, 3])
      #input_imgs[i] = tf.image.resize_image_with_crop_or_pad(input_imgs[i], 30, 30)
      input_imgs[i] = tf.image.resize_images(input_imgs[i], 30,30)
      input_imgs[i] = tf.reshape(input_imgs[i], [1, 30, 30, 3])
    input_img1 = tf.concat(1, input_imgs[0:7])
    input_img2 = tf.concat(1, input_imgs[7:14])
    input_img3 = tf.concat(1, input_imgs[14:21])
    input_img4 = tf.concat(1, input_imgs[21:28])
    input_img5 = tf.concat(1, input_imgs[28:35])
    input_img6 = tf.concat(1, input_imgs[35:42])
    input_img7 = tf.concat(1, input_imgs[42:49])

    input_img = tf.concat(2, [input_img1, input_img2, input_img3, input_img4, input_img5, input_img6, input_img7])

    return input_img

def draw_hconv(h_conv1,IMAGE_SIZE,layer):
    zero_pad = tf.zeros([1,IMAGE_SIZE,IMAGE_SIZE,1])       
    h_conv1_imgs = tf.slice(h_conv1, [0,0,0,0], [49,-1,-1,-1]) 
    h_conv1_imgs = tf.split(0, 49, h_conv1_imgs) # (1,28,28,32) * 49 (images)
        
    for i in range(49):
        h_conv1_imgs[i] = tf.concat(3, [h_conv1_imgs[i],zero_pad,zero_pad,zero_pad,zero_pad]) # (1,28,28,36)
        h_conv1_imgs_part = tf.split(3, 36, h_conv1_imgs[i]) # (1,96,96,1) * 36
        for j in range(36):
            h_conv1_imgs_part[j] = tf.reshape(h_conv1_imgs_part[j], [IMAGE_SIZE,IMAGE_SIZE,1])
            h_conv1_imgs_part[j] = tf.image.resize_images(h_conv1_imgs_part[j], IMAGE_SIZE,IMAGE_SIZE)
            #h_conv1_imgs_part[j] = tf.image.resize_image_with_crop_or_pad(h_conv1_imgs_part[j], 60, 60)
            h_conv1_imgs_part[j] = tf.reshape(h_conv1_imgs_part[j], [1,IMAGE_SIZE,IMAGE_SIZE,1])
        h_conv1_part_img1 = tf.concat(1, h_conv1_imgs_part[0:6])
        h_conv1_part_img2 = tf.concat(1, h_conv1_imgs_part[6:12])
        h_conv1_part_img3 = tf.concat(1, h_conv1_imgs_part[12:18])
        h_conv1_part_img4 = tf.concat(1, h_conv1_imgs_part[18:24])
        h_conv1_part_img5 = tf.concat(1, h_conv1_imgs_part[24:30])
        h_conv1_part_img6 = tf.concat(1, h_conv1_imgs_part[30:36])
        h_conv1_part_img = tf.concat(2, [h_conv1_part_img1, h_conv1_part_img2, h_conv1_part_img3, h_conv1_part_img4, h_conv1_part_img5, h_conv1_part_img6])
        h_conv1_part_img = tf.reshape(h_conv1_part_img, [IMAGE_SIZE*6,IMAGE_SIZE*6,1])
        h_conv1_part_img = tf.image.resize_image_with_crop_or_pad(h_conv1_part_img, IMAGE_SIZE*6+20, IMAGE_SIZE*6+20)
        h_conv1_part_img = tf.reshape(h_conv1_part_img, [1,IMAGE_SIZE*6+20,IMAGE_SIZE*6+20,1])
        h_conv1_imgs[i] = h_conv1_part_img
    h_conv1_img1 = tf.concat(1, h_conv1_imgs[0:7])
    h_conv1_img2 = tf.concat(1, h_conv1_imgs[7:14])
    h_conv1_img3 = tf.concat(1, h_conv1_imgs[14:21])
    h_conv1_img4 = tf.concat(1, h_conv1_imgs[21:28])
    h_conv1_img5 = tf.concat(1, h_conv1_imgs[28:35])
    h_conv1_img6 = tf.concat(1, h_conv1_imgs[35:42])
    h_conv1_img7 = tf.concat(1, h_conv1_imgs[42:49])
    h_conv1_img = tf.concat(2, [h_conv1_img1, h_conv1_img2, h_conv1_img3, h_conv1_img4, h_conv1_img5, h_conv1_img6, h_conv1_img7])
    
    return h_conv1_img
def draw_hpool(h_pool1,IMAGE_SIZE,layer):
    '''
    visualize pool1
    (?, 14, 14, 32)
    '''
    IMAGE_SIZE = int(IMAGE_SIZE/2)
    zero_pad = tf.zeros([1,IMAGE_SIZE,IMAGE_SIZE,1])
    h_pool1_imgs = tf.slice(h_pool1, [0,0,0,0], [49,-1,-1,-1])
    h_pool1_imgs = tf.split(0, 49, h_pool1_imgs) # (1,14,14,32) * 49
    
    for i in range(49):
      h_pool1_imgs[i] = tf.concat(3, [h_pool1_imgs[i],zero_pad,zero_pad,zero_pad,zero_pad]) # (1,14,14,36)
      h_pool1_imgs_part = tf.split(3, 36, h_pool1_imgs[i]) # (1,14,14,1) * 36
      for j in range(36):
        h_pool1_imgs_part[j] = tf.reshape(h_pool1_imgs_part[j], [IMAGE_SIZE,IMAGE_SIZE,1])
        h_pool1_imgs_part[j] = tf.image.resize_images(h_pool1_imgs_part[j], IMAGE_SIZE,IMAGE_SIZE)
        #h_pool1_imgs_part[j] = tf.image.resize_image_with_crop_or_pad(h_pool1_imgs_part[j], 32, 32)
        h_pool1_imgs_part[j] = tf.reshape(h_pool1_imgs_part[j], [1,IMAGE_SIZE,IMAGE_SIZE,1])
      h_pool1_part_img1 = tf.concat(1, h_pool1_imgs_part[0:6])
      h_pool1_part_img2 = tf.concat(1, h_pool1_imgs_part[6:12])
      h_pool1_part_img3 = tf.concat(1, h_pool1_imgs_part[12:18])
      h_pool1_part_img4 = tf.concat(1, h_pool1_imgs_part[18:24])
      h_pool1_part_img5 = tf.concat(1, h_pool1_imgs_part[24:30])
      h_pool1_part_img6 = tf.concat(1, h_pool1_imgs_part[30:36])
      h_pool1_part_img = tf.concat(2, [h_pool1_part_img1, h_pool1_part_img2, h_pool1_part_img3, h_pool1_part_img4, h_pool1_part_img5, h_pool1_part_img6])
      h_pool1_part_img = tf.reshape(h_pool1_part_img, [IMAGE_SIZE*6,IMAGE_SIZE*6,1])
      h_pool1_part_img = tf.image.resize_image_with_crop_or_pad(h_pool1_part_img, IMAGE_SIZE*6+20, IMAGE_SIZE*6+20)
      h_pool1_part_img = tf.reshape(h_pool1_part_img, [1,IMAGE_SIZE*6+20,IMAGE_SIZE*6+20,1])
      h_pool1_imgs[i] = h_pool1_part_img
    h_pool1_img1 = tf.concat(1, h_pool1_imgs[0:7])
    h_pool1_img2 = tf.concat(1, h_pool1_imgs[7:14])
    h_pool1_img3 = tf.concat(1, h_pool1_imgs[14:21])
    h_pool1_img4 = tf.concat(1, h_pool1_imgs[21:28])
    h_pool1_img5 = tf.concat(1, h_pool1_imgs[28:35])
    h_pool1_img6 = tf.concat(1, h_pool1_imgs[35:42])
    h_pool1_img7 = tf.concat(1, h_pool1_imgs[42:49])
    h_pool1_img = tf.concat(2, [h_pool1_img1, h_pool1_img2, h_pool1_img3, h_pool1_img4, h_pool1_img5, h_pool1_img6, h_pool1_img7])

    h_pool1_sum = tf.image_summary("h_pool1_Visualize", h_pool1_img, max_images=10)
    
    return h_pool1_img