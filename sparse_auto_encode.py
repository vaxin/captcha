# - coding: utf-8 -
import tensorflow as tf
import numpy as np
import util
import img as img_util

def getCost(X, encoder_W, encoder_b, decoder_W, decoder_b):
  mid = tf.matmul(X, encoder_W)
  if encoder_b is not None:
    mid = mid + encoder_b
  mid = tf.nn.sigmoid(mid)
    
  output = tf.matmul(mid, decoder_W)
  if decoder_b is not None:
    output = output + decoder_b
  output = tf.nn.sigmoid(output)

  beta = tf.constant(3.)
  sparsity = tf.constant(0.01)
  weight_decay = tf.constant(0.0001)
  
  square_error = tf.reduce_mean(0.5 * tf.reduce_sum(tf.pow(X - output, 2), reduction_indices = 1))
  avg_activate = tf.reduce_mean(mid)
  sparsity_penalty = beta * tf.reduce_sum(sparsity * tf.log(sparsity / avg_activate) + (tf.constant(1.) - sparsity) * tf.log((tf.constant(1.) - sparsity) / (tf.constant(1.) - avg_activate)))
  regularization = 0.5 * weight_decay * (tf.reduce_sum(tf.pow(encoder_W, 2)) + tf.reduce_sum(tf.pow(decoder_W, 2)))
  cost = square_error + sparsity_penalty + regularization

  return {
    'cost'             : cost,
    'mid'              : mid,
    'output'           : output,
    'square_error'     : square_error,
    'sparsity_penalty' : sparsity_penalty,
    'regularization'   : regularization
  }

def train(training_set, size, pre_net_op, n_feature):
  '''
  input_data -> pre_net -> layer, the layer can be conv_layer, if not you should make segments by yourself
  size: is img_size, size[0] --> width, size[1] --> height
  '''

  learning_rate = 0.003
  training_epochs = 10000
  batch_size = 500

  row = size[1]
  col = size[0]

  # layer will trained by autoencoder

  n_input = row * col
  X = tf.placeholder(tf.float32, [ None, n_input ])

  encoder_W = tf.Variable(tf.random_normal([ n_input, n_feature ]))
  encoder_b = tf.Variable(tf.zeros([ n_feature ]))
  decoder_W = tf.Variable(tf.random_normal([ n_feature, n_input ]))
  decoder_b = tf.Variable(tf.zeros([ n_input ]))

  all = getCost(X, encoder_W, encoder_b, decoder_W, decoder_b) 
  cost = all['cost']
  optimizer = tf.train.RMSPropOptimizer(learning_rate).minimize(cost)
  
  # it seems the checker has bug for watching gradient descent for tf.pow 
  check = tf.add_check_numerics_ops()
  
  print 'Start Training'
  init = tf.initialize_all_variables()
  with tf.Session() as sess:
    sess.run(init)

    weights_0_to_255 = tf.image.convert_image_dtype(encoder_W, dtype=tf.uint8)
    tmp = tf.transpose(weights_0_to_255)
    for i in range(tf.shape(tmp)[0].eval()):
      tw = tmp[i]
      tw = tf.reshape(tw, (1, row, col, 1))
      tf.image_summary('encoder_Weight_' + str(i), tw)

    tf.scalar_summary('cost', cost)

    summary_writer = tf.train.SummaryWriter("/tmp/logs", sess.graph)
    all_sum = tf.merge_all_summaries()

    num = training_set.shape[0]
    total_batch = int(num / batch_size)

    try:
      for epoch in range(training_epochs):
        # Loop over all batches
        for i in range(total_batch):
          batch_x = training_set[i * batch_size : (i+1) * batch_size]
          _, c, ws = sess.run([ optimizer, cost, all_sum ], feed_dict = { X: batch_x })

        print "Epoch:", epoch, ", cost=", c
        summary_writer.add_summary(ws, epoch)
    #except Exception as e:
    #  print e
    except:
      print 'Early Stop Manually'

    w = encoder_W.eval()

  return w

