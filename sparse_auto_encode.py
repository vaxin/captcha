# - coding: utf-8 -
import tensorflow as tf
import numpy as np
import util
import img as img_util

def getCost(X, encoder_W, encoder_b, decoder_W, decoder_b):
  mid = tf.nn.sigmoid(tf.matmul(X, encoder_W) + encoder_b)
  output = tf.nn.sigmoid(tf.matmul(mid, decoder_W) + decoder_b)

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


def train(training_set, img_size, conv_size, pre_net_op, n_feature):
  '''
  input_data -> pre_net -> layer, the layer can be conv_layer, if not you should make segments by yourself
  '''
  img_w = img_size[0]
  img_h = img_size[1]
  n_pixel = img_w * img_h

  # convert training_set to segment
  training_set = training_set.reshape(-1, img_w * img_h)
  '''
  new_set = []
  training_set = training_set.reshape(-1, img_w, img_h)

  for img in training_set:
    for item in img_util.conv(img, conv_size, 10):
      item = np.asarray(item).reshape(conv_size * conv_size)
      new_set.append(item)
  training_set = np.asarray(new_set)
  '''
  print training_set

  learning_rate = 1e-5
  training_epochs = 10000
  batch_size = 500

  # layer will trained by autoencoder

  n_input = conv_size ** 2
  X = tf.placeholder(tf.float32, [ None, n_input ])

  encoder_W = tf.Variable(tf.random_normal([ n_input, n_feature ]))
  encoder_b = tf.Variable(tf.random_normal([ n_feature ]))
  decoder_W = tf.Variable(tf.random_normal([ n_feature, n_input ]))
  decoder_b = tf.Variable(tf.random_normal([ n_input ]))

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
      tw = tf.reshape(tw, (1, conv_size, conv_size, 1))
      tf.image_summary('encoder_Weight_' + str(i), tw)

    tf.scalar_summary('cost', cost)

    summary_writer = tf.train.SummaryWriter("/tmp/logs", sess.graph)
    all_sum = tf.merge_all_summaries()

    num = training_set.shape[0]
    total_batch = int(num / batch_size)

    for epoch in range(training_epochs):
      # Loop over all batches
      for i in range(total_batch):
        batch_x = training_set[i * batch_size : (i+1) * batch_size]
        _, c, ws = sess.run([ optimizer, cost, all_sum ], feed_dict = { X: batch_x })

      print "Epoch:", epoch, ", cost=", c
      summary_writer.add_summary(ws, epoch)
    w = encoder_W.eval()

  return w
  

if __name__ == "__main__":
  '''
  import generator
  w = 10
  n_feature = 25
  training_set = generator.generate((w, w))
  '''
  img = img_util.gray(img_util.getImage('desk.jpg'))
  img_size = img.size
  img = img_util.getImageArray(img)
  training_set = np.asarray([ img ])
  n_feature = 50
  
  encoder_W = train(training_set, img_size, 15, None, n_feature)
