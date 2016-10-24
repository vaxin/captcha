# - coding: utf-8 -
import tensorflow as tf
import numpy as np

def train(training_set, n_pixel, pre_net_op, n_input, n_hidden):
  '''
  input_data -> pre_net -> layer, the layer can be conv_layer, if not you should make segments by yourself
  '''

  learning_rate = 0.0001
  training_epochs = 20
  batch_size = 1

  # layer will trained by autoencoder
  X = tf.placeholder(tf.float32, [ None, n_pixel ])

  encoder_h = tf.Variable(tf.random_normal([ n_input, n_hidden ]))
  encoder_b = tf.Variable(tf.random_normal([ n_hidden ]))
  decoder_h = tf.Variable(tf.random_normal([ n_hidden, n_input ]))
  decoder_b = tf.Variable(tf.random_normal([ n_input ]))

  mid = tf.nn.sigmoid(tf.matmul(X, encoder_h) + encoder_b)
  output = tf.nn.sigmoid(tf.matmul(mid, decoder_h) + decoder_b)

  cost = tf.reduce_mean(tf.pow(X - output, 2))
  optimizer = tf.train.RMSPropOptimizer(learning_rate).minimize(cost)

  # it seems the checker has bug for watching gradient descent for tf.pow 
  # check = tf.add_check_numerics_ops()

  init = tf.initialize_all_variables()
  with tf.Session() as sess:
    sess.run(init)

    # test
    #print sess.run(mid, feed_dict={ X: training_set })
    #print sess.run(output, feed_dict = { X: training_set })
    #print sess.run(cost, feed_dict = { X: training_set })

    num = len(training_set)
    total_batch = int(num/batch_size)
    for epoch in range(training_epochs):
      # Loop over all batches
      for i in range(total_batch):
        batch_x = training_set[i * batch_size : i+1 * batch_size]
        _, c = sess.run([ optimizer, cost ], feed_dict = { X: batch_x })

      print "Epoch:", epoch, ", cost=", c

  return (encoder_h, encoder_b)

if __name__ == "__main__":
  import generator
  training_set = generator.generate((50, 50))
  n_pixel = 50 * 50
  
  '''
  '''
  one = []
  import random
  for i in range(n_pixel):
    one.append(random.random())
  
  one = training_set[0].tolist()
  print one
  #training_set = [ one ]
  (encoder_h, encoder_b) = train(training_set, n_pixel, None, n_pixel, 50)
