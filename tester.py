import line
import sparse_auto_encode
import tensorflow as tf
import img as img_util
import numpy as np
import util

def genOneLineGrayImageArray(S):
  return line.genImage(S) / 255.

def testLines():
  # test the sparse auto encoder with a line generator

  S = 8
  n_feature = 25

  training_set = []
  for _  in range(10000):
    training_set.append(genOneLineGrayImageArray(S))
  weights = sparse_auto_encode.train(np.asarray(training_set), (S, S), None, n_feature)
  
  imgs = util.visualize(weights, (S, S))
  saveImages(imgs)

def saveImages(images):
  i = 0
  for x in images:
    x.save('test/test_' + str(i) + '.tiff')
    i += 1

def computeCost():
  n_input = 25
  n_feature = 1
  X = tf.placeholder(tf.float32, [ None, 25 ])
  line = np.array([ [ 1, 0, 0, 0, 0,   0, 1, 0, 0, 0,   0, 0, 1, 0, 0,   0, 0, 0, 1, 0,   0, 0, 0, 0, 1  ] ], dtype = np.float32)
  line_c = tf.constant(line, dtype=tf.float32)

  encoder_W = tf.transpose(line_c)
  encoder_b = tf.constant([ 0. ])

  decoder_W = line_c
  decoder_b = tf.zeros([ n_input ])

  all = sparse_auto_encode.getCost(X, encoder_W, encoder_b, decoder_W, decoder_b)
  cost = all['cost']
  sess = tf.Session()
  xs = line
  print sess.run(cost, feed_dict = { X: xs })

def testWithImage():
  img = img_util.gray(img_util.getImage('desk.jpg'))
  img_size = img.size
  img = img_util.getImageArray(img)
  training_set = np.asarray([ img ])
  training_set = img_util.explode(training_set, img_size, 15)
  training_set = training_set.reshape(-1, 15, 15)
  img_util.saveTIFFsFromArray(training_set, 'source')

  return
  n_feature =25 
  conv_size = 15
  
  weights = sparse_auto_encode.train(training_set, img_size, conv_size, None, n_feature)

  imgs = util.visualize(weights, (conv_size, conv_size))
  saveImages(imgs)

#computeCost()
#testLines()
testWithImage()
