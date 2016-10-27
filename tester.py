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
  training_set = np.asarray(training_set)

  img_util.saveTIFFsFromArray(training_set[0:10], 'source') 
  
  training_set = training_set.reshape(-1, S * S)
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
  ix = img_util.ImageX('desk.jpg')
  ix.gray()
  arr = ix.getArray()
  img_size = ix.getSize()
  conv_size = 30

  training_set = np.asarray([ arr ])
  training_set = img_util.explode(training_set, img_size, conv_size = conv_size, stride = conv_size)

  #img_util.saveTIFFsFromArray(training_set[0:10], 'source') 

  training_set = training_set.reshape(-1, conv_size * conv_size)

  n_feature =25 
  weights = sparse_auto_encode.train(training_set, (conv_size, conv_size), None, n_feature)

  imgs = util.visualize(weights, (conv_size, conv_size))
  saveImages(imgs)

def testConv():
  img = [ [ 11, 12, 13, 14, 15 ], [ 21, 22, 23, 24, 25 ], [ 31, 32, 33, 34, 35 ] ]
  print img_util.conv(img, 2, 1).tolist()

#computeCost()
#testLines()
testWithImage()
#testConv()
