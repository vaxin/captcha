import line
import sparse_auto_encode
import tensorflow as tf
import img as img_util
import numpy as np
import util
import random

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

def pick(arr, count):
  ''' randomly pick up n elements from a list '''
  l = len(arr)
  ret = []
  for _ in range(count):
    ret.append(arr[random.randint(0, l)])
  return ret

def convTrain(training_set, img_size, conv_size = 10, stride = 1,  n_feature = 100):
  training_set = img_util.explode(training_set, img_size, conv_size = conv_size, stride = stride)
  img_util.saveTIFFsFromArray(pick(training_set, 30), 'source') 
  training_set = training_set.reshape(-1, conv_size * conv_size)
  weights = sparse_auto_encode.train(training_set, (conv_size, conv_size), None, n_feature)
  imgs = util.visualize(weights, (conv_size, conv_size))
  saveImages(imgs)


def testWithImage():
  ix = img_util.ImageX('desk.jpg')
  ix.gray()
  arr = ix.getArray()
  img_size = ix.getSize()

  training_set = np.asarray([ arr ])
  convTrain(training_set, img_size, conv_size = 30, stride = 30, n_feature = 500)

def testConv():
  img = [ [ 11, 12, 13, 14, 15 ], [ 21, 22, 23, 24, 25 ], [ 31, 32, 33, 34, 35 ] ]
  print img_util.conv(img, 2, 1).tolist()


def testWithChars():
  import generator
  size = 80
  training_set = generator.generateChars(size = size, count = 1000)
  convTrain(training_set, img_size = (size, size), conv_size = 10, stride = 5, n_feature = 100)

#computeCost()
#testLines()
#testWithImage()
#testConv()
testWithChars()
