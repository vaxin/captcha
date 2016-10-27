import line
import sparse_auto_encode
import tensorflow as tf
import img
import numpy as np

def test_lines():
  # test the sparse auto encoder with a line generator

  S = 8

  def genOneLineGrayImageArray():
    return line.genImage(S) / 255.

  training_set = []
  for _  in range(100000):
    training_set.append(genOneLineGrayImageArray())
  print training_set[0].tolist()
  weights = sparse_auto_encode.train(np.asarray(training_set), (S, S), S, None, 25)

  import util
  i = 0
  for x in util.visualize(weights, (S, S)):
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


#computeCost()
test_lines()
