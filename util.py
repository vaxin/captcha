#- coding: utf-8 -
import tensorflow as tf
import numpy as np
import img

def saveNet(net, path):
  pass

def loadNet(path):
  pass
 
def visualize(weights, input_shape):
  # Visualize
  weights = np.transpose(weights)
  max = weights.max()
  min = weights.min()
  #weights = weights - weights.mean()
  #size = weights.max()
  #weights = weights / size
  #weights = weights * 0.8 + 0.5
  weights = (weights - min) / (max - min) # force to -> [ 0, 1 ]

  base_num = weights.shape[0]
  con_num_per_base = weights.shape[1]
  image_per_row = int(con_num_per_base ** 0.5)

  result = []
  index = 0
  row = []

  images = []
  for one in weights:
    one = one.reshape(input_shape[0], input_shape[1])
    #one *= 255
      
    #ix = img.ImageX(np.array(one, dtype=np.float32))
    ix = img.ImageX(one)
    images.append(ix.getImage())

  return images


if __name__ == '__main__':
  arr = [ [ 2., 0., 0., 0., 0.,   0., 1, 0., 0., 0.,   0., 0., 1., 0., 0.,    0., 0., 0., 1., 0.,   0., 0., 0., 0., 1. ] ] 
  arr2 = np.transpose(np.array(arr))
  
  images = visualize(arr2, (5, 5))
  images[0].save('test.tiff')
