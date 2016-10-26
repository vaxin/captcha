import line
import sparse_auto_encode
import img
import numpy as np

# test the sparse auto encoder with a line generator

S = 40

def genOneLineGrayImageArray():
  array = line.genImage(S)
  imgx = img.ImageX(array)
  imgx.gray()
  return imgx.getArray()

training_set = []
for _  in range(10000):
  training_set.append(genOneLineGrayImageArray())

sparse_auto_encode.train(np.asarray(training_set), (S, S), None, 25)
