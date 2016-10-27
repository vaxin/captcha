'''
Image Utils
read image file, get image array, convert array to image of various formats, convertion and so on.
'''

import numpy as np
from PIL import Image

def arr2Image(arr):
  return Image.fromarray(source)

def getImage(filename):
  return Image.open(filename)

def gray(img):
  return img.convert('L')

def rotate(img, rotate):
  return img.rotate(rotate)

def scale(img, scale):
  w = int(float(img.size[0]) * scale)
  h = int(float(img.size[1]) * scale)
  return img.resize((w, h), Image.ANTIALIAS)

def getImageArray(img, reshape = None):
  img = np.asarray(img, dtype=np.float32)
  res = img / 255.
  if reshape is not None:
    res = res.reshape(reshape)
  return res

def conv(img, conv_size, stride):
  if type(img) == Image:
    array = getImageArray(img)
  else:
    array = img
  ret = []
  row = len(img)
  col = len(img[0])

  for x in range(0, row - conv_size + 1, stride):
    for y in range(0, col - conv_size + 1, stride):
      tmp = img[x: x + conv_size, y: y + conv_size]
      ret.append(tmp)

  return np.asarray(ret)

def explode(training_set, img_size, conv_size):
  img_w = img_size[0]
  img_h = img_size[1]

  # convert training_set to segment
  training_set = training_set.reshape(-1, img_w * img_h)

  new_set = []
  training_set = training_set.reshape(-1, img_w, img_h)

  for img in training_set:
    for item in conv(img, conv_size, 10):
      item = item.reshape(conv_size * conv_size)
      new_set.append(item)
  return np.asarray(new_set)

def saveImageFromArray(arr, path):
  img = Image.fromarray(arr)
  img.save(path)

def saveTIFFsFromArray(arrs, path):
  i = 0
  for one in arrs:
    saveImageFromArray(one, path + '/' + str(i) + '.tiff')
    i += 1

def saveImage(img, path):
  img.save(path, 'PNG')

class ImageX:
  ''' a bridge class for image processing '''
  def __init__(self, source):
    ''' when source is array, the correctness of the format will be ensured by the invoker '''
    if type(source) == str:
      # filename
      self.image = Image.open(source)
    elif type(source) == list or type(source) == np.ndarray:
      # array
      if type(source) == list:
        source = np.array(source)
      self.image = Image.fromarray(source)

  def getImage(self):
    return self.image

  def getArray(self, reshape = None):
    return getImageArray(self.image, reshape)

  def gray(self):
    self.image = self.image.convert('L')

  def rotate(self, rotate):
    self.image = self.image.rotate(rotate)

  def scale(self, scale):
    self.image = scale(self.image, scale)

  def conv(self, conv_size, stride):
    return explode(self.image, conv_size, stride)

  def save(self, path):
    saveImage(self.image, path)
    
