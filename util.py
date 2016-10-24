#- coding: utf-8 -
from PIL import Image
from pylab import *
import numpy
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy

def showImages(imgs):
  f, arr = plt.subplots(10, 10)
  for i in range(10):
    one = None
    for j in range(10):
      plt.gray()
      tmp = numpy.asarray(imgs[i * 8 + j]).reshape(8, 8)
      arr[i][j].imshow(tmp)
  plt.show()

S = 8

def getImg(filename):
  return Image.open(filename)

def gray(img):
  #读取图片,灰度化，并转为数组
  return img.convert('L')

def rotate(img, rotate):
  return img.rotate(rotate)

def scale(img, scale):
  w = int(float(img.size[0]) * scale)
  h = int(float(img.size[1]) * scale)
  return img.resize((w, h), Image.ANTIALIAS)

def getImageData(img, reshape = None):
  img = numpy.asarray(img, dtype=numpy.float32)
  res = img / 255.
  if reshape is not None:
    res = res.reshape(reshape)
  return res

def getFragments(img, step):
  img = getImageData(img)
  ret = []
  row = len(img)
  col = len(img[0])
  print row, col

  for x in range(0, row - S, step):
    for y in range(0, col - S, step):
      tmp = img[x:x+S, y:y+S]
      ret.append(tmp)
  return ret

def getMore(img):
  # self
  r = getFragments(img, 20)
  
  # scale 9 levels 0.1 -0.9
  for s in range(1, 10):
    r += getFragments(scale(img, float(s) * 0.1), s * 2)

  # rotate 19 levels -90 - 90
  for s in xrange(-90, 91, 10):
    r += getFragments(rotate(img, s), 20)

  return r

def getSineImage(freq, angle, contrast):
  # 先横着画，然后在旋转
  pass


def showImg(im):
  gray()
  imshow(im)  
  show()

if __name__ == "__main__":
  img = getImg('test.jpg')
  #scale(img, 0.1).show()
  #rotate(img, 45).show()
  #gray(img).show()

  #rs = getFragments(img, 5)

  rs = getMore(gray(img))
  print len(rs)
