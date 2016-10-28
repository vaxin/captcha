# - coding: utf-8 -
from PIL import Image, ImageDraw, ImageFont
import random
import util
import numpy as np
import img as img_util
import sys
# just 0-9a-zA-Z
def getCharImage(char, font, size, rotation):
  # always white background, the network should be able to learn seperating the char from the background
  # in fact, the NN should only care about the contrast and not the white backgroud and black font.
  # the input channels to the image will be four channels (Red, Green, Blue, Illumination)
  txt = Image.new('RGBA', (size, size), (255, 255, 255, 0))
  font = ImageFont.truetype('fonts/' + font + '.ttf', size)

  draw = ImageDraw.Draw(txt)
  draw.text((size / 3, 0), char, font=font, fill = (0, 0, 0, 255))
  
  txt = txt.rotate(rotation, expand = False)
  bg = Image.new('RGBA', (size, size), (255, 255, 255, 255))
  bg.paste(txt, (0, 0), txt)
  return bg

fonts = [ 'simhei' ]
chars = '0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'
def generateChars(size, count = None):
  font_len = len(fonts)
  char_len = len(chars)
  ret = []
  n = 0
  for font in fonts:
    for char in chars:
      # rotate 100 times
      for i in range(20):
        # rotate
        img = getCharImage(char, font, size, rotation = random.randint(-60, 60))
        ix = img_util.ImageX(img)
        ix.gray()
        data = ix.getArray()
        ret.append(data)
        n += 1
        if count is not None and n > count:
          return np.asarray(ret)

  return np.asarray(ret)

def generateSequenceImage(seq, size, font):
  bg = Image.new('RGBA', (len(seq) * size, size), (255,) * 4)
  for i in range(len(seq)):
    char = seq[i]
    one = getCharImage(char, font, size, rotation = random.randint(-60, 60))
    bg.paste(one, (i * size, 0), one)
  return bg

def generateSequences(count, size, len_range):
  ''' 
  generate sequences with random length under len_range
  The image size will be also random but mainly the type of long strip.
  '''

  len = random.randint(len_range[0], len_range[1])
  ret = []
  for _ in range(count):
    sequence = np.random.choice(list(chars), len)
    img = generateSequenceImage(sequence, size, fonts[0] )
    ix = img_util.ImageX(img)
    ix.gray()
    data = ix.getArray()
    ret.append(data)
  return np.asarray(ret)


if __name__ == '__main__':
  #generateChars(30)
  generateSequences(1, 50, (10, 20))

