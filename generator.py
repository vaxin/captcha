# - coding: utf-8 -
from PIL import Image, ImageDraw, ImageFont
import random
import util
import numpy as np
import img as img_util

# just 0-9a-zA-Z
def getCharImage(char, font_path, size):
  # always white background, the network should be able to learn seperating the char from the background
  # in fact, the NN should only care about the contrast and not the white backgroud and black font.
  # the input channels to the image will be four channels (Red, Green, Blue, Illumination)
  txt = Image.new('RGBA', (size, size), (255, 255, 255, 255))
  font = ImageFont.truetype(font_path, size)

  draw = ImageDraw.Draw(txt)
  draw.text((0, 0), char, font=font, fill = (0, 0, 0, 255))
  
  return txt

def generate(size):
  fonts = [ 'simhei' ]
  font_len = len(fonts)
  chars = '0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'
  char_len = len(chars)
  ret = []
  for font in fonts:
    for char in chars:
      img = getCharImage(chars[random.randint(0, char_len - 1)], 'fonts/' + fonts[random.randint(0, font_len - 1)] + '.ttf', size)
      
      # rotate 100 times
      for i in range(100):
        # rotate
        ix = img_util.ImageX(img)
        ix.rotate(random.randint(-60, 60))
        ix.save('test.png')
        ix.gray()
        data = ix.getArray()
        print data.tolist()

        tt = Image.fromarray(data)
        tt.save('test.tiff')
        break
      break
    break

  return np.asarray(ret)


if __name__ == '__main__':
  generate(100)
