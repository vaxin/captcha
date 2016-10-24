# - coding: utf-8 -
from PIL import Image, ImageDraw, ImageFont
import random
import util
import numpy as np

# just 0-9a-zA-Z
def getCharImage(char, font_path, size):
  # always white background, the network should be able to learn seperating the char from the background
  # in fact, the NN should only care about the contrast and not the white backgroud and black font.
  # the input channels to the image will be four channels (Red, Green, Blue, Illumination)
  txt = Image.new('RGBA', size, (255, 255, 255, 255))
  font = ImageFont.truetype(font_path, 50)

  draw = ImageDraw.Draw(txt)
  draw.text((0, 0), char, font=font, fill = (0, 0, 0, 255))
  
  return txt

def generate(size):
  fonts = [ 'simhei' ]
  font_len = len(fonts)
  chars = '0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'
  char_len = len(chars)
  set = []
  for font in fonts:
    for char in chars:
      img = getCharImage(chars[random.randint(0, char_len - 1)], 'fonts/' + fonts[random.randint(0, font_len - 1)] + '.ttf', size)
      
      # rotate 100 times
      for i in range(100):
        # rotate
        img_x = img.rotate(random.randint(-60, 60))
        #img.save('test.png', 'PNG')
        img_x = img_x.convert('L')
        data = util.getImageData(img_x, (size[0] * size[1]))
        set.append(data)

  return np.asarray(set)


if __name__ == '__main__':
  generate()
