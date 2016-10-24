# - coding: utf-8 -
from PIL import Image, ImageDraw, ImageFont

# just 0-9a-zA-Z
def getCharImage(char, font_path):
  # TODO Size is very important, will deep into it soon
  # The first stage will only use a fixed size image
  size = (150, 150)
  # always white background, the network should be able to learn seperating the char from the background
  # in fact, the NN should only care about the contrast and not the white backgroud and black font.
  # the input channels to the image will be four channels (Red, Green, Blue, Illumination)
  txt = Image.new('RGBA', size, (255, 255, 255, 255))
  font = ImageFont.truetype(font_path, 50)

  draw = ImageDraw.Draw(txt)
  draw.text((50, 50), char, font=font, fill = (0, 0, 0, 255))
  
  return txt

def generate():
  fonts = [ 'simhei.ttf' ]


if __name__ == '__main__':
  img = getCharImage('a', 'fonts/simhei.ttf')
  img.save('test.png', 'PNG')
