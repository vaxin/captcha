import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import random
import util

def _fpart(x):
  return x - int(x)
 
def _rfpart(x):
  return 1 - _fpart(x)

def getpixel(img, xy):
  if xy[0] >= len(img) or xy[1] >= len(img[xy[0]]):
    return (255., 255., 255.)
  return img[xy[0]][xy[1]]

def putpixel(img, xy, color, alpha=1):
  if xy[0] >= len(img) or xy[1] >= len(img[xy[0]]):
    return
  """Paints color over the background at the point xy in img.
 
  Use alpha for blending. alpha=1 means a completely opaque foreground.
 
  """
  c = tuple(map(lambda bg, fg: int(round(alpha * fg + (1-alpha) * bg)),
          getpixel(img, xy), color))
  
  img[xy[0]][xy[1]] = c
 
def draw_line(img, p1, p2, color):
  """Draws an anti-aliased line in img from p1 to p2 with the given color."""
  x1, y1, x2, y2 = p1 + p2
  dx, dy = x2-x1, y2-y1
  steep = abs(dx) < abs(dy)
  p = lambda px, py: ((px,py), (py,px))[steep]

  if abs(dx) > abs(dy):
    p = lambda px, py: (px,py)
  else:
    p = lambda px, py: (py, px)
    x1, y1, x2, y2, dx, dy = y1, x1, y2, x2, dy, dx

  if x2 < x1:
    x1, x2, y1, y2 = x2, x1, y2, y1
 
  grad = dy/dx
  intery = y1 + _rfpart(x1) * grad
  def draw_endpoint(pt):
    x, y = pt
    xend = round(x)
    yend = y + grad * (xend - x)
    xgap = _rfpart(x + 0.5)
    px, py = int(xend), int(yend)
    putpixel(img, p(px, py), color, _rfpart(yend) * xgap)
    putpixel(img, p(px, py+1), color, _fpart(yend) * xgap)
    return px
 
  xstart = draw_endpoint(p(*p1)) + 1
  xend = draw_endpoint(p(*p2))
 
  if xstart > xend:
    xstart, xend = xend, xstart
  for x in range(xstart, xend):
    y = int(intery)
    putpixel(img, p(x, y), color, _rfpart(intery))
    putpixel(img, p(x, y+1), color, _fpart(intery))
    intery += grad

def showImg(arr):
  implot = plt.imshow(arr, cmap=cm.Greys_r, vmin=0, vmax=255)
  implot.set_interpolation('nearest')
  plt.show()

def genImage(size):
  ''' return 2d array with elements like [ 255, 0, 127 ] np.uint8 '''
  img = []

  for i in range(size):
    img.append([])
    for j in range(size):
      img[i].append((255., 255., 255.))

  # 2 x or x 2, x ~ (2, size - 2)
  start_x = float(int(random.random() * (size - 2)) + 1)
  start_y = 1
  if random.random() > 0.5:
    start_y = start_x
    start_x = 1

  end_x = size - 1 - start_x
  end_y = size - 1 - start_y 

  #print (start_x, start_y), (end_x, end_y)

  if start_x - end_x == 0 and start_y - end_y == 0:
    return genImage(size)
  
  if start_x > end_x:
    start_x, start_y, end_x, end_y = end_x, end_y, start_x, start_y
  
  #start_x, start_y, end_x, end_y = 41.0, 55.0, 45.0, 16.0
  draw_line(img, (start_x, start_y), (end_x, end_y), (0., 0., 0.))

  res = []
  for x in range(size):  
    row = []
    for y in range(size):
      #row.append([ int(one) for one in img[x][y] ])
      row.append(img[x][y][0])
    res.append(row)

  return np.array(res, dtype = np.uint8)

if __name__ == '__main__':
  res = genImage(30)
  #import img
  #img.saveImageFromArray(res, 'line.png')
  from PIL import Image
  img = Image.fromarray(res / 255.)
  img.save('test.tiff')
