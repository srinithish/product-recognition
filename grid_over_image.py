
#https://stackoverflow.com/questions/8209568/how-do-i-draw-a-grid-onto-a-plot-in-python
import numpy as np
import matplotlib.pyplot as plt

img2 = plt.imread("image.jpg")
img = img2.copy()
x_nums = 5
y_nums = 5
dx, dy = img2.shape[1]/x_nums,img2.shape[0]/y_nums


plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['axes.edgecolor'] = 'white'
plt.rcParams['grid.alpha'] = 1
plt.rcParams['grid.color'] = "#cccccc"
plt.xticks(np.arange(0, img2.shape[1], dx))

plt.yticks(np.arange(0, img2.shape[0], dy))
plt.grid(True)
plt.imshow(img,'gray',vmin=-1,vmax=1)
plt.show()
