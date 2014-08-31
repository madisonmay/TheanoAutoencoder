import numpy as np
from matplotlib import pyplot as plt
from scipy.misc import imsave

def unit_scale(x, eps=1e-8):
	""" Scales all values in the ndarray ndar to be between 0 and 1 """
	x = x.copy()
	x -= x.min()
	x *= 1.0 / (x.max() + eps)
	return x

def grayscale_grid_vis(X,show=True,save=False,transform=False):
	ngrid = int(np.ceil(np.sqrt(len(X))))
	npxs = np.sqrt(X[0].size)
	img = np.zeros((npxs*ngrid+ngrid-1,npxs*ngrid+ngrid-1))
	for i,x in enumerate(X):
		j = i%ngrid
		i = i/ngrid
		if transform:
			x = transform(x)
		img[i*npxs+i:(i*npxs)+npxs+i,j*npxs+j:(j*npxs)+npxs+j] = x
	if show:
		plt.imshow(img,cmap='gray')
		plt.show()
	if save:
		imsave(save,img)
	return img

def color_grid_vis(X,show=True,save=False,transform=False):
	ngrid = int(np.ceil(np.sqrt(len(X))))
	npxs = np.sqrt(X[0].size/3)
	img = np.zeros((npxs*ngrid+ngrid-1,npxs*ngrid+ngrid-1,3))
	for i,x in enumerate(X):
		j = i%ngrid
		i = i/ngrid
		if transform:
			x = transform(x)
		img[i*npxs+i:(i*npxs)+npxs+i,j*npxs+j:(j*npxs)+npxs+j] = x
	if show:
		plt.imshow(img,interpolation='nearest')
		plt.show()
	if save:
		imsave(save,img)
	return img