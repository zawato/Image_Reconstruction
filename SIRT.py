import numpy as np
import matplotlib.pyplot as plt

from skimage.transform import radon, iradon

def SIRT(img, sinogram, alpha=0.1, niter=1e2):
    
    THETA = np.linspace(0., 180., 400, endpoint=False)
    
    # initial
    image = np.ones(img.shape) / 2
    
    for i in range(int(niter)):
        predict_img = radon(image, THETA, circle=True)
        image = image + alpha * iradon(sinogram - predict_img, THETA, circle=True)
        
        image[image < 0] = 0
        image[image > 1] = 1
        
        plt.imshow(image)
        plt.colorbar()
        plt.title("%d / %d" % (i + 1, niter))
        plt.pause(1)
        plt.close()
    
    return image