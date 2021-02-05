import numpy as np
import matplotlib.pyplot as plt
import os

from skimage.draw import circle as cir
from skimage.metrics.simple_metrics import mean_squared_error
from skimage.transform import radon, rescale, iradon

from tqdm import tqdm_notebook as tqdm

def SIRT(image, sinogram, theta, circle=True, alpha=0.1, n_iter=1e2, init=None):
    
    """
    image 再構成画像
    sinogram 投影データ
    theta 投影角度
    n_iter イテレーション
    """
    
    # initial
    if init is None:
        if circle==True:
            recon = np.zeros(image.shape)
            rr, cc = cir(image.shape[0] / 2, image.shape[1] / 2, image.shape[0] / 2 - 1)
            recon[rr, cc] = 1
        else:
            recon = np.ones(image.shape)
    else:
        recon = init
    
    mse = []
    
    for iter in tqdm(range(n_iter), desc="sirt iter", leave=False):
        predict_image = radon(recon, theta, circle=circle)
        recon = recon + alpha * iradon(sinogram - predict_image, theta, circle=circle)
        
        recon[recon < 0] = 0
        recon[recon > 1] = 1
        
        if iter > 0 : 
            mse.append(mean_squared_error(recon, image))
        
        # make dir
        make_dir("result")
        make_dir("result/recon")
        make_dir("result/recon/sirt")
        make_dir("result/mse")

        # plot
        plt.imshow(recon)
        plt.colorbar()
        plt.title("sirt recon, iteration: %d" % (iter + 1))
        
        # save
        filename = "./result/recon/sirt/recon_sirt_iter"+str(iter+1).zfill(3)+".png"
        plt.savefig(filename)
        plt.pause(1)
        plt.close()
        
    np.save("./result/mse/mse_sirt.npy", mse)

    return recon

def make_dir(DIR_PATH):
    if not os.path.exists(DIR_PATH):
        os.makedirs(DIR_PATH)