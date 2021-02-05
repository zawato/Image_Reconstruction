import numpy as np
import matplotlib.pyplot as plt
import os

from skimage.draw import circle as cir
from skimage.metrics.simple_metrics import mean_squared_error
from skimage.transform import radon, rescale, iradon

from tqdm import tqdm_notebook as tqdm

def OSEM(image, sinogram, theta, circle=True, n_iter=10, n_sub=1, init = None):
    
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
        recon = np.copy(init)

    # normalization matrix
    nview = len(theta)
    norm = np.ones(sinogram.shape)
    wgts = []
    for sub in range(n_sub):
        views = range(sub, nview, n_sub)
        wgt = iradon(norm[:, views], theta=theta[views], filter=None, circle=circle)
        wgts.append(wgt)

    mse = []

    # iteration
    for iter in tqdm(range(n_iter), desc="osem iter", leave=False):
        order = np.random.permutation(range(n_sub))
        for sub in tqdm(order, desc="osem sub", leave=False):
            views = range(sub, nview, n_sub)
            fp = radon(recon, theta=theta[views], circle=circle)
            ratio = sinogram[:, views] / (fp + 1e-6)
            bp = iradon(ratio, theta=theta[views], filter=None, circle=circle)     
            recon *= bp / (wgts[sub] + 1e-6)
        if iter > 0 : 
            mse.append(mean_squared_error(image,np.array(recon)))

        # make dir
        make_dir("result")
        make_dir("result/recon")
        make_dir("result/recon/osem")
        make_dir("result/mse")

        # plot
        plt.imshow(recon)
        plt.colorbar()
        plt.title("osem recon, iteration: %d" % (iter + 1))
        
        # save
        filename = "./result/recon/osem/recon_osem_iter"+str(iter+1).zfill(3)+".png"
        plt.savefig(filename)
        plt.pause(1)
        plt.close()
        
    np.save("./result/mse/mse_osem.npy", mse)

    return recon

def make_dir(DIR_PATH):
    if not os.path.exists(DIR_PATH):
        os.makedirs(DIR_PATH)