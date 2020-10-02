import numpy as np
import matplotlib.pyplot as plt

from skimage.draw import circle
from skimage.metrics.simple_metrics import mean_squared_error
from skimage.transform import radon, rescale, iradon

from tqdm import tqdm_notebook as tqdm

def OSEM(img, image, theta, n_iter=10, n_sub=1):

    """
    img 再構成画像
    image 投影データ
    theta 投影角度
    n_iter イテレーション
    n_sub　サブプロットの数
    """

    # sinogram
    shape = image.shape
    sinogram = image
   
    # initial
    recon = np.zeros(img.shape)
    rr, cc = circle(img.shape[0] / 2, img.shape[1] / 2, img.shape[0] / 2 - 1)
    recon[rr, cc] = 1

    # normalization matrix
    nview   = len(theta)
    norm    = np.ones(image.shape)
    wgts    = []
    for sub in range(n_sub):
        views   = range(sub, nview, n_sub)
        wgt = iradon(norm[:, views], theta=theta[views], filter=None, circle=True)
        wgts.append(wgt)

    mse = []

    # iteration
    for iter in tqdm(range(n_iter), desc="osem iter", leave=False):
        order   = np.random.permutation(range(n_sub))
        for sub in tqdm(order, desc="osem sub", leave=False):
            views   = range(sub, nview, n_sub)
            fp  = radon(recon, theta=theta[views], circle=True)
            ratio   = sinogram[:, views] / (fp + 1e-6)
            
            ################################################### bp = my_iradon(ratio, theta=theta[views], filter=None, circle=True)
            bp  = iradon(ratio, theta=theta[views], filter=None, circle=True)
            
            recon *= bp / (wgts[sub] + 1e-6)
#         print("osem loop"+str(iter))
        if iter > 0 : 
            #recon_sai = OSEM_sai(img= img, image =  image,theta=theta,n_iter = iter-1,n_sub = 1)
            #mse.append(compare_mse(recon_sai,np.array(recon)))
            mse.append(mean_squared_error(recon_sai,np.array(recon)))
            #psnr.append(compare_psnr(recon_sai,recon))
            #ssim.append(compare_ssim(recon_sai,recon))
        recon_sai = np.copy(recon)
    
    fig, ax = plt.subplots(figsize=(7,5), dpi=50)
    literature = ['2','4','6','8','10','12','14','16','18','20']

    plt.plot(mse)
    plt.tick_params(labelsize=14)
    ax.set_xticklabels(literature)

    #plt.title('ROI4 SUM', fontsize=18)
    plt.ylabel('MSE', fontsize=18)
    plt.xlabel('iterations', fontsize=18)
    #plt.legend(loc='best', fontsize=12)
    plt.grid()
    #plt.savefig('20181105_recon_imagenet_51200_iter200_L20/SSIM_python.png', dpi=100)
    plt.show()

    return recon