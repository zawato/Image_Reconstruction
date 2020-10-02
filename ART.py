import numpy as np
import matplotlib.pyplot as plt

from skimage.draw import circle
from skimage.metrics.simple_metrics import mean_squared_error
from skimage.transform import radon, rescale, iradon

from tqdm import tqdm_notebook as tqdm

def ART(img, image, theta, n_iter=10):

    """
    img 再構成画像
    image 投影データ
    theta 投影角度
    n_iter イテレーション
    """

    # sinogram
    shape = image.shape
    sinogram = image

    # initial
    recon = np.zeros(shape)
    rr, cc = circle(shape[0] / 2, shape[1] / 2, shape[0] / 2 - 1)
    recon[rr, cc] = 1

    # normalization matrix
    n_sub = len(theta)
    nview = len(theta)
    norm = np.ones(shape)
    wgts = []
    for sub in range(n_sub):
        views = range(sub, nview, n_sub)
        wgt = iradon(norm[:, views], theta=theta[views], filter=None, circle=True)
        wgts.append(wgt)

    mse = []

    # iteration
    for iter in tqdm(range(n_iter), desc="art iter", leave=False):
        order = np.random.permutation(range(n_sub))
        for sub in tqdm(order, desc="art sub", leave=False):
            views = range(sub, nview, n_sub)
            fp = radon(recon, theta=theta[views], circle=True)
            diff = sinogram[:, views] - fp
            bp = iradon(diff, theta=theta[views], filter=None, circle=True) 
            recon += bp / (wgts[sub] + 1e-6)

#             ratio = sinogram[:, views] / (fp + 1e-6)
#             bp = iradon(ratio, theta=theta[views], filter=None, circle=True) 
#             recon *= bp / (wgts[sub] + 1e-6)
        
        plt.imshow(recon, cmap='gray')
        plt.title("%d / %d" % (iter + 1, n_iter))
        plt.pause(1)
        plt.close()
#         if iter > 0 : 
#             mse.append(mean_squared_error(img, np.array(recon)))
            
#         recon = np.copy(recon)
        
#     fig, ax = plt.subplots(figsize=(7,5), dpi=50)
#     literature = ['2','4','6','8','10','12','14','16','18','20']

#     plt.plot(mse)
#     plt.tick_params(labelsize=14)
#     ax.set_xticklabels(literature)

#     plt.ylabel('MSE', fontsize=18)
#     plt.xlabel('iterations', fontsize=18)
#     plt.grid()
#     plt.show()

# def ART(A, AT, b, x, mu=1e0, niter=1e2, bpos=True):

#     ATA = AT(A(np.ones_like(x)))

#     for i in range(int(niter)):

#         x = x + np.divide(mu * AT(b - A(x)), ATA)

#         if bpos:
#             x[x < 0] = 0

#         plt.imshow(x, cmap='gray')
#         plt.title("%d / %d" % (i + 1, niter))
#         plt.pause(1)
#         plt.close()

#     return x

    return recon