import numpy as np
import matplotlib.pyplot as plt

from scipy.fftpack import fft, ifft, fftfreq

from skimage.draw import circle
from skimage.metrics.simple_metrics import mean_squared_error
from skimage.transform import radon, rescale, iradon

from tqdm import tqdm_notebook as tqdm

def OSEM_TV(img, image, theta, n_iter=10, n_sub=1,):
    
    # PET sinogram
    shape = image.shape
    sinogram = image
   
    # initial
    recon = np.zeros(img.shape)
    rr, cc = circle(img.shape[0] / 2, img.shape[1] / 2, img.shape[0] / 2 - 1)
    recon[rr, cc] = 1

    # normalization matrix
    nview = len(theta)
    norm = np.ones(image.shape)
    wgts = []
    for sub in range(n_sub):
        views = range(sub, nview, n_sub)
        wgt = iradon(norm[:, views], theta=theta[views], filter=None, circle=True)
        wgts.append(wgt)
    
    mse = []
    
    # iteration
    for iter in tqdm(range(n_iter), desc="osem iter", leave=False):
        order = np.random.permutation(range(n_sub))
        for sub in tqdm(order, desc="osem sub", leave=False):
            views = range(sub, nview, n_sub)
            fp = radon(recon, theta=theta[views], circle=True)
            ratio = sinogram[:, views] / (fp + 1e-6)
            bp = iradon_TV(ratio, recon, theta=theta[views], filter=None, circle=True)
            recon *= bp / (wgts[sub] + 1e-6)
#         print("osem tv loop"+str(iter))
        if iter > 0 : 
            #recon_sai = OSEM_sai(img= img, image =  image,theta=theta,n_iter = iter-1,n_sub = 1)
            #mse.append(compare_mse(recon_sai,np.array(recon)))
            mse.append(mean_squared_error(recon_sai,np.array(recon)))
            #psnr.append(compare_psnr(recon_sai,recon))
            #ssim.append(compare_ssim(recon_sai,recon))
        recon_sai = np.copy(recon)
    
    fig, ax = plt.subplots(figsize=(7,5),dpi=50)
    literature = ['2','4','6','8','10','12','14','16','18','20']

    plt.plot(mse)
    plt.tick_params(labelsize=14)
    ax.set_xticklabels(literature)

    #plt.title('ROI4 SUM', fontsize=18)
    plt.ylabel('MSE', fontsize=18)
    plt.xlabel('iterations', fontsize=18)
    #plt.legend(loc='best',fontsize=12)
    plt.grid()
    #plt.savefig('20181105_recon_imagenet_51200_iter200_L20/SSIM_python.png', dpi=100)
    plt.show()
    
    return recon

def iradon_TV(radon_image, old_recon, theta=None, output_size=None, filter="ramp", interpolation="linear", circle=True):
    if radon_image.ndim != 2:
        raise ValueError('The input image must be 2-D')
    if theta is None:
        m, n = radon_image.shape
        theta = np.linspace(0, 180, n, endpoint=False)
    else:
        theta = np.asarray(theta)
    if len(theta) != radon_image.shape[1]:
        raise ValueError("The given ``theta`` does not match the number of "
                         "projections in ``radon_image``.")
    interpolation_types = ('linear', 'nearest', 'cubic')
    if interpolation not in interpolation_types:
        raise ValueError("Unknown interpolation: %s" % interpolation)
    if not output_size:
        # If output size not specified, estimate from input radon image
        if circle:
            output_size = radon_image.shape[0]
        else:
            output_size = int(np.floor(np.sqrt((radon_image.shape[0]) ** 2
                                               / 2.0)))
    if circle:
        radon_image = _sinogram_circle_to_square(radon_image)
    
    th = (np.pi / 180.0) * theta
    # resize image to next power of two (but no less than 64) for
    # Fourier analysis; speeds up Fourier and lessens artifacts
    projection_size_padded = \
        max(64, int(2 ** np.ceil(np.log2(2 * radon_image.shape[0]))))
    pad_width = ((0, projection_size_padded - radon_image.shape[0]), (0, 0))
    img = np.pad(radon_image, pad_width, mode='constant', constant_values=0)

    # Construct the Fourier filter
    f = fftfreq(projection_size_padded).reshape(-1, 1)   # digital frequency
    omega = 2 * np.pi * f                                # angular frequency
    fourier_filter = 2 * np.abs(f)                       # ramp filter
    if filter == "ramp":
        pass
    elif filter == "shepp-logan":
        # Start from first element to avoid divide by zero
        fourier_filter[1:] = fourier_filter[1:] * np.sin(omega[1:]) / omega[1:]
    elif filter == "cosine":
        fourier_filter *= np.cos(omega)
    elif filter == "hamming":
        fourier_filter *= (0.54 + 0.46 * np.cos(omega / 2))
    elif filter == "hann":
        fourier_filter *= (1 + np.cos(omega / 2)) / 2
    elif filter is None:
        fourier_filter[:] = 1
    else:
        raise ValueError("Unknown filter: %s" % filter)
    # Apply filter in Fourier domain
    projection = fft(img, axis=0) * fourier_filter
    radon_filtered = np.real(ifft(projection, axis=0))

    # Resize filtered image back to original size
    radon_filtered = radon_filtered[:radon_image.shape[0], :]
    reconstructed = np.zeros((output_size, output_size))
    # Determine the center of the projections (= center of sinogram)
    mid_index = radon_image.shape[0] // 2

    [X, Y] = np.mgrid[0:output_size, 0:output_size]
    xpr = X - int(output_size) // 2
    ypr = Y - int(output_size) // 2
    #print(xpr)
    #print(ypr)
    
    old_recon = (0.1*TV_Differential(old_recon))
    
    # Reconstruct image by interpolation
    for i in range(len(theta)):
        t = ypr * np.cos(th[i]) - xpr * np.sin(th[i])
        x = np.arange(radon_filtered.shape[0]) - mid_index
        if interpolation == 'linear':
            backprojected = np.interp(t, x, radon_filtered[:, i],
                                      left=0, right=0)
        else:
            interpolant = interp1d(x, radon_filtered[:, i], kind=interpolation,
                                   bounds_error=False, fill_value=0)
            backprojected = interpolant(t)
        reconstructed += backprojected
        #print(np.max(reconstructed))

    if circle:
        radius = output_size // 2
        reconstruction_circle = (xpr ** 2 + ypr ** 2) <= radius ** 2
        reconstructed[~reconstruction_circle] = 0.
    #fig = plt.figure(dpi=50)
    #plt.imshow(old_recon + (2 * len(th)),cmap="jet" ,interpolation='none')
    #plt.show()

    return (reconstructed * np.pi) / (old_recon + (2 * len(th)))
    #return (reconstructed * np.pi) / (2 * len(th))
    
def TV_Differential(lamda):
    TV_result = np.zeros((lamda.shape))
    old_lamda = lamda
    lamda = np.pad(lamda, [(1,1),(1,1)], 'constant')
    for i in range(1,old_lamda.shape[0]+1):
        for j in range(1,old_lamda.shape[1]+1):
            TV_result[i-1,j-1] = (((lamda[i,j]-lamda[i-1,j]) / TV_Norm(lamda, i-1, j)) + 
                                  ((lamda[i,j]-lamda[i,j-1]) / TV_Norm(lamda, i, j-1)) - 
                                  ((lamda[i+1,j] + lamda[i,j+1] - ( 2*lamda[i,j] )) / TV_Norm(lamda, i, j)))
    #fig = plt.figure(dpi=50)
    #plt.imshow(TV_result,cmap="jet" ,interpolation='none')
    #plt.show()
    return TV_result

def TV_Norm(lamda, m, n, epsilon=1e-6):
    return (np.sqrt((lamda[m+1,n] - lamda[m,n])**2 + (lamda[m, n+1] - lamda[m,n])**2 + epsilon))
    #return ((lamda[m+1,n]-lamda[m,n])**2 + (lamda[m,n+1]-lamda[m,n])**2 + epsilon)

def _sinogram_circle_to_square(sinogram):
    diagonal = int(np.ceil(np.sqrt(2) * sinogram.shape[0]))
    pad = diagonal - sinogram.shape[0]
    old_center = sinogram.shape[0] // 2
    new_center = diagonal // 2
    pad_before = new_center - old_center
    pad_width = ((pad_before, pad - pad_before), (0, 0))
    return np.pad(sinogram, pad_width, mode='constant', constant_values=0)