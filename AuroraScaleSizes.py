
import numpy as np
from skimage.restoration import denoise_tv_chambolle,calibrate_denoiser
from skimage.morphology import binary_closing,remove_small_holes,remove_small_objects
from sklearn.mixture import GaussianMixture as GMM
from scipy.ndimage import distance_transform_edt
from skimage.draw import disk

def gaussian(x,A,mu,sigma):
    """
    Gaussian function
    A: float, amplitude
    mu: float, mean
    sigma: float, standard deviation
    """
    y = (A/(np.sqrt(2*np.pi*sigma**2)))*np.exp(-((x-mu)**2)/(2*sigma**2))
    return y

def assess(stack_sum,recon,means):
    ### instead of evaluating the brightness residuals between the ideal construction (based on the GMM) and the reconstruction
    ### based on the accepted points, we evaluate the residuals in the labels as this more accurately reflects the performance of the algorithm.
    ### e.g. a label resiudal of +/- 1 is disproportionately penalised if this is evaluated using the brightness values.
    
    ### we set the lowest brightness value in the reconstruction to zero as these are likely to be noise or very large scale background, which we aren't really interesting in.
    recon = np.where(recon == np.min(means),0,recon)
    uni = np.unique(recon)
    numbers_rc = np.zeros_like(recon)
    numbers_ss = np.zeros_like(stack_sum)
    for i in range(len(uni)):
        idx1 = np.where(recon == uni[i])
        ### create the label array of the ideal reconstruction
        numbers_rc[idx1] = i
        idx2 = np.where(stack_sum ==uni[i])
        ### create the label array of the found reconstruction
        numbers_ss[idx2] = i
        
    
    stack_sum = np.where(stack_sum == np.min(means),0,stack_sum)

    ### exclude the edges as they are more likely to be missed and artificially inflate the label residuals
    ss = stack_sum[25:231,25:231]
    ssn = numbers_ss[25:231,25:231]
    rc = recon[25:231,25:231]
    rcn = numbers_rc[25:231,25:231]


    err = np.sum(abs(rcn-ssn)) ### sum of absolute label residuals

    total_pix = np.where(rc!=0,1,0)

    #### total pixels with non-zero values in the reduced reconstruction array
    total_pix = np.sum(total_pix)

    ### mean label error
    err = err/total_pix

    ### pixels missed by the algorithm
    binary = np.where((rc!=0)&(ss==0),1,0)

    ### fraction of missed pixels
    missed_pix = np.sum(binary)/total_pix
    
    if np.isnan(missed_pix):
        return 1,np.nan,0
    else:
        return missed_pix,err,total_pix



def get_sizes(image,pred,mpp,NCL,threshes,base_mpp = 40,check_scale = 6*np.sqrt(2),require_neighbours = 3,require_local_levels = 3,size_delta = 1,neighbourhood1 = 5,neighbourhood2 = 3):

    """
    ----------------------
    PARAMETERS:
    ----------------------
    image: 2D array,  denoised image
    pred:  2D array,  reconstructed thresholded image using threshold means
    mpp: float, metres per pixel
    NCL: float, number of gaussians in the GMM
    threshes: 1D array, thresholds
    base_mpp: float, the bin spacing
    check_scale: float, if found scales have a radius <= to this then they are checked for neighbours and local levels
    require_neighbours: int, the number of neighbours of a similar size to found scale size radius (r) in the range: found_diameter/2 -size_delta <= r <= found_diameter/2 +size_delta
    size_delta: float, as defined above
    require_local_levels: the number of unique brightness values in pred required in the neighbourhood of the found point. Designed to remove 'noise' in low brightness gradient areas.
    neighbourhood1: float, the disk radius = np.max([r*neighbourhood1,neighbourhood1]), defining the neighbourhood to check for require_local_levels
    neighbourhood2: float, the disk radius = np.max([r*neighbourhood2,neighbourhood2]), defining the neighbourhood to check for require_neighbours
    ----------------------
    RETURNS:
    ----------------------
    ALL_DISTS: list, width of structures in metres for all accepted points
    ALL_POWS: list, brightness values assigned to each accepted point
    ALL_IX: list, pixel x coordinates of accepted points
    ALL_IY: list, pixel y coordinates of accepted points
    STACK: 3d float array, dimensions (rows,columns,scale sizes). When a scale size is found, a disk of radius scale size/2 and centred on the accepted point
            with coordinates (rows,columns) is filled in. Which level of STACK is filled in is determined by scale size. This keeps the power at each scale separate for later analysis
            and also prevents counting pixels more than once for overlapping circles.
    means: float array, the brightness values assigned to each pair of thesholds
    NCL: int, number of gaussians in the GMM
    TEMP: float array, the thresholds, len should be len(means)+1
    ----------------------
    ----------------------
    """
    im_dims = np.shape(image) ### image dimensions
    im_dim_y = im_dims[0]
    im_dim_x = im_dims[1]
    
    #### if there is very little brightness range, force len(TEMP) such that this frame is skipped and return empty arrays ###
    if ((np.max(image)-np.min(image) < 2500)): #2000
        n_cl = 1 
        TEMP = np.array([np.mean(image)])

    else:
        n_cl = NCL  ### number of gaussians in the GMM
        TEMP = np.copy(threshes)  ### the brightness thresholds based on the GMM


    means = np.unique(pred) ### the brightness values assigned to each threshold

    imeans = np.argsort(means)
    means = means[imeans] ### sorted in ascending order
    
    ALL_DISTS = []  ### initialise list to hold distances between centre of feature and masked background
    ALL_POWS = [] ### initialise list to hold the assigned brightness values
    
    
    ALL_IX = [] ### initialise list to hold the x pixel coordinates of each found point
    ALL_IY = [] ### initialise list to hold the y pixel coordinates of each found point
    
    
    STACK = np.zeros((im_dim_y,im_dim_x,256)) ### dims: im_dim_y,im_dim_x,scale size
    
    BINS = np.linspace(base_mpp,256*base_mpp,257)
    
    for ii in range(1,len(TEMP)-1):  ### loop through the thresholds in increasing order of brightness
        subA = np.where(pred<TEMP[ii],0,1)  ### mask pixels below this threshold
        subB = remove_small_holes(subA)  ### tries to remove noise
        subC = remove_small_objects(subB) ### tries to remove noise
        sub1 = binary_closing(subC) ### tries to remove noise

        
        ### Array is padded so edge sizes are included in the euclidean distance transform.
        ### You may want to exclude edges as the extent of these structures outside the FOV is uncertain; in this case do distance_transform_edt on sub1, i.e., without padding:
        ### edt = distance_transform_edt(sub1)
        edt = distance_transform_edt(np.pad(sub1,1))[1:im_dim_y+1,1:im_dim_x+1]

        ### the gradient of the distance transform; if the larger gradients will be associated with points away from the centre of features and therefore
        ### will no represent the scale size of the feature
        egrad = np.gradient(edt)
        egrad = np.sqrt(egrad[0]**2+egrad[1]**2)

        ### the gradient is evaluated as (I_{x+1} - I_{x-1})/2h, so where I_{x} = 0 at the masked/unmasked boundary there will be a non-zero gradient value. Multiplying by sub1
        ### makes these boundary cases zero
        
        mod_egrad = egrad*sub1

        
        ### the first sift. Want values away from the edges of the mask and not in the masked pixels (i.e, where edt > 0) and where the gradient in edt is small.
        ### For example, consider an ideal arc that doesn't change width along its length. Along the centre of the arc, parrallel to the arc axis
        ### the edt is constant, so the edt gradient is zero. Perpendicular to the arc axis, the edt gradient is also zero, as the perpendicular cut of edt
        ### looks like [..E-2,E-1,E,E-1,E-2...] where E is the edt value at the centre of the arc, and the gradient is calculated as ((E-1) - (E-1))/2 = 0

        ### it turns out mod_egrad <=0.5 is the threshold required to detect the smallest structures that are not parallel to either image axis
        mask = np.where((((edt>0)&(mod_egrad <= 0.5))), edt, 0)

        mask2 = np.copy(mask)
        
        

            
        
        idx = np.where(mask2!=0)
        
        mask3 = np.copy(mask2)
        
        
        
        INDEX_X = idx[1]
        INDEX_Y = idx[0]
        
        DISTS = edt[idx]

        dsort = np.argsort(-DISTS)
        DISTS = DISTS[dsort]
        INDEX_X = INDEX_X[dsort]
        INDEX_Y = INDEX_Y[dsort]


        ### Identify points for checking below a certain scale size. This is a denoising step. Smaller check_scale increases the chances of allowing noise,
        ### but a larger check_scale increases the chances of missing subtle structures.
        IDXa = np.where(DISTS <=check_scale)
           
        
        for kk in range(len(INDEX_X[IDXa])):
            ### distance to the edge of the mask in pixels
            r = DISTS[IDXa][kk]
            if r:
                ### defining the radius of the neighbourhood to check

                ### for a brightness gradient
                plus = np.max([r*neighbourhood1,neighbourhood1])

                ### for neighbours of a similar scale size
                plus2 = np.max([r*neighbourhood2,neighbourhood2])
                
                ### pixel coordinates of the disk neighbourhoods
                rr,cc = disk((INDEX_Y[IDXa][kk],INDEX_X[IDXa][kk]),plus,shape = np.shape(image))
                rr2,cc2 = disk((INDEX_Y[IDXa][kk],INDEX_X[IDXa][kk]),plus2,shape = np.shape(image))

                ### the two patches to check
                patch = np.copy(pred[rr,cc])
                patch2 = np.copy(mask2[rr2,cc2])

                ### size_delta determines how different neighbouring scale sizes can be to still be counted as 'similiar' in terms of scale size.
                ### Smaller size delta is more strict and decreses the noise, but increases the chances of excluded true points.
                sub_patch2 = np.where(((patch2 <= r+size_delta)&(patch2 !=0)&(patch2>=r-size_delta)))  ### finds all non-zero edt values within this scale size +/- size_delta
                uni_patch= np.unique(patch)
                if len(uni_patch)<require_local_levels:
                    ### noise is more likely to be in regions where there are only two brightness values in the pred array
                    mask3[INDEX_Y[IDXa][kk],INDEX_X[IDXa][kk]] = 0
                if len(sub_patch2[0])<require_neighbours:
                    ### if insufficient neighbours, set to zero in mask3 (mask2 remains unchanged so they can still contribute to the evaluation of other points)
                    mask3[INDEX_Y[IDXa][kk],INDEX_X[IDXa][kk]] = 0

        ### this is the indices of final accepted points after efforts to denoise
        idx = np.where((mask3!=0))
        
        

        ### the x and y pixel indices of the accepted points
                    
        INDEX_X = idx[1]
        INDEX_Y = idx[0]


        #### the distance of the accepted points to the edge of the mask in pixels (this is a radius; the diameter is the width of the structure)
        DISTS = edt[idx]
        
        
            
        ### this bit fills in STACK 
        for kk in range(len(INDEX_X)):
            
            r = DISTS[kk]
            id0 = np.digitize([r*2*mpp],BINS*2)[0]
            
            id0-=1
            id0 = np.max([id0,0])
            if id0 not in [-1,256]:
                ### disk coordinate
                rr,cc = disk((INDEX_Y[kk],INDEX_X[kk]),r,shape = np.shape(image))
    
                ### id0 is the index of the scale size in the stack
                ### rr,cc are the coordinates of a disk
                STACK[rr,cc,id0] = means[ii]
                    
        ### these are the scale sizes (diameters), now in metres
        ALL_DISTS+=list(DISTS*mpp*2)

        ### these are the brightness values
        ALL_POWS+=list(pred[idx])
        
        ### these are the pixel indices
        ALL_IX+=list(INDEX_X)
        ALL_IY+=list(INDEX_Y)
        
        
    return ALL_DISTS,ALL_POWS,ALL_IX,ALL_IY,STACK,means,n_cl,TEMP


def aurora_power(image0,mpp,base_mpp = 40,check_scale = 6*np.sqrt(2),require_neighbours = 3,require_local_levels = 3,size_delta = 1,\
                 neighbourhood1 = 5,neighbourhood2 = 3,flat_cut = 2500,prev_denoise = None):
    '''
    -------------------------
    PARAMETERS:
    -------------------------
    image0: 2d array; the original (noisy) image
    
    mpp: float; metres per pixel
    
    base_mpp: float; determines the scale bin sizes (metres) – should be similar to mpp (e.g., 40 for ASK without telescope; 20 with telescope)
    
    check_scale: float; if found scales have a radius <= to this then they are checked for neighbours and local levels
    
    require_neighbours: int; the number of neighbours of a similar size to found scale size radius (r) in the range: found_diameter/2 -size_delta <= r <= found_diameter/2 +size_delta

    size_delta: float; as defined above
    
    require_local_levels: int; the number of unique brightness values in pred required in the neighbourhood of the found point. Designed to remove 'noise' in low brightness gradient areas.
    
    neighbourhood1: float; the disk radius = np.max([r*neighbourhood1,neighbourhood1]), defining the neighbourhood to check for require_local_levels
    
    neighbourhood2: float; the disk radius = np.max([r*neighbourhood2,neighbourhood2]), defining the neighbourhood to check for require_neighbours
    
    flat_cut: int; if the denoised image max-min < flat_cut, then the frame is skipped as it is likely that any scales found will be due to noise.
    -------------------------
    RETURNS:
    Note: INT_BRIGHT_SCALE, STACK_FRAC, and CENTRES are probably the most useful for assessing scale-dependent power.
          To get the ideal reconstruction based on the GMM, plt.imshow(recon).
          To get the reconstruction based on the found scale sizes, plt.imshow(np.max(STACK,axis = 2)).
    -------------------------
    im0: 2d float array, the denoised image
    
    recon: 2d float array, the ideal reconstructed image based on the GMM. We don't actually want to be able to perfectly reconstruct this as this will inevitably have some noise in it.
    
    STACK: 3d float array, dimensions (rows,columns,scale sizes). When a scale size is found, a disk of radius scale size/2 and centred on the accepted point
            with coordinates (rows,columns) is filled in. Which level of STACK is filled in is determined by scale size. This keeps the power at each scale separate for later analysis
            and also prevents counting pixels more than once for overlapping circles.
            
    INT_BRIGHT_SCALE: float array, the integrated brightness at each scale size. It is the sum of brightnesses at each scale size level in stack (i.e., np.sum(STACK, axis = (0,1)))
    
    STACK_FRAC: float array, number of pixels found at each scale size divided by total image pixels (gives indication of how much of the image is covered by each scale size)
    
    BINARY_STACK: 3d binary array, 1 indicates a found pixel in STACK, 0 indicates no found pixel
    
    LABEL_STACK: 3d float array, labels assigned to each brightness value in STACK
    
    FOUND_PIX: int, total number of pixels found (not edge-reduced)
    
    TOTAL_PIX: float, total pixels to be found in the edge-reduced array (see assess function)
    
    ASSESS: float, fraction of pixels missed in the edge-reduced array by the algorithm (see assess function)
    
    ERR: float, the mean absolute 'label' error (rather than brightness error) (see assess function)
    
    BINS: float array, the bins of feature scale sizes /2 (i.e., radii)
    
    CENTRES: float array, the bin centres of feature scale sizes (diameters)
    
    TEMP: float arrray, the thresholds determined from the intersection of each gaussian in the GMM
    
    means: float array, the brightness values assigned to each pair of thresholds
    
    GAUSSES: list of float arrays, the idividual guassians in the GMM evaluated using gauss_x (see below)
    
    TOTAL_GAUSS: float array, the sum of all the gaussians in GAUSSES. Can be plotted to see how well the GMM fitted to the denoised image histogram.
    
    gauss_x: float array, range of brightness values used to evaluate the thresholds of the GMM. It's more for interest if you want to plot them, but you don't really need it.
    
    prev_denoise: int, the denoiser parameter determined for the previous image in the sequence. If provided, reduces the parameter space search time for this image.
    -------------------------
    '''
    BINS = np.linspace(base_mpp,256*base_mpp,257) ### radius bins
    CENTRES = (BINS[1:]+BINS[:-1])*0.5*2  ### the diameter bin centres
    if np.max(image0)-np.min(image0)>6000:
        if prev_denoise:  ### assume the noise level doesn't change too much between frames, so only search parameter space in a small range either side of the previous value. Takes less time.
            parameters = {'weight':np.arange(np.max([500,prev_denoise-1000]),prev_denoise+1000+250,250)}
            denoising_function = calibrate_denoiser(image0, denoise_tv_chambolle,denoise_parameters=parameters,stride = 4)
            denoising_function.keywords['denoiser_kwargs']['weight']=np.max([1000,denoising_function.keywords['denoiser_kwargs']['weight']-250])
            prev_denoise = denoising_function.keywords['denoiser_kwargs']['weight']


        else:    ### get the first denoise parameter; search a broad parameter space. Takes a while.
            parameters = {'weight':np.arange(1000,9500,250)}
            denoising_function = calibrate_denoiser(image0, denoise_tv_chambolle,denoise_parameters=parameters,stride = 4)
            denoising_function.keywords['denoiser_kwargs']['weight']=np.max([1000,denoising_function.keywords['denoiser_kwargs']['weight']-250])
            prev_denoise = denoising_function.keywords['denoiser_kwargs']['weight']


        ## the denoise image
        im0 = denoising_function(image0)
        
        
        ### only proceed if the image isn't too flat ###
        if ((np.max(im0)-np.min(im0) > flat_cut)): # 2500
               
                
                
            NCL = np.arange(3,16)#20
            X = np.ravel(im0).reshape(-1, 1)
            best_aic = 10000000
            best_ncl = 1
            aics = []


            #### determining the optimal number of gaussians in the GMM ####
            for ii in range(len(NCL)):
                if NCL[ii]:#<len(bin_ed)-2:

                    gmm = GMM(n_components=NCL[ii]).fit(X)
                    means0 = gmm.means_
                    covs0 = gmm.covariances_
                    weights0 = gmm.weights_
                    under_weight = weights0[weights0<0.05]
                    cov = covs0[:,0,0]
                    low_cov = cov[cov<250000]
                    
                    
                    AIC = gmm.aic(X)+ (2*(NCL[ii]*3)**3.75 + 2*(3*NCL[ii]))/(np.max([np.std(im0)/(np.max([len(under_weight)+len(low_cov),1])),3*NCL[ii]+2]) - 3*NCL[ii] - 1)
                    
                    aics.append(AIC)
                    

                    if AIC<best_aic:
                        best_aic = AIC
                        best_ncl = NCL[ii]
                        best_means = means0
                        best_covs = covs0
                        best_weights = weights0
            aics = np.array(aics)
            
            GAUSSES = []
            gauss_x = np.arange(np.min(im0),np.max(im0)+1)
            TOTAL_GAUSS = np.zeros(len(gauss_x))
            best_means = best_means[:,0]
            best_covs = best_covs[:,0,0]
            sor = np.argsort(best_means)
            best_means = best_means[sor]
            best_covs = best_covs[sor]
            best_weights = best_weights[sor]
            for ii in range(len(best_means)):
                gau = gaussian(gauss_x,best_weights[ii]*256*256,best_means[ii],np.sqrt(best_covs[ii]))
                TOTAL_GAUSS = TOTAL_GAUSS+gau
                GAUSSES.append(gau)
                
            THRESHES = []
            for ii in range(len(GAUSSES)-1):
                try:
                    f = GAUSSES[ii]
                    g = GAUSSES[ii+1]
                    idx = np.argwhere(np.diff(np.sign(f - g))).flatten()
                    yvals = f[idx]
                    idx2 = np.argmax(yvals)
                    thr = gauss_x[idx[idx2]]
                    THRESHES.append(thr)
                except:
                    THRESHES.append(best_means[ii]-np.sqrt(best_covs[ii])*2.355/2)
            THRESHES.append(np.max(im0)+1)
            THRESHES = np.array(THRESHES)
            THRESHES = THRESHES[THRESHES>2500]
            THRESHES = list(THRESHES)
            THRESHES.append(np.min(im0)-1)
            THRESHES = np.array(THRESHES)
            THRESHES = THRESHES[np.argsort(THRESHES)]
            
            
            if len(THRESHES)>1:
                MEANS = 0.5*(THRESHES[1:]+THRESHES[:-1])
                
                
                
                recon = np.zeros(np.shape(im0))
        
                for ii in range(len(THRESHES)-1):
                    ix = np.where((im0<THRESHES[ii+1])&(im0>=THRESHES[ii]))
                    recon[ix] = MEANS[ii]
                recon = np.where(im0>=THRESHES[-1],MEANS[-1],recon)
                recon = np.where(im0<THRESHES[0],MEANS[0],recon)
             
                
                image = np.copy(im0)

                
                ALL_DISTS,ALL_POWS,ALL_IX,ALL_IY,STACK,means,NCL,TEMP = get_sizes(im0,recon,mpp,best_ncl,THRESHES,base_mpp=base_mpp,\
                                                                                  check_scale = check_scale,require_neighbours = require_neighbours,require_local_levels = require_local_levels,\
                                                                                  size_delta = size_delta,neighbourhood1 = neighbourhood1,neighbourhood2 = neighbourhood2)
                assert np.allclose(means,MEANS)
                if len(means)==0:
                    means = np.array([np.mean(im0)])
                     
                INT_BRIGHT_SCALE = np.sum(STACK,axis = (0,1))
                LABEL_STACK = np.zeros_like(STACK)
                for ii in range(len(STACK[0][0])):
                    idxl = np.where(STACK[:,:,ii] !=0)
                    LABEL_STACK[idxl[0],idxl[1],ii] = len(STACK[0][0])-ii
                LABEL_STACK = np.max(LABEL_STACK,axis = 2)
                    
                BINARY_STACK = np.where(STACK ==0,0,1)
                FOUND_PIX = np.sum(np.max(BINARY_STACK,axis = 2))
                STACK_FRAC = np.sum(BINARY_STACK,axis = (0,1))/len(np.ravel(image))
                ASSESS,ERR,TOTAL_PIX = assess(np.max(STACK,axis = 2),recon,means)
            return im0,recon,STACK,INT_BRIGHT_SCALE,STACK_FRAC,BINARY_STACK,LABEL_STACK,FOUND_PIX,TOTAL_PIX,ASSESS,ERR,BINS,CENTRES,TEMP,means,GAUSSES,TOTAL_GAUSS,gauss_x,prev_denoise
        else:
            print('Brightness range is insufficient; the image probably lacks features and/or is noisy')
            return image0,im0,CENTRES,prev_denoise
    else:
        print('Brightness range is insufficient; the image probably lacks features and/or is noisy')
        return image0,image0,CENTRES,prev_denoise



def plotting(image,im0,recon,CENTRES,STACK,INT_BRIGHT_SCALE,STACK_FRAC):
                
    fig = plt.figure()
    ax =fig.add_subplot(321)
    ax2 = fig.add_subplot(322)
    ax3 = fig.add_subplot(323)
    ax4 = fig.add_subplot(324)
    ax5 = fig.add_subplot(325)
    ax6 = fig.add_subplot(326)
    ax.imshow(image,origin = 'lower')
    ax.set_title('Original')
    ax2.imshow(im0,origin = 'lower')
    ax2.set_title('Denoised')
    ax3.imshow(recon,origin = 'lower')
    ax3.set_title('Ideal reconstruction')
    ax4.imshow(np.max(STACK,axis = 2),origin = 'lower')
    ax4.set_title('Scale size reconstruction')
    ax5.plot(CENTRES,INT_BRIGHT_SCALE)
    ax5.set_xscale('log')
    ax5.set_yscale('log')
    ax5.set_ylabel('Inegrated Brightness')
    ax5.set_xlabel('Scale')
    ax6.plot(CENTRES,STACK_FRAC)
    ax6.set_xscale('log')
    ax6.set_ylabel('Fractional Coverage')
    ax6.set_xlabel('Scale')
    plt.show()
