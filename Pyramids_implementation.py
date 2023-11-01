import numpy as np
import matplotlib.pyplot as plt
from scipy import signal








def matlab_style_gauss2D(shape=(3,3),sigma=0.5):
    """
    2D gaussian mask - should give the same result as MATLAB's
    fspecial('gaussian',[shape],[sigma])
    """
    m,n = [(ss-1.)/2. for ss in shape]
    y,x = np.ogrid[-m:m+1,-n:n+1]
    h = np.exp( -(x*x + y*y) / (2.*sigma*sigma) )
    h[ h < np.finfo(h.dtype).eps*h.max() ] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    return h

def downsample_by_a_factor_of_2(img):
    #if img.shape mod2!= 0 then padding
   
    arr = []
    
    if(img.shape[0]%2):
        
        #remove the last row
        img = np.delete(img , img.shape[0]-1 , axis=0)
        
        #pad one row of zeros
        #img = np.concatenate( (img,np.zeros( (1,img.shape[1]) ).astype(int) ) )
        
    if(img.shape[1]%2):
        
        #remove the last column
        img = np.delete(img , img.shape[1]-1 , axis=1)
        
        #pad one row of zeros
        #img = np.concatenate( ( img,np.zeros( (img.shape[0],1) ).astype(int) ) , axis=1 ) 
        
    
    for i in range(0,img.shape[0]-1,2):
        for j in range(0,img.shape[1]-1,2):
            arr.append((img[i,j]+img[i+1,j]+img[i,j+1]+img[i+1,j+1])/4)
            
    arr = np.array(arr)
    downsampled_img = arr.reshape(img.shape[0]//2,img.shape[1]//2)
            
    return downsampled_img

def upsampling_by_a_factor_of_2(a):
    #UPSAMPLING by a factor of 2 (we fill between 2 samples a 0 for each row and column respectively)
    a = np.insert( a, slice(1,None) , 0 , axis=0 )
    a = np.insert( a , slice(1,None) , 0 , axis=1 )
    
    #pad one row of zeros
    a = np.concatenate( (a ,np.zeros( (1,a.shape[1]) ).astype(int) ) )
    
    #pad one row of zeros
    a = np.concatenate( ( a,np.zeros( (a.shape[0],1) ).astype(int) ) , axis=1 ) 
    
    return a

def pyrDown(img , filter_kernel):
    
    #filtering stage
    #img_filtered_gaussian_for_the_current_lvl = np.real(ifft2(ifftshift(fftshift(fft2(img))*fftshift(fft2(filter_kernel,img.shape)))))
    img_filtered_gaussian_for_the_current_lvl = signal.convolve2d(img, filter_kernel ,mode="same")
    
    '''
    #getting the laplacian out of gaussian
    img_filtered_laplacian_for_the_current_lvl = img - img_filtered_gaussian_for_the_current_lvl
    '''
    
    #downsampling by a factor of 2 stage
    img_filtered_and_subsampled_for_the_next_lvl = downsample_by_a_factor_of_2(img_filtered_gaussian_for_the_current_lvl)
    
    return img_filtered_and_subsampled_for_the_next_lvl 

def pyrUp(low_res_img  , interpolation_filt , out_shape):
      
    img_upsampled = upsampling_by_a_factor_of_2(low_res_img)
    
    
    #checking compatibility of rows 
    if(img_upsampled.shape[0] > out_shape[0]):
        #remove the last row
        img_upsampled = np.delete(img_upsampled , img_upsampled.shape[0]-1 , axis=0)
        
    elif(img_upsampled.shape[0] < out_shape[0]):
        #pad one row of zeros
        img_upsampled = np.concatenate( (img_upsampled,np.zeros( (1,img_upsampled.shape[1]) ).astype(int) ) )
        
        
    #checking compadibility of columns
    if(img_upsampled.shape[1] > out_shape[1]):
        #remove the last column
        img_upsampled = np.delete(img_upsampled , img_upsampled.shape[1]-1 , axis=1)
             
    elif(img_upsampled.shape[1] < out_shape[1]):
        #pad one row of zeros
        img_upsampled = np.concatenate( ( img_upsampled,np.zeros( (img_upsampled.shape[0],1) ).astype(int) ) , axis=1 ) 
        
    
    
    #smooth out the discontinueties created by the upsampling (we use the same kernel as for the pyr_down multiplied by 4)
    #img_upsampled = np.real(ifft2(ifftshift(fftshift(fft2(img_upsampled))*fftshift(fft2(h,img_upsampled.shape)))))
    img_upsampled = signal.convolve2d(img_upsampled, interpolation_filt*4 ,mode="same")

    
    #img_next_lvl_recon = img_upsampled + L_minus_one_lapl_img
    
    img_next_lvl_recon = img_upsampled
    
    return img_next_lvl_recon


def get_arbitary_level_of_gaus_pyr( sig , filter_kernel ,  desired_level ):
    

    signal_input_to_the_nxt_lvl = signal.convolve2d(sig, filter_kernel ,mode="same")
    #signal_input_to_the_nxt_lvl = sig
    dictionary = {}
    dictionary["level0"] = signal_input_to_the_nxt_lvl
    
    for i in range(desired_level):
        signal_input_to_the_nxt_lvl   = pyrDown(signal_input_to_the_nxt_lvl , filter_kernel )
        dictionary["level"+str(i+1)] = signal_input_to_the_nxt_lvl 
    
    return dictionary


def get_lapl_pyr(gauss_pyr,filter_kernel):
    dictionary = {}
    #I_hat = gauss_pyr["level"+str(len(gauss_pyr)-1)]

    for i in range(len(gauss_pyr)-2 , -1 , -1):
        dst = pyrUp( gauss_pyr["level"+str(i+1)] , filter_kernel , gauss_pyr["level"+str(i)].shape )

        
        dictionary["level"+str(i)] =  gauss_pyr["level"+str(i)] - dst
        
    return dictionary


def reconstruct_blended_img(B_dict,filter_kernel):
    dictionary = {}
    recon = []
    
    dst = pyrUp(B_dict["g_0_L"] , filter_kernel , B_dict["b"+str(len(B_dict)-2)].shape )
    recon_tmp =  B_dict["b"+str(len(B_dict)-2)] + dst
    recon.append( B_dict["b"+str(len(B_dict)-2)] + dst )
    
    for i in range(len(B_dict)-3 , -1 , -1):
        
        dst = pyrUp(recon_tmp, filter_kernel , B_dict["b"+str(i)].shape )
        recon_tmp = B_dict["b"+str(i)] + dst
        recon.append( B_dict["b"+str(i)] + dst ) 
        
    return recon

def get_B_dict(Gaus_pyr_mask,Gaus_pyr_for_imgs,Lapl_pyr_for_imgs,pyramid_lvl):
    B_dict = {}
    for i in range(pyramid_lvl):
        B_dict["b"+str(i)] = Gaus_pyr_mask["level"+str(i)]*Lapl_pyr_for_imgs[0]["level"+str(i)] + (1 - Gaus_pyr_mask["level"+str(i)])*Lapl_pyr_for_imgs[1]["level"+str(i)]
                            
        g_0_L = Gaus_pyr_mask["level"+str(pyramid_lvl)]*Gaus_pyr_for_imgs[0]["level"+str(pyramid_lvl)] + (1 - Gaus_pyr_mask["level"+str(pyramid_lvl)])*(Gaus_pyr_for_imgs[1]["level"+str(pyramid_lvl)])
        
        B_dict["g_0_L"] = g_0_L

    return B_dict    

def plotting_pyramid(pyr_lvl,pyramid,pyr_type):

    if (pyr_type=="gaussian"):
        #gaussian pyr 
        f, axarr = plt.subplots(pyr_lvl+1)
        f.tight_layout()
        for i in range(pyr_lvl):
            axarr[i].imshow(pyramid["level"+str(i)] , cmap="gray" )
            axarr[i].title.set_text(pyr_type+'_level_'+str(i))
        axarr[pyr_lvl].imshow(pyramid["level"+str(pyr_lvl)] , cmap="gray" )
        axarr[pyr_lvl].title.set_text(pyr_type+'_level_'+str(pyr_lvl))


    if (pyr_type=="laplacian"):
        #laplacian pyr 
        f, axarr = plt.subplots(pyr_lvl+1)
        f.tight_layout()
        for i in range(pyr_lvl):
            axarr[i].imshow(pyramid["level"+str(i)] , cmap="gray" )
            axarr[i].title.set_text(pyr_type+'_level_'+str(i))

    if (pyr_type=="B_dict"):
        #Blended pyr
        f, axarr = plt.subplots(pyr_lvl+1,figsize=(40,40))
        f.tight_layout()
        for i in range(pyr_lvl):
            axarr[i].imshow(pyramid["b"+str(i)] , cmap="gray" )
            axarr[i].title.set_text("blended_laplacians_with\nsmooth_transition_lvl"+str(i))
        
        axarr[pyr_lvl].title.set_text("blended_gaussians_with\nsmooth_transition_lvl"+str(pyr_lvl))
        axarr[pyr_lvl].imshow( pyramid["g_0_L"] , cmap="gray")
        #plt.subplots_adjust(hspace=0.8)

    if (pyr_type=="cross-corellation"):
        #laplacian pyr 
        f, axarr = plt.subplots(pyr_lvl+1)
        f.tight_layout()
        for i in range(pyr_lvl):
            axarr[i].imshow(pyramid["level"+str(i)] , cmap="gray" )
            axarr[i].title.set_text(pyr_type+'response at level'+str(i))

 