from Pyramids_implementation import *
import cv2
from scipy.signal import fftconvolve




dims = input("Give the dimensions of the gaussian kernel FOR THE IMAGES\n ( expected structure rows,cols ):")
dims = tuple(map(int, dims.split(','))) 
var = input("Give the variance of the gaussian kernel FOR THE IMAGES:")

dims1 = input("Give the dimensions of the gaussian kernel FOR THE MASK\n ( expected structure rows,cols ):")
dims1 = tuple(map(int, dims1.split(','))) 
var1 = input("Give the variance of the gaussian kernel FOR THE MASK:")




#SOME FUNCTIONS-----------------------------------------------------------------------------------------
def normxcorr2(template, image, mode="full"):
    """
    Input arrays should be floating point numbers.
    :param template: N-D array, of template or filter you are using for cross-correlation.
    Must be less or equal dimensions to image.
    Length of each dimension must be less than length of image.
    :param image: N-D array
    :param mode: Options, "full", "valid", "same"
    full (Default): The output of fftconvolve is the full discrete linear convolution of the inputs. 
    Output size will be image size + 1/2 template size in each dimension.
    valid: The output consists only of those elements that do not rely on the zero-padding.
    same: The output is the same size as image, centered with respect to the ‘full’ output.
    :return: N-D array of same dimensions as image. Size depends on mode parameter.
    """

    # If this happens, it is probably a mistake
    if np.ndim(template) > np.ndim(image) or \
            len([i for i in range(np.ndim(template)) if template.shape[i] > image.shape[i]]) > 0:
        print("normxcorr2: TEMPLATE larger than IMG. Arguments may be swapped.")

    template = template - np.mean(template)
    image = image - np.mean(image)

    a1 = np.ones(template.shape)
    # Faster to flip up down and left right then use fftconvolve instead of scipy's correlate
    ar = np.flipud(np.fliplr(template))
    out = fftconvolve(image, ar.conj(), mode=mode)

    image = fftconvolve(np.square(image), a1, mode=mode) - \
            np.square(fftconvolve(image, a1, mode=mode)) / (np.prod(template.shape))

    # Remove small machine precision errors after subtraction
    image[np.where(image < 0)] = 0

    template = np.sum(np.square(template))
    out = out / np.sqrt(image * template)

    # Remove any divisions by 0 or very close to 0
    out[np.where(np.logical_not(np.isfinite(out)))] = 0

    return out

def display_image_in_actual_size(img,title):

    dpi = 80
    im_data = img
    height, width = im_data.shape

    # What size does the figure need to be in inches to fit the image?
    figsize = width / float(dpi), height / float(dpi)

    # Create a figure of the right size with one axes that takes up the full figure
    fig = plt.figure(figsize=figsize)
    ax = fig.add_axes([0, 0, 1, 1])

    # Hide spines, ticks, etc.
    ax.axis('off')

    # Display the image.
    ax.imshow(im_data, cmap='gray')

    #add title
    ax.set_title(title)

def downscale_image(img,scale_percent):

    #image resize (because the scene image is too large and the convolutions to find the template are slow)
    # percent of original size
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)
    
    # resize image
    resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)

    return resized
    #----------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------------





noisy = int(input("Enter 1 for noisy patern or 0 for non noisy patern\n: "))
if noisy:
    pattern = cv2.imread('noisy_pattern.bmp',0).astype(np.float64)
else:
    pattern = cv2.imread('pattern.bmp',0).astype(np.float64)
scene = cv2.imread('scene.bmp',0).astype(np.float64)
    
#resize images for faster processing (perform the fft faster)
scale_percent = 50
pattern = downscale_image(pattern,50)
scene = downscale_image(scene,scale_percent)   

#Getting the gaussian and then the laplacian pyramid of the scene (first aproach)
h = matlab_style_gauss2D((5,5),1)
pyr_dict = get_arbitary_level_of_gaus_pyr(scene,h,4)
lapl_pyr = get_lapl_pyr(pyr_dict,h)

dicti = {}
max_list = []
for i in range(len(lapl_pyr)):

    dicti["level"+str(i)] = normxcorr2(pattern, lapl_pyr["level"+str(i)]  , mode="same")
    max_list.append ( np.max(dicti["level"+str(i)]) )


#finding the highest response of the cross corellation at the apropriate scale
aprop_level = np.argmax(max_list)
ind = np.unravel_index(np.argmax(dicti["level"+str(aprop_level)], axis=None), lapl_pyr["level"+str(aprop_level)].shape) 

#drawing a rectangle around that point at the apropriate level of the lapl pyramid 
(height,width) = np.shape(pattern)
cv2.rectangle(lapl_pyr["level"+str(aprop_level)], (ind[1]-width//2, ind[0]+height//2), (ind[1]+width//2, ind[0]-height//2), (255,0,0), 2)
lel1 = pyr_dict["level"+str(aprop_level)] + lapl_pyr["level"+str(aprop_level)]
tmp = pyrUp(lel1, h , lapl_pyr["level"+str(aprop_level-1)].shape)

#reconstructing image with the recognized template (with drawn rectangle indicator) using the already computed gaussian pyramid
for i in range(aprop_level-2,0,-1):
    tmp = pyrUp(tmp,h,i)



plotting_pyramid(len(dicti),dicti,"cross-corellation")
plt.plot(  max_list)
plt.title("Normalized cross-corellation maxima response")
plt.xlabel("level of lapl pyr")
plt.ylabel("response")
plt.figure()
display_image_in_actual_size(tmp,'')
plt.show()