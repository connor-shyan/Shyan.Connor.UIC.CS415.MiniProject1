#
# CS 415 - Mini Project 1: Programming
# Connor Shyan, UIN 650119775
# UIC Fall 2022
# Built upon Image Filtering code tutorial
#

import cv2
import numpy as np

#
# Correlation Function
#
def correlation(im, kernel):
    im_height, im_width, im_channels = im.shape
    kernel_size = kernel.shape[0]
    pad_size = int((kernel_size-1)/2)
    im_padded = np.zeros((im_height+pad_size*2, im_width+pad_size*2, im_channels))
    im_padded[pad_size:-pad_size, pad_size:-pad_size, :] = im

    im_out = np.zeros_like(im)
    for c in range(im_channels):
        for x in range(im_width):
            for y in range(im_height):
                im_patch = im_padded[y:y+kernel_size, x:x+kernel_size, c]
                new_value = np.sum(kernel*im_patch)
                new_value = (int)(min(max(0, new_value), 255)) # bound the pixel within (0, 255)
                im_out[y, x, c]= new_value
    return im_out

#
# Convolution Function (P1)
#
def convolution(im, kernel):
    # Flipping the kernel
    kernel = np.fliplr(np.flipud(kernel))
    # Returning output from correlation with flipped kernel
    return correlation(im, kernel)

#
# Function to generate Gaussian kernel
# Referenced from Edge Detection code tutorial
#
def get_gaussian_kernel(kernel_size, sigma):
    kernel_x = np.linspace(-(kernel_size // 2), kernel_size // 2, kernel_size)
    for i in range(kernel_size):
        kernel_x[i] = np.exp(-(kernel_x[i] / sigma)**2 / 2)
    kernel = np.outer(kernel_x.T, kernel_x.T)

    kernel *= 1.0 / kernel.sum()
    return kernel

#
# Median Filter Function (P2)
#
def median_filter(im, kernel_size):
    im_height, im_width, im_channels = im.shape
    pad_size = int((kernel_size - 1) / 2)
    im_padded = np.zeros((im_height + pad_size * 2, im_width + pad_size * 2, im_channels))
    im_padded[pad_size:-pad_size, pad_size:-pad_size, :] = im

    im_out = np.zeros_like(im)
    for c in range(im_channels):
        for x in range(im_width):
            for y in range(im_height):
                im_patch = im_padded[y:y + kernel_size, x:x + kernel_size, c]
                new_value = np.median(1 * im_patch)
                im_out[y, x, c] = new_value
    return im_out

#
# From code tutorial
#
# kernel = np.array([[1,1,1],
#                    [1,1,1],
#                    [1,1,1]])/9
# im = cv2.imread("lena.png")
# im = im.astype(float)
# im_out = correlation(im, kernel)
# im_out = im_out.astype(np.uint8)
# cv2.imwrite('output_image.png', im_out)
# cv2.imshow("Output", im_out)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

#
# Using Convolution to apply mean filter to lena.png with kernel size 3x3
#
# kernel = np.array([[1,1,1],
#                    [1,1,1],
#                    [1,1,1]])/9
# im = cv2.imread("lena.png")
# im = im.astype(float)
# im_out = convolution(im, kernel)
# im_out = im_out.astype(np.uint8)
# cv2.imwrite('lena_convolution_mean_3x3.png', im_out)
# cv2.imshow("Output", im_out)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

#
# Using Convolution to apply Gaussian (std=1) filter to lena.png with kernel size 3x3
#
# kernel = get_gaussian_kernel(3, 1)
# im = cv2.imread("lena.png")
# im = im.astype(float)
# im_out = convolution(im, kernel)
# im_out = im_out.astype(np.uint8)
# cv2.imwrite('lena_convolution_gaussian_3x3.png', im_out)
# cv2.imshow("Output", im_out)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

#
# Using Convolution to apply sharpening filter to lena.png with kernel size 3x3
#
# kernel = np.array([[0,0,0],
#                    [0,2.0,0],
#                    [0,0,0]])
# kernel -= np.array([[1,1,1],
#                    [1,1,1],
#                    [1,1,1]])/9
# im = cv2.imread("lena.png")
# im = im.astype(float)
# im_out = convolution(im, kernel)
# im_out = im_out.astype(np.uint8)
# cv2.imwrite('lena_convolution_sharpening_3x3.png', im_out)
# cv2.imshow("Output", im_out)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

#
# Using Convolution to apply mean filter to lena.png with kernel size 5x5
#
# kernel = np.array([[1,1,1,1,1],
#                    [1,1,1,1,1],
#                    [1,1,1,1,1],
#                    [1,1,1,1,1],
#                    [1,1,1,1,1]])/25
# im = cv2.imread("lena.png")
# im = im.astype(float)
# im_out = convolution(im, kernel)
# im_out = im_out.astype(np.uint8)
# cv2.imwrite('lena_convolution_mean_5x5.png', im_out)
# cv2.imshow("Output", im_out)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

#
# Using Convolution to apply Gaussian (std=1) filter to lena.png with kernel size 5x5
#
# kernel = get_gaussian_kernel(5, 1)
# im = cv2.imread("lena.png")
# im = im.astype(float)
# im_out = convolution(im, kernel)
# im_out = im_out.astype(np.uint8)
# cv2.imwrite('lena_convolution_gaussian_5x5.png', im_out)
# cv2.imshow("Output", im_out)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

#
# Using Convolution to apply sharpening filter to lena.png with kernel size 5x5
#
# kernel = np.array([[0,0,0,0,0],
#                    [0,0,0,0,0],
#                    [0,0,2.0,0,0],
#                    [0,0,0,0,0],
#                    [0,0,0,0,0]])
# kernel -= np.array([[1,1,1,1,1],
#                    [1,1,1,1,1],
#                    [1,1,1,1,1],
#                    [1,1,1,1,1],
#                    [1,1,1,1,1]])/25
# im = cv2.imread("lena.png")
# im = im.astype(float)
# im_out = convolution(im, kernel)
# im_out = im_out.astype(np.uint8)
# cv2.imwrite('lena_convolution_sharpening_5x5.png', im_out)
# cv2.imshow("Output", im_out)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

#
# Using Convolution to apply mean filter to lena.png with kernel size 7x7
#
# kernel = np.array([[1,1,1,1,1,1,1],
#                    [1,1,1,1,1,1,1],
#                    [1,1,1,1,1,1,1],
#                    [1,1,1,1,1,1,1],
#                    [1,1,1,1,1,1,1],
#                    [1,1,1,1,1,1,1],
#                    [1,1,1,1,1,1,1]])/49
# im = cv2.imread("lena.png")
# im = im.astype(float)
# im_out = convolution(im, kernel)
# im_out = im_out.astype(np.uint8)
# cv2.imwrite('lena_convolution_mean_7x7.png', im_out)
# cv2.imshow("Output", im_out)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

#
# Using Convolution to apply Gaussian (std=1) filter to lena.png with kernel size 7x7
#
# kernel = get_gaussian_kernel(7, 1)
# im = cv2.imread("lena.png")
# im = im.astype(float)
# im_out = convolution(im, kernel)
# im_out = im_out.astype(np.uint8)
# cv2.imwrite('lena_convolution_gaussian_7x7.png', im_out)
# cv2.imshow("Output", im_out)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

#
# Using Convolution to apply sharpening filter to lena.png with kernel size 7x7
#
# kernel = np.array([[0,0,0,0,0,0,0],
#                    [0,0,0,0,0,0,0],
#                    [0,0,0,0,0,0,0],
#                    [0,0,0,2.0,0,0,0],
#                    [0,0,0,0,0,0,0],
#                    [0,0,0,0,0,0,0],
#                    [0,0,0,0,0,0,0]])
# kernel -= np.array([[1,1,1,1,1,1,1],
#                    [1,1,1,1,1,1,1],
#                    [1,1,1,1,1,1,1],
#                    [1,1,1,1,1,1,1],
#                    [1,1,1,1,1,1,1],
#                    [1,1,1,1,1,1,1],
#                    [1,1,1,1,1,1,1]])/49
# im = cv2.imread("lena.png")
# im = im.astype(float)
# im_out = convolution(im, kernel)
# im_out = im_out.astype(np.uint8)
# cv2.imwrite('lena_convolution_sharpening_7x7.png', im_out)
# cv2.imshow("Output", im_out)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

#
# Using Convolution to apply mean filter to art.png with kernel size 3x3
#
# kernel = np.array([[1,1,1],
#                    [1,1,1],
#                    [1,1,1]])/9
# im = cv2.imread("art.png")
# im = im.astype(float)
# im_out = convolution(im, kernel)
# im_out = im_out.astype(np.uint8)
# cv2.imwrite('art_convolution_mean_3x3.png', im_out)
# cv2.imshow("Output", im_out)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

#
# Applying Median Filter to art.png with kernel size 3x3
#
# im = cv2.imread("art.png")
# im = im.astype(float)
# im_out = median_filter(im, 3)
# im_out = im_out.astype(np.uint8)
# cv2.imwrite('art_median_3x3.png', im_out)
# cv2.imshow("Output", im_out)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

#
# Using Convolution to apply mean filter to art.png with kernel size 5x5
#
# kernel = np.array([[1,1,1,1,1],
#                    [1,1,1,1,1],
#                    [1,1,1,1,1],
#                    [1,1,1,1,1],
#                    [1,1,1,1,1]])/25
# im = cv2.imread("art.png")
# im = im.astype(float)
# im_out = convolution(im, kernel)
# im_out = im_out.astype(np.uint8)
# cv2.imwrite('art_convolution_mean_5x5.png', im_out)
# cv2.imshow("Output", im_out)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

#
# Applying Median Filter to art.png with kernel size 5x5
#
# im = cv2.imread("art.png")
# im = im.astype(float)
# im_out = median_filter(im, 5)
# im_out = im_out.astype(np.uint8)
# cv2.imwrite('art_median_5x5.png', im_out)
# cv2.imshow("Output", im_out)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

#
# Using Convolution to apply mean filter to art.png with kernel size 7x7
#
# kernel = np.array([[1,1,1,1,1,1,1],
#                    [1,1,1,1,1,1,1],
#                    [1,1,1,1,1,1,1],
#                    [1,1,1,1,1,1,1],
#                    [1,1,1,1,1,1,1],
#                    [1,1,1,1,1,1,1],
#                    [1,1,1,1,1,1,1]])/49
# im = cv2.imread("art.png")
# im = im.astype(float)
# im_out = convolution(im, kernel)
# im_out = im_out.astype(np.uint8)
# cv2.imwrite('art_convolution_mean_7x7.png', im_out)
# cv2.imshow("Output", im_out)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

#
# Applying Median Filter to art.png with kernel size 7x7
#
# im = cv2.imread("art.png")
# im = im.astype(float)
# im_out = median_filter(im, 7)
# im_out = im_out.astype(np.uint8)
# cv2.imwrite('art_median_7x7.png', im_out)
# cv2.imshow("Output", im_out)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

#
# Using Convolution to apply mean filter to art.png with kernel size 9x9
#
# kernel = np.array([[1,1,1,1,1,1,1,1,1],
#                    [1,1,1,1,1,1,1,1,1],
#                    [1,1,1,1,1,1,1,1,1],
#                    [1,1,1,1,1,1,1,1,1],
#                    [1,1,1,1,1,1,1,1,1],
#                    [1,1,1,1,1,1,1,1,1],
#                    [1,1,1,1,1,1,1,1,1],
#                    [1,1,1,1,1,1,1,1,1],
#                    [1,1,1,1,1,1,1,1,1]])/81
# im = cv2.imread("art.png")
# im = im.astype(float)
# im_out = convolution(im, kernel)
# im_out = im_out.astype(np.uint8)
# cv2.imwrite('art_convolution_mean_9x9.png', im_out)
# cv2.imshow("Output", im_out)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

#
# Applying Median Filter to art.png with kernel size 9x9
#
# im = cv2.imread("art.png")
# im = im.astype(float)
# im_out = median_filter(im, 9)
# im_out = im_out.astype(np.uint8)
# cv2.imwrite('art_median_9x9.png', im_out)
# cv2.imshow("Output", im_out)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

#
# Applying Gaussian Filter to lena.png with kernal size 3x3 using filter2D (P3)
#
# kernel = get_gaussian_kernel(3, 1)
# im = cv2.imread("lena.png")
# im = im.astype(float)
# im_out = cv2.filter2D(src = im, ddepth = -1, kernel = kernel)
# im_out = im_out.astype(np.uint8)
# cv2.imwrite('lena_filter2d_gaussian_3x3.png', im_out)
# cv2.imshow("Output", im_out)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

#
# Applying Gaussian Filter to lena.png with kernal size 5x5 using filter2D (P3)
#
# kernel = get_gaussian_kernel(5, 1)
# im = cv2.imread("lena.png")
# im = im.astype(float)
# im_out = cv2.filter2D(src = im, ddepth = -1, kernel = kernel)
# im_out = im_out.astype(np.uint8)
# cv2.imwrite('lena_filter2d_gaussian_5x5.png', im_out)
# cv2.imshow("Output", im_out)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

#
# Applying Gaussian Filter to lena.png with kernal size 7x7 using filter2D (P3)
#
# kernel = get_gaussian_kernel(7, 1)
# im = cv2.imread("lena.png")
# im = im.astype(float)
# im_out = cv2.filter2D(src = im, ddepth = -1, kernel = kernel)
# im_out = im_out.astype(np.uint8)
# cv2.imwrite('lena_filter2d_gaussian_7x7.png', im_out)
# cv2.imshow("Output", im_out)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

#
# P3. The images from filter2D seems just a little bit clearer, especially on the edges
#
