import numpy as np
import cv2
import matplotlib.pyplot as plt


"""
    Convolve a 1-D array with a given kernel
    :param inSignal: 1-D array
    :param kernel1: 1-D array as a kernel
    :return: The convolved array
"""
def conv1D(inSignal:np.ndarray,kernel1:np.ndarray)->np.ndarray:

    flip = np.flip(kernel1)  # flip the vector kernel
    res = np.zeros(inSignal.size + kernel1.size - 1)  # init the result array with the new size ("full")
    num_of_zeros = kernel1.size - 1  # number of zeros to padding each side of the signal
    zero_padding = np.zeros(num_of_zeros)
    new_signal = np.append(zero_padding, np.append(inSignal, zero_padding))

    for i in range(res.size):  # from the starting of new_signal to the last element of inSignal
        res[i] = np.dot(new_signal[i : i+num_of_zeros+1], flip)

    return res
    pass


"""
    Convolve a 2-D array with a given kernel
    :param inImage: 2D image
    :param kernel2: A kernel
    :return: The convolved image
"""
def conv2D(inImage:np.ndarray,kernel2:np.ndarray)->np.ndarray:

    # flip = kernel2[-1::-1, -1::-1]  # ?
    flip = np.flip(kernel2)
    res = np.zeros(inImage.shape)

    x_width, y_width = find_pad_width(flip)
    padding_img = np.pad(inImage, [(x_width,), (y_width,)], mode='constant')  # zero padding to the image

    for i in range(x_width, padding_img.shape[0]-x_width):
        for j in range(y_width, padding_img.shape[1]-y_width):
            from_i = i - x_width
            to_i = i - x_width + flip.shape[0]
            from_j = j - y_width
            to_j = j - y_width + flip.shape[1]
            signal_part = padding_img[from_i:to_i, from_j:to_j]  # size of the flip kernel always
            res[from_i, from_j] = np.sum(np.multiply(signal_part, flip))

    return res
    pass


# function to determine the pad width in each axis
# returns: (width on x axis, width on y axis)

def find_pad_width(kernel:np.ndarray) -> (int, int):
    x_width = np.floor(kernel.shape[0] / 2).astype(int)
    if x_width < 1: x_width = 1
    y_width = np.floor(kernel.shape[1] / 2).astype(int)
    if y_width < 1: y_width = 1
    return x_width, y_width
    pass


"""
    Calculate gradient of an image
    :param inImage: Grayscale iamge
    :return: (directions, magnitude,x_der,y_der)
"""
def convDerivative(inImage:np.ndarray) -> (np.ndarray,np.ndarray,np.ndarray,np.ndarray):
    kernel_x = np.array([1, 0, -1]).reshape((1, 3))
    kernel_y = kernel_x.reshape((3, 1))
    im_derive_x = cv2.filter2D(inImage, -1, kernel_x, borderType=cv2.BORDER_REPLICATE)
    im_derive_y = cv2.filter2D(inImage, -1, kernel_y, borderType=cv2.BORDER_REPLICATE)
    magnitude = np.sqrt(np.square(im_derive_x) + np.square(im_derive_y))
    directions = np.arctan(np.divide(im_derive_y, im_derive_x))
    return directions, magnitude, im_derive_x, im_derive_y
    pass


"""
    Detects edges using the Sobel method
    :param img: Input image
    :param thresh: The minimum threshold for the edge response
    :return: opencv solution, my implementation
"""
def edgeDetectionSobel(img: np.ndarray, thresh: float = 0.7) -> (np.ndarray, np.ndarray):
    their = cv2_sobel(img, thresh)
    Sx_kernel = np.array([[1, 0, -1],
                          [2, 0, -2],
                          [1, 0, -1]])
    Sy_kernel = np.array([[1, 2, 1],
                          [0, 0, 0],
                          [-1, -2, -1]])
    sobelX = cv2.filter2D(img, -1, Sx_kernel, borderType=cv2.BORDER_REPLICATE)
    sobelY = cv2.filter2D(img, -1, Sy_kernel, borderType=cv2.BORDER_REPLICATE)
    magnitude = np.sqrt(np.square(sobelX) + np.square(sobelY))
    mine = np.zeros(img.shape)
    mine[magnitude < thresh] = 0
    mine[magnitude >= thresh] = 1
    return their, mine
    pass


def cv2_sobel(img: np.ndarray, thresh: float) -> np.ndarray:
    sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=5)  # x
    sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=5)  # y
    magnitude = cv2.magnitude(sobelx, sobely)
    ans = np.zeros(img.shape)
    ans[magnitude < thresh] = 0
    ans[magnitude >= thresh] = 1
    # laplacian = cv2.Laplacian(img, cv2.CV_64F)
    return ans
    pass


"""
    Detecting edges using the "ZeroCrossingLOG" method
    :param I: Input image
    :return: Edge matrix
"""
def edgeDetectionZeroCrossingLOG(img:np.ndarray)->(np.ndarray):
    # smoothing with 2D Gaussian
    gaussian = np.array([[1, 2, 1],
                        [2, 4, 2],
                        [1, 2, 1]]) * (1/16)
    smooth = cv2.filter2D(img, -1, gaussian, borderType=cv2.BORDER_REPLICATE)
    plt.imshow(smooth)
    plt.show()
    pass


"""
    Detecting edges usint "Canny Edge" method
    :param img: Input image
    :param thrs_1: T1
    :param thrs_2: T2
    :return: opencv solution, my implementation
"""
def edgeDetectionCanny(img: np.ndarray, thrs_1: float, thrs_2: float) -> (np.ndarray, np.ndarray):
    pass
