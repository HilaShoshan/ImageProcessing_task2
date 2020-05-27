import numpy as np
import cv2
import matplotlib.pyplot as plt


""" 1.1
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


""" 1.2
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


# function to determine the pad width in each axis
# returns: (width on x axis, width on y axis)

def find_pad_width(kernel:np.ndarray) -> (int, int):
    x_width = np.floor(kernel.shape[0] / 2).astype(int)
    if x_width < 1: x_width = 1
    y_width = np.floor(kernel.shape[1] / 2).astype(int)
    if y_width < 1: y_width = 1
    return x_width, y_width


""" 2.1
    Calculate gradient of an image
    :param inImage: Grayscale iamge
    :return: (directions, magnitude,x_der,y_der)
"""
def convDerivative(inImage:np.ndarray) -> (np.ndarray,np.ndarray,np.ndarray,np.ndarray):
    inImage = cv2.GaussianBlur(inImage, (5, 5), 1)
    kernel_x = np.array([1, 0, -1]).reshape((1, 3))
    kernel_y = kernel_x.reshape((3, 1))
    im_derive_x = cv2.filter2D(inImage, -1, kernel_x, borderType=cv2.BORDER_REPLICATE)
    im_derive_y = cv2.filter2D(inImage, -1, kernel_y, borderType=cv2.BORDER_REPLICATE)
    magnitude = np.sqrt(np.square(im_derive_x) + np.square(im_derive_y)).astype('uint8')
    directions = np.arctan2(im_derive_y, im_derive_x) * 180 / np.pi
    # directions = directions.astype('int')  # ??
    return directions, magnitude, im_derive_x, im_derive_y


""" 2.2
    Blur an image using a Gaussian kernel
    :param inImage: Input image
    :param kernelSize: Kernel size
    :return: The Blurred image
"""
def blurImage1(in_image:np.ndarray,kernel_size:np.ndarray)->np.ndarray:
    kernel = np.array(kernel_size)
    sigma = 0.3 * ((kernel_size[0] - 1) * 0.5 - 1) + 0.8
    for i in range(kernel_size[0]):
        for j in range(kernel_size[1]):
            kernel[i, j] = ((1 / 2*np.pi) * np.e) - ((i**2 + j**2) / 2)
    return conv2D(in_image, kernel)


"""
    Blur an image using a Gaussian kernel using OpenCV built-in functions
    :param inImage: Input image
    :param kernelSize: Kernel size
    :return: The Blurred image
"""
def blurImage2(in_image:np.ndarray,kernel_size:np.ndarray)->np.ndarray:
    kernel = cv2.getGaussianKernel(kernel_size)
    blur = cv2.filter2D(in_image, -1, kernel, borderType=cv2.BORDER_REPLICATE)
    return blur


""" 3.1
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
    mine[magnitude >= thresh] = 1
    return their, mine


def cv2_sobel(img: np.ndarray, thresh: float) -> np.ndarray:
    sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)  # x
    sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)  # y
    magnitude = cv2.magnitude(sobelx, sobely)
    ans = np.zeros(img.shape)
    ans[magnitude >= thresh] = 1
    # laplacian = cv2.Laplacian
    return ans


""" 3.2
    Detecting edges using the "ZeroCrossingLOG" method
    :param I: Input image
    :return: Edge matrix
"""
def edgeDetectionZeroCrossingLOG(img:np.ndarray) -> (np.ndarray):
    # smoothing with 2D Gaussian filter
    smooth = cv2.GaussianBlur(img, (5, 5), 1)
    # convolve the smoothed image with the Laplacian filter
    laplacian = np.array([[0, 1, 0],
                          [1, -4, 1],
                          [0, 1, 0]])
    lap_img = cv2.filter2D(smooth, -1, laplacian, borderType=cv2.BORDER_REPLICATE)
    ans = zeroCrossing(lap_img)  # a binary image (0,1) that representing the edges
    return ans


# function that should find edges in the given image (second derivative)
# looking for {+,-} or {+,0,-}

def zeroCrossing(img:np.ndarray) -> np.ndarray:
    ans = np.zeros(img.shape)
    row = col = 1  # starting the loop from (1,1) pixel
    pairs_list = np.zeros(8)  # list of all the couples of the current pixel (those around it, in the 8 directions)
    while row < img.shape[0] - 1:
        while col < img.shape[1] - 1:
            pairs_list[0] = img[row - 1][col]  # up
            pairs_list[1] = img[row - 1][col + 1]  # top right diagonal            7  0  1
            pairs_list[2] = img[row][col + 1]  # right                              \ | /
            pairs_list[3] = img[row + 1][col + 1]  # lower right diagonal         6 - * - 2
            pairs_list[4] = img[row + 1][col]  # down                               / | \
            pairs_list[5] = img[row + 1][col - 1]  # lower left diagonal           5  4   3
            pairs_list[6] = img[row][col - 1]  # left
            pairs_list[7] = img[row - 1][col - 1]  # top left diagonal
            ans = find_edges(img, ans, pairs_list, row, col)  # update ans
            col += 1
        row += 1
    return ans


def find_edges(img:np.ndarray, ans:np.ndarray, pairs_list:np.ndarray, row:int, col:int) -> np.ndarray:
    pixel = img[row][col]
    posIndx = np.where(pairs_list > 0)[0]  # array representing where there are positive elements
    zerosIndx = np.where(pairs_list == 0)[0]  # all the indexes that there are zeros
    numNeg = pairs_list.size - posIndx.size - zerosIndx.size
    if pixel < 0:  # {+,-}
        if posIndx.size > 0:  # there is at least one positive number around
            ans[row][col] = 1.0
            print("{+,-}")
    elif pixel > 0:  # {-,+}
        if numNeg > 0:  # there is at least one negative number around
            ans[row][col] = 1.0
            print("{-,+}")
    else:  # pixel == 0, {+,0,-}
        comp_list = [pairs_list[0] < 0 and pairs_list[4] > 0, pairs_list[0] > 0 and pairs_list[4] < 0,
            pairs_list[1] < 0 and pairs_list[5] > 0, pairs_list[1] > 0 and pairs_list[5] < 0,
            pairs_list[2] < 0 and pairs_list[6] > 0, pairs_list[2] > 0 and pairs_list[6] < 0,
            pairs_list[3] < 0 and pairs_list[7] > 0, pairs_list[3] > 0 and pairs_list[7] < 0]
        if any(comp_list):
            ans[row][col] = 1.0
            print("{+,0,-}")
    return ans


""" 3.3
    Detecting edges usint "Canny Edge" method
    :param img: Input image
    :param thrs_1: T1
    :param thrs_2: T2
    :return: opencv solution, my implementation
"""
def edgeDetectionCanny(img: np.ndarray, thrs_1: float, thrs_2: float) -> (np.ndarray, np.ndarray):
    cv_sol = cv2.Canny(img, thrs_1, thrs_2)

    # blur and compute the partial derivatives, magnitude and direction
    sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)  # x
    sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)  # y
    magnitude = cv2.magnitude(sobelx, sobely)
    directions = np.arctan2(sobelx, sobely)  # * 180 / np.pi

    magnitude = magnitude / magnitude.max() * 255

    # quantize the gradient directions
    # quant_dir = np.zeros(directions.shape)
    # quant_dir = quantGradientDirections(quant_dir, directions)

    # perform non-maximum suppression
    thin_edges = nonMaxSupression(magnitude, directions)

    thresh, weak, strong = threshold(thin_edges, thrs_2, thrs_1)

    # find all edges - hysteresis
    ans = hysteresis(thresh, weak, strong)

    return cv_sol, thin_edges


# quantize the gradient directions to 4 parts
def quantGradientDirections(img: np.ndarray, directions:np.ndarray) -> np.ndarray:
    quant1 = np.logical_and(directions >= 0, directions < 45)
    quant2 = np.logical_and(directions >= 45, directions < 90)
    quant3 = np.logical_and(directions >= 90, directions < 135)
    quant4 = np.logical_and(directions >= 135, directions <= 180)
    img[directions in quant1] = 22.5
    img[directions in quant2] = 67.5
    img[directions in quant3] = 112.5
    img[directions in quant4] = 157.5
    img[img < 0] += 180  # ??
    return img


def nonMaxSupression(img, D):
    M, N = img.shape
    Z = np.zeros((M, N), dtype=np.int32)
    angle = D * 180. / np.pi
    angle[angle < 0] += 180

    for i in range(1, M - 1):
        for j in range(1, N - 1):
            try:
                q = 255
                r = 255

                # angle 0
                if (0 <= angle[i, j] < 22.5) or (157.5 <= angle[i, j] <= 180):
                    q = img[i, j + 1]
                    r = img[i, j - 1]
                # angle 45
                elif (22.5 <= angle[i, j] < 67.5):
                    q = img[i + 1, j - 1]
                    r = img[i - 1, j + 1]
                # angle 90
                elif (67.5 <= angle[i, j] < 112.5):
                    q = img[i + 1, j]
                    r = img[i - 1, j]
                # angle 135
                elif (112.5 <= angle[i, j] < 157.5):
                    q = img[i - 1, j - 1]
                    r = img[i + 1, j + 1]

                if (img[i, j] >= q) and (img[i, j] >= r):
                    Z[i, j] = img[i, j]
                else:
                    Z[i, j] = 0

            except IndexError as e:
                pass

    return Z


def threshold(img, lowThresholdRatio=0.05, highThresholdRatio=0.09):
    highThreshold = img.max() * highThresholdRatio;
    lowThreshold = highThreshold * lowThresholdRatio;

    M, N = img.shape
    res = np.zeros((M, N), dtype=np.int32)

    weak = np.int32(25)
    strong = np.int32(255)

    strong_i, strong_j = np.where(img >= highThreshold)
    zeros_i, zeros_j = np.where(img < lowThreshold)

    weak_i, weak_j = np.where((img <= highThreshold) & (img >= lowThreshold))

    res[strong_i, strong_j] = strong
    res[weak_i, weak_j] = weak

    return res, weak, strong


def hysteresis(img, weak, strong=255):
    M, N = img.shape
    for i in range(1, M-1):
        for j in range(1, N-1):
            if (img[i,j] == weak):
                try:
                    if ((img[i+1, j-1] == strong) or (img[i+1, j] == strong) or (img[i+1, j+1] == strong)
                        or (img[i, j-1] == strong) or (img[i, j+1] == strong)
                        or (img[i-1, j-1] == strong) or (img[i-1, j] == strong) or (img[i-1, j+1] == strong)):
                        img[i, j] = strong
                    else:
                        img[i, j] = 0
                except IndexError as e:
                    pass
    return img


""" 4
    Find Circles in an image using a Hough Transform algorithm extension
    :param I: Input image
    :param minRadius: Minimum circle radius
    :param maxRadius: Maximum circle radius
    :return: A list containing the detected circles,
    [(x,y,radius),(x,y,radius),...]
"""
def houghCircle(img:np.ndarray, min_radius:float, max_radius:float) -> list:
    if min_radius <= 0 or max_radius <= 0 or min_radius >= max_radius:
        print("There is some problem with the given radius values")
        return []

    blur_img = cv2.GaussianBlur(img, (5, 5), 1)
    edged_img = cv2.Canny(blur_img, 75, 150)
    circles_list = list()  # the answer to return

    filter3D = np.ones((30, 30, 100))

    return circles_list
