import numpy as np
import cv2
import matplotlib.pyplot as plt
from ex2_utils import *


def test() -> np.ndarray:
    a = np.arange(9)
    print("a: ", a)
    b = np.ones(9)
    print("b: ", b)
    c = np.dot(a, b)
    print("c: ", c.shape, c)
    return c
    pass


def test_conv1D():
    signal = np.array([1, 2, 3, 7, 8])
    kernel = np.array([0, 1, 0.5])
    print("numpy: ", np.convolve(signal, kernel, "full"))
    print("mine: ", conv1D(signal, kernel))


def test_conv2D():
    signal2D = cv2.imread("beach.jpg", cv2.IMREAD_GRAYSCALE)
    kernel = np.ones(shape=(5, 5))*(1/9)
    print(kernel)
    print("signal:\n", signal2D)
    their = cv2.filter2D(signal2D, -1, kernel, borderType=cv2.BORDER_REPLICATE)
    print("their:\n", their)
    mine = conv2D(signal2D, kernel)
    print("mine:\n", mine)
    plt.gray()
    plt.imshow(their)
    plt.show()
    plt.imshow(mine)
    plt.show()


def test_convDerivative():
    img = cv2.imread("boxman.jpg")
    directions, magnitude, im_derive_x, im_derive_y = convDerivative(img)
    plt.imshow(im_derive_x)
    plt.show()
    plt.imshow(im_derive_y)
    plt.show()


def test_edgeDetectionSobel():
    img = cv2.imread("codeMonkey.jpeg")
    their, mine = edgeDetectionSobel(img, 0.3)

    plt.subplot(2, 2, 1), plt.imshow(img)
    plt.title('Original'), plt.xticks([]), plt.yticks([])
    plt.subplot(2, 2, 3), plt.imshow(their)
    plt.title('their'), plt.xticks([]), plt.yticks([])
    plt.subplot(2, 2, 4), plt.imshow(mine)
    plt.title('mine'), plt.xticks([]), plt.yticks([])

    plt.show()


def test_edgeDetectionZeroCrossingLOG():
    img = cv2.imread("boxman.jpg", cv2.IMREAD_GRAYSCALE)
    plt.gray()
    plt.imshow(img)
    plt.show()
    ans = edgeDetectionZeroCrossingLOG(img)
    plt.imshow(ans)
    plt.show()


def main():
    test_conv1D()
    # test_conv2D()
    test_convDerivative()
    # test_edgeDetectionSobel()
    # test_edgeDetectionZeroCrossingLOG()
    # canny
    # hough


if __name__ == '__main__':
    main()