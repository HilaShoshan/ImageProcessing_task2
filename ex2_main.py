import numpy as np
import cv2
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
    print(np.convolve(signal, kernel, "full"))
    print(conv1D(signal, kernel))


def test_conv2D():
    signal2D = cv2.imread("beach.jpg", cv2.IMREAD_GRAYSCALE)
    kernel = np.array([[1/9, 1/9, 1/9],
              [1/9, 1/9, 1/9],
              [1/9, 1/9, 1/9]])
    #their = cv2.filter2D(signal2D, -1, kernel, borderType=cv2.BORDER_REPLICATE)
    #print("their:\n", their)
    mine = conv2D(signal2D, kernel)
    print("mine:\n", mine)


def main():
    test_conv2D()


if __name__ == '__main__':
    main()