from ex2_utils import *
import matplotlib.pyplot as plt


def test_conv1D():
    signal = np.array([1, 2, 3, 7, 8])
    kernel = np.array([0, 1, 0.5])
    print("numpy: ", np.convolve(signal, kernel, "full"))
    print("mine: ", conv1D(signal, kernel))


def test_conv2D():
    signal2D = cv2.imread("coins.jpg", cv2.IMREAD_GRAYSCALE)
    kernel = np.ones(shape=(5, 5))*(1/25)
    their = cv2.filter2D(signal2D, -1, kernel, borderType=cv2.BORDER_REPLICATE)
    mine = conv2D(signal2D, kernel)
    plt.gray()
    plt.imshow(their)
    plt.show()
    plt.imshow(mine)
    plt.show()


def test_convDerivative():
    img = cv2.imread("boxman.jpg", cv2.IMREAD_GRAYSCALE)
    directions, magnitude, im_derive_x, im_derive_y = convDerivative(img)

    plt.gray()
    plt.subplot(2, 2, 1), plt.imshow(img)
    plt.title('Original'), plt.xticks([]), plt.yticks([])
    plt.subplot(2, 2, 2), plt.imshow(magnitude)
    plt.title('Magnitude'), plt.xticks([]), plt.yticks([])
    plt.subplot(2, 2, 3), plt.imshow(im_derive_x)
    plt.title('Derivative X'), plt.xticks([]), plt.yticks([])
    plt.subplot(2, 2, 4), plt.imshow(im_derive_y)
    plt.title('Derivative Y'), plt.xticks([]), plt.yticks([])
    plt.show()

    print(directions)

    plt.imshow(directions)
    plt.show()


def test_blurImage():
    img = cv2.imread("beach.jpg", cv2.IMREAD_GRAYSCALE)
    blur = blurImage2(img)
    plt.gray()
    plt.imshow(img)
    plt.show()
    plt.imshow(blur)
    plt.show()


def test_edgeDetectionSobel():
    img = cv2.imread("codeMonkey.jpeg", cv2.IMREAD_GRAYSCALE)
    their, mine = edgeDetectionSobel(img, 0.3)

    plt.gray()

    plt.subplot(2, 2, 1), plt.imshow(img)
    plt.title('Original'), plt.xticks([]), plt.yticks([])
    plt.subplot(2, 2, 3), plt.imshow(their)
    plt.title('their'), plt.xticks([]), plt.yticks([])
    plt.subplot(2, 2, 4), plt.imshow(mine)
    plt.title('mine'), plt.xticks([]), plt.yticks([])

    plt.show()


def test_edgeDetectionZeroCrossingLOG():
    img = cv2.imread("boxman.jpg", cv2.IMREAD_GRAYSCALE)
    img = img / 255.0
    plt.gray()
    plt.imshow(img)
    plt.show()
    ans = edgeDetectionZeroCrossingLOG(img)
    plt.imshow(ans)
    plt.show()


def test_edgeDetectionCanny():
    img = cv2.imread("coins.jpg", cv2.IMREAD_GRAYSCALE)
    plt.gray()
    plt.imshow(img)
    plt.show()
    canny, mine = edgeDetectionCanny(img, 50, 200)
    plt.imshow(mine)
    plt.show()


def test_houghCircle():
    img = cv2.imread("bubbles.jpg", cv2.IMREAD_GRAYSCALE)
    houghCircle(img, 10, 20)


def main():
    # test_conv1D()
    # test_conv2D()
    # test_convDerivative()
    # test_edgeDetectionSobel()
    # test_edgeDetectionZeroCrossingLOG()
    # test_edgeDetectionCanny()
    test_houghCircle()


if __name__ == '__main__':
    main()