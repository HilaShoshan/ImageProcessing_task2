import numpy as np


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

    flip = kernel2[-1::-1, -1::-1]  # ?
    res = np.zeros(inImage.shape)
    pad_width = np.floor(kernel2.shape[0] / 2)  # assume that the kernel is square
    if pad_width < 1: pad_width = 1
    padding_img = np.pad(inImage, 1, mode='constant')  # zero padding to the image
    print("************************************")
    print(padding_img)

    for i in range(padding_img.shape[0] - pad_width):
        for j in range(padding_img.shape[1] - pad_width):
            signal_part = padding_img[i:i+pad_width+1, j:j+pad_width+1]  # size of the kernel
            print(signal_part)
            res[i, j] = np.sum(np.multiply(signal_part, flip))

    return res
    pass
