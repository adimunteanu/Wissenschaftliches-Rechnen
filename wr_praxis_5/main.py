import numpy as np

####################################################################################################
# Exercise 1: DFT

def dft_matrix(n: int) -> np.ndarray:
    """
    Construct DFT matrix of size n.

    Arguments:
    n: size of DFT matrix

    Return:
    F: DFT matrix of size n

    Forbidden:
    - numpy.fft.*
    """
    # TODO: initialize matrix with proper size
    F = np.zeros((1, 1), dtype='complex128')

    # TODO: create principal term for DFT matrix

    # TODO: fill matrix with values

    # TODO: normalize dft matrix


    return F


def is_unitary(matrix: np.ndarray) -> bool:
    """
    Check if the passed in matrix of size (n times n) is unitary.

    Arguments:
    matrix: the matrix which is checked

    Return:
    unitary: True if the matrix is unitary
    """
    unitary = True
    # TODO: check that F is unitary, if not return false

    return unitary


def create_harmonics(n: int = 128) -> (list, list):
    """
    Create delta impulse signals and perform the fourier transform on each signal.

    Arguments:
    n: the length of each signal

    Return:
    sigs: list of np.ndarrays that store the delta impulse signals
    fsigs: list of np.ndarrays with the fourier transforms of the signals
    """

    # list to store input signals to DFT
    sigs = []
    # Fourier-transformed signals
    fsigs = []

    # TODO: create signals and extract harmonics out of DFT matrix

    return sigs, fsigs


####################################################################################################
# Exercise 2: FFT

def shuffle_bit_reversed_order(data: np.ndarray) -> np.ndarray:
    """
    Shuffle elements of data using bit reversal of list index.

    Arguments:
    data: data to be transformed (shape=(n,), dtype='float64')

    Return:
    data: shuffled data array
    """

    # TODO: implement shuffling by reversing index bits
    
    return data


def fft(data: np.ndarray) -> np.ndarray:
    """
    Perform real-valued discrete Fourier transform of data using fast Fourier transform.

    Arguments:
    data: data to be transformed (shape=(n,), dtype='float64')

    Return:
    fdata: Fourier transformed data

    Note:
    This is not an optimized implementation but one to demonstrate the essential ideas
    of the fast Fourier transform.

    Forbidden:
    - numpy.fft.*
    """

    fdata = np.asarray(data, dtype='complex128')
    n = fdata.size

    # check if input length is power of two
    if not n > 0 or (n & (n - 1)) != 0:
        raise ValueError

    # TODO: first step of FFT: shuffle data


    # TODO: second step, recursively merge transforms

    # TODO: normalize fft signal

    return fdata


def generate_tone(f: float = 261.626, num_samples: int = 44100) -> np.ndarray:
    """
    Generate tone of length 1s with frequency f (default mid C: f = 261.626 Hz) and return the signal.

    Arguments:
    f: frequency of the tone

    Return:
    data: the generated signal
    """

    # sampling range
    x_min = 0.0
    x_max = 1.0

    data = np.zeros(num_samples)

    # TODO: Generate sine wave with proper frequency

    return data


def low_pass_filter(adata: np.ndarray, bandlimit: int = 1000, sampling_rate: int = 44100) -> np.ndarray:
    """
    Filter high frequencies above bandlimit.

    Arguments:
    adata: data to be filtered
    bandlimit: bandlimit in Hz above which to cut off frequencies
    sampling_rate: sampling rate in samples/second

    Return:
    adata_filtered: filtered data
    """
    
    # translate bandlimit from Hz to dataindex according to sampling rate and data size
    bandlimit_index = int(bandlimit*adata.size/sampling_rate)

    # TODO: compute Fourier transform of input data

    # TODO: set high frequencies above bandlimit to zero, make sure the almost symmetry of the transform is respected.

    # TODO: compute inverse transform and extract real component
    adata_filtered = np.zeros(adata.shape[0])

    return adata_filtered


if __name__ == '__main__':
    print("All requested functions for the assignment have to be implemented in this file and uploaded to the "
          "server for the grading.\nTo test your implemented functions you can "
          "implement/run tests in the file tests.py (> python3 -v test.py [Tests.<test_function>]).")
