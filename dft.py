import matplotlib.pyplot as plt
import numpy as np
import math

#https://pythonnumericalmethods.studentorg.berkeley.edu/notebooks/chapter24.03-Fast-Fourier-Transform.html
#https://mecha-mind.medium.com/fast-fourier-transform-optimizations-5c1fd108a8ed
#https://jakevdp.github.io/blog/2013/08/28/understanding-the-fft/

def plot_twiddle_factors(N):
    
    twiddle_factor = np.exp(-2j*np.pi*np.arange(N)/ N)
    print("twiddle factor N=" + str(N))
    print(twiddle_factor)

    # New code to plot twiddle factor in a circle
    plt.figure(figsize=(6,6))
    plt.scatter(twiddle_factor.real, twiddle_factor.imag, color='red')
    # Add a circle with the same radius
    circle = plt.Circle((0, 0), 1, color='black', fill=False)
    plt.gca().add_artist(circle)

    plt.xlabel('Real')
    plt.ylabel('Imaginary')
    plt.title('Twiddle Factor')
    plt.grid(True)
    plt.gca().set_aspect('equal', adjustable='box')  # To ensure the plot is a circle
    plt.show()

def DFT(x):
    """
    Function to calculate the
    discrete Fourier Transform
    of a 1D real-valued signal x
    """

    N = len(x)
    n = np.arange(N)
    k = n.reshape((N, 1))

    e = np.exp(-2j * np.pi * k * n / N)
    X = np.dot(e, x)

    return X


def DFT_len2(x):
    """
    Function to calculate the
    discrete Fourier Transform
    of a 1D real-valued signal x
    """

    if( len(x) != 2):
        print("ERROR: len(x) is " + str(len(x)) + " instead of 2" )
        return [0]

    N = 2
    n = np.arange(N)
    k = n.reshape((N, 1))

    e = np.exp(-2j * np.pi * k * n / N)
    print(e)

    X = np.dot(e, x)

    return X


def recursive_FFT(x):
    """
    A recursive implementation of
    the 1D Cooley-Tukey FFT, the
    input should have a length of
    power of 2.
    """
    N = len(x)

    if N == 1:
        return x
    else:
        X_even = recursive_FFT(x[::2])
        X_odd = recursive_FFT(x[1::2])
        twiddle_factor = \
          np.exp(-2j*np.pi*np.arange(N)/ N)

        X = np.concatenate(\
            [X_even+twiddle_factor[:int(N/2)]*X_odd,
             X_even+twiddle_factor[int(N/2):]*X_odd])
        return X

def recursive_FFT_radix2(x):
    """
    A recursive implementation of
    the 1D Cooley-Tukey FFT, the
    input should have a length of
    power of 2.
    """
    N = len(x)

    if N == 2:
        return DFT_len2(x)
    else:
        X_even = recursive_FFT(x[::2])
        X_odd = recursive_FFT(x[1::2])
        twiddle_factor = \
          np.exp(-2j*np.pi*np.arange(N)/ N)

        X = np.concatenate(\
            [X_even+twiddle_factor[:int(N/2)]*X_odd,
             X_even+twiddle_factor[int(N/2):]*X_odd])
        return X


def bit_reversal(n, num_bits, base=2):
    result = 0
    shiftamt = int(np.log2(base))
    mask     = base - 1

    for _ in range(num_bits):
        # Shift result one bit to the left
        result <<= shiftamt
        # Add the least significant bit of n
        result |= n & mask
        # Shift n one bit to the right
        n >>= shiftamt
    return result

def get_bit_reversed_seq(N, num_bits,base=2):
    return [bit_reversal(i, num_bits,base) for i in range(N)]

def iterative_FFT_radix2(x, debug=False):
    """
    A recursive implementation of
    the 1D Cooley-Tukey FFT, the
    input should have a length of
    power of 2.
    """
    N = len(x)

    nbits = math.log2(N)
    bit_reversed_seq = get_bit_reversed_seq(N, int(nbits))

    xrev = np.zeros(N)

    xrev = [x[i] for i in bit_reversed_seq]

    if(debug):
        print("iterative_FFT_radix2")
        print("bit_reversed_seq = " + str(bit_reversed_seq))
        print("xrev = " + str(xrev))

    n_stages = int(math.log2(N))
    X_stage = np.zeros((n_stages+1, len(x)),dtype = 'complex_')

    k = 2
    stage = 1
    X_stage[0] = xrev

    while k <= N:        
        w = np.exp(-2j*np.pi/k)
        u = int(k/2)
        for i in range(0, N, k):
            h = 1
            for j in range(u):
                a, b = X_stage[stage-1][i+j], X_stage[stage-1][i+j+u]
                X_stage[stage][i+j] = a + h*b
                X_stage[stage][i+j+u] = a - h*b
                if(debug):
                    print("X_stage[" + str(stage-1) + "][" + str(i+j) + "] = " + format(X_stage[stage-1][i+j], ".3f"))
                    print("X_stage[" + str(stage-1) + "][" + str(i+j+u) + "] = " + format(X_stage[stage-1][i+j+u], ".3f"))
                    print("X_stage[" + str(stage) + "][" + str(i+j) + "] = " + format(X_stage[stage][i+j], ".3f"))
                    print("X_stage[" + str(stage) + "][" + str(i+j+u) + "] = " + format(X_stage[stage][i+j+u], ".3f"))
                    print("h = " + format(h, ".3f"))
                    print("j = " + str(j))
                h *= w

        if(debug):
            print("X_stage[" + str(stage) + "] is ")
            print([format(val, ".3f") for val in X_stage[stage]])

        k *= 2
        stage += 1    

    X = X_stage[-1]

    return X

def iterative_FFT_radix4(x, debug=False):
    """
    A recursive implementation of
    the 1D Cooley-Tukey FFT, the
    input should have a length of
    power of 2.
    """
    N = len(x)
    nbits = math.log(N,4)
    bit_reversed_seq = get_bit_reversed_seq(N, int(nbits),base=4)

    xrev = np.zeros(N)

    xrev = [x[i] for i in bit_reversed_seq]

    if(debug):
        print("iterative_FFT_radix4")
        print("bit_reversed_seq = " + str(bit_reversed_seq))
        print("xrev = " + str(xrev))

    n_stages = int(math.log(N,4))
    X_stage = np.zeros((n_stages+1, len(x)),dtype = 'complex_')

    k = 4
    stage = 1
    X_stage[0] = xrev

    while k <= N:        
        w = np.exp(-2j*np.pi/k)
        u = int(k/4)
        if(debug):
            print("k = " + str(k))
            print("u = " + str(u))
        for i in range(0, N, k):
            h = 1
            for j in range(u):
                a, b = X_stage[stage-1][i+j], X_stage[stage-1][i+j+u]
                c, d = X_stage[stage-1][i+j+2*u], X_stage[stage-1][i+j+3*u]
                X_stage[stage][i+j] = a + h*b + h*h*c + h*h*h*d
                X_stage[stage][i+j+u] = a - complex(0, 1)*h*b - h*h*c + complex(0, 1)*h*h*h*d
                X_stage[stage][i+j+2*u] = a - h*b + h*h*c - h*h*h*d
                X_stage[stage][i+j+3*u] = a + complex(0, 1)*h*b - h*h*c - complex(0, 1)*h*h*h*d
                if(debug):
                    print("X_stage[" + str(stage-1) + "][" + str(i+j) + "] = " + format(X_stage[stage-1][i+j], ".3f"))
                    print("X_stage[" + str(stage-1) + "][" + str(i+j+u) + "] = " + format(X_stage[stage-1][i+j+u], ".3f"))
                    print("X_stage[" + str(stage-1) + "][" + str(i+j+2*u) + "] = " + format(X_stage[stage-1][i+j+2*u], ".3f"))
                    print("X_stage[" + str(stage-1) + "][" + str(i+j+3*u) + "] = " + format(X_stage[stage-1][i+j+3*u], ".3f"))
                    print("X_stage[" + str(stage) + "][" + str(i+j) + "] = " + format(X_stage[stage][i+j], ".3f"))
                    print("X_stage[" + str(stage) + "][" + str(i+j+u) + "] = " + format(X_stage[stage][i+j+u], ".3f"))
                    print("X_stage[" + str(stage) + "][" + str(i+j+2*u) + "] = " + format(X_stage[stage][i+j+2*u], ".3f"))
                    print("X_stage[" + str(stage) + "][" + str(i+j+3*u) + "] = " + format(X_stage[stage][i+j+3*u], ".3f"))
                    print("h = " + format(h, ".3f"))
                    print("j = " + str(j))
                h *= w

        if(debug):
            print("X_stage[" + str(stage) + "] is ")
            print([format(val, ".3f") for val in X_stage[stage]])

        k *= 4
        stage += 1    

    X = X_stage[-1]

    return X
# sampling rate
sr = 1024
# sampling interval
ts = 1.0/sr

freq_1 = 0.5
freq_2 = 1.

t = np.arange(0,1,ts)

x_1 = 1*np.sin(2*np.pi*freq_1*t)
x_2 = 1.1*np.sin(2*np.pi*freq_2*t)

plt.figure(figsize = (8, 6))
plt.plot(t, x_1, 'r')
plt.plot(t, x_2, 'b')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')

dft_1 = DFT(x_1)
fft_2 = np.fft.fft(x_2)
fft_2_recursive = recursive_FFT(x_2)
fft_2_recursive_radix2 = recursive_FFT_radix2(x_2)
fft_2_iterative_radix2 = iterative_FFT_radix2(x_2)

if (np.allclose(fft_2, fft_2_recursive) == False):
    print("ERROR FFT version 1")
    exit(-1)

if (np.allclose(fft_2, fft_2_recursive_radix2) == False):
    print("ERROR FFT version 2")
    exit(-1)

if (np.allclose(fft_2_recursive_radix2, fft_2_recursive) == False):
    print("ERROR FFT version 3")
    exit(-1)

if (np.allclose(fft_2, fft_2_iterative_radix2) == False):
    print("ERROR FFT version 4")
    exit(-1)

fft_2_iterative_radix2 = iterative_FFT_radix2(np.arange(4*4*4*4))
fft_2_iterative_radix4 = iterative_FFT_radix4(np.arange(4*4*4*4))

if (np.allclose(fft_2_iterative_radix4, fft_2_iterative_radix2) == False):
    print("ERROR FFT version 5")
    exit(-1)

# calculate the frequency
N = len(dft_1) #this is equal to len(fft_2)
n = np.arange(N)
T = N/sr
freq = n/T

plt.figure(figsize = (8, 6))

plt.stem(freq, abs(dft_1), 'r', \
         markerfmt=" ", basefmt="-b")

plt.stem(freq, abs(fft_2), 'b', \
         markerfmt=" ", basefmt="-b")

plt.stem(freq, abs(fft_2_recursive_radix2), 'k', \
         markerfmt=" ", basefmt="-b")

plt.xlabel('Freq (Hz)')
plt.ylabel('DFT Amplitude |X(freq)|')

plt.show()

plot_twiddle_factors(sr)
