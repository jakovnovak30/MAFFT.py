#!/bin/python3

import numpy as np

class Seq:
    def __init__(self, seq : str, debug : bool = False):
        chars = ['A', 'C', 'G', 'T']

        nums = list(map(lambda x: chars.index(x)+1, seq))
        avg = np.average(nums)
        std = np.std(nums)

        # normalized signal
        self.signal = [(x-avg)/std for x in nums]
        # original sequence string
        self.seq = seq
        # enable extra output
        self.debug = debug

    def __str__(self):
        return \
            'Seq{' +\
            f'seq={self.seq.__repr__()}, signal={self.signal.__repr__()}' \
            '}'

    def correlate(self, other : Seq) -> tuple[int, list[float]]:
        # length and padding
        len1 = len(self.signal)
        len2 = len(other.signal)
        pad1 = [0]*(len2-1)
        pad2 = [0]*(len1-1)

        # corr array
        k_arr = range(-len2+1, len1)

        # RFFT
        X1 = np.fft.rfft(pad1 + self.signal)
        X2 = np.fft.rfft(other.signal + pad2)
        # NB: bez konjugacije bi bila konvolucija
        Corr = np.conjugate(X2)*X1
        corr = np.fft.irfft(Corr)

        k = k_arr[corr.argmax()]
        if self.debug:
            print("Correlation by FFT:")
            print(corr)
            print(f"k={k}")
        return k, list(map(float, corr))

if __name__ == '__main__':
    s1 = Seq('GATTTGGG')
    s2 = Seq('CCGATCTA')
    print(s1.correlate(s2))

    # TODO: dodaj vise korelacija (spikes) i vizualizaciju poravnanja
