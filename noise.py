from numbers import Integral
from functools import wraps

import numpy
from scipy import signal

def record(method):
    @wraps(method)
    def recorded_method(recorder, *args, **kwargs):

        # If this isn't being called from within a recorded method,
        # then we should add the call to the object's record. We
        # also need to indicate that other calls are coming from
        # within a recorded method. Otherwise, we just call the
        # method without touching anything.
        if not recorder._record_lock:
            recorder._record_lock = True
            print("recording lock acquired")
            recorder._record.append((recorded_method, args, kwargs))
            result = method(recorder, *args, **kwargs)
            recorder._record_lock = False
            print("recording lock released")
            return result
        else:
            print("recording locked")
            return method(recorder, *args, **kwargs)

    return recorded_method

class Noise(object):
    def __init__(self, duration=1, sample_rate=44100):
        self.duration = duration
        self.sample_rate = sample_rate
        self.samples = None
        self._record = []
        self._record_lock = False

    def copy(self):
        new = type(self)()
        new.duration = self.duration
        new.sample_rate = self.sample_rate
        new.samples = numpy.array(self.samples)
        new._record = list(self._record)
        new._record_lock = self._record_lock
        return new

    @property
    def n_original_samples(self):
        return int(self.duration * self.sample_rate)

    @property
    def n_samples(self):
        if self.samples is not None:
            return len(self.samples)
        else:
            return self.n_original_samples

    def resample(self):
        n = Noise(self.duration, self.sample_rate)
        n._record = self._record
        n._playback()
        return n

    def resample_array(self, n_cols, wav=True):
        if wav:
            return numpy.hstack([self.resample().wav().reshape(-1, 1)
                                 for x in range(n_cols)])
        else:
            return numpy.hstack([self.resample().samples.reshape(-1, 1)
                                 for x in range(n_cols)])

    def _playback(self):
        recorded_methods = self._record
        self._record = []  # Same record will be re-recorded
        for method, args, kwargs in recorded_methods:
            method(self, *args, **kwargs)

    @record
    def _mix_in(self, np_rand, integrate=False, proportion=1.0):
        """Take a series of random samples from a gaussian distribution.
        Replace the previous samples by default. If `0 < proportion < 1.0`,
        combine the gaussian samples with existing samples such that
        `final == (1.0 - proportion) * original + proportion * new)`.
        """
        if not 0 <= proportion <= 1.0:
            raise ValueError('Proportion must be greater than zero '
                             'and less than one.')

        if self.samples is None:
            self.samples = numpy.zeros(self.n_samples)

        old_samples = self.samples
        self.samples = np_rand(-1, 1, self.n_samples)
        if integrate:
            self.integrate()
            # self._record.pop()  # Don't record inside a recorded method
            print("would have popped here, mixing in")

        self.samples *= proportion
        self.samples += old_samples * (1.0 - proportion)
        self.samples -= self.samples.mean()
        return self

    def white(self, proportion=1.0):
        """Take a series of random samples from a gaussian distribution."""
        return self._mix_in(numpy.random.normal, proportion=proportion)

    def laplacian(self, proportion=1.0):
        """Take a series of random samples from a laplacian distribution."""
        return self._mix_in(numpy.random.laplace, proportion=proportion)

    def brownian(self, proportion=1.0):
        """Integrate a series of random samples from a gaussian distribution."""
        return self._mix_in(numpy.random.normal,
                            integrate=True, proportion=proportion)

    @record
    def integrate(self):
        """'Integrate' (i.e. do a cumulative sum of) the current sample set.
        This is useful because integrating white noise produces Brownian
        noise.

        There's a wonderfully concrete way to understand that fact. Imagine
        you're a particle that wiggles, and that each time you move, you
        decide how to move by taking a sample from a Gaussian distribution.
        If you plotted all your movements by starting from zero each time,
        you'd get a white noise (Gaussian noise) pattern. But if you plotted
        the resulting path, you'd get a Brownian noise pattern. The
        resulting path is just the sum of the individual movements; hence
        integrating (summing) white noise produces Brownian noise.
        """
        # Subtract the mean to avoid positive or negative explosion
        self.samples -= self.samples.mean()
        self.samples = self.samples.cumsum()

        # Re-center the data
        self.samples -= self.samples.mean()
        return self

    @record
    def butter_high(self, freq=2 ** -8, order=1):
        b, a = signal.butter(order, freq, 'high', analog=False)
        self.samples[:] = signal.lfilter(b, a, self.samples)
        return self

    @record
    def butter_low(self, freq=2 ** -1, order=1):
        b, a = signal.butter(order, freq, 'low', analog=False)
        self.samples[:] = signal.lfilter(b, a, self.samples)
        return self

    # No need to record here, since this just calls recorded methods
    def butter_filter(self, lowpass=2 ** -1, highpass=2 ** -8, order=1):
        """Perform a high- and low-pass butterworth filter."""
        self.butter_low(lowpass, order)
        self.butter_high(highpass, order)
        return self

    @record
    def gauss_filter(self, sample_width=0.02, edge_policy='same'):
        """Convolve with a unit area Gaussian kernel. This is the
        same thing as a weighted moving average with a Gaussian
        weight curve.

        `sample_width` may be a floating point number in the range
        `(0, 1)`, representing the width of the kernel relative to
        the original sample set. It may also be an integer specifying
        the precise width of the kernel in samples.

        `edge_policy` determines the way the edges of the sample are
        handled; `'valid'` avoids zero-padding but reduces the total
        number of samples, and `'same'` uses zero-padding to guarantee
        that the number of samples remains the same.
        """

        if 0 < sample_width < 1:
            sample_width = int(self.duration * self.sample_rate * sample_width)
        elif not isinstance(sample_width, Integral) or sample_width <= 0:
            raise ValueError('sample_width must be a floating point number '
                             'in the range (0, 1], or an integer greater '
                             'than zero.')

        kernel = numpy.exp(-numpy.linspace(-3, 3, sample_width) ** 2)
        kernel /= kernel.sum()
        self.samples = signal.convolve(self.samples, kernel, edge_policy)
        return self

    @record
    def square_filter(self, sample_width=0.02, edge_policy='same'):
        """Convolve with a unit area constant kernel. This is the
        same thing as an unweighted moving average.

        `sample_width` may be a floating point number in the range
        `(0, 1)`, representing the width of the kernel relative to
        the original sample set. It may also be an integer specifying
        the precise width of the kernel in samples.

        `edge_policy` determines the way the edges of the sample are
        handled; `'valid'` avoids zero-padding but reduces the total
        number of samples, and `'same'` uses zero-padding to guarantee
        that the number of samples remains the same.
        """

        if 0 < sample_width < 1:
            sample_width = int(self.duration * self.sample_rate * sample_width)
        elif not isinstance(sample_width, Integral) or sample_width <= 0:
            raise ValueError('sample_width must be a floating point number '
                             'in the range (0, 1), or an integer greater '
                             'than zero.')

        kernel = numpy.ones(sample_width)
        kernel /= kernel.sum()
        self.samples = signal.convolve(self.samples, kernel, edge_policy)
        return self

    @record
    def autofilter(self, sample_width=0.002, mean=False, median=False):
        """Bin-sum the signal. This will attenuate noise that has
        no local correlation, while amplifying noise that does have
        local correlation."""

        if 0 < sample_width < 1:
            sample_width = int(self.duration * self.sample_rate * sample_width)
        elif not isinstance(sample_width, Integral) or sample_width <= 0:
            raise ValueError('sample_width must be a floating point number '
                             'in the range (0, 1), or an integer greater '
                             'than zero.')

        truncate_len = len(self.samples) - len(self.samples) % sample_width
        bins = self.samples[:truncate_len].reshape(-1, sample_width)

        if mean:
            self.samples = bins.mean(axis=1).ravel()
        elif median:
            self.samples = numpy.median(bins, axis=1).ravel()
        else:
            self.samples = bins.sum(axis=1).ravel()
            self.samples -= bins.max(axis=1).ravel()
            self.samples -= bins.min(axis=1).ravel()
            self.samples /= sample_width - 2
        return self

    @record
    def autoresample(self, sample_width=0.002, mean=False):
        """Bin-sum a "bootstrapped" resampling of the signal. I tried
        this, but it performed worse than `autofilter` above. Resampling
        strategies at the ensemble level might be more useful."""
        if 0 < sample_width < 1:
            sample_width = int(self.duration * self.sample_rate * sample_width)
        elif not isinstance(sample_width, Integral) or sample_width <= 0:
            raise ValueError('sample_width must be a floating point number '
                             'in the range (0, 1), or an integer greater '
                             'than zero.')

        truncate_len = len(self.samples) - len(self.samples) % sample_width
        bins = self.samples[:truncate_len].reshape(-1, sample_width)
        for i, bn in enumerate(bins):
            resample = numpy.random.choice(bn, (sample_width, sample_width))
            resample = resample.sum(axis=0)
            if mean:
                resample /= sample_width
            bins[i, :] = resample

        self.samples = bins.ravel()
        self.autofilter(sample_width, mean)
        # self._record.pop()  # Don't record inside a recorded method
        print("would have popped here, resampling")
        return self

    @record
    def autoconvolve(self, edge_policy='same'):
        self.samples = signal.convolve(self.samples, self.samples, edge_policy)
        return self

    @record
    def fade(self, sample_width=0.1):
        """Fade in at the beginning and out at the end. This softens the
        perceived 'click' at the beginning and end of the noise.
        """
        sample_width = int(self.duration * self.sample_rate * sample_width)
        self.samples[:sample_width] *= numpy.linspace(0, 1, sample_width)
        self.samples[-sample_width:] *= numpy.linspace(1, 0, sample_width)
        return self

    @record
    def scale(self):
        """Scale the current data by the absolute maximum. This maximizes
        volume without causing clipping artifacts.
        """
        self.samples -= self.samples.mean()
        self.samples /= numpy.max(numpy.abs(self.samples))
        self.samples *= 32767  # max amplitude at 16 bits per sample
        return self

    @record
    def amplify(self, amplitude=1):
        """Amplify the tone by the given amplitude.

        This may produce clipping.
        """
        self.samples *= amplitude
        return self

    def wav(self):
        """Return data suitable for saving as a PCM or .wav file."""
        newobj = self.copy()
        newobj.scale()
        return numpy.int16(newobj.samples)

if __name__ == '__main__':
    n = Noise().brownian()
    print(n.samples.sum())
    print(n._record)
    print(n.resample().samples.sum())
    print(n._record)
    print(n.resample().samples.sum())
    print(n._record)
    print(n.wav())
    print(n._record)
