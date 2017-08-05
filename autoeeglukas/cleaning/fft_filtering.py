#!/usr/bin/python2.7.12
from scipy.fftpack import rfft, irfft, fftfreq
from matplotlib import pyplot as plt
import numpy as np

sample_freq = 250.0
signal_length = 60.0
nyq_freq = sample_freq / 2
sample_period = 1 / sample_freq
n_samples_in_signal = sample_freq * signal_length
time_vector = np.arange(0, signal_length, sample_period)
print len(time_vector)

# create artificial signal
carrier = 0.7*np.sin(2*np.pi*50*time_vector) \
        +     np.sin(2*np.pi*120*time_vector) \
        +   2*np.sin(2*np.pi*9*time_vector)
print len(carrier)

np.random.seed(0)
noise = 2 * np.random.randn(n_samples_in_signal)
carrier += noise

carrier = np.array([carrier, carrier])
print carrier.shape

# split into time windows of length 2 seconds
window_length = 2
n_windows = signal_length/window_length
n_samples_in_window = n_samples_in_signal/n_windows
carrier_windows = np.array(np.array_split(carrier, n_windows, axis=1))
carrier_windows = carrier_windows.reshape(2, n_windows, n_samples_in_signal/n_windows)
print carrier_windows.shape

# plt.plot(time_vector,carrier[0])
# plt.show()

# compute power spectrum of window
# rfft does not compute two sided spectrum, since it is redundant
carrier_ft = np.fft.rfft(carrier_windows[0][0])
print carrier_ft[:10], len(carrier_ft)
# carrier_windows_ft = np.fft.rfft(carrier_windows,axis=2)
# print carrier_windows_ft[0][0][:10], carrier_windows_ft.shape
spectrum = abs(carrier_ft)/n_samples_in_window
# at pos 0 is 0*sampling freq due to symmetry reasons
spectrum = spectrum[1:]
# multiply with 2 to get from 2-sided to 1-sided
spectrum = 2*np.pi*spectrum
frequency_vector = sample_freq*np.arange(0,n_samples_in_window/2)/n_samples_in_window

# filter frequency ranges
# spectrum[:16] = 0
# spectrum[25:] = 0
plt.plot(frequency_vector,spectrum)
plt.show()