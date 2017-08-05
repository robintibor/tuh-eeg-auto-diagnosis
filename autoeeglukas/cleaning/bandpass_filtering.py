#!/usr/bin/python2.7.12
from scipy.signal import butter, filtfilt, freqz, buttord, resample, spectrogram
from matplotlib.widgets import Slider, Button, RadioButtons
from collections import OrderedDict
import matplotlib.pyplot as plt
import numpy as np
import pyedflib
import sys

# explanation of linspace and arange
# >>> np.linspace(0,1,10,endpoint=False)
# array([ 0. ,  0.1,  0.2,  0.3,  0.4,  0.5,  0.6,  0.7,  0.8,  0.9])

# >>> np.arange(0,1,0.1)
# array([ 0. ,  0.1,  0.2,  0.3,  0.4,  0.5,  0.6,  0.7,  0.8,  0.9])


class Butter(object):
    def __init__(self, a=None, b=None):
        self.a = a
        self.b = b
        self.lowcut = 8
        self.highcut = 12

if len(sys.argv) != 3:
    print "Usage: python filter.py <path> <electrode name>"

file_path = sys.argv[1]

edf_file = pyedflib.EdfReader(file_path)
print edf_file.getSignalLabels(), edf_file.getNSamples()[0]
# print signal
electrode = sys.argv[2]
signal = edf_file.readSignal(edf_file.getSignalLabels().index(electrode))

# print edf_file.getSampleFrequency(0), edf_file.getNSamples()

# sampling frequency
Fs = 250.0
# nyquist frequency
nyq = Fs/2
# sampling period
T = 1.0/Fs
# length of signal
L = 10 * Fs
# time vector
t = np.arange(0,L*T,T)
# print t, len(t)

# signal
# S = 0.7*np.sin(2*np.pi*50*t) + np.sin(2*np.pi*120*t) + 2*np.sin(2*np.pi*9*t)
# # signal + noise
# np.random.seed(0)
# noise = 2*np.random.randn(L) # 36 for noise stuff


# noise = np.array(map(lambda x: [30*[x]], noise)).flatten()
# noise = noise[:1000]

# X = S + noise
mean = signal.mean()
if mean > 1:
    signal = signal/signal.mean()
X = np.array(signal)[0:2500]
# L=len(signal[230000:232500])
signal_min, signal_max = min(X), max(X)


fig, ((ax3, ax4),(ax1, ax2)) = plt.subplots(2, 2)



# ax4.plot(t,noise)
# ax4.set_ylim([-8,8])

# print len(X), len(t)

[sigplt]=ax3.plot(t,X)
ax3.set_ylim([signal_min, signal_max])
ax3.set_title("Raw signal")
ax3.set_ylabel('X(t)')
ax3.set_xlabel('t [seconds]')
# ax3.set_ylim([-8,8])
# plt.show()

# # resample the signal. what does it actually do?! and why?!
# resample_frequency2 = 18
# resamples = L / Fs * resample_frequency2
# X_prime_prime = resample(X, resamples)
# X_prime_prime = np.array(map(lambda x: [x,x], X_prime_prime)).flatten()

# resample_frequency = 36
# resamples = L / Fs * resample_frequency
# X_prime = resample(X, resamples)
# X_prime -= X_prime_prime

# t_prime = np.arange(0,4,4./resamples)
# ax1.plot(t_prime, X_prime, 'r-')
# ax1.plot(t_prime, X_prime_prime, 'k-')
# ax1.set_ylim([-8,8])
# ax1.axhline(-2, color='g')
# ax1.axhline(2, color='g')

# S_prime = 2*np.sin(2*np.pi*9*t)
# ax2.plot(t,S_prime)
# ax2.set_ylim([-8,8])
# plt.show()


butt = Butter()

def plot_power_spectrum(X, clear, axis_handle, color):
    if clear:
        axis_handle.cla()
    # compute dft
    Y = np.fft.rfft(X)
    # two sided spectrum
    P2 = abs(Y/L)
    # one sided spectrum
    P1 = P2[1:int(L/2)+1]
    # multiply with 2 to get from 2-sided to 1-sided
    P1[2:-1] = 2*np.pi*P1[2:-1]

    f = Fs*np.arange(0,L/2)/L

    axis_handle.plot(f,P1,color=color)
    axis_handle.set_title("Power spectrum")
    axis_handle.set_ylabel('|P1(f)|')
    axis_handle.set_xlabel('f [Hz]')
    axis_handle.set_xlim([0,butt.highcut+5])
    # axis_handle.legend()

plot_power_spectrum(X, True, ax2, 'blue')

def plot_butter_filter(axis_handle, lowcut, highcut):
    lowcut_norm, highcut_norm = lowcut/nyq, highcut/nyq

    # find order and butterworth natural frequency. don't use buttord?
    (N,Wn) = buttord(wp=[lowcut_norm,highcut_norm],ws=[0.1/nyq,nyq],gpass=3,gstop=40,analog=0)
    b, a = butter(N, Wn, 'bandpass', analog=False, output='ba')
    w, h = freqz(b, a)

    axis_handle.cla()
    axis_handle.plot((nyq/np.pi)*w, abs(h), color='green')

    axis_handle.set_title('Butterworth filter frequency response')
    axis_handle.set_xlabel('Frequency [radians / second]')
    # ax4.set_ylabel('Amplitude [dB]')
    axis_handle.set_xlim([0,highcut+5])
    axis_handle.grid(which='both', axis='both')
    axis_handle.axvline(lowcut, color='black') # cutoff frequency
    axis_handle.axvline(highcut, color='black') # cutoff frequency
    return a,b


a,b = plot_butter_filter(ax4, butt.lowcut, butt.highcut)
butt.a = a
butt.b = b

lowpassed = filtfilt(butt.b,butt.a,X)


def plot_filtered_signal(X,axis_handle,color):
    axis_handle.cla()
    axis_handle.set_title('Filtered signal')
    axis_handle.plot(t,X, color='red')
    axis_handle.set_ylabel("X'(t)")
    axis_handle.set_xlabel('t [seconds]')
    # ax1.set_ylim([-8,8])

plot_filtered_signal(lowpassed,ax1,'red')

# compute dft
# Y = np.fft.rfft(lowpassed)
# # two sided spectrum
# P2 = abs(Y/L)
# # one sided spectrum
# P1 = P2[1:int(L/2)+1]
# P1[2:-1] = 2*np.pi*P1[2:-1]


# # frequency domain

# f = Fs*np.arange(0,L/2)/L
# ax2.plot(f,P1,color='red')
# ax2.set_xlim([0,highcut*1.5])

plot_power_spectrum(lowpassed, False, ax2, 'red')


# Add two sliders for tweaking the parameters
time_slider_ax  = fig.add_axes([0.08, 0.02, 0.85, 0.025], axisbg='white')
time_slider = Slider(time_slider_ax, 'Time', 0., len(signal)-L)
time_slider.valtext.set_text('{} s'.format(0))
# freq_slider_ax = fig.add_axes([0.25, 0.1, 0.65, 0.03], axisbg=axis_color)
# freq_slider = Slider(freq_slider_ax, 'Freq', 0.1, 30.0, valinit=freq_0)

def sliders_on_changed(val):
    samples =signal[int(val):int(val+L)]
    signal_max, signal_min = max(samples), min(samples)
    ax3.set_ylim([signal_min,signal_max])
    time_slider.valtext.set_text('{} s'.format(int(val/Fs)))

    labels = [item.get_text() for item in ax3.get_xticklabels()]
    for i in range(len(labels)):
        labels[i] = str(int(val/Fs)+2*i)
    ax3.set_xticklabels(labels)
    sigplt.set_ydata(samples)

    plot_power_spectrum(samples,True,ax2,'blue')
    lowpassed = filtfilt(butt.b,butt.a,samples)
    plot_filtered_signal(lowpassed, ax1,'red')
    plot_power_spectrum(lowpassed, False, ax2,'red')
    fig.canvas.draw_idle()

time_slider.on_changed(sliders_on_changed)

frequency_ranges_dict = OrderedDict([(r'$\delta$ (0.5-4Hz)', (0.5,4)),
                                (r'$\theta$ (4-8Hz)', (4,8)), 
                                (r'$\alpha$ (8-12Hz)', (8,12)),
                                (r'$\beta$ (12-30Hz)', (12,30)),
                                (r'$\gamma$ (30-50Hz)', (30,50))])
print frequency_ranges_dict.keys()


freq_ranges_ax = fig.add_axes([0.02, 0.45, 0.075, 0.12], axisbg='white')
freq_ranges = RadioButtons(freq_ranges_ax, frequency_ranges_dict.keys(), active=2)

def change_freq_band_on_click(label):
    (lowcut, highcut) = frequency_ranges_dict[label]

    butt.lowcut = lowcut
    butt.highcut = highcut
    butt.a,butt.b = plot_butter_filter(ax4, lowcut, highcut)

    sliders_on_changed(time_slider.val)

    fig.canvas.draw_idle()

freq_ranges.on_clicked(change_freq_band_on_click)

subject = file_path.split('/')
subject = '/'.join(subject[-4:])

plt.suptitle("Electrode EEG T4-LE\n" + subject, size=16)
plt.show()
