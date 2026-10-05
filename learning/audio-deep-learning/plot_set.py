#画图函数
from random import sample
import librosa
import numpy as np
from matplotlib import pyplot as plt

def plot_waveform(waveform,sr,title="Waveform"):
    waveform = np.array(waveform)
    samples = waveform.size
    time_scale = np.linspace(0,samples/sr,num=samples)
    plt.figure(figsize=(10,10))
    plt.plot(time_scale,waveform,linewidth=1)
    plt.title(title)
    plt.grid(True)
    plt.show()

def plot_waveform_fft(waveform,sr,n_fft,title="Waveform_FT"):
    waveform = np.array(waveform)
    samples = waveform.size
    waveform_fft = np.fft.rfft(waveform,n_fft)
    freq_scale = np.linspace(0,sr/2,num=int(n_fft/2)+1)
    plt.figure(figsize=(10,10))
    plt.plot(freq_scale,waveform_fft,linewidth=1)
    plt.title(title)
    plt.grid(True)
    plt.show()

def plot_spectrogram(spectrogram,title="Speactrogram(dB)"):
    plt.imshow(librosa.amplitude_to_db(spectrogram))
    plt.title(title)
    plt.xlabel("Frame/s")
    plt.ylabel("Frequency/Hz")
    plt.colorbar()
    plt.show()