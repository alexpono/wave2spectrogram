#%%

import numpy as np
from scipy import signal
from scipy.fft import fftshift
import matplotlib.pyplot as plt
rng = np.random.default_rng()

#%%
fs = 10e3
N = 1e5
amp = 2 * np.sqrt(2)
noise_power = 0.01 * fs / 2
time = np.arange(N) / float(fs)
mod = 500*np.cos(2*np.pi*0.25*time)
carrier = amp * np.sin(2*np.pi*3e3*time + mod)
noise = rng.normal(scale=np.sqrt(noise_power), size=time.shape)
noise *= np.exp(-time/5)
x = carrier + noise

#%%
f, t, Sxx = signal.spectrogram(x, fs)
plt.pcolormesh(t, f, Sxx, shading='gouraud')
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [sec]')
plt.show()

#%%
import librosa
import wave
soundPath = r"..\data\XC169082_Chardonneret_jaune_Spinus_tristis.mp3"
# wav_file = wave.open(soundPath)

#%%
x, fs = librosa.load(soundPath)
#%%
import matplotlib.colors as colors

f, t, Sxx = signal.spectrogram(x[50000:100000], fs)
plt.pcolormesh(t, f, Sxx,
                   norm=colors.LogNorm(vmin=Sxx.min(), vmax=Sxx.max()),
                   cmap='gray', shading='gouraud')
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [sec]')
plt.show()