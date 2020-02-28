import librosa
import os
import numpy as np
import librosa.display
from aco import ACOio, datetime, timedelta, Mp3Loader
import pydub 


from scipy import signal
from scipy.fft import fftshift
import matplotlib.pyplot as plt


#%%

# C:\Users\cruze\Documents\ocean-sounds\2015\02\01

file_dir = os.path.join('C:\\', 'Users', 'cruze', 'Documents', 'ocean-sounds')

loader = ACOio(file_dir, Mp3Loader)
target = datetime(day=1, month=2, year=2015)


# if needed, select a duration (defaults to 5 minutes)
# dur = timedelta(seconds=20)
# src = loader.load(target, dur)
src = loader.load(target)



# file_dir = os.path.join('2015', '02', '01')
# sample = os.path.join(file_dir, '2015-02-01--00.00.mp3')

# y, sr = librosa.load(sample)

# testout = os.path.join(os.getcwd(), 'test.wav')
# librosa.output.write_wav(testout, y, sr)

#%%

src.View()

src._data.shape


#%%




# C:\Users\cruze\Documents\ocean-sounds\2015\02\01
file_dir = os.path.join('C:\\', 'Users', 'cruze', 'Documents', 'ocean-sounds', '2015', '02', '01')
file_name = os.path.join(file_dir, '2015-02-01--00.00.mp3')



def read(f, normalized=False):
    """MP3 to numpy array"""
    a = pydub.AudioSegment.from_mp3(f)
    y = np.array(a.get_array_of_samples())
    if a.channels == 2:
        y = y.reshape((-1, 2))
    if normalized:
        return a.frame_rate, np.float32(y) / 2**15
    else:
        return a.frame_rate, y

def write(f, sr, x, normalized=False):
    """numpy array to MP3"""
    channels = 2 if (x.ndim == 2 and x.shape[1] == 2) else 1
    if normalized:  # normalized array - each item should be a float in [-1, 1)
        y = np.int16(x * 2 ** 15)
    else:
        y = np.int16(x)
    song = pydub.AudioSegment(y.tobytes(), frame_rate=sr, sample_width=2, channels=channels)
    song.export(f, format="mp3", bitrate="320k")



sr, x = read(file_name)
print(x)

x.shape

x[0:480000]


f, t, Sxx = signal.spectrogram(x[0:480000], sr)
plt.pcolormesh(t, f, Sxx)
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [sec]')
plt.show()


# scipy.signal.spectrogram(x, fs=1.0, window=('tukey', 0.25), nperseg=None, noverlap=None, 
# nfft=None, detrend='constant', return_onesided=True, scaling='density', axis=-1, mode='psd')[source]



# def spectrogram(
#         self, frame_duration=.08, frame_shift=.02, wtype='hanning'
#     ):
#         unit = self._Frame(frame_duration, frame_shift)
#         mat = unit.data * signal.get_window(wtype, unit.data.shape[1])
#         N = 2 ** int(np.ceil(np.log2(mat.shape[0])))
#         return unit._replace(data=np.fft.rfft(mat, n=N))

testout = os.path.join(os.getcwd(), 'test.mp3')
write(testout, sr, x)



#%%

from pylab import*
from scipy.io import wavfile

sampFreq, snd = wavfile.read('440_sine.wav')

snd.shape[0] / sampFreq
snd = snd / (2.**15)


s1 = snd[:,0] 


timeArray = np.arange(0, 5292, 1)
timeArray = timeArray / sampFreq
timeArray = timeArray * 1000  #scale to milliseconds

plot(timeArray, s1, color='k')
ylabel('Amplitude')
xlabel('Time (ms)')

n = len(s1) 
p = fft(s1) # take the fourier transform 

f, t, Sxx = signal.spectrogram(x, sr)
plt.pcolormesh(t, f, Sxx)
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [sec]')
plt.show()

# %%
