import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import warnings
import xarray as xr
warnings.simplefilter('ignore')
from pylab import *
from math import *
from numpy.fft import fftfreq
from scipy.fftpack import *
from scipy.signal import butter, filtfilt, freqz

pwd = 'Modul_2/Tugas/'

# Baca Data
prData =  xr.open_dataarray(pwd + 'MSWEP_MON_INA_197902-202011.nc')

# Resample Data ke Bulanan Rata-Rata dan Potong Data di Koordinat Kota Kediri
prKediri = prData.resample(time='1M').mean().sel(lat=-7.8239, lon=111.9842, method='nearest')
prKediri20 = prKediri.sel(time=slice('1995-01-01', '2015-01-01'))
t = np.linspace(1, 240, 240)
n = len(t)
Y = prKediri20

# Plot Data Time-Series Curah Hujan
plt.figure(figsize=(15,6))
plt.plot(prKediri20, color='fuchsia')
plt.xlabel('Bulan',fontsize=15)
plt.ylabel('Curah Hujan (mm/bulan)', fontsize=15)
plt.title('Time Series Bulanan Kota Kediri Tahun 1995-2015')
plt.grid()
plt.xticks(np.arange(1, 240, 12))
plt.xlim(1, 240)
plt.savefig(pwd + 'Time-Series Kediri.png')
plt.close()

# Melakukan Standardisasi Data dan Plot Ulang
ybar = np.nanmean(Y) 
stdev = np.std(Y)
Y = (Y - ybar) / stdev
plt.figure(figsize=(15,6))
plt.plot(t,Y, color='darkgreen')
plt.xlabel('Bulan', fontsize=15)
plt.ylabel('Magnitude', fontsize=15)
plt.title('Data Time Series Hasil Standardisasi', fontsize=20)
plt.grid()
plt.xticks(np.arange(1, 240, 12))
plt.xlim(1, 240)
plt.savefig(pwd + 'Standarized Time Series Data CH Kota Kediri 1995-2015.png')
plt.close()

# Definisikan Fungsi Untuk Filtering
def butter_lpf(data, cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    yfilt = filtfilt(b, a, data, padlen=None)
    return yfilt

def butter_hpf(data, cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    yfilt = filtfilt(b, a, data, padlen=None)
    return yfilt

def butter_bpf(data, cutoff_low, cutoff_hi, fs, order=5):
    cutoff = np.array([cutoff_low, cutoff_hi])
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='bandpass', analog=False)
    yfilt = filtfilt(b, a, data, padlen=None)
    return yfilt

# Definisikan variabel-variabel untuk FFT
dt = t[1]-t[0]
F = fft(Y.values)
w = fftfreq(n,dt)
T = n/t[0:119]
indices = where(w > 0)
w_pos = abs(w[indices])
F_pos = abs(F[indices])

# Plot FFT Frekuensi
plt.figure(figsize=(15,6))
plt.plot(w_pos,abs(F_pos), color = 'orangered')
plt.xlabel('Frekuensi', fontsize=15)
plt.ylabel('Magnitude', fontsize=15)
plt.title('Periodogram FFT Frekuensi')
plt.grid()
plt.savefig(pwd + 'FFT Freq.png')
plt.close()

# Plot FFT Periode
plt.figure(figsize=(15,6))
plt.plot(T,abs(F_pos), color='slateblue')
plt.xlabel('Periode FFT (Bulan)', fontsize=15)
plt.ylabel('Magnitude', fontsize=15)
plt.title('Periodogram FFT Periode')
plt.grid()
plt.savefig(pwd + 'FFT Period.png')
plt.close()

# Lakukan Filter Lowpass dan Lakukan FFT
yfiltlow = butter_lpf(Y, 1/(12*7), 1)
Ffiltlow = fft(yfiltlow)
wfiltlow = fftfreq(n,dt)
indices = where(wfiltlow > 0)
w_posfilow = wfiltlow[indices]
F_posfilow = Ffiltlow[indices]

# Plot hasil FFT dari Lowpass Filtering
fig1, ax1 = plt.subplots(2, 1)
ax1[0].plot(T,abs(F_pos),'b')
ax1[0].set_xlabel('Periode (Bulan)',fontsize=13)
ax1[0].set_ylabel('Magnitude',fontsize=13)
ax1[0].set_title('Periodogram FFT Sebelum LPF',fontsize=15)
ax1[0].set_xlim([0,119])
ax1[0].tick_params(labelsize=13)
plt.grid()
ax1[1].plot(T,abs(F_posfilow),'b')
ax1[1].set_xlabel('Periode (Bulan)',fontsize=13)
ax1[1].set_ylabel('Magnitude',fontsize=13)
ax1[1].set_title('Periodogram FFT Setelah LPF',fontsize=15)
ax1[1].set_xlim([0,119])
ax1[1].tick_params(labelsize=13)
fig1.tight_layout()
plt.grid()
plt.savefig(pwd + 'FFT LPF.png')
plt.close()

# Lakukan Filter Highpass dan Lakukan FFT
yfilthi = butter_hpf(Y, 1/(12*3), 1)
Ffilthi = fft(yfilthi)
wfilthi = fftfreq(n,dt)
indices = where(wfilthi > 0)
w_posfihi = wfilthi[indices]
F_posfihi = Ffilthi[indices]

# Plot hasil FFT dari Highpass Filtering
fig2, ax2 = plt.subplots(2, 1)
ax2[0].plot(T,abs(F_pos),'b')
ax2[0].set_xlabel('Periode (Bulan)',fontsize=13)
ax2[0].set_ylabel('Magnitude',fontsize=13)
ax2[0].set_title('Periodogram FFT Sebelum HPF',fontsize=15)
ax2[0].set_xlim([0,119])
ax2[0].tick_params(labelsize=13)
ax2[1].plot(T,abs(F_posfihi),'b')
ax2[1].set_xlabel('Periode (Bulan)',fontsize=13)
ax2[1].set_ylabel('Magnitude',fontsize=13)
ax2[1].set_title('Periodogram FFT Setelah HPF',fontsize=15)
ax2[1].set_xlim([0,119])
ax2[1].tick_params(labelsize=13)
fig2.tight_layout()
plt.grid()
plt.savefig(pwd + 'FFT HPF.png')
plt.close()

# Lakukan Filter Bandpass dan Lakukan FFT
yfiltband = butter_bpf(Y, 1/(12*7), 1/(12*3), 1)
Ffiltband = fft(yfiltband)
wfiltband = fftfreq(n,dt)
indices = where(wfiltband > 0)
w_posfiband = wfiltband[indices]
F_posfiband = Ffiltband[indices]

# Plot hasil FFT dari Bandpass Filtering
fig3, ax3 = plt.subplots(2, 1)
ax3[0].plot(T,abs(F_pos),'b')
ax3[0].set_xlabel('Periode (Bulan)',fontsize=13)
ax3[0].set_ylabel('Magnitude',fontsize=13)
ax3[0].set_title('Periodogram FFT Sebelum BPF',fontsize=15)
ax3[0].set_xlim([0,119])
ax3[0].tick_params(labelsize=13)
ax3[1].plot(T,abs(F_posfiband),'b')
ax3[1].set_xlabel('Periode (Bulan)',fontsize=13)
ax3[1].set_ylabel('Magnitude',fontsize=13)
ax3[1].set_title('Periodogram FFT Setelah BPF',fontsize=15)
ax3[1].set_xlim([0,119])
ax3[1].tick_params(labelsize=13)
fig3.tight_layout()
plt.grid()
plt.savefig(pwd + 'FFT BPF.png')
plt.close()

# Plot Seluruh Grafik
plt.plot(t, Y, label='Data Asli')
plt.plot(t, yfiltlow, label='Low Pass')
plt.plot(t, yfilthi, label='High Pass')
plt.plot(t, yfiltband, label='Band Pass')
plt.xlabel('Periode')
plt.ylabel('Magnitude')
plt.title('Data Asli dan Hasil Filtering')
plt.legend()
plt.savefig(pwd + 'ORIFILTER.png')
plt.close()