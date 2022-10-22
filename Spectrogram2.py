import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft
from scipy.io.wavfile import read
treshold = 300000#75000
plt.style.use('dark_background')
fig,(ax,ax2) = plt.subplots(nrows = 2,figsize = (500,60))
a = read('Toccata.wav')
Fs = a[0]
dt = 1/Fs
N = 1024
Dt = N*dt #longitud del bloque
df = Fs/N
def sigmoid(x):
  return (2/(1 + np.e**(-(x-treshold)/(3*treshold)))-1)

n = np.arange(int(N/2))
# si la onda tiene frecuencia (Fs - f) me queda igual a una con f por eso queda simetrica la fft por eso solo escribo hasta la mitad

freq_spectrum = n/Dt 
x = []
y = []
for pair in a[1]:
  x.append(pair[0])
x = np.array(x)
num_points = len(x)
num_blocks = int(num_points/N)
F = []
for k in range(num_blocks):
  block_wave = x[k*N:(k+1)*N]
  f = fft(block_wave)
  h = sigmoid(abs(f))*1*(abs(f) > treshold)
  F.append(h)
  if k % 1000 == 0:
    print('generando f:',k)

start = 0

for k in range(int(num_blocks/8)):
  k = k + start*int(num_blocks/8)
  f = F[k]
  if k % 100 == 0:
    print('graficando f:',k)
  for j in range(int(N/16)):
    t0 = k*Dt - start*int(num_blocks/8)*Dt
    f0 = j*df
    val = f[j]
    if val:
      ax.plot((t0,t0+Dt/2,t0+Dt/2,t0,t0),(np.log(f0)/np.log(2**(1/12)),np.log(f0)/np.log(2**(1/12)),np.log(f0)/np.log(2**(1/12))+1,np.log(f0)/np.log(2**(1/12))+1,np.log(f0)/np.log(2**(1/12))),'-',linewidth = 4,color = (val,1,val,val))
      #ax.plot((t0,t0+Dt/2,t0+Dt/2,t0,t0),(f0,f0,f0+df/2,f0+df/2,f0),'-',linewidth = 4,color = (1,val,val,val))
t = np.arange(0,num_points*dt,dt)
ax2.plot(t[0:int(num_points/8)],x[start*int(num_points/8):(start + 1)*int(num_points/8)],'.')
ax.plot(num_points/8*dt,60)
ax2.plot(num_points/8*dt,60)
ax.plot(0,60)
ax2.plot(0,60)
