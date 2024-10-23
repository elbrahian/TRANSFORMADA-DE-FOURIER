import soundfile as sf
import numpy as np
import matplotlib.pyplot as plt

rate = 11400

def plotAudio(audio,k,n):

    plt.subplot(6, 2, n)
    plt.plot(audio)
    plt.title('Waveform of Audio ' + str(k))

def plotFreq(F, k,n):

    plt.subplot(6, 2, n)
    plt.plot(np.abs(F))
    plt.title('Frequency Spectrum of Audio ' + str(k))

# Lectura de audios
audio1,sr = sf.read('/home/smorenova/Audio/01_Drums.wav')
audio2,_ = sf.read('/home/smorenova/Audio/02_Bass.wav')
audio3,_ = sf.read('/home/smorenova/Audio/03_Electric_Left.wav')
audio4,_ = sf.read('/home/smorenova/Audio/04_Electric_Right.wav')
audio5,_ = sf.read('/home/smorenova/Audio/05_Lead_Vocal.wav')

# Aplicar la transformada de fourier a cada audio

audio1 *= 1
audio2 *= 1
audio3 *= 1
audio4 *= 1
audio5 *= 1

print("Relizando la transformada del audio 1...")
F1 = np.fft.fft(audio1)
print("Relizando la transformada del audio 2...")
F2 = np.fft.fft(audio2)
print("Relizando la transformada del audio 3...")
F3 = np.fft.fft(audio3)
print("Relizando la transformada del audio 4...")
F4 = np.fft.fft(audio4)
print("Relizando la transformada del audio 5...")
F5 = np.fft.fft(audio5)
print("Listo.")

#Ajustar los tamaños de cada matrix para que sean iguales (llenar con ceros)
n = max(F1.shape[0], F2.shape[0], F3.shape[0], F4.shape[0], F5.shape[0])

minLength = min(F1.shape[0], F2.shape[0], F3.shape[0], F4.shape[0], F5.shape[0])

print("Ajustando audio 1")
F1 = F1[:minLength]
print("Ajustando audio 2")
F2 = F2[:minLength]

print("Ajustando audio 3")
F3 = F3[:minLength]

print("Ajustando audio 4")
F4 = F4[:minLength]
print("Ajustando audio 5")
F5 = F5[:minLength]

print(F1.shape)
print(F2.shape)
print(F3.shape)
print(F4.shape)
print(F5.shape)


# Sumar cada transformada
mixed_F = F1 + F2 + F3 + F4 + F5

# Aplicar la transformada inversa de fourier
mixed_audio = np.fft.ifft(mixed_F).real  # Real part for audio data

# Guardar el nuevo archivo de audio
sf.write('mixed_audio.wav', mixed_audio, sr)

# Graficar
print("Dibujando la gráfica...")
plt.figure(figsize=(12, 8))

# Audio 1
plotAudio(audio1, 1, 1)
plotFreq(F1, 1, 2)

# Audio 2
plotAudio(audio2, 2, 3)
plotFreq(F2, 2, 4)

# Audio 3
plotAudio(audio3, 3, 5)
plotFreq(F3, 3, 6)

# Audio 4
plotAudio(audio4, 4, 7)
plotFreq(F4, 4, 8)

# Audio 5
plotAudio(audio5, 5, 9)
plotFreq(F5, 5, 10)

# Audio mezclado
plotAudio(mixed_audio, 6, 11)
plotFreq(mixed_F, 6, 12)

plt.tight_layout()

# Escribir la gráfica a un nuevo archivo
plt.savefig('audio_analysis.png')
plt.close()

print("Listo.")
