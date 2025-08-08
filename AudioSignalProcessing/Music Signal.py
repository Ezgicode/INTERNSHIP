# ses kütüphanelerini import ediyoruz
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

# o videoyu import ediyoruz y ses  sinyalinin kendisi yani amplitüd değerleri
# sr sampling rate yani örnekleme hızı sr none diyerek orijinalini koruyoruz
y, sr = librosa.load(r"C:\Users\ezgii\OneDrive\Masaüstü\videoplayback.wav", sr=None)

# mfccs konuşma ve müzik analizi için yaygın ses öznitelikleridir ilk değer olarak 13 alıyoruz
mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)

# mfccs vektörlerini grafik olarak çizer x_axis time zaman eksenine göre çizimdir
plt.figure(figsize=(10,4))
librosa.display.specshow(mfccs, x_axis ='time')
plt.colorbar()
plt.title('MFCC')
#çizim sıkıştırılır boşluklar azalır
plt.tight_layout()
plt.show()

# mel spectrogram çıkarılır 128 farklı mel-band kullanılır sesin
# frekans içeriğinin zamanla nasıl değiştiğini mel ölçeğinde gösterir
S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)

# mel spectrogramdaki değerler dB desibel birimine çevrilir  ref=np.max en yüksek değere göre normalize eder
s_dB = librosa.power_to_db(S, ref=np.max)
plt.figure(figsize=(10,4))
librosa.display.specshow(s_dB, x_axis ='time', y_axis='mel', sr=sr, fmax=8000)
plt.colorbar(format='%+2.0f dB')
plt.title('Mel Spectrogram')
plt.tight_layout()
plt.show()


