from symbol import continue_stmt
from unittest import skipIf

import librosa
import numpy as np
import os
import soundfile as sf
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import joblib

sample_rate= 16000# örnekleme oranı
hedefsüre= 2.0
sabituzunluk= int(sample_rate*hedefsüre)

#catörnek=(r"C:\Users\ezgii\OneDrive\Masaüstü\archive (6)\cats_dogs\test\cats\cat_3.wav")
cat= (r"C:\Users\ezgii\OneDrive\Masaüstü\archive (6)\cats_dogs\train\cat")
dog= (r"C:\Users\ezgii\OneDrive\Masaüstü\archive (6)\cats_dogs\train\dog")
test_cats = r"C:\Users\ezgii\OneDrive\Masaüstü\archive (6)\cats_dogs\test\cats"
test_dogs = r"C:\Users\ezgii\OneDrive\Masaüstü\archive (6)\cats_dogs\test\dogs"

def process_audio(path):
    y,sr =librosa.load(path, sr=sample_rate)
    if len(y)== 0:
       return np.zeros(sabituzunluk)

    max_val=np.max(np.abs(y))# sesin en büyük mutlak değerinin buluyoruz

    if max_val > 0:
        y=y/max_val
    y,_= librosa.effects.trim(y, top_db=20)

    if len(y)> sabituzunluk:
        y= y[:sabituzunluk]
    else:
        eksik = sabituzunluk - len(y)
        y  = np.pad(y,(0,eksik), mode='constant' ,constant_values=0)
    return y


# ================== AUGMENTATION ==================
def add_noise_snr(y, snr_db):
    sig_power = np.mean(y**2) + 1e-9
    snr = 10**(snr_db/10)
    noise_power = sig_power / snr
    noise = np.random.normal(0, np.sqrt(noise_power), size=y.shape)
    return y + noise

def time_shift(y, shift_max=0.2):
    n = len(y)
    s = int(np.random.uniform(-shift_max, shift_max) * n)
    return np.roll(y, s)

def time_stretch(y, rate):
    return librosa.effects.time_stretch(y, rate=rate)  # rate ~ [0.95, 1.05]

def pitch_shift(y, sr, n_steps):
    return librosa.effects.pitch_shift(y, sr=sr, n_steps=n_steps)  # n_steps ∈ [-2, +2]

def random_bandpass(y, sr, low=200, high=4000):
    S = librosa.stft(y, n_fft=1024, hop_length=256, win_length=1024)
    freqs = librosa.fft_frequencies(sr=sr, n_fft=1024)
    mask = (freqs >= low) & (freqs <= high)
    M = np.where(mask[:, None], 1.0, 0.3)  # bant dışını kıs
    return librosa.istft(S * M, hop_length=256, win_length=1024)

def random_augment(y, sr):
    """Rastgele en fazla 2 augment uygular."""
    ops = []
    if np.random.rand() < 0.6:
        ops.append(lambda x: add_noise_snr(x, snr_db=np.random.choice([10, 15, 20])))
    if np.random.rand() < 0.5:
        ops.append(lambda x: time_shift(x, 0.15))
    if np.random.rand() < 0.4:
        ops.append(lambda x: time_stretch(x, rate=np.random.uniform(0.95, 1.05)))
    if np.random.rand() < 0.4:
        ops.append(lambda x: pitch_shift(x, sr, n_steps=np.random.randint(-2, 3)))
    if np.random.rand() < 0.3:
        ops.append(lambda x: random_bandpass(x, sr))
    np.random.shuffle(ops)
    z = y.copy()
    for fn in ops[:2]:
        z = fn(z)
    return z
#yeniiiiiiiiiiiiiiiiiii

if __name__ == "__main__":
    cat_örnek = (r"C:\Users\ezgii\OneDrive\Masaüstü\archive (6)\cats_dogs\test\cats\cat_3.wav")
    output=  (r"C:\Users\ezgii\OneDrive\Masaüstü\sesdeneme\örnek1.wav")
    y_processed= process_audio(cat_örnek)
    sf.write(output,y_processed,sample_rate)

    print(len(y_processed), len(y_processed)/sample_rate)


output_cats=r"C:\Users\ezgii\OneDrive\Masaüstü\cat_dog_output\cats"
output_dogs=r"C:\Users\ezgii\OneDrive\Masaüstü\cat_dog_output\dogs"

os.makedirs(output_cats,exist_ok=True)
os.makedirs(output_dogs,exist_ok=True)



### dosya hazırlama kısmı değişti
for in_file, out_file in [(cat, output_cats), (dog, output_dogs)]:
    for name in os.listdir(in_file):
        if not name.lower().endswith(".wav"):
            continue
        new_path = os.path.join(in_file, name)
        if not os.path.exists(new_path):
            print("YOK:", new_path);
            continue

        y_processed = process_audio(new_path)

        # Orijinal kaydet
        out_path = os.path.join(out_file, name)
        sf.write(out_path, y_processed, sample_rate)

        # +3 augmentation kopyası üret ve kaydet
        for i in range(3):
            y_aug = random_augment(y_processed, sample_rate)
            aug_name = name.replace(".wav", f"_aug{i+1}.wav")
            out_path_aug = os.path.join(out_file, aug_name)
            sf.write(out_path_aug, y_aug, sample_rate)

        print("hazırladım + augment:", out_path)




#• Her ses dosyasından MFCC, Mel Spectrogram, Zero Crossing Rate, vb.
#çıkarınız.
#• Bu çıkardığınız özellikleri sabit boyutta bir vektör haline getirmeniz
#gerekmektedir.

def extract_features_from_processed(y, sr=sample_rate):
    # ---- 1. MFCC ----
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfcc_mean = np.mean(mfcc, axis=1)
    mfcc_std = np.std(mfcc, axis=1)

    # ---- 2. Mel Spectrogram ----
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=40)
    mel_mean = np.mean(mel, axis=1)
    mel_std = np.std(mel, axis=1)

    # ---- 3. Zero Crossing Rate ----
    zcr = librosa.feature.zero_crossing_rate(y)
    zcr_mean = np.mean(zcr)
    zcr_std = np.std(zcr)

    # Tek vektörde birleştir
    feature_vector = np.hstack([
        mfcc_mean, mfcc_std,
        mel_mean, mel_std,
        zcr_mean, zcr_std
    ])
    return feature_vector

# Örnek kullanım


# ==== TOPLU ÖZELLİK ÇIKARMA ve KAYDETME ====
features_dir = r"C:\Users\ezgii\OneDrive\Masaüstü\cat_dog_output\features"
os.makedirs(features_dir, exist_ok=True)

X = []
y_labels = []

for in_dir, label in [(output_cats, 0), (output_dogs, 1)]:
    for fname in os.listdir(in_dir):
        if not fname.lower().endswith(".wav"):
            continue
        fpath = os.path.join(in_dir, fname)

        # İşlenmiş wav'ı yükle (zaten normalize+pad edilmiş)
        y, sr = librosa.load(fpath, sr=sample_rate)

        # Özellik vektörü çıkar
        feat = extract_features_from_processed(y, sr=sample_rate)

        X.append(feat)
        y_labels.append(label)

# NumPy dizilerine çevir ve kaydet
X = np.array(X, dtype=np.float32)
y = np.array(y_labels, dtype=np.int64)

np.save(os.path.join(features_dir, "X.npy"), X)
np.save(os.path.join(features_dir, "y.npy"), y)

print("Kaydedildi:", X.shape, y.shape, "->", features_dir)

# ==== TEST VERİLERİNİ TEMİZLE VE KAYDET ====
output_test_cats = r"C:\Users\ezgii\OneDrive\Masaüstü\cat_dog_output\test_cats"
output_test_dogs = r"C:\Users\ezgii\OneDrive\Masaüstü\cat_dog_output\test_dogs"

os.makedirs(output_test_cats, exist_ok=True)
os.makedirs(output_test_dogs, exist_ok=True)

for in_file, out_file in [(test_cats, output_test_cats), (test_dogs, output_test_dogs)]:
    for name in os.listdir(in_file):
        if name.lower().endswith(".wav"):
            new_path = os.path.join(in_file, name)
            y_processed = process_audio(new_path)
            out_path = os.path.join(out_file, name)
            sf.write(out_path, y_processed, sample_rate)
            print("Test verisi hazırlandı:", out_path)

#data sızıntısı önleme kodu
    # ==== TEST VERİLERİNDEN ÖZELLİK ÇIKARMA ve KAYDETME ====
    X_test = []
    y_test_labels = []

    for in_dir, label in [(output_test_cats, 0), (output_test_dogs, 1)]:
        for fname in os.listdir(in_dir):
            if not fname.lower().endswith(".wav"):
                continue
            fpath = os.path.join(in_dir, fname)
            y, sr = librosa.load(fpath, sr=sample_rate)
            feat = extract_features_from_processed(y, sr=sample_rate)
            X_test.append(feat)
            y_test_labels.append(label)

    X_test = np.array(X_test, dtype=np.float32)
    y_test = np.array(y_test_labels, dtype=np.int64)

    np.save(os.path.join(features_dir, "X_test.npy"), X_test)
    np.save(os.path.join(features_dir, "y_test.npy"), y_test)

    print("Test verileri kaydedildi:", X_test.shape, y_test.shape)

#veri sızıntısı önleme kodunun bitişiymiş


    # Özelliklerin kayıtlı olduğu klasör
    features_dir = r"C:\Users\ezgii\OneDrive\Masaüstü\cat_dog_output\features"

    # 1) Verileri yükle
    X_train = np.load(os.path.join(features_dir, "X.npy"))
    y_train = np.load(os.path.join(features_dir, "y.npy"))
    X_test = np.load(os.path.join(features_dir, "X_test.npy"))
    y_test = np.load(os.path.join(features_dir, "y_test.npy"))

    print("Train:", X_train.shape, " Test:", X_test.shape)

    # 2) Ölçekleme (sadece train'e fit et)
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    # 3) Model (Logistic Regression)
    clf = LogisticRegression(max_iter=1000, n_jobs=-1)
    clf.fit(X_train_s, y_train)

    # 4) Değerlendirme
    y_pred = clf.predict(X_test_s)
    print("Accuracy:", round(accuracy_score(y_test, y_pred), 4))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred, target_names=["cat", "dog"]))

    # 5) Kaydet (model + scaler)
    joblib.dump(clf, os.path.join(features_dir, "model_logreg.joblib"))
    joblib.dump(scaler, os.path.join(features_dir, "scaler.joblib"))
    print("Kaydedildi:", os.path.join(features_dir, "model_logreg.joblib"))
    print("Kaydedildi:", os.path.join(features_dir, "scaler.joblib"))


#yeni test köpek üzerine #


features_dir = r"C:\Users\ezgii\OneDrive\Masaüstü\cat_dog_output\features"
model  = joblib.load(os.path.join(features_dir, "model_logreg.joblib"))
scaler = joblib.load(os.path.join(features_dir, "scaler.joblib"))

def predict_file(wav_path):
    y_proc = process_audio(wav_path)                                    # normalize + trim + sabitle
    feat   = extract_features_from_processed(y_proc, sr=sample_rate)    # vektör
    feat_s = scaler.transform([feat])                                   # ölçekle
    pred   = model.predict(feat_s)[0]                                   # 0=cat, 1=dog
    prob   = model.predict_proba(feat_s)[0]
    label  = "cat" if pred==0 else "dog"
    print(f"Dosya: {wav_path}\nTahmin: {label} (cat:{prob[0]:.2f}, dog:{prob[1]:.2f})")
    return label

# ÖRNEK:
test_wav = r"C:\Users\ezgii\OneDrive\Masaüstü\mivhaw\kedimiw1.wav"   # eğitimde hiç kullanılmamış yeni bir dosya
predict_file(test_wav)
#minik miw
test_wav = r"C:\Users\ezgii\OneDrive\Masaüstü\mivhaw\miw2.wav"   # eğitimde hiç kullanılmamış yeni bir dosya
predict_file(test_wav)
#tatlı gır
test_wav = r"C:\Users\ezgii\OneDrive\Masaüstü\mivhaw\miw3.wav"   # eğitimde hiç kullanılmamış yeni bir dosya
predict_file(test_wav)

print("köpekleri deniyoz")
#köpekleri deneylim ayrı veri setinde

# ÖRNEK:
test_wav = r"C:\Users\ezgii\OneDrive\Masaüstü\mivhaw\dog_34.wav"
predict_file(test_wav)

test_wav = r"C:\Users\ezgii\OneDrive\Masaüstü\mivhaw\dog_36.wav"
predict_file(test_wav)



####
import numpy as np
import os

features_dir = r"C:\Users\ezgii\OneDrive\Masaüstü\cat_dog_output\features"

X_train = np.load(os.path.join(features_dir, "X.npy"))
X_test = np.load(os.path.join(features_dir, "X_test.npy"))

print("Train shape:", X_train.shape)
print("Test shape:", X_test.shape)

# Ortak satır var mı diye kontrol
same_count = 0
for xt in X_test:
    if any(np.array_equal(xt, xtr) for xtr in X_train):
        same_count += 1

print(f"Test verisinde train verisiyle birebir aynı olan {same_count} örnek var.")


