import cv2
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.optimizers import Adam

# Conv2D özellik katmanı MaxPooling2D havuzlama katmanı boyut küçültür
# Flatten 1D ye çevirir Dense karar verir.

#Model oluşturuyoruz ilk katmanları ekliyoruz

model = Sequential() # Katmanları tek tek ekleyeceğim demek Sequential katman ekleyeceğimiz model tipi

model.add(Conv2D(32,(3,3), activation = "relu" , input_shape = (128,128,3))) #32 tane 3-3 filtrelerle tara 128-128 fotoğrafın boyutu renk kanalı
model.add(MaxPooling2D(2,2))# Havuzlama Katmanı boyutları küçültür 2-2 pencerelerle küçültür
model.add(Conv2D(64,(3,3), activation = "relu"))
model.add(MaxPooling2D(2,2))
model.add(Flatten())
model.add(Dense(64, activation = "relu"))
model.add(Dense(1,activation = "sigmoid"))


model2 = Sequential()

model2.add(Conv2D(32,(3,3), activation = "relu", input_shape = (128,128,3)))
model2.add(MaxPooling2D(2,2))
model2.add(Dropout(0.25))
model2.add(Conv2D(64,(3,3), activation = "relu"))
model2.add(MaxPooling2D(2,2))
model2.add(Dropout(0.25))
model2.add(Flatten())
model2.add(Dense(64, activation = "relu"))
model2.add(Dropout(0.5))
model2.add(Dense(1, activation ="sigmoid"))


path_train = r"C:\Users\ezgii\OneDrive\Masaüstü\test_set"

#imread yapmamıza gerek yok ana klasörden kendi okuyacak

#Resimlerdeki pikseller 0-255 arasında ama derin öğrenme modeli iyi çalışsın
# diye 255 e bölüyoruz pikselleri 0-1 arasına getiriyoruz.(rescale=1./255)

# Verinin yüzde 20si teste doğrulamaya 80 i eğitime ayırıyoruz

train_datagen = ImageDataGenerator (
    rescale = 1./255,
    validation_split = 0.2
)

# Resimleri eşitlik için 128-128 boyutuna indirgedik ve bir eğitimde 32 resim
# almasını sağladık veri setimizi  eğitim ve test olarak böldük

train_data = train_datagen.flow_from_directory(
    path_train,
    target_size = (128,128),
    batch_size = 32,
    class_mode = "binary",
    subset = "training"
)

validation_data1 = train_datagen.flow_from_directory(
    path_train,
    target_size = (128,128),
    batch_size = 32,
    class_mode = "binary",
    subset = "validation"
)

# Şimdi modeli derliyoruz yani nasıl öğreniceğini söyleyeceğiz

model.compile(
    optimizer = Adam(learning_rate=0.0005) , #En dengeleri Öğrenme algoritması hız ve yönü ayarlar
    loss = "binary_crossentropy", # Modelin doğruluğunu ölçer iki sınıf binary ölçüm yapar model yanılıyorsa loss büyük
    metrics = ["accuracy", "mse"], # Ekrana yazdırmak istediğimiz metrics biz doğruluk istedik
)

#Modeli eğitiyoruz her seferinde 10 epoch(tur)
train_stats = model.fit(
    train_data,
    validation_data = validation_data1,
    epochs = 8
)

model2.compile(
    optimizer = Adam(learning_rate=0.0005) ,#Adam a öğrenme yavaşlığı getirmeseydik import etmek zorunda kalmazdık
    loss = "binary_crossentropy",
    metrics = ["accuracy", "mse"]
)



train_stats2 = model2.fit(
    train_data,
    validation_data = validation_data1,
    epochs = 8
)




#Doğruluk Grafiği
plt.plot(train_stats.history["accuracy"],label = "Accuracy Static")
plt.plot(train_stats.history["val_accuracy"], label = "Accuracy Value Static")
plt.title("Accuracy Status")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.show()

#Loss Grafiği
plt.plot(train_stats.history["loss"],label= "Loss Static")
plt.plot(train_stats.history["val_loss"], label = "Loss Value Static")
plt.title("Loss Status")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()

# Mean square error Grafiği tahminin ne kadar saptığını
# ölçen metrik genelde regresyona kullnalılr

plt.plot(train_stats.history["mse"], label= "Mse Static")
plt.plot(train_stats.history["val_mse"], label = "Mse Value Static")
plt.title("MSE Status")
plt.xlabel("Epochs")
plt.ylabel("Mse")
plt.legend()
plt.show()


#------------------------------------------------------------------

plt.plot(train_stats2.history["accuracy"], label = "Accuracy Static")
plt.plot(train_stats2.history["val_accuracy"], label = "Accuracy Value Static")
plt.title("Accuracy Status")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.show()


plt.plot(train_stats2.history["loss"], label = "Loss Value Static")
plt.plot(train_stats2.history["val_loss"], label = "Loss Value Static")
plt.title("Loss Status")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()



plt.plot(train_stats2.history["mse"], label = "Mse Value Static")
plt.plot(train_stats2.history["val_mse"], label = " Mse Value Static")
plt.title("Mse Status")
plt.xlabel("Epochs")
plt.ylabel("Mse")
plt.legend()
plt.show()




