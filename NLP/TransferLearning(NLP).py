import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Embedding, LSTM, Dense, Bidirectional
from tensorflow.keras.models import Sequential
import numpy as np
import pandas as pd
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard
import os


df = pd.read_csv(r"C:\datasets\sentiment_analysis.csv")

#Bizim veri setimizde 0-1 olarak değil o yüzden
# sentiment sütununu sayılara çeviriyoruz

df["sentiment"] = df["sentiment"].map({
    "positive": 1,
    "neutral":0,
    "negative" : -1
})

x = df["text"]
y = df["sentiment"]

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split (x,y,test_size = 0.2)

#----------------------------------------------------------------------

# ilk kelimeleri sayı yapmalıyz ki makine anlayabilsin o kelime ne bilsin

vocab_size = 10000
oov_token = ("bilmiyom ki ;{")
tokenizer = Tokenizer(num_words= vocab_size, oov_token = oov_token)
# Eğitim verisindeki metinleri kullanarak kelime sayı sözlüğünü oluşturur
tokenizer.fit_on_texts(x_train)

# her kelimenin nasıl numaralandırıldığını görürüz
word_index = tokenizer.word_index

padding_type = "post"
truncation_type = "post"
max_length = 100

# her cümleyi bi sayı listesine çeviriyruz
#fonksiyon kelimeyi sözlükten bulup sayıya çeviriyor
x_train_sequences = tokenizer.texts_to_sequences(x_train)

x_train_padded = pad_sequences(
    x_train_sequences,
    maxlen=max_length,
    padding=padding_type,
    truncating=truncation_type
)


x_test_sequences = tokenizer.texts_to_sequences(x_test)

x_test_padded = pad_sequences(
    x_test_sequences,
    maxlen=max_length,
    padding=padding_type,
    truncating=truncation_type
)
# cümlelerin uzunlukları farklı olduğu için bu sequence lerin
# de uzunlukları farklı oluyr ama eşit olmalı truncating
# longer sentence and padding shorter ones with zeros




embeddings_index = {}
#veri setini kendi dict e kopyaladık
with open(r"C:\datasets\glove.6B.100d.txt\glove.6B.100d.txt", encoding="utf8") as f:
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype = "float32")
        embeddings_index[word] =coefs
f.close()

print("Found %s word vectors."% len(embeddings_index))

#glovebedem yüklediğimiz embeddings_index sözlüğünü
# kullanıp kelimeleri temsil eden embedding matrix oluşturuyoruz

embedding_matrix = np.zeros((len(word_index)+1,100))

for word ,i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector

embedding_layer = Embedding(len(word_index) + 1,
                            max_length,
                            weights = [embedding_matrix],
                            input_length = max_length,
                            trainable = False)

model = Sequential([
    embedding_layer,
    Bidirectional(LSTM(150, return_sequences = True)),
    Bidirectional(LSTM(150)),# false yazmıyoruz default zaten
    Dense(6, activation ="relu"),
    Dense (1,activation = "sigmoid")
])

model.compile(
    loss = "binary_crossentropy",
    optimizer = "adam",
    metrics = ["accuracy"]
)
"""
log_folder = os.path.join('logs',"train")
if not os.path.exists(log_folder):
    os.makedirs(log_folder)"""

callbacks = [
            EarlyStopping(patience = 10),
            #TensorBoard(log_dir=log_folder)
            ]

num_epochs = 600
history = model.fit(x_train_padded, y_train,
                    epochs=num_epochs,
                    validation_data=(x_test_padded, y_test),
                    callbacks=callbacks)

loss, accuracy = model.evaluate(x_test_padded,y_test)
print('Test accuracy :', accuracy)









