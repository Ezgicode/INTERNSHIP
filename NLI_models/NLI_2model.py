# bir öncül premise- ve bir hipotez
# çıkarım mı çelişki mi bwlirsiz mi
# nlı sınıflandırma modeli geliştirilmeli 2 model

# tfıdf vectorize logistic regression naive bayes basit model
# Cnn ve BİLSTM gibi derin öğrenme modelleriyyle performans larşılaştır
# derin öğrenme modelleri için token sequence ye dönüştürülmelidir
# tokenizer kullanıp sequence üretimi gerekliymiş

# confusion matrix, accuracy  ve f1 score ile yorumla en iyi en kötü rapor

import pandas as pd
import numpy as np
import tensorflow as tf
import sklearn as sk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression


data_dev = pd.read_csv(r"C:\Users\ezgii\OneDrive\Masaüstü\snli_1.0\snli_1.0\snli_1.0_dev.txt",
                       encoding="utf-8", sep="\t", on_bad_lines="skip")

data_train = pd.read_csv(r"C:\Users\ezgii\OneDrive\Masaüstü\snli_1.0\snli_1.0\snli_1.0_train.txt",
                         encoding="utf-8", sep="\t", on_bad_lines="skip")

data_test = pd.read_csv(r"C:\Users\ezgii\OneDrive\Masaüstü\snli_1.0\snli_1.0\snli_1.0_test.txt",
                        encoding="utf-8", sep="\t", on_bad_lines="skip")




data_dev=data_dev[["gold_label","sentence1","sentence2"]]
data_train=data_train[["gold_label","sentence1","sentence2"]]
data_test=data_test[["gold_label","sentence1","sentence2"]]

label_map = {'entailment': 0, 'neutral': 1, 'contradiction': 2}
data_dev["label"]=data_dev["gold_label"].map(label_map)
data_train["label"]=data_train["gold_label"].map(label_map)
data_test["label"]=data_test["gold_label"].map(label_map)


"""
vectorizer = TfidfVectorizer()
dev = vectorizer.fit_transform(data_dev["sentence1","sentence2"])
train = vectorizer.transform(data_train["sentence1","sentence2"])
test = vectorizer.transform(data_test["sentence1","sentence2"])
"""

#ŞUNLARA RAPOR YAZ SİLMEDEN YAPILMIYO MU VE NEDEN ESKİLER GİBİ DEĞİL BÖL BİRLEŞTİR NİYE
#VECTORİZE ETME DE FARKLI NEDEN KARIŞIYO BU KADAR

data_dev = data_dev.dropna(subset=["label", "sentence1", "sentence2"])
data_train = data_train.dropna(subset=["label", "sentence1", "sentence2"])
data_test = data_test.dropna(subset=["label", "sentence1", "sentence2"])


data_dev["combined"] = data_dev["sentence1"] + " " + data_dev["sentence2"]
data_train["combined"] = data_train["sentence1"] + " " + data_train["sentence2"]
data_test["combined"] = data_test["sentence1"] + " " + data_test["sentence2"]

vectorizer = TfidfVectorizer()
train = vectorizer.fit_transform(data_train["combined"])
dev = vectorizer.transform(data_dev["combined"])
test = vectorizer.transform(data_test["combined"])

model= MultinomialNB()
model.fit(train,data_train["label"])
predictions=model.predict(test)

print("Model 1")
print("Confusion matrix ile gösterim")
print(confusion_matrix(data_test["label"],predictions))
print("Classificstion raporu ile gösterim")
print(classification_report(data_test["label"],predictions,target_names=['entailment','neutral','contradiction']))


model=LogisticRegression(max_iter=10000)
model.fit(train,data_train["label"])
predictions=model.predict(test)

print("Model 2")
print("Confusion Matrix Raporu")
print(confusion_matrix(data_test["label"],predictions))
print("classification raporu ile gösterim")
print(classification_report(data_test["label"],predictions,target_names=['entailment','neutral','contradiction']))

