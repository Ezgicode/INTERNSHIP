import pandas as pd
import numpy as np
import tensorflow as tf
import sklearn as sk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix,classification_report



train = pd.read_csv(r"C:\Users\ezgii\OneDrive\Masaüstü\NLP_model\train.csv")
test = pd.read_csv(r"C:\Users\ezgii\OneDrive\Masaüstü\NLP_model\test.csv")

train.columns= ["tür","başlık","açıklama"]
test.columns=["tür","başlık","açıklama"]


text_train=train["başlık"]
label_train=train["tür"]

text_test=test["başlık"]
label_test=test["tür"]


def cleaning(text):
    clean_data = []
    delete_this = "!,'^/&()[]{}?\-|_"
    for i in text:
        if i  not in delete_this:
            clean_data.append(i)
    return "".join(clean_data)


text_train=text_train.apply(cleaning)
label_train=label_train-1

text_test=text_test.apply(cleaning)
label_test=label_test-1



vectorizer= TfidfVectorizer()
train_tf= vectorizer.fit_transform(text_train)
test_tf= vectorizer.transform(text_test)


#max_iter belirtmezsek varsayılanı 100 alır bu da model için iterasyon sayısı yeterli olmayabilir o yüzden elle bir değer veriyoruz

model = LogisticRegression(max_iter= 1000)
model.fit(train_tf,label_train)

tahmin=model.predict(test_tf)

# recall precision confusion_matrix

print(confusion_matrix(label_test,tahmin))
print(classification_report(label_test,tahmin,target_names=["World","Sport","Business","Sci-Fci"]))
