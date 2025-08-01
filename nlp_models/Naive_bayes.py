import pandas as pd
import numpy as np
import tensorflow as tf
import sklearn as sk

# tfidf kütüphanesi
from sklearn.feature_extraction.text import TfidfVectorizer

# modelin türünü import et 3 tane türü var
# gaussian, sürekli veriler için
# bernolli, ikili sınıflar için
# multinomial çok kategorili veriler
from sklearn.naive_bayes import MultinomialNB

#doğruluk değerlendirmek için confusion_matrix recall ve precision import ediyoruz
from sklearn.metrics import confusion_matrix, classification_report
# 1. model naive bayes dünya spor ticaret bilim 1234
# model temizleme lowercase noktalama işareti kaldırma
# tokenization ve tf ıdf vektörleme

train= pd.read_csv(r"C:\Users\ezgii\OneDrive\Masaüstü\NLP_model\train.csv")
test = pd.read_csv(r"C:\Users\ezgii\OneDrive\Masaüstü\NLP_model\test.csv")

#1- Sütunları ayır ve isimle

train.columns= ["tür","başlık","açıklama"]
test.columns=["tür","başlık","açıklama"]

#2- Kullanıcağımız başlıklar için kod için anlamlandır

text1 =train["başlık"]
label1=train["tür"]

text2=test["başlık"]
label2=test["tür"]

#3- Veriyi temizle

def cleaning(x):
    x=x.lower()
    clean_data = []
    delete_this = "!,'^/&()[]{}?\-|_"
    for i in x:
        if i not in delete_this:
            clean_data.append(i)
    return"".join(clean_data)

text1 = train["başlık"].apply(cleaning)
label1 = train["tür"]-1

text2= test["başlık"].apply(cleaning)
label2=test["tür"]-1



# 4- tf-idf ayarını yapıp sayısal hale getir o kelimeyi nereden anlasın kelimelerin önemini sayılarla ifade eder

vectorizer = TfidfVectorizer()
text1_tf = vectorizer.fit_transform(text1)
text2_tf = vectorizer.transform(text2)

# 5 Model Eğit sadece train ile

model= MultinomialNB()
model.fit(text1_tf, label1)

# 6- Test verisinde tahmin yap

tahmin = model.predict(text2_tf)

# 7- modeli doğruluğunu değerlendir confusion_matrix recall precision

print("İşte confusion_matrix ile doğruluk")
print(confusion_matrix(label2,tahmin))

print("Şimdi de precision ve recall açısından bakalım")
print(classification_report(label2,tahmin,target_names=["World","Sport","Business","Sci_Fic"]))


