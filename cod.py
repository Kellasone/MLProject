#import pachetele de care am nevoie. 
#numpy pentru array-uri, pandas pentru citire mai usoara din csv, csv-ul pentru scrierea in fisier. 
#As fi putut folosi tot pandas dar imi adauga o coloana in plus si nu am stat sa ma chinui sa rezolv problema.
#Din keras am importat modelul Sequential, layer de tip Dense si functiile de activare si de asemenea optimizers pentru 
#loss function
#confusion_matrix din sklearn.metrics pentru matricea de confuzie
import numpy as np
import pandas as pd
import csv
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras import optimizers
from sklearn.metrics import confusion_matrix
#citirea datelor din fisiere. si introducerea lor in array-uri.
#array-urile for avea data type float pentru samples ca sa poata sustina numerele reale 
#si int pentru numerele intregi din labels.
train_labels = np.array(pd.read_csv('train_labels.csv', header=None), dtype='int8')
train_samples= np.array(pd.read_csv('train_samples.csv', header=None), dtype='float32')

test_samples= np.array(pd.read_csv('test_samples.csv', header=None), dtype='float32')
#impart datele din setul de antrenare in 3 fold-uri
labels1= np.array(train_labels[0:5000,])
samples1=np.array(train_samples[0:5000,:])
labels2=train_labels[5000:10000,]
samples2=train_samples[5000:10000,:]
labels3=train_labels[10000:15000,]
samples3=train_samples[10000:15000,:]

#deoarece raspunsurile mi se salveaza intr-un array de forma (:,1) si eu am nevoie de un array de forma (:,) folosesc functia
#flatten din pachetul numpy. imi este returnat un array intr-o singura dimensiune.
labels1 = labels1.flatten()
labels2 = labels2.flatten()
labels3 = labels3.flatten()

#instantiez modelul cu 3 layere. Primul layer, primeste input-ul de forma (4096,) iar ca output ofera un tensor de 32 de unitati.
#Functia de activare pentru acest layer este "relu" iar aceasta functie am ales-o pentru rapiditatea ei.
#Al 2-lea layer va avea un output de 16 unitati si va folosi aceeasi functie de activare.
#Ultimul layer va avea un output de 8 unitati, numarul de clase pe care noi il avem ca raspunsuri iar functia de activare va fi softmax.
#Functia softmax intoarce valori normalizate, pentru fiecare clasa cate un numar reprezentand probabilitatea raspunsului.
#Clasa cu cea mai mare probabilitate reprezinta raspunsul nostru. Insumarea tuturor probabilitatilor claselor are ca rezultat valoarea 1
model = Sequential([
    Dense(32, input_shape=(4096,)),
    Activation('relu'),
    Dense(16),
    Activation('relu'),
    Dense(8),
    Activation('softmax'),
])

#compilam modelul cu urmatorii parametrii:
#optimizator: sochastic gradent descent
#functia loss : categorical cross entropy in formatul in care preia integer, nu posibilitati de clase
#metrics reprezinta functia care ne arata cat de buna este reteaua dupa parametrul primit, in acest caz ne va arata acuratetea.
model.compile(optimizer='SGD',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

#incep antrenarea modelului pe sample-urile si label-urile de antrenare din foldul 1, in 25 de ere cu pachete de cate 64 de probe
model.fit(samples1, labels1, epochs=25, batch_size=64)

#afisez acuratetea si matricea de confuzie pe primul fold, dupa ce s-a antrenat pe el
score = model.evaluate(samples1, labels1, batch_size=128)
print(score[1])
x=np.array(model.predict_classes(samples1))
print(confusion_matrix(x, labels1))

#continui antrenarea modelului pe sample-urile si label-urile de antrenare din foldul 2
model.fit(samples2, labels2, epochs=25, batch_size=64)

#afisez acuratetea si matricea de confuzie pe primul fold si al 2-lea, dupa ce s-a antrenat pe amandoua
score = model.evaluate(samples1, labels1, batch_size=128)
print(score[1])
x=np.array(model.predict_classes(samples1))
print(confusion_matrix(x, labels1))

score = model.evaluate(samples2, labels2, batch_size=128)
print(score[1])
x=np.array(model.predict_classes(samples2))
print(confusion_matrix(x, labels2))

#afisez acuratetea si matricea de confuzie pe cel de-al treilea fold, pe care nu s-a antrenat
score = model.evaluate(samples3, labels3, batch_size=128)
print(score[1])
x=np.array(model.predict_classes(samples3))
print(confusion_matrix(x, labels3))

#Antrenez si pe ultimul fold ca antrenarea sa fie completa si sa il pun sa prezica ulterior pentru test_samples
model.fit(samples3, labels3, epochs=25, batch_size=64)

#Prezicerea si scrierea in csv in vederea uploadarii pe kaggle.
predicted = model.predict_classes(test_samples)
help=np.arange(5001)
import csv
with open('dinprezentare.csv', mode='w', newline='') as sm:
    writer = csv.writer(sm, delimiter=',')
    writer.writerow(['Id', 'Prediction'])
    for i in range(len(predicted)):
        writer.writerow([i+1,predicted[i]])