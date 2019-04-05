# Proiect Machine Learning - Albu Mihail-Alexandru
### De ce am ales un algoritm bazat pe retele neuronale

Prin natura curiozitatii mele, am testat datele downloadate de pe kaggle cu toti clasificatorii invatati. Unii clasificatori au fost mai greu de inteles, altii mai usor. Intr-un final, am ales retelele neuronale prin prisma a doua aspecte:
>Biblioteca KERAS este foarte usor de folsit si destul de puternica, cu mult mai multe optiuni de implementare fata de sklearn si deasemenea buna interactiune pe care o are cu jupyter notebook, ceea ce m-a facut sa ma simt cu adevarat in mijlocul problemei.

>Acuratetea pe ultimele 3000 de date de antrenare, dupa ce s-a antrenat pe primele 12000 a fost maxima, fata de ceilalti clasificatori.

>Parerea personala asupra faptului ca retelele neuronale sunt cele mai puternice, ele nu doar invatand din trecut dar avand si capabilitatea sa se adapteze noilor date.

### Detalii tehnice

Pentru acest model am testat mai multe configuratii de parametrii. Am ramas totusi la cei default in mare parte pentru ca:
        --acuratetea e suficient de mare 
        --mult mai usor de implementat
Am folosit ca functii de activare, ReLU pentru straturile non-terminale si softmax pentru stratul terminal. Numarul de straturi l-am ales arbitrar, dupa cum se poate vedea in sumarizarea urmatoare: 

```
Layer (type)                 Output Shape              Param #   
=================================================================
dense_1 (Dense)              (None, 32)                131104    
_________________________________________________________________
activation_1 (Activation)    (None, 32)                0         
_________________________________________________________________
dense_2 (Dense)              (None, 16)                528       
_________________________________________________________________
activation_2 (Activation)    (None, 16)                0         
_________________________________________________________________
dense_3 (Dense)              (None, 8)                 136       
_________________________________________________________________
activation_3 (Activation)    (None, 8)                 0         
=================================================================
Total params: 131,768
Trainable params: 131,768
Non-trainable params: 0
```
Am folosit un layer initial, unul final si unul ascuns.
Dupa cum se poate vedea, rata de invatare nu a fost modificata, cea default fiind de 0.01.

Pierderea este calculata in functie de optimizatorul "SGD" - Sochastic Gradient Descend si functia loss "sparse_categorical_crossentropy".

Nu am folosit regularizare, optiunea default fiind " NONE " pentru cele 3 tipuri de regularizari posibile:

>kernel_regularizer
bias_regularizer
activity_regularizer


### Acuratetea modelului.

Am impartit datele de antrenare in 3 folduri, dupa cum urmeaza:
>foldul 1: primele 5000 de date.
>foldul 2: urmatoarele 5000 de date, de la 5000 la 9999
>foldul 3: ultimele 5000 de date, de la 10000 la 14999

Am antrenat reteaua pe primul fold, dupa care am facut matricea de confuzie si acuratetea pe acelasi fold, rezultatele fiind urmatoarele: 

```
    Accuracy:0.9682
    [[1719    6    0   20    1    0    0   17]
     [   4 1466    5    4    1    0    0    1]
     [   3    3  613    0    0    0   14    0]
     [  20    1    1  571    3    1    6    3]
     [   7    3    4    2  394   19    0    6]
     [   0    0    0    0    0    0    0    0]
     [   1    0    2    1    0    0   78    0]
     [   0    0    0    0    0    0    0    0]]
```
Am continuat apoi antrenarea si pe foldul al 2-lea, si am calculat acuratetea si matricea de confuzie pentru primele doua folduri, rezultatele fiind urmatoarele: 

 Pentru foldul 1
```
    Accuracy: 0.9682
    [[1719    6    0   20    1    0    0   17]
     [   4 1466    5    4    1    0    0    1]
     [   3    3  613    0    0    0   14    0]
     [  20    1    1  571    3    1    6    3]
     [   7    3    4    2  394   19    0    6]
     [   0    0    0    0    0    0    0    0]
     [   1    0    2    1    0    0   78    0]
     [   0    0    0    0    0    0    0    0]]
```
Pentru foldul 2
```
    Accuracy: 0.967
    [[1721    3    2   12    5    0    0   14]
     [   5 1453    3    2    4    0    0    2]
     [   6    4  589    0    2    0   12    0]
     [  49    1    0  557    5    3    4    4]
     [   1    1    1    0  409   11    1    2]
     [   0    0    0    0    2   11    0    1]
     [   0    0    3    0    0    0   94    0]
     [   0    0    0    0    0    0    0    1]]

```

Intr-un final, am calculat acuratetea si matricea de confuzie pentru cel de-al treilea fold, peste care nu s-a antrenat, rezultatele fiind urmatoarele:

```
    Accuracy: 0.9612
    [[1655    2    2   18    3    0    0   10]
     [   8 1462    2    2    3    0    0    1]
     [   8    4  576    1    1    0   10    0]
     [  61    3    0  639    5    3    3    3]
     [   1    3    3    0  392   11    1    1]
     [   2    0    0    1    1    0    0    1]
     [   1    1    8    4    0    1   80    0]
     [   0    1    0    0    0    0    0    2]]
```





