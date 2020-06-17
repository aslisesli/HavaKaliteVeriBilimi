
#gerekli kütüphanelerin yüklenmesi
from keras.datasets import imdb
from keras.preprocessing import sequence
import numpy as np

#imdb verisetinin yüklenmesi
(x_eğitim, y_eğitim),(x_test, y_test) = imdb.load_data(num_words=10000)

#boyutların düzenelnmesi
x_eğitim = sequence.pad_sequences(x_eğitim, 500)
x_test = sequence.pad_sequences(x_test, 500)

from keras.layers import Dense, Embedding, Flatten, SimpleRNN
from keras.models import Sequential 

#RNN modelinin oluşturulması
model = Sequential()
model.add(Embedding(10000,32)) #10000 özellik vektörümüz, 32 çekirdek değerimiz
model.add(SimpleRNN(64))
model.add(Dense(1,activation="sigmoid"))
model.summary()

#modelin derlenmesi
model.compile(optimizer="rmsprop", loss="binary_crossentropy", metrics=["accuracy"])

#modelin eğitilmesi
eğitimModel = model.fit(x_eğitim,y_eğitim, epochs=15, validation_data=(x_test,y_test))

doğrulamaBaşarımı = eğitimModel.history["val_acc"]
print("Doğrulama başarımı: ", np.mean(doğrulamaBaşarımı))
