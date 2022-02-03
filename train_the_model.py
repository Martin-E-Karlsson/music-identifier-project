from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import pandas as pd
import numpy as np
from keras import models
from keras import layers
from keras.callbacks import TensorBoard
import time

from tensorflow import keras

model_name = "music-year-cnn-{}".format(int(time.time()))

tensorboard = TensorBoard(log_dir='logs\{}'.format(model_name))



data = pd.read_csv('data.csv')
data.head()
data = data.drop(['filename'], axis=1)

genre_list = data.iloc[:, -1]
encoder = LabelEncoder()
y = encoder.fit_transform(genre_list)
test_ = data.iloc[:, :-1]
scaler = StandardScaler()
X = scaler.fit_transform(np.array(data.iloc[:, :-1], dtype=float))

print()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


def train_model(X_train, X_test, y_train, y_test, tensorboard):

    model = models.Sequential()
    model.add(layers.Dense(256, activation='relu', input_shape=(X_train.shape[1],)))
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(7, activation='softmax'))

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    history = model.fit(X_train, y_train, epochs=20, batch_size=128)

    test_loss, test_acc = model.evaluate(X_test, y_test)
    print('test_acc: ', test_acc)

    x_val = X_train[:2]
    partial_x_train = X_train[2:]
    y_val = y_train[:2]
    partial_y_train = y_train[2:]

    model = models.Sequential()
    model.add(layers.Dense(512, activation='relu', input_shape=(X_train.shape[1],)))
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(7, activation='softmax'))

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.fit(partial_x_train, partial_y_train, epochs=30, batch_size=512, validation_data=(x_val, y_val), callbacks=[tensorboard])
    results = model.evaluate(X_test, y_test)
    print('Test loss:', results[0])
    print('Test accuracy:', results[1])
    print("******************************************************************")
    print(model.summary())

    model.save("saved_model\model.h5")

    return



train_model(X_train, X_test, y_train, y_test, tensorboard)


