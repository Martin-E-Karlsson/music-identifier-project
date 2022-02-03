import pandas as pd
import numpy as np
import os
import librosa
from sklearn.preprocessing import LabelEncoder, StandardScaler
import csv
import tensorflow as tf
from keras.models import load_model





def prepare_song(filepath):

    y, sr = librosa.load(filepath, mono=True, duration=30)
    chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
    spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)
    spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    zcr = librosa.feature.zero_crossing_rate(y)
    mfcc = librosa.feature.mfcc(y=y, sr=sr)
    rmse = librosa.feature.rms(y=y)
    to_append = f'{np.mean(chroma_stft)} {np.mean(rmse)} {np.mean(spec_cent)} {np.mean(spec_bw)} {np.mean(rolloff)} {np.mean(zcr)}'
    for e in mfcc:
        to_append += f' {np.mean(e)}'

    X = []
    for i in to_append.split(" "):
        num = float(i)
        X.append(num)

    return X



def prediction(X):

    categories = ['1940s', '1950s', '1960s', '1970s', '1980s', '1990s', '2000s']

    X = np.array([X])
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    model = load_model('saved_model\model.h5')
    predictions = model.predict(X)
    np.sum(predictions[0])

    result = np.argmax(predictions[0])

    result = print(f'Your Song Is From: {categories[int(result)]}')


    return result


filepath = 'data/1940s/1940s.0.mp3'

X = prepare_song(filepath)

X = prediction(X)












