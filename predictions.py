import math
import librosa
import numpy as np
from keras.models import load_model


def load_data(file_path, num_mfcc=13, n_fft=2048, hop_length=512, num_segments=80):
    SAMPLE_RATE = 22050
    TRACK_DURATION = 30
    SAMPLES_PER_TRACK = SAMPLE_RATE * TRACK_DURATION

    data = {
        "mfcc": []
    }
    samples_per_segment = int(SAMPLES_PER_TRACK / num_segments)
    num_mfcc_vectors_per_segment = math.ceil(samples_per_segment / hop_length)

    signal, sample_rate = librosa.load(file_path, sr=SAMPLE_RATE)
    for d in range(num_segments):
        start = samples_per_segment * d
        finish = start + samples_per_segment
        mfcc = librosa.feature.mfcc(signal[start:finish], sample_rate, n_mfcc=num_mfcc, n_fft=n_fft,
                                    hop_length=hop_length)
        mfcc = mfcc.T
        if len(mfcc) == num_mfcc_vectors_per_segment:
            data["mfcc"].append(mfcc.tolist())

    return data


def predict_genre(file_path):

    categories = ['Blues', 'Country', 'Disco', 'Hiphop', 'Jazz', 'Metal', 'Pop', 'Reggae', 'Rock']

    data = load_data(file_path)
    X = np.array(data["mfcc"])
    model = load_model('neural_networks/saved_models/model_multilayer_overfitting_Gtzan_mixed_genre.h5')
    prediction = model.predict(X)

    np.sum(prediction[0])
    result = np.argmax(prediction[0])
    genre = categories[int(result)]


    return print(f'Genre: {genre}')

def predict_genre_and_year(file_path):

    categories = ['Blues', 'Country', 'Disco', 'Hiphop', 'Jazz', 'Metal', 'Pop', 'Reggae', 'Rock']

    data = load_data(file_path)
    X = np.array(data["mfcc"])
    model = load_model('neural_networks/saved_models/model_multilayer_overfitting_Gtzan_mixed_genre.h5')
    prediction = model.predict(X)

    np.sum(prediction[0])
    result = np.argmax(prediction[0])
    genre = categories[int(result)]

    running = True
    while running:
        result = result
        genre = genre

        if result == 0:
            year = blues(file_path)
            print(f'Genre: {genre} From: {year}')
            break
        elif result == 1:
            year = country(file_path)
            print(f'Genre: {genre} From: {year}')
            break
        elif result == 2:
            year = disco(file_path)
            print(f'Genre: {genre} From: {year}')
            break
        elif result == 3:
            year = hiphop(file_path)
            print(f'Genre: {genre} From: {year}')
            break
        elif result == 4:
            year = jazz(file_path)
            print(f'Genre: {genre} From: {year}')
            break
        elif result == 5:
            year = metal(file_path)
            print(f'Genre: {genre} From: {year}')
            break
        elif result == 6:
            year = pop(file_path)
            print(f'Genre: {genre} From: {year}')
            break
        elif result == 7:
            year = reggae(file_path)
            print(f'Genre: {genre} From: {year}')
            break
        elif result == 8:
            year = rock(file_path)
            print(f'Genre: {genre} From: {year}')
            break
        else:
            running = False
    return


def blues(file_path):
    categories = ['1940s', '1950s', '1960s', '1970s', '1980s', '1990s', '2000s']

    data = load_data(file_path)
    X = np.array(data["mfcc"])
    model = load_model('neural_networks/saved_models/model_multilayer_overfitting_blues.h5')
    prediction = model.predict(X)
    np.sum(prediction[0])
    result = np.argmax(prediction[0])
    year = categories[int(result)]
    return year

def country(file_path):
    categories = ['1940s', '1950s', '1960s', '1970s', '1980s', '1990s', '2000s']

    data = load_data(file_path)
    X = np.array(data["mfcc"])
    model = load_model('neural_networks/saved_models/model_multilayer_overfitting_country.h5')
    prediction = model.predict(X)
    np.sum(prediction[0])
    result = np.argmax(prediction[0])
    year = categories[int(result)]
    return year

def disco(file_path):
    categories = ['1970s', '1980s', '1990s', '2000s']

    data = load_data(file_path)
    X = np.array(data["mfcc"])
    model = load_model('neural_networks/saved_models/model_multilayer_overfitting_disco.h5')
    prediction = model.predict(X)
    np.sum(prediction[0])
    result = np.argmax(prediction[0])
    year = categories[int(result)]
    return year

def hiphop(file_path):
    categories = ['1980s', '1990s', '2000s']

    data = load_data(file_path)
    X = np.array(data["mfcc"])
    model = load_model('neural_networks/saved_models/model_multilayer_overfitting_hiphop.h5')
    prediction = model.predict(X)
    np.sum(prediction[0])
    result = np.argmax(prediction[0])
    year = categories[int(result)]
    return year

def jazz(file_path):
    categories = ['1940s', '1950s', '1960s', '1970s', '1980s', '1990s', '2000s']

    data = load_data(file_path)
    X = np.array(data["mfcc"])
    model = load_model('neural_networks/saved_models/model_multilayer_overfitting_jazz.h5')
    prediction = model.predict(X)
    np.sum(prediction[0])
    result = np.argmax(prediction[0])
    year = categories[int(result)]
    return year

def metal(file_path):
    categories = ['1970s', '1980s', '1990s', '2000s']

    data = load_data(file_path)
    X = np.array(data["mfcc"])
    model = load_model('neural_networks/saved_models/model_multilayer_overfitting_metal.h5')
    prediction = model.predict(X)
    np.sum(prediction[0])
    result = np.argmax(prediction[0])
    year = categories[int(result)]
    return year


def pop(file_path):
    categories = ['1960s', '1970s', '1980s', '1990s', '2000s']

    data = load_data(file_path)
    X = np.array(data["mfcc"])
    model = load_model('neural_networks/saved_models/model_multilayer_overfitting_pop.h5')
    prediction = model.predict(X)
    np.sum(prediction[0])
    result = np.argmax(prediction[0])
    year = categories[int(result)]
    return year

def reggae(file_path):
    categories = ['1960s', '1970s', '1980s', '1990s', '2000s']


    data = load_data(file_path)
    X = np.array(data["mfcc"])
    model = load_model('neural_networks/saved_models/model_multilayer_overfitting_reggae.h5')
    prediction = model.predict(X)
    np.sum(prediction[0])
    result = np.argmax(prediction[0])
    year = categories[int(result)]
    return year

def rock(file_path):
    categories = ['1950s', '1960s', '1970s', '1980s', '1990s', '2000s']

    data = load_data(file_path)
    X = np.array(data["mfcc"])
    model = load_model('neural_networks/saved_models/model_multilayer_overfitting_rock.h5')
    prediction = model.predict(X)
    np.sum(prediction[0])
    result = np.argmax(prediction[0])
    year = categories[int(result)]
    return year










