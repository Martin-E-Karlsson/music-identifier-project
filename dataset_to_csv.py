
import librosa
import numpy as np
import matplotlib.pyplot as plt
import os
import pathlib
import csv
import warnings
warnings.filterwarnings('ignore')




genres = ['1940s', '1950s', '1960s', '1970s', '1980s', '1990s', '2000s']

def get_img_pictures(genres):

    '''This function made melspectrogram of every song in the data set.'''

    cmap = plt.get_cmap('inferno')
    plt.figure(figsize=(10, 10))
    for g in genres:
        pathlib.Path(f'img_data/{g}').mkdir(parents=True, exist_ok=True)
        for filename in os.listdir(f'data/{g}'):
            songname = f'data/{g}/{filename}'
            y, sr = librosa.load(songname, mono=True, duration=5)
            plt.specgram(y, NFFT=2048, Fs=2, Fc=0, noverlap=128, cmap=cmap, sides='default', mode='default',
                         scale='dB')
            plt.axis('off')
            plt.savefig(f'img_data/{g}/{filename[:-3].replace(".", "")}.png')
            plt.clf()



def csv_template():

    '''This function create a csv template. where we can put our data from the songs.'''

    header = 'filename chroma_stft rmse spectral_centroid spectral_bandwidth rolloff zero_crossing_rate'
    for i in range(1, 21):
        header += f' mfcc{i}'
    header += ' label'
    header = header.split()

    file = open('data.csv', 'w', newline='')
    with file:
        writer = csv.writer(file)
        writer.writerow(header)




def get_data_to_csv(genres):

    '''this function extracts data/melspectrogram from each song and then puts it in the csv file.'''

    for g in genres:
        for filename in os.listdir(f'data/{g}'):
            songname = f'data/{g}/{filename}'
            y, sr = librosa.load(songname, mono=True, duration=30)
            chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
            spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)
            spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
            rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
            zcr = librosa.feature.zero_crossing_rate(y)
            mfcc = librosa.feature.mfcc(y=y, sr=sr)
            rmse = librosa.feature.rms(y=y)
            to_append = f'{filename} {np.mean(chroma_stft)} {np.mean(rmse)} {np.mean(spec_cent)} {np.mean(spec_bw)} {np.mean(rolloff)} {np.mean(zcr)}'
            for e in mfcc:
                to_append += f' {np.mean(e)}'
            to_append += f' {g}'
            file = open('data.csv', 'a', newline='')
            with file:
                writer = csv.writer(file)
                writer.writerow(to_append.split())


csv_template()
get_data_to_csv(genres)