"""Running this script requires having a functioning FFmpeg entry in Path in Environment Variables."""
import os
from data.config import *


def remove_spaces(directory, filename):
	newName = filename.replace(" ", "_")
	os.rename(f'{directory}\{filename}', f'{directory}\{newName}')
	return newName

def mp3_to_wav(mp3Name):
	wavName = f'{mp3Name[:-4]}.wav'
	os.system(f"""ffmpeg -y -i {MP3DIR}\{mp3Name} {WAVDIR}\{wavName}""")
	return wavName

def wav_to_mel_spectogram(wavName):
	imageName = f'{wavName[:-4]}_{IMGRES}.png'
	os.system(f"""ffmpeg -y -i {WAVDIR}\{wavName} -lavfi showspectrumpic=s={IMGRES}:mode=separate:legend={LEGEND} {IMGDIR}\{imageName}""")
	return imageName

def convert_data():
	# Create a directories for wav files and images they do not exist.
	if not os.path.exists(WAVDIR):
		os.mkdir(WAVDIR)
	if not os.path.exists(IMGDIR):
		os.mkdir(IMGDIR)

	# Loops through each file in the mp3 directory and applies selected functions.
	for filename in os.listdir(MP3DIR):
		# Since the windows command line can't handle spaces in filenames they are replaced with underscores.
		mp3Name = remove_spaces(MP3DIR, filename)

		# Convert the mp3 file to a wav file and place it in the \wav directory.
		wavName = mp3_to_wav(mp3Name)

		# Convert the wav file to a mel spectogram and place it in the \images directory.
		wav_to_mel_spectogram(wavName)