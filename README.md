# music-identifier-project
A project attempting to create an AI-model able to predict the genre and year a song was released.



# Running

 Preparing data:

    1. conversion_utils.py : If you want to convert mp3 song to wav.
    2. first_preparation_of_dataset.py : Changes name and add a number to every song.
    3. prepare_dataset.py : Mapping and pick out mfcc values to a json dataset.

 Model training:
    
    1. We have four different neural networks you can train with, multilayer_model_overfitting.py is the one that works best and is also the one on which the app runs.
       You can find all neural networks in neural_networks/

    2. Run multilayer_model_overfitting.py on all files in data_json/

 App:

    Run the app by running main.py

    The app is built with the help of 10 different models, the first model predict what genre it is on the song. 
    Then depending on what genre it predict it is. 
    Then it goes on to predict what year the song is from, in the specific genre.

