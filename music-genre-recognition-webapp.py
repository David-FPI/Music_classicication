import numpy as np
import streamlit as st
import base64
import pytube
import os
import subprocess 
import librosa
import tempfile 
from pydub import AudioSegment
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize
import tensorflow as tf
from statistics import mode
from tensorflow import keras
from keras import regularizers
from keras.preprocessing.image import load_img,img_to_array
from tensorflow.keras.models import Model, Sequential, load_model
from tensorflow.keras.layers import (Input, Add, Dense, Activation, ZeroPadding2D, BatchNormalization, 
                                     Flatten, Conv2D, AveragePooling2D, MaxPooling2D, GlobalMaxPooling2D,
                                     Dropout)



st.set_page_config(page_title="Music Genre Recognition App", layout="centered")


page_bg_img = """
<style>
[data-testid="stAppViewContainer"] {
background-image: url("https://images.unsplash.com/photo-1530176928500-2372a88e00b5?q=80&w=1974&auto=format&fit=crop&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D");
background-size: cover;
background-position: top left;
background-repeat: no-repeat;
}

[data-testid="stHeader"] {
background: rgba(0,0,0,0);
}

[data-testid="stToolbar"] {
right: 2rem;
}

</style>
"""

#define function to convert mp3 to wav format
def convert_mp3_to_wav(music_file):  
    sound = AudioSegment.from_mp3(music_file)
    sound.export("music_file.wav",format="wav")
  
#define funciton to produce and save mel spectogram
def create_melspectrogram(wav_file):  
    y,sr = librosa.load(wav_file)  
    mel_spec = librosa.power_to_db(librosa.feature.melspectrogram(y=y,sr=sr))    
    plt.figure(figsize=(10, 5))
    plt.axes([0., 0., 1., 1.], frameon=False, xticks=[], yticks=[])
    librosa.display.specshow(mel_spec, x_axis="time", y_axis='mel', sr=sr)
    plt.margins(0)
    plt.savefig('melspectrogram.png')

#define function to build CNN model
def GenreModel(input_shape = (100,200,4),classes=9):
    
    X_input = Input(input_shape)
    
    classifier = Sequential()

    classifier.add(Conv2D(8, (3, 3), input_shape = input_shape, activation = 'relu'))
    classifier.add(Activation('relu'))
    classifier.add(MaxPooling2D(pool_size = (2, 2)))

    classifier.add(Conv2D(16, (3, 3), activation = 'relu'))
    classifier.add(Activation('relu'))
    classifier.add(MaxPooling2D(pool_size = (2, 2)))

    classifier.add(Conv2D(32, (3, 3), activation = 'relu'))
    classifier.add(Activation('relu'))
    classifier.add(MaxPooling2D(pool_size = (2, 2)))

    classifier.add(Conv2D(64, (3, 3), activation = 'relu'))
    classifier.add(Activation('relu'))
    classifier.add(MaxPooling2D(pool_size = (2, 2)))

    classifier.add(Conv2D(128, (3, 3), activation = 'relu'))
    classifier.add(Activation('relu'))
    classifier.add(MaxPooling2D(pool_size = (2, 2)))

    classifier.add(Flatten())
    
    classifier.add(Dropout(0.5))
    classifier.add(Dense(units = 256, activation = 'relu', kernel_regularizer=regularizers.l2(0.0001)))
    classifier.add(Dropout(0.25))
    classifier.add(Dense(units = 10, activation = 'softmax'))

    classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics='accuracy')
    
    return classifier

#define function to predict music genre based on mel spectogram
def predict(image_data, model):   
    image = img_to_array(image_data)   
    image = np.reshape(image,(1,100,200,4))   
    prediction = model.predict(image/255)   
    prediction = prediction.reshape((10,))     
    class_label = np.argmax(prediction)     
    return class_label, prediction


class_labels = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal',  'pop',  'reggae', 'rock']
    
    
#configure bacckground
st.markdown(page_bg_img, unsafe_allow_html=True)


#configure main titles
st.markdown("<h1 style='text-align: center; color: black;'>Music Genre Recognition App</h1>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align: center; color: black;'>Know the genre of your favorite musics! </h3>", unsafe_allow_html=True)  


#configure sidebar - define social medias
col1,col2 = st.sidebar.columns(2)
linked_in = '''[![LinkedIn](https://upload.wikimedia.org/wikipedia/commons/thumb/0/01/LinkedIn_Logo.svg/120px-LinkedIn_Logo.svg.png)](https://www.linkedin.com/in/david-iss%C3%A1/)'''
col1.markdown(linked_in, unsafe_allow_html=True)
github = '''[![LinkedIn](https://upload.wikimedia.org/wikipedia/commons/thumb/2/29/GitHub_logo_2013.svg/120px-GitHub_logo_2013.svg.png)](https://github.com/davidissa99)'''
col2.markdown(github, unsafe_allow_html=True)


#configure sidebar - upload mp3 file and visualize it
st.sidebar.write("## Upload the mp3 file of a music of your choice:")
mp3_file = st.sidebar.file_uploader("Upload an audio file", type=["mp3"], label_visibility="collapsed")    


#configure model prediction and content to appear when music is uploaded
if mp3_file is not None:    
  st.sidebar.write("**Play the song below if you want!**")
  st.sidebar.audio(mp3_file,"audio/mp3")
  
  model = GenreModel(input_shape=(100,200,4),classes=10)
  model.load_weights("music_genre_recog_model.h5")
  
  convert_mp3_to_wav(mp3_file)
  audio_full = AudioSegment.from_wav('music_file.wav')
  
  class_labels_total = []
  predictions_total = []
  for w in range(int(round(len(audio_full)/3000,0))):
      audio_3sec = audio_full[3*(w)*1000:3*(w+1)*1000]
      audio_3sec.export(out_f = "audio_3sec.wav", format = "wav")
      create_melspectrogram("audio_3sec.wav")
      image_data = load_img('melspectrogram.png', color_mode='rgba', target_size=(100,200))   
      class_label, prediction = predict(image_data, model)
      prediction = prediction.reshape((10,)) 
      class_labels_total.append(class_label)
      predictions_total.append(prediction)
  
  class_label_final = mode(class_labels_total)
  predictions_final = np.mean(predictions_total, axis=0)
  
  color_data = [1,2,3,4,5,6,7,8,9,10]
  my_cmap = cm.get_cmap('Blues')
  my_norm = Normalize(vmin=0, vmax=10)
  fig,ax= plt.subplots(figsize=(10,5))
  ax.bar(x=class_labels,height=predictions_final, color=my_cmap(my_norm(color_data)))
  ax.spines['top'].set_visible(False)
  ax.spines['right'].set_visible(False)
  ax.spines['left'].set_visible(False)
  ax.spines['bottom'].set_color('#DDDDDD')
  ax.tick_params(bottom=False, left=False)
  plt.savefig("prob_distribution_genres.png",format='png', dpi=1000, transparent=True)
  
  st.markdown("<h4 style='text-align: center; color: black;'>The genre of your song is: {} </h4>".format(class_labels[class_label_final]), unsafe_allow_html=True) 
  st.markdown("<h4 style='text-align: center; color: black;'></h4>", unsafe_allow_html=True) 
  st.image("prob_distribution_genres.png", use_column_width=True, caption="Probability Distribution Of The Given Song Over Different Genres")
