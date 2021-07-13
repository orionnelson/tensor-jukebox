# tensor-jukebox
A Jukebox that guesses the music you might want to play based on age and accent.

The Idea Behind Tensor Jukebox is is to be able to better predict what music a person will like based on their age gender accent race and emotion.

Accent Detection in Machine Learning models is a newer concept so this project will consist of learning Tensorflow and then eventually working to reverse engineer the submissions for INTERSPEECH AESRC 2020 Challenge.


> Speech Recognition Training Using an Breakthrough Wav2Vec Model: https://www.kdnuggets.com/2021/03/speech-text-wav2vec.html

> Speaker Identification with NeMo : https://github.com/NVIDIA/NeMo/blob/main/tutorials/speaker_recognition/Speaker_Recognition_Verification.ipynb

> Speach Parsing and Recognition with Nvida Jasper : https://ngc.nvidia.com/catalog/models/nvidia:tlt-jarvis:speechtotext_english_jasper

> Nvidia auto Punctuation Methods : https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/stable/nlp/punctuation_and_capitalization.html

```A Python3 Project ```

Shortlink: or9.ca/jbox

# Libraries In Use
 *BACKGROUND BASED ON ACCENT - Not Final*
> https://gitee.com/ephemeroptera/arnet.git
 
 *ETHNICITY GENDER AND AGE*
 
> https://github.com/serengil/deepface

# Future Libraries and Needed Features to look out for

> Spotify Bluetooth Implementation

> Apple Music Bluetooth Implementation

> Better Emotion Tracking Libraries

> Statistics on Peak Music Stagnation

> List of Intergenerational Songs

> Music Prefrences Based on Age Gender and Major Catagories of that Group.

> Boring Database Stuff


## Intructions to install Complete:

Install MinGW and make sure that gcc is working on system path.

- Do this like me first install choco in powershell https://chocolatey.org/install and run the ps1 script.
- make sure choco is on system path
- Then do  ```choco install mingw``` 

Follow this tutorial to get a general understanding of Git bash https://kbroman.org/github_tutorial/pages/first_time.html

## The Command Needed to download the repo is

```git lfs clone --recurse-submodules git@github.com:orionnelson/tensor-jukebox```

## Before Installing Requirements First Install both Cmake and Visual Studio Cmake addons.

Follow this tutorial here https://medium.com/analytics-vidhya/how-to-install-dlib-library-for-python-in-windows-10-57348ba1117f

Next I would recommend setting up ```pyvenv``` https://docs.python.org/3/library/venv.html use ```python -m ``` not ```python3```
```
1 : Get cmake  >> pip install cmake
2 : Install dlib  >> pip install dlib -vvv
3 : Install CUDA and Tensorflow.
4 : >> pip install torch==1.7.1+cu110 torchvision==0.8.2+cu110 torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.htm
 Then >> pip install -r requirements.txt
5 : If you get an error that is not 'face was not detected' try reinstalling keras 2.6.0rc0 and it will fix the issue.
```
Install takes >30 min due to slow repo's will fix later.

## Make sure a webcam is connected in order to test ```DeepFaceExample.py```
