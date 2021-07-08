# tensor-jukebox
A Jukebox that guesses the music you might want to play based on age and accent.

```A Python3 Project ```


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


## Intructions to install:

Follow this tutorial to get a general understanding of Git bash https://kbroman.org/github_tutorial/pages/first_time.html

## The Command Needed to download the repo is

```git clone --recurse-submodules git@github.com:orionnelson/tensor-jukebox```

## Before Installing Requirements First Install both Cmake and Visual Studio Cmake addons.

Follow this tutorial here https://medium.com/analytics-vidhya/how-to-install-dlib-library-for-python-in-windows-10-57348ba1117f

Next I would recommend setting up ```pyvenv```

```
1 : Get cmake  >> pip install cmake
2 : Install dlib  >> pip install dlib
 Then >> pip install -r requirements.txt
3 : If you get an error that is not 'face was not detected' try reinstalling keras 2.6.0rc0 and it will fix the issue.
```
Install takes >30 min due to slow repo's will fix later.

## Make sure a webcam is connected in order to test ```DeepFaceExample.py```
