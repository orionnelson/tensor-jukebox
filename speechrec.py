import soundfile as sf
import os
import torch
from datasets import load_dataset
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import librosa
#Librispeech training.
import glob
global wavpath
import numpy as np
wavpath = 'lib\wav' #Bunch of files length might need to be ~5 seconds at 16000.
processor = Wav2Vec2Processor.from_pretrained('lib\wav2vec2-large-960h-lv60')
model = Wav2Vec2ForCTC.from_pretrained('lib\wav2vec2-large-960h-lv60')
import sounddevice as sd
import time
from scipy.io.wavfile import write
global num
num = 0

class speechrec(object):
    @staticmethod
    def record(seconds):
        global num
        sd.wait()
        fs = 16000  # Sample rate

        myrecording = sd.rec(int(seconds * fs), samplerate=fs, channels=1, dtype=np.int16)
        sd.wait()  # Wait until recording is finished
        file = wavpath +'/test'+ str(num)+ '.wav'
        write(file, fs, myrecording)
        num = num + 1
        return file


    @staticmethod
    def transcript(wav):
        # load audio
        audio_input, sample_rate = librosa.load(wav,sr= 16000)
        # pad input values and return pt tensor
        input_values = processor(audio_input, sampling_rate=sample_rate, return_tensors="pt", padding="longest").input_values
        # INFERENCE
        # retrieve logits & take argmax
        logits = model(input_values).logits
        predicted_ids = torch.argmax(logits, dim=-1)
        # transcribe
        transcription = processor.batch_decode(predicted_ids)[0]
        return (transcription)

    @staticmethod
    def run():
        for file in glob.glob(os.path.join(wavpath,'*.wav')):
            speechrec.transcript(file)

def example1():
    print("\n\n\nSPEAK  NOW TO HAVE THE MODEL TRANSCRIBE YOUR VOICE: \n\n\n")
    # Tests Audio Recording This can be modified for future usage but currently records to lib/wav/test%d.wav    
    start_time = 0
    while True:
            seconds = 30  # Duration of recording
            file = speechrec.record(seconds)
            sd.wait()
            #print(file)
            os.system('vad.py  3 '+str(file))
            # in case recording proccessing time fails / does somthing stupid we have this method that waits till sd is finished
            transcript = speechrec.transcript(file)
            print(transcript)

def example2():
    #Transcribes all audio in recordings.
    speechrec.run()




#example1()
