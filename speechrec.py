import soundfile as sf
import os
import torch
from datasets import load_dataset
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import librosa
#Librispeech training.
import glob
global wavpath
wavpath = 'lib\wav' #Bunch of files length might need to be ~5 seconds at 16000.
processor = Wav2Vec2Processor.from_pretrained('lib\wav2vec2-large-960h-lv60')
model = Wav2Vec2ForCTC.from_pretrained('lib\wav2vec2-large-960h-lv60')


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
    print(transcription)

def run():
    for file in glob.glob(os.path.join(wavpath,'*.wav')):
        transcript(file)

run()
