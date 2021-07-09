import soundfile as sf
import torch
from datasets import load_dataset
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

# load pretrained model
processor = Wav2Vec2Processor.from_pretrained('lib\wav2vec2-large-960h-lv60')
model = Wav2Vec2ForCTC.from_pretrained('lib\wav2vec2-large-960h-lv60')

# load audio
audio_input, sample_rate = sf.read('lib\wav\Test.wav')

# pad input values and return pt tensor
input_values = processor(audio_input, sampling_rate=sample_rate, return_tensors="pt").input_values

# INFERENCE

# retrieve logits & take argmax
logits = model(input_values).logits
predicted_ids = torch.argmax(logits, dim=-1)

# transcribe
transcription = processor.decode(predicted_ids[0])

print(transcription)
