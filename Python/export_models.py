import nemo
import importlib


def get_class(cls_path):
    package_path = '.'.join(cls_path.split('.')[:-1])
    cls_name = cls_path.split('.')[-1]
    package = importlib.import_module(package_path)
    return getattr(package, cls_name)


def export(cls_path: str, model_name: str):
    cls = get_class(cls_path)
    model = cls.from_pretrained(model_name)
    print(model)
    model.export(f'{model_name}.onnx')



#export("nemo_asr.models.EncDecClassificationModel")

#import nemo
#import nemo.collections.asr as nemo_asr
#import nemo.collections.asr


cls_path = "nemo.collections.asr.models.EncDecClassificationModel"
#export(cls_path, 'vad_marblenet')
cls = get_class(cls_path)
model = cls.from_pretrained('vad_marblenet')
print(model.preprocessor)
print(model._cfg.preprocessor)
import librosa
import torch
import numpy as np
import struct
from glob import glob
for wave_file in glob("../test_data/*.wav"):
    audio, sample_rate = librosa.load(wave_file, sr=16000)
    audio_signal = torch.from_numpy(audio / 32768.0).to(torch.float32)
    audio_signal = torch.unsqueeze(audio_signal, 0)
    audio_signal_len = torch.tensor([audio.shape[0]], dtype=torch.int64)
    processed_signal, processed_signal_len = model.preprocessor(
        input_signal=audio_signal, length=audio_signal_len,
    )
    print(processed_signal.shape, processed_signal_len)
    processed_signal = processed_signal[0]
    bin_file = wave_file.replace('.wav', '.bin')
    with open(bin_file, "wb") as fp:
        fp.write(struct.pack('2i', *list(processed_signal.shape)))
        fp.write(processed_signal.numpy().tobytes())

# audio_signal, audio_signal_len = batch
#     audio_signal, audio_signal_len = audio_signal.to(vad_model.device), audio_signal_len.to(vad_model.device)
#     processed_signal, processed_signal_len = vad_model.preprocessor(
#         input_signal=audio_signal, length=audio_signal_len,
#     )