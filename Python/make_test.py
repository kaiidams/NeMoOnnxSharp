import librosa
import torch
from nemo.collections.asr.modules import (
    AudioToMelSpectrogramPreprocessor,
    AudioToMFCCPreprocessor
)


def main():
    wavpath = "61-70968-0000.wav"
    sr = 16000
    audio_signal, sr = librosa.load(wavpath, sr=sr, mono=True)
    assert audio_signal.ndim == 1
    audio_signal = torch.from_numpy(audio_signal)
    audio_signal = torch.unsqueeze(audio_signal, 0)
    length = torch.tensor([audio_signal.shape[1]], dtype=torch.int64)
    convert_mfcc(audio_signal, length)
    convert_mel_spectrogram(audio_signal, length)


def convert_mel_spectrogram(audio_signal, length):
    print(audio_signal.shape)
    preprocessor = AudioToMelSpectrogramPreprocessor(
        normalize="per_feature",
        window_size=0.02,
        sample_rate=16000,
        window_stride=0.01,
        window="hann",
        features=64,
        n_fft=512,
        frame_splicing=1,
        dither=0.00001,
        stft_conv=False)
    with torch.no_grad():
        processed_signal, processed_signal_length = preprocessor(input_signal=audio_signal, length=length)
    print(processed_signal, processed_signal_length)
    print(processed_signal.shape, processed_signal_length)
    with open('mel_spectrogram.bin', 'wb') as fp:
        fp.write(processed_signal[0].T.numpy().tobytes("C"))


def convert_mfcc(audio_signal, length):
    print(audio_signal.shape)
    preprocessor = AudioToMFCCPreprocessor(
        window_size=0.025,
        window_stride=0.01,
        window="hann",
        n_mels=64,
        n_mfcc=64,
        n_fft=512)
    with torch.no_grad():
        processed_signal, processed_signal_length = preprocessor(input_signal=audio_signal, length=length)
    print(processed_signal, processed_signal_length)
    print(processed_signal.shape, processed_signal_length)
    with open('mfcc.bin', 'wb') as fp:
        fp.write(processed_signal[0].T.numpy().tobytes("C"))


if __name__ == "__main__":
    main()
