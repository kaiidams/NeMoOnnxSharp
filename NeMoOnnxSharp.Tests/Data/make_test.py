from nemo.collections.asr.modules import (
    AudioToMelSpectrogramPreprocessor,
    AudioToMFCCPreprocessor
)

import librosa
import torch


def readwav(filepath="61-70968-0000.wav", sr=16000):
    waveform, sr = librosa.load(filepath, sr=sr)

    print(f"  - Length {len(waveform):6}")
    print(f"  - Max    {waveform.max():6.3f}")
    print(f"  - Min    {waveform.min():6.3f}")
    print(f"  - Mean   {waveform.mean():6.3f}")

    return waveform


def pad_waveform(waveform):
    return np.concatenate([
        np.zeros((512-400)//2),
        waveform,
        np.zeros((512-400)//2)
    ])


def spectrogram(waveform, log_offset=1e-6):
    waveform = pad_waveform(waveform)

    S = librosa.stft(
        waveform,
        n_fft=512,
        hop_length=160,
        win_length=400,
        window="hann",
        center=False)
    S = np.log(np.abs(S) ** 2 + log_offset)

    return S.T.astype(np.float32)


def melspectrogram(waveform, sr=16000, log_offset=1e-6):
    waveform = pad_waveform(waveform)

    M = librosa.feature.melspectrogram(
        y=waveform,
        sr=sr,
        n_fft=512,
        hop_length=160,
        win_length=400,
        window="hann",
        center=False,
        n_mels=64,
        htk=True,
        norm=None)
    M = np.log(M + log_offset)

    return M.T.astype(np.float32)


def main():
    wavpath = "61-70968-0000.wav"
    sr = 16000
    audio_signal, sr = librosa.load(wavpath, sr=sr, mono=True)
    assert audio_signal.ndim == 1
    audio_signal = torch.from_numpy(audio_signal)
    audio_signal = torch.unsqueeze(audio_signal, 0)
    length = torch.tensor([audio_signal.shape[1]], dtype=torch.int64)
    convert_mfcc(audio_signal, length)
    return

    print("Spectrogram")

    X = spectrogram(waveform)
    print(f"  - Output {X.shape}")

    with open('spectrogram.bin', 'wb') as f:
        f.write(X.tobytes("C"))

    print("Mel-Spectrogram")

    X = melspectrogram(waveform)
    print(f"  - Output {X.shape}")

    with open('melspectrogram.bin', 'wb') as f:
        f.write(X.tobytes("C"))


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
