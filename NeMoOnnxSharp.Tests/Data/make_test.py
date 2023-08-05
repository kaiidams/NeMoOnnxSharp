import librosa
import numpy as np


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
	waveform = readwav()

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


if __name__ == "__main__":
	main()