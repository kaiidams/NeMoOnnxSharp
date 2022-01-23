from typing import List, Text

audio_file = "test4.wav"
onnx_file = "QuartzNet15x5Base-En.onnx"
config_path = "./examples/asr/conf/quartznet/quartznet_15x5.yaml"


def test_nemo():
    import nemo.collections.asr as nemo_asr
    quartznet = nemo_asr.models.EncDecCTCModel.from_pretrained(model_name="QuartzNet15x5Base-En")
    files = [audio_file]
    texts = quartznet.transcribe(paths2audio_files=files)
    for fname, transcription in zip(files, texts):
        print(f"Audio in {fname} was recognized as: {transcription}")


def export():
    import nemo.collections.asr as nemo_asr
    quartznet = nemo_asr.models.EncDecCTCModel.from_pretrained(model_name="QuartzNet15x5Base-En")
    quartznet.export(onnx_file)


def preprocess(audio_files: List[Text]):
    import nemo.collections.asr as nemo_asr
    import torch
    import soundfile as sf
    import librosa
    from omegaconf import OmegaConf
    import numpy as np

    config = OmegaConf.load(config_path)
    preprocessor = nemo_asr.models.EncDecCTCModel.from_config_dict(config.model.preprocessor)
    vocabulary = config["model"]["decoder"]["vocabulary"]

    preprocessor.eval()

    input_signal = []
    input_signal_length = []
    for audio_file in audio_files:
        x, sample_rate = sf.read(audio_file)
        x = librosa.resample(x, sample_rate, 16000)
        x = torch.from_numpy(x).float()
        assert x.dim() == 1
        input_signal.append(x)
        input_signal_length.append(x.shape[0])

    input_signal = torch.nn.utils.rnn.pad_sequence(input_signal, batch_first=True)
    input_signal_length = torch.tensor(input_signal_length, dtype=torch.long)
    sample_rate = 16000
    print(input_signal.shape)

    input_signal = input_signal * 0.8 / torch.max(torch.abs(input_signal))

    audio_signal, audio_signal_length = preprocessor(
        input_signal=input_signal,
        length=input_signal_length)

    x: np.ndarray = audio_signal.float().numpy()
    with open("test.bin2", "wb") as f:
        f.write(x.tobytes())

    print(audio_signal.shape)
    np.savez(
        "test.npz",
        audio_signal=audio_signal.numpy(),
        audio_signal_length=audio_signal_length.numpy(),
        vocabulary=vocabulary)


def preprocess2(audio_files):
    import soundfile as sf
    import librosa
    import torch

    sample_rate = 16000
    n_fft = 512
    preemph = 0.97
    hop_length = 160
    win_length = 320
    nfilt = 64
    lowfreq = 0
    highfreq = sample_rate / 2
    log_zero_guard_value = 2 ** -24

    # Read data
    input_signal = []
    input_signal_length = []
    for audio_file in audio_files[:1]:
        x, orig_sample_rate = sf.read(audio_file)
        x = librosa.resample(x, orig_sample_rate, sample_rate)
        x = torch.from_numpy(x).float()
        x = 0.8 * x / torch.max(torch.abs(x))
        assert x.dim() == 1
        input_signal.append(x)
        input_signal_length.append(x.shape[0])
    input_signal = torch.nn.utils.rnn.pad_sequence(input_signal, batch_first=True)
    input_signal_length = torch.tensor(input_signal_length, dtype=torch.long)

    # Window
    window = torch.hann_window(win_length, periodic=False, dtype=torch.float32)

    # Filterbanks
    filterbanks = torch.tensor(
        librosa.filters.mel(sample_rate, n_fft, n_mels=nfilt, fmin=lowfreq, fmax=highfreq), dtype=torch.float
    ).unsqueeze(0)

    x = input_signal

    seq_len = input_signal_length // hop_length + 1

    # Preemphasize
    x = torch.cat((x[:, 0].unsqueeze(1), x[:, 1:] - preemph * x[:, :-1]), dim=1)

    # Short-time Fourier transform
    with torch.cuda.amp.autocast(enabled=False):
        x = torch.stft(
            x,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            center=True,
            window=window,
            return_complex=False)

    # To magnitude
    x = x.pow(2).sum(-1)
    print(x.shape)

    x = torch.matmul(filterbanks, x)
    print(x.shape)

    x = torch.log(x + log_zero_guard_value)

    print(seq_len)
    x_mean = torch.zeros((seq_len.shape[0], x.shape[1]), dtype=x.dtype, device=x.device)
    x_std = torch.zeros((seq_len.shape[0], x.shape[1]), dtype=x.dtype, device=x.device)
    for i in range(x.shape[0]):
        x_mean[i, :] = x[i, :, : seq_len[i]].mean(dim=1)
        x_std[i, :] = x[i, :, : seq_len[i]].std(dim=1)

    x_std += 1e-5
    x = (x - x_mean.unsqueeze(2)) / x_std.unsqueeze(2)

    print(x.shape)

    import numpy as np
    with np.load("test.npz") as f:
        audio_signal = f["audio_signal"]
        audio_signal_length = f["audio_signal_length"]
        vocabulary = f["vocabulary"]

    audio_signal = audio_signal[:1, :, :audio_signal_length[0]]
    d = torch.sum(torch.pow(x - audio_signal, 2))
    print(audio_signal[0, :3, :10])
    print(x[0, :3, :10])
    print(d)

    return x


def infer():
    import re
    import numpy as np
    import onnxruntime
    sess = onnxruntime.InferenceSession(onnx_file)
    with np.load("test.npz") as f:
        audio_signal = f["audio_signal"]
        audio_signal_length = f["audio_signal_length"]
        vocabulary = f["vocabulary"]
    input_name = sess.get_inputs()[0].name
    output_name = sess.get_outputs()[0].name
    for i in range(audio_signal.shape[0]):
        l = audio_signal_length[i]
        x = audio_signal[i:i + 1, :l]
        logprobs, = sess.run(
            [output_name],
            {
                input_name: x
            }
        )
        predicts = logprobs.argmax(axis=2)
        transcription = ''.join([
            vocabulary[x] if x < len(vocabulary) else '_'
            for x in predicts[0, :]])
        # print(f"Timing: {transcription}")
        transcription = re.sub(r"(.)\1+", r"\1", transcription)
        transcription = transcription.replace("_", "")
        print(f"Recognized as: {transcription}")


# test_nemo()
from glob import glob
audio_files = sorted(glob("/home/kaiida/Data/LibriSpeech/train-clean-100/322/124146/*.flac"))
audio_files = ["test.wav"]
# preprocess(audio_files[:1])
preprocess2(audio_files[:1])
# infer()
import numpy as np
with open("test.bin", "rb") as f:
    x = np.frombuffer(f.read(), dtype=np.float32)
with open("test.bin2", "rb") as f:
    y = np.frombuffer(f.read(), dtype=np.float32)
print(x)
print(y[:x.shape[0]])
