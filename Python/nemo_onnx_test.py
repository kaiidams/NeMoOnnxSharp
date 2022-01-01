import unittest
import os
import torch
import soundfile as sf
import librosa
import numpy as np
import re
from typing import List, Text, Tuple
from collections import namedtuple

TestData = namedtuple("TestData", ["name", "file", "text"])


class NeMoOnnxTest(unittest.TestCase):

    def setUp(self):
        self.sample_rate = 16000
        self.model_name = "QuartzNet15x5Base-En"
        self.base_dir = os.path.dirname(__file__)
        self.onnx_file = os.path.join(self.base_dir, "..", "NeMoOnnxSharp", "QuartzNet15x5Base-En.onnx")
        self.config_file = os.path.join(self.base_dir, "quartznet_15x5.yaml")
        self.test_dir = os.path.join(self.base_dir, "..", "test_data")
        self.test_file = os.path.join(self.test_dir, "transcript.txt")
        self.test_data = []
        with open(self.test_file, "rt") as f:
            for line in f:
                name, text = line.rstrip("\r\n").split("|")
                file = os.path.join(self.test_dir, name)
                self.test_data.append(TestData(name, file, text))

    def test_nemo_transcribe(self):
        import nemo.collections.asr as nemo_asr
        quartznet = nemo_asr.models.EncDecCTCModel.from_pretrained(model_name=self.model_name)
        files = [test.file for test in self.test_data]
        texts = quartznet.transcribe(paths2audio_files=files)
        for fname, transcription in zip(files, texts):
            print(f"Audio in {fname} was recognized as: {transcription}")

    def test_nemo_preprocess(self):
        """Do pre-processing using NeMo."""
        import nemo.collections.asr as nemo_asr
        from omegaconf import OmegaConf

        # Instanciate a preprocessor from Hydra config file.
        config = OmegaConf.load(self.config_file)
        preprocessor = nemo_asr.models.EncDecCTCModel.from_config_dict(config.model.preprocessor)
        self.assertIsInstance(preprocessor, nemo_asr.modules.AudioToMelSpectrogramPreprocessor)

        # Make sure that the preprocessor is not in the training mode.
        preprocessor.eval()

        # Read batch-padded audio
        input_signal, input_signal_length = self.read_audio()
        print("input_signal.shape:", input_signal.shape)

        # Call preprocess and get batch-padded mel-spectrogram
        audio_signal, audio_signal_length = preprocessor(
            input_signal=input_signal,
            length=input_signal_length)
        print("audio_signal.shape:", audio_signal.shape)

        # Save the result
        np.savez(
            "preprocessed_nemo.npz",
            audio_signal=audio_signal.float().numpy(),
            audio_signal_length=audio_signal_length.numpy())

    def test_torch_preprocess(self):
        """Do pre-processing using PyTorch."""
        sample_rate = 16000
        n_fft = 512
        preemph = 0.97
        hop_length = 160
        win_length = 320
        nfilt = 64
        lowfreq = 0
        highfreq = sample_rate / 2
        log_zero_guard_value = 2 ** -24
        std_zero_guard_value = 1e-5

        # Read batch-padded audio
        input_signal, input_signal_length = self.read_audio()
        print("input_signal.shape:", input_signal.shape)

        # Hann window (https://en.wikipedia.org/wiki/Hann_function)
        window = torch.hann_window(win_length, periodic=False, dtype=torch.float32)

        # Mel filterbanks
        filterbanks = torch.tensor(
            librosa.filters.mel(sample_rate, n_fft, n_mels=nfilt, fmin=lowfreq, fmax=highfreq), dtype=torch.float
        ).unsqueeze(0)

        x = input_signal

        # Length of mel-spectrogram
        seq_len = input_signal_length // hop_length + 1

        # Pre-emphasize
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
        print("x.shape after STFT:", x.shape)

        # To squared magnitude
        x = x.pow(2).sum(-1)

        # To mel-spectrogram
        x = torch.matmul(filterbanks, x)
        print("x.shape after Filterbanks:", x.shape)

        # To log mel-spectrogram
        x = torch.log(x + log_zero_guard_value)

        # Normalize per feature
        x_mean = torch.zeros((seq_len.shape[0], x.shape[1]), dtype=x.dtype, device=x.device)
        x_std = torch.zeros((seq_len.shape[0], x.shape[1]), dtype=x.dtype, device=x.device)
        for i in range(x.shape[0]):
            x_mean[i, :] = x[i, :, : seq_len[i]].mean(dim=1)
            x_std[i, :] = x[i, :, : seq_len[i]].std(dim=1)

        x_std += std_zero_guard_value
        x = (x - x_mean.unsqueeze(2)) / x_std.unsqueeze(2)

        audio_signal = x
        audio_signal_length = seq_len

        # Save the result
        np.savez(
            "preprocessed_torch.npz",
            audio_signal=audio_signal.float().numpy(),
            audio_signal_length=audio_signal_length.numpy())

    @unittest.skipUnless(os.path.exists("preprocessed_torch.npz"), "requires preprocessing")
    def infer_with_onnx(self):
        import onnxruntime
        from omegaconf import OmegaConf

        # Read vocabulary for post-processing
        config = OmegaConf.load(self.config_file)
        vocabulary = config["model"]["decoder"]["vocabulary"]

        # Load ONNX model
        sess = onnxruntime.InferenceSession(self.onnx_file)

        # Load pre-processed data
        with np.load("preprocessed_torch.npz") as f:
            audio_signal = f["audio_signal"]
            audio_signal_length = f["audio_signal_length"]

        input_name = sess.get_inputs()[0].name
        output_name = sess.get_outputs()[0].name

        for i in range(audio_signal.shape[0]):
            seq_len = audio_signal_length[i]
            x = audio_signal[None, i, :seq_len]

            # Run ONNX model
            logprobs, = sess.run(
                [output_name],
                {
                    input_name: x
                }
            )
            # logprobs: [batch_size, audio_signal_length, vocab_size]

            transcription = self.postprocess(logprobs, vocabulary)

            print(f"Recognized as: {transcription}")

    @unittest.skipUnless(
        os.path.exists("preprocessed_nemo.npz") and os.path.exists("preprocessed_torch.npz"),
        "requires preprocessing")
    def compare_preprocess(self):
        with np.load("preprocessed_nemo.npz") as f:
            audio_signal_nemo = f["audio_signal"]
            audio_signal_length_nemo = f["audio_signal_length"]
        with np.load("preprocessed_torch.npz") as f:
            audio_signal_torch = f["audio_signal"]
            audio_signal_length_torch = f["audio_signal_length"]
        print("Shape of audio_signal from NeMo:", audio_signal_nemo.shape)
        print("Shape of audio_signal from Torch:", audio_signal_torch.shape)
        self.assertEqual(audio_signal_nemo.shape[0], audio_signal_torch.shape[0])
        for i in range(audio_signal_nemo.shape[0]):
            x = audio_signal_nemo[i]
            xlen = audio_signal_length_nemo[i]
            y = audio_signal_torch[i]
            ylen = audio_signal_length_torch[i]
            self.assertEqual(xlen, ylen)
            diff = np.mean((x[:, :xlen] - y[:, :ylen]) ** 2)
            self.assertLess(diff, 1e-5)

    def read_audio(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Read audio files and returns batch-padded audio
        input_signal: [batch_size, audio_len]
        input_signal_length: [batch_size]
        """
        input_signal = [
            self.read_wav(test.file)
            for test in self.test_data
        ]
        input_signal_length = [
            x.shape[0]
            for x in input_signal
        ]

        input_signal = torch.nn.utils.rnn.pad_sequence(input_signal, batch_first=True)
        input_signal_length = torch.tensor(input_signal_length, dtype=torch.long)

        return input_signal, input_signal_length

    def read_wav(self, audio_file: Text) -> torch.Tensor:
        """Read a WAV file and make it 32-bit float 16000Hz mono."""
        x, sample_rate = sf.read(audio_file)
        x = librosa.resample(x, sample_rate, self.sample_rate)
        x = torch.from_numpy(x).float()
        assert x.dim() == 1
        return x

    def postprocess(self, logprobs: torch.Tensor, vocabulary: List[Text]) -> Text:
        # Get the most probable labels
        predicts = logprobs.argmax(axis=2)

        # Decode the label into characters
        transcription = ''.join([
            vocabulary[int(x)] if x < len(vocabulary) else '_'
            for x in predicts[0].astype(int)])

        # Decode CTC prediction
        transcription = re.sub(r"(.)\1+", r"\1", transcription)
        transcription = transcription.replace("_", "")

        return transcription


if __name__ == "__main__":
    unittest.main()
