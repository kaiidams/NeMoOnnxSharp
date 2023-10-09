// Copyright (c) Katsuya Iida.  All Rights Reserved.
// See LICENSE in the project root for license information.

using System;
using System.IO;
using System.Text;
using System.Threading.Tasks;
using System.Collections.Generic;
using System.Runtime.InteropServices;

namespace NeMoOnnxSharp.Example
{
    internal static class Program
    {
        static async Task Main(string[] args)
        {
            string task = args.Length == 0 ? "streamaudio" : args[0];

            if (task == "transcribe")
            {
                await Transcribe();
            }
            else if (task == "speak")
            {
                await Speak();
            }
            else if (task == "vad")
            {
                await FramePredict(false);
            }
            else if (task == "mbn")
            {
                await FramePredict(true);
            }
            else if (task == "streamaudio")
            {
                await StreamAudio();
            }
            else
            {
                throw new InvalidDataException(task);
            }
        }

        static async Task Transcribe()
        {
            string appDirPath = AppDomain.CurrentDomain.BaseDirectory;
            string modelPath = await DownloadModelAsync("stt_en_quartznet15x5");
            string inputDirPath = Path.Combine(appDirPath, "..", "..", "..", "..", "test_data");
            string inputPath = Path.Combine(inputDirPath, "transcript.txt");

            using var model = new EncDecCTCModel(modelPath);
            using var reader = File.OpenText(inputPath);
            string? line;
            while ((line = reader.ReadLine()) != null)
            {
                string[] parts = line.Split("|");
                string name = parts[0];
                string targetText = parts[1];
                string waveFile = Path.Combine(inputDirPath, name);
                var audioSignal = WaveFile.ReadWAV(waveFile, 16000);
                string predictText = model.Transcribe(audioSignal);
                Console.WriteLine("{0}|{1}|{2}", name, targetText, predictText);
            }
        }

        static async Task Speak()
        {
            string appDirPath = AppDomain.CurrentDomain.BaseDirectory;
            string phonemeDict = await DownloadModelAsync("cmudict-0.7b_nv22.10");
            string heteronyms = await DownloadModelAsync("heteronyms-052722");
            string specGenModelPath = await DownloadModelAsync("tts_en_fastpitch");
            string vocoderModelPath = await DownloadModelAsync("tts_en_hifigan");
            string inputDirPath = Path.Combine(appDirPath, "..", "..", "..", "..", "test_data");
            string inputPath = Path.Combine(inputDirPath, "transcript.txt");

            var specGen = new SpectrogramGenerator(specGenModelPath, phonemeDict, heteronyms);
            var vocoder = new Vocoder(vocoderModelPath);
            using var reader = File.OpenText(inputPath);
            string? line;
            while ((line = reader.ReadLine()) != null)
            {
                string[] parts = line.Split("|");
                string name = "generated-" + parts[0];
                string targetText = parts[1];
                Console.WriteLine("Generating {0}...", name);
                string waveFile = Path.Combine(inputDirPath, name);
                var parsed = specGen.Parse(targetText);
                var spec = specGen.GenerateSpectrogram(parsed, pace: 1.0);
                var audio = vocoder.ConvertSpectrogramToAudio(spec);
                WaveFile.WriteWAV(waveFile, audio, vocoder.SampleRate);
            }
        }

        static async Task FramePredict(bool mbn)
        {
            string appDirPath = AppDomain.CurrentDomain.BaseDirectory;
            string modelPath = await DownloadModelAsync(
                mbn ? "commandrecognition_en_matchboxnet3x1x64_v2" : "vad_marblenet");
            string inputDirPath = Path.Combine(appDirPath, "..", "..", "..", "..", "test_data");
            string waveFile = Path.Combine(inputDirPath, "SpeechCommands_demo.wav");
            using var model = new EncDecClassificationModel(modelPath, mbn);
            var audioSignal = WaveFile.ReadWAV(waveFile, 16000);
            double windowStride = 0.10;
            double windowSize = mbn ? 1.28 : 0.15;
            int sampleRate = 16000;
            int nWindowStride = (int)(windowStride * sampleRate);
            int nWindowSize = (int)(windowSize * sampleRate);
            var buffer = new short[audioSignal.Length + nWindowSize];
            audioSignal.CopyTo(buffer.AsSpan(nWindowSize / 2));
            for (int offset = 0; offset + nWindowSize <= buffer.Length; offset += nWindowStride)
            {
                string predictedText = model.Transcribe(buffer.AsSpan(offset, nWindowSize));
                double t = (double)offset / sampleRate;
                Console.WriteLine("time: {0:0.000}, predicted: {1}", t, predictedText);
            }
        }

        static async Task StreamAudio()
        {
            string appDirPath = AppDomain.CurrentDomain.BaseDirectory;
            string vadModelPath = await DownloadModelAsync("vad_marblenet");
            string asrModelPath = await DownloadModelAsync("stt_en_quartznet15x5");
            string inputDirPath = Path.Combine(appDirPath, "..", "..", "..", "..", "test_data");
            string inputPath = Path.Combine(inputDirPath, "transcript.txt");

            using var recognizer = new SpeechRecognizer(vadModelPath, asrModelPath);
            using var ostream = new FileStream(Path.Combine(inputDirPath, "result.txt"), FileMode.Create);
            using var writer = new StreamWriter(ostream);
            int index = 0;
            recognizer.SpeechStartDetected += (s, e) =>
            {
                double t = (double)e.Offset / recognizer.SampleRate;
                Console.WriteLine("SpeechStartDetected {0}", t);
            };
            recognizer.SpeechEndDetected += (s, e) =>
            {
                double t = (double)e.Offset / recognizer.SampleRate;
                Console.WriteLine("SpeechEndDetected {0}", t);
            };
            recognizer.Recognized += (s, e) =>
            {
                double t = (double)e.Offset / recognizer.SampleRate;
                Console.WriteLine("Recognized {0} {1} {2}", t, e.Audio?.Length, e.Text);
                string fileName = string.Format("recognized-{0}.wav", index);
                writer.WriteLine("{0}|{1}|{2}", fileName, e.Audio?.Length, e.Text);
                if (e.Audio != null)
                {
                    WaveFile.WriteWAV(Path.Combine(inputDirPath, fileName), e.Audio, recognizer.SampleRate);
                }
                index += 1;
            };
            var stream = GetAllAudioStream(inputDirPath);
            var buffer = new byte[1024];
            while (true)
            {
                int bytesRead = stream.Read(buffer);
                if (bytesRead == 0)
                {
                    break;
                }
                recognizer.Transcribe(buffer.AsSpan(0, bytesRead));
            }
        }

        private static MemoryStream GetAllAudioStream(
            string inputDirPath,
            int sampleRate = 16000,
            double gapSeconds = 1.0)
        {
            string inputPath = Path.Combine(inputDirPath, "transcript.txt");
            using var reader = File.OpenText(inputPath);
            string? line;
            var stream = new MemoryStream();
            var waveform = new short[(int)(sampleRate * gapSeconds)];
            var bytes = MemoryMarshal.Cast<short, byte>(waveform);
            stream.Write(bytes);
            while ((line = reader.ReadLine()) != null)
            {
                string[] parts = line.Split("|");
                string name = parts[0];
                string waveFile = Path.Combine(inputDirPath, name);
                waveform = WaveFile.ReadWAV(waveFile, sampleRate);
                bytes = MemoryMarshal.Cast<short, byte>(waveform);
                stream.Write(bytes);
                waveform = new short[(int)(sampleRate * gapSeconds)];
                bytes = MemoryMarshal.Cast<short, byte>(waveform);
                stream.Write(bytes);
            }
            stream.Seek(0, SeekOrigin.Begin);
            return stream;
        }

        private static async Task<string> DownloadModelAsync(string model)
        {
            using var downloader = new ModelDownloader();
            var info = PretrainedModelInfo.Get(model);
            string fileName = GetFileNameFromUrl(info.Location);
            Console.WriteLine("Model: {0}", model);
            await downloader.MayDownloadAsync(fileName, info.Location, info.Hash);
            return fileName;
        }

        private static string GetFileNameFromUrl(string url)
        {
            int slashIndex = url.LastIndexOf("/");
            if (slashIndex == -1)
            {
                throw new ArgumentException();
            }
            string fileName = url.Substring(slashIndex + 1);
            if (string.IsNullOrWhiteSpace(fileName))
            {
                throw new ArgumentException();
            }
            return fileName;
        }
    }
}