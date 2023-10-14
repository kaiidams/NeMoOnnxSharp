// Copyright (c) Katsuya Iida.  All Rights Reserved.
// See LICENSE in the project root for license information.

using System;
using System.IO;
using System.Text;
using System.Threading.Tasks;
using System.Runtime.InteropServices;
using System.Net.Sockets;
using System.Collections.Generic;
using NeMoOnnxSharp.Models;

namespace NeMoOnnxSharp.Example
{
    internal static class Program
    {
        private const string AppName = "NeMoOnnxSharp";

        static async Task Main(string[] args)
        {
            string task = args.Length > 0 ? args[0] : "speak_german";

            if (task == "transcribe")
            {
                await TranscribeAsync();
            }
            else if (task == "speak")
            {
                await SpeakAsync();
            }
            else if (task == "speak_german")
            {
                await SpeakGermanAsync();
            }
            else if (task == "vad")
            {
                await FramePredictAsync(false);
            }
            else if (task == "mbn")
            {
                await FramePredictAsync(true);
            }
            else if (task == "streamaudio")
            {
                await StreamAudioAsync();
            }
            else if (task == "socketaudio")
            {
                await SocketAudioAsync();
            }
            else
            {
                throw new InvalidDataException(task);
            }
        }

        /// <summary>
        /// Using EncDecCTCModel with QuartzNet to transcribe texts from speech audio.
        /// </summary>
        /// <returns></returns>
        static async Task TranscribeAsync()
        {
            string appDirPath = AppDomain.CurrentDomain.BaseDirectory;
            string modelPath = await DownloadModelAsync("stt_de_quartznet15x5");
            string inputDirPath = Path.Combine(appDirPath, "..", "..", "..", "..", "test_data");
            string inputPath = Path.Combine(inputDirPath, "transcript.txt");
            var config = new EncDecCTCConfig
            {
                modelPath = modelPath,
                vocabulary = EncDecCTCConfig.GermanVocabulary
            };
            using var model = new EncDecCTCModel(config);
            using var reader = File.OpenText(inputPath);
            string? line;
            while ((line = reader.ReadLine()) != null)
            {
                string[] parts = line.Split("|");
                string name = parts[0];
                string targetText = parts[1];
                string waveFile = Path.Combine(inputDirPath, name);
                var audioSignal = WaveFile.ReadWAV(waveFile, model.SampleRate);
                string predictText = model.Transcribe(audioSignal);
                Console.WriteLine("{0}|{1}|{2}", name, targetText, predictText);
            }
        }

        /// <summary>
        /// Use high level API SpeechSynthesizer with FastSpeech and HifiGAN
        /// </summary>
        /// <returns></returns>
        /// <exception cref="InvalidDataException"></exception>
        static async Task SpeakAsync()
        {
            string appDirPath = AppDomain.CurrentDomain.BaseDirectory;
            string phonemeDict = await DownloadModelAsync("cmudict-0.7b_nv22.10");
            string heteronyms = await DownloadModelAsync("heteronyms-052722");
            string specGenModelPath = await DownloadModelAsync("tts_en_fastpitch");
            string vocoderModelPath = await DownloadModelAsync("tts_en_hifigan");
            string inputDirPath = Path.Combine(appDirPath, "..", "..", "..", "..", "test_data");
            string inputPath = Path.Combine(inputDirPath, "transcript.txt");
            var config = new SpeechConfig
            {
                specGen = new SpectrogramGeneratorConfig
                {
                    modelPath = specGenModelPath,
                    phonemeDictPath = phonemeDict,
                    heteronymsPath = heteronyms
                },
                vocoder = new VocoderConfig
                {
                    modelPath = vocoderModelPath
                }
            };
            using var reader = File.OpenText(inputPath);
            using var synthesizer = new SpeechSynthesizer(config);
            string? line;
            while ((line = reader.ReadLine()) != null)
            {
                string[] parts = line.Split("|");
                string name = "generated-" + parts[0];
                string targetText = parts[1];
                Console.WriteLine("Generating {0}...", name);
                string waveFile = Path.Combine(inputDirPath, name);
                var result = synthesizer.SpeakText(targetText);
                if (result.AudioData == null) throw new InvalidDataException();
                WaveFile.WriteWAV(waveFile, result.AudioData, result.SampleRate);
            }
        }

        static async Task SpeakGermanAsync()
        {
            string appDirPath = AppDomain.CurrentDomain.BaseDirectory;
            string specGenModelPath = await DownloadModelAsync("tts_de_fastpitch_singleSpeaker_thorstenNeutral_2210");
            string vocoderModelPath = await DownloadModelAsync("tts_de_hifigan_singleSpeaker_thorstenNeutral_2210");
            var config = new SpeechConfig
            {
                specGen = new SpectrogramGeneratorConfig
                {
                    modelPath = specGenModelPath,
                    textTokenizer = "GermanCharsTokenizer"
                },
                vocoder = new VocoderConfig
                {
                    modelPath = vocoderModelPath
                },
            };
            using var synthesizer = new SpeechSynthesizer(config);
            string inputDirPath = Path.Combine(appDirPath, "..", "..", "..", "..", "test_data");
            string name = "generated-samples_thorsten-21.06-emotional_neutral.wav";
            string targetText = "Mist, wieder nichts geschafft.";
            Console.WriteLine("Generating {0}...", name);
            string waveFile = Path.Combine(inputDirPath, name);
            var result = synthesizer.SpeakText(targetText);
            if (result.AudioData == null) throw new InvalidDataException();
            WaveFile.WriteWAV(waveFile, result.AudioData, result.SampleRate);
        }

        /// <summary>
        /// Use EncDecClassficationModel with MarbleNet for VAD or speech classification
        /// </summary>
        /// <param name="mbn"></param>
        /// <returns></returns>
        static async Task FramePredictAsync(bool mbn)
        {
            string appDirPath = AppDomain.CurrentDomain.BaseDirectory;
            string modelPath = await DownloadModelAsync(
                mbn ? "commandrecognition_en_matchboxnet3x1x64_v2" : "vad_marblenet");
            string inputDirPath = Path.Combine(appDirPath, "..", "..", "..", "..", "test_data");
            string waveFile = Path.Combine(inputDirPath, "SpeechCommands_demo.wav");
            var config = new EncDecClassificationConfig
            {
                modelPath = modelPath,
                labels = mbn ? EncDecClassificationConfig.SpeechCommandsLabels : EncDecClassificationConfig.VADLabels,
            };
            using var model = new EncDecClassificationModel(config);
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

        /// <summary>
        /// Use high level API SpeechRecognizer with MarbleNet and QuartzNet
        /// </summary>
        /// <returns></returns>
        private static async Task StreamAudioAsync()
        {
            string appDirPath = AppDomain.CurrentDomain.BaseDirectory;
            string inputDirPath = Path.Combine(appDirPath, "..", "..", "..", "..", "test_data");
            var modelPaths = await DownloadModelsAsync(new string?[]
            {
                    "vad_marblenet", "stt_en_quartznet15x5"
            });
            var config = new SpeechConfig
            {
                vad = new EncDecClassificationConfig
                {
                    modelPath = modelPaths[0],
                    labels = EncDecClassificationConfig.VADLabels
                },
                asr = new EncDecCTCConfig
                {
                    modelPath = modelPaths[1],
                    vocabulary = EncDecCTCConfig.EnglishVocabulary
                }
            };
            using var recognizer = new SpeechRecognizer(config);
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
                recognizer.Write(buffer.AsSpan(0, bytesRead));
            }
        }

        /// <summary>
        /// Use high level API SpeechRecognizer with MarbleNet and QuartzNet
        /// </summary>
        /// <returns></returns>
        private static async Task SocketAudioAsync()
        {
            var modelPaths = await DownloadModelsAsync(new string?[]
            {
                    "vad_marblenet", "stt_en_quartznet15x5"
            });
            var config = new SpeechConfig
            {
                vad = new EncDecClassificationConfig
                {
                    modelPath = modelPaths[0],
                    labels = EncDecClassificationConfig.VADLabels
                },
                asr = new EncDecCTCConfig
                {
                    modelPath = modelPaths[1],
                    vocabulary = EncDecCTCConfig.EnglishVocabulary
                }
            };
            using var recognizer = new SpeechRecognizer(config);
            using Socket socket = new Socket(SocketType.Stream, ProtocolType.Tcp);
            socket.Connect("127.0.0.1", 17843);
            Console.WriteLine("Connected");
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
            };
            var buffer = new byte[1024];
            while (true)
            {
                int bytesRead = socket.Receive(buffer);
                if (bytesRead == 0)
                {
                    break;
                }
                recognizer.Write(buffer.AsSpan(0, bytesRead));
            }
        }

        private static async Task<string> DownloadModelAsync(string? model)
        {
            var modelPaths = await DownloadModelsAsync(new string?[] { model });
            return modelPaths[0];
        }

        private static async Task<string[]> DownloadModelsAsync(string?[] models)
        {
            string appDirPath = Path.Combine(
                Environment.GetFolderPath(
                    Environment.SpecialFolder.LocalApplicationData,
                    Environment.SpecialFolderOption.DoNotVerify),
                AppName);
            string cacheDirectoryPath = Path.Combine(appDirPath, "Cache");
            Directory.CreateDirectory(cacheDirectoryPath);
            using var downloader = new ModelDownloader();
            var modelPaths = new List<string>();
            foreach (string? model in models)
            {
                if (string.IsNullOrEmpty(model))
                {
                    throw new InvalidDataException();
                }
                var info = PretrainedModelInfo.Get(model);
                string fileName = GetFileNameFromUrl(info.Location);
                string filePath = Path.Combine(cacheDirectoryPath, fileName);
                Console.WriteLine("Model: {0}", model);
                await downloader.MayDownloadAsync(filePath, info.Location, info.Hash);
                modelPaths.Add(filePath);
            }
            return modelPaths.ToArray();
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
            var rng = new Random();
            while ((line = reader.ReadLine()) != null)
            {
                string[] parts = line.Split("|");
                string name = parts[0];
                string waveFile = Path.Combine(inputDirPath, name);
                waveform = WaveFile.ReadWAV(waveFile, sampleRate);
                for (int i = 0; i < waveform.Length; i++)
                {
                    //waveform[i] += (short)(rng.NextDouble() * 2000 - 1000);
                }
                bytes = MemoryMarshal.Cast<short, byte>(waveform);
                stream.Write(bytes);
                waveform = new short[(int)(sampleRate * gapSeconds)];
                for (int i = 0; i < waveform.Length; i++)
                {
                    //waveform[i] = (short)(rng.NextDouble() * 2000 - 1000);
                }
                bytes = MemoryMarshal.Cast<short, byte>(waveform);
                stream.Write(bytes);
            }
            stream.Seek(0, SeekOrigin.Begin);
            return stream;
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