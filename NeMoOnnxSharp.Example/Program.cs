// Copyright (c) Katsuya Iida.  All Rights Reserved.
// See LICENSE in the project root for license information.

using System;
using System.IO;
using System.Text;
using System.Threading.Tasks;
using System.Runtime.InteropServices;
using System.Net.Sockets;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;

namespace NeMoOnnxSharp.Example
{
    internal static class Program
    {
        private const string AppName = "NeMoOnnxSharp";

        static async Task Main(string[] args)
        {
            string basePath = AppDomain.CurrentDomain.BaseDirectory;
            string task = args.Length > 0 ? args[0] : "transcribe";

            if (task == "transcribe")
            {
                await Transcribe();
            }
            else if (task == "vad")
            {
                await FramePredict(false);
            }
            else if (task == "mbn")
            {
                await FramePredict(true);
            }
            else if (task == "speak")
            {
                await Speak();
            }
            else if (task == "socketaudio")
            {
                string modelPath = await DownloadModelAsync("vad_marblenet");
                RunSocketAudio(modelPath);
                return;
            }
            else if (task == "streamaudio")
            {
                var modelPaths = await DownloadModelsAsync(new string?[]
                {
                    "vad_marblenet", "stt_en_quartznet15x5"
                });
                RunFileStreamAudio(basePath, modelPaths);
                return;
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
            string strText =
                "K|AA1|N|SH|AH0|S| |o|f| |i|t|s| |S|P|IH1|R|IH0|CH|UW2|AH0|"
                + "L| |a|n|d| |M|AO1|R|AH0|L| |h|e|r|i|t|a|g|e|,| |t|h|e| |Y|"
                + "UW1|N|Y|AH0|N| |i|s| |F|AW1|N|D|IH0|D| |o|n| |t|h|e| |IH2|"
                + "N|D|IH0|V|IH1|S|IH0|B|AH0|L|,| |Y|UW2|N|AH0|V|ER1|S|AH0|L|"
                + " |V|AE1|L|Y|UW0|Z| |o|f| |h|u|m|a|n| |D|IH1|G|N|AH0|T|IY0|"
                + ",| |F|R|IY1|D|AH0|M|,| |IH0|K|W|AA1|L|AH0|T|IY0| |a|n|d| |"
                + "S|AA2|L|AH0|D|EH1|R|AH0|T|IY0";
            string appDirPath = AppDomain.CurrentDomain.BaseDirectory;
            string specGenModelPath = await DownloadModelAsync("tts_en_fastpitch");
            string vocoderModelPath = await DownloadModelAsync("tts_en_hifigan");
            var specGen = new SpectrogramGenerator(specGenModelPath);
            var vocoder = new Vocoder(vocoderModelPath);
            var parsed = specGen.Parse(strText);
            var spec = specGen.GenerateSpectrogram(parsed, pace: 1.0);
            var audio = vocoder.ConvertSpectrogramToAudio(spec);
            string inputDirPath = Path.Combine(appDirPath, "..", "..", "..", "..", "test_data");
            string waveFile = Path.Combine(inputDirPath, "FastPitch_HiFiGAN_demo.wav");
            WaveFile.WriteWAV(waveFile, audio, 22050);
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

        private static void RunSocketAudio(string modelPath)
        {
            using var vad = new EncDecClassificationModel(modelPath);
            using Socket socket = new Socket(SocketType.Stream, ProtocolType.Tcp);
            socket.Connect("127.0.0.1", 17843);
            Console.WriteLine("Connected");
            byte[] responseBytes = new byte[1024];
            var audioSignal = new List<short>();
            while (true)
            {
                int bytesReceived = socket.Receive(responseBytes);
                if (bytesReceived == 0) break;
                if (bytesReceived % 2 != 0)
                {
                    // TODO
                    throw new InvalidDataException();
                }
                audioSignal.AddRange(MemoryMarshal.Cast<byte, short>(responseBytes.AsSpan(0, bytesReceived)).ToArray());
                if (audioSignal.Count > 16000)
                {
                    string text = vad.Transcribe(audioSignal.ToArray());
                    Console.WriteLine("text: {0}", text);
                    audioSignal.Clear();
                }
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

        private static void RunFileStreamAudio(string basePath, string[] modelPaths)
        {
            var stream = GetAllAudioStream(basePath);
            int sampleRate = 16000;
            var transform = new MFCC(
                sampleRate: sampleRate,
                window: WindowFunction.Hann,
                winLength: 400,
                nFFT: 512,
                nMels: 64,
                nMFCC: 64,
                fMin: 0.0,
                fMax: null,
                logMels: true,
                melScale: MelScale.HTK,
                melNorm: MelNorm.None);
            var buffer = new AudioFeatureBuffer<short, float>(
                transform,
                hopLength: 160);
            using var recognizer = new EncDecCTCModel(modelPaths[1]);
            using var vad = new EncDecClassificationModel(modelPaths[0]);
            byte[] responseBytes = new byte[1024];
            int count = 0;
            int vadWinLength = (int)(sampleRate * 0.31 / buffer.HopLength * buffer.NumOutputChannels);
            int vadHopLength = (int)(sampleRate * 0.01 / buffer.HopLength * buffer.NumOutputChannels);
            int scoresLength = vadWinLength / vadHopLength;
            int scoresIndex = 0;
            var sw = new Stopwatch();
            sw.Reset();
            sw.Start();
            var scores = new double[scoresLength];
            double scoreSum = 0.0;
            string displayChars = ".-=*#";
            double recordStartThreshold = 0.75;
            double recordEndThreshold = 0.25;
            bool recording = false;
            int recordStartShift = 100 * 16; // # 100ms delay
            int recordEndPad = 5; // # 50ms
            int repeatCount = 0;
            int recordedAudioIndex = 0;
            var recordedAudio = new List<short>();
            int recordedIndex = 0;
            using var scoreStream = File.OpenWrite("score.dat");
            while (true)
            {
                int bytesReceived = stream.Read(responseBytes);
                if (bytesReceived == 0) break;
                if (bytesReceived % 2 != 0)
                {
                    // TODO
                    throw new InvalidDataException();
                }

                var audioSignal = MemoryMarshal.Cast<byte, short>(responseBytes.AsSpan(0, bytesReceived));
                recordedAudio.AddRange(audioSignal.ToArray());
                for (int offset = 0; offset < audioSignal.Length;)
                {
                    int written = buffer.Write(audioSignal.Slice(offset, audioSignal.Length - offset));
                    offset += written;
                    while (buffer.OutputCount >= vadWinLength)
                    {
                        var x = buffer.OutputBuffer.AsSpan(0, vadWinLength);
                        var logits = vad.Predict(x);
                        double score = 1.0 / (1.0 + Math.Exp(logits[0] - logits[1]));
                        scoreSum += score - scores[scoresIndex];
                        scores[scoresIndex] = score;
                        scoresIndex++;
                        if (scoresIndex >= scoresLength) scoresIndex = 0;
                        score = scoreSum / scoresLength;
                        scoreStream.Write(new byte[1] { (byte)(score * 255) });
                        char recordingChar = ' ';
                        recordedAudioIndex += buffer.HopLength;
                        if (!recording)
                        {
                            if (score >= recordStartThreshold)
                            {
                                recording = true;
                                recordingChar = '[';
                                repeatCount = 0;
                                if (recordedAudioIndex + recordStartShift > 0)
                                {
                                    recordedAudio.RemoveRange(0, recordedAudioIndex + recordStartShift);
                                    recordedAudioIndex = -recordStartShift;
                                }
                            }
                        }
                        else
                        {
                            repeatCount = score < recordEndThreshold ? repeatCount + 1 : 0;
                            if (repeatCount > recordEndPad)
                            {
                                recording = false;
                                recordingChar = ']';
                                WaveFile.WriteWAV(
                                    string.Format("recorded-{0:0000}.wav", recordedIndex),
                                    recordedAudio.ToArray(),
                                    16000);
                                recordedIndex++;
                                string text = recognizer.Transcribe(recordedAudio.ToArray());
                                Console.WriteLine();
                                Console.WriteLine("text: {0}", text);
                                //recordedAudio.Clear();
                                //recordedAudioIndex = 0;
                            }
                        }

                        if (recordingChar != ' ')
                        {
                            Console.Write(recordingChar);
                        }
                        else
                        {
                            Console.Write(displayChars[(int)(score * displayChars.Length)]);
                        }
                        ++count;
                        if (count % 50 == 0)
                        {
                            count = 0;
                            Console.WriteLine();
                        }
                        buffer.ConsumeOutput(vadHopLength);
                    }
                }
            }
            sw.Stop();
            Console.WriteLine();
            double audioTime = (double)(stream.Position / 2) / sampleRate;
            double clockTime = sw.ElapsedMilliseconds / 1000.0;
            Console.WriteLine("{0} sec audio processed in {1} sec", audioTime, clockTime);
        }

        private static MemoryStream GetAllAudioStream(string basePath)
        {
            string inputDirPath = Path.Combine(basePath, "..", "..", "..", "..", "test_data");
            string inputPath = Path.Combine(inputDirPath, "transcript.txt");
            using var reader = File.OpenText(inputPath);
            string? line;
            var stream = new MemoryStream();
            stream.Write(new byte[32000]);
            while ((line = reader.ReadLine()) != null)
            {
                string[] parts = line.Split("|");
                string name = parts[0];
                string waveFile = Path.Combine(inputDirPath, name);
                var waveform = WaveFile.ReadWAV(waveFile, 16000);
                var bytes = MemoryMarshal.Cast<short, byte>(waveform);
                stream.Write(bytes);
                stream.Write(new byte[32000]);
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