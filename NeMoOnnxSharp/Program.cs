// Copyright (c) Katsuya Iida.  All Rights Reserved.
// See LICENSE in the project root for license information.

using Microsoft.Extensions.Configuration;
using System;
using System.IO;
using System.Net.Http;
using System.Text;
using System.Threading.Tasks;
using System.Runtime.InteropServices;
using System.Net.Sockets;
using System.Collections.Generic;
using System.Diagnostics;

namespace NeMoOnnxSharp
{
    internal static class Program
    {
        private const string AppName = "NeMoOnnxSharp";

        static async Task Main(string[] args)
        {
            IConfiguration config = new ConfigurationBuilder()
                .AddJsonFile("appsettings.json")
                .AddEnvironmentVariables()
                .Build();
            var settings = config.GetRequiredSection("Settings").Get<Settings>();
            string basePath = AppDomain.CurrentDomain.BaseDirectory;
            string appDirPath = Path.Combine(
                Environment.GetFolderPath(
                    Environment.SpecialFolder.LocalApplicationData,
                    Environment.SpecialFolderOption.DoNotVerify),
                AppName);
            string cacheDirectoryPath = Path.Combine(appDirPath, "Cache");
            var bundle = ModelBundle.GetBundle(settings.Model);
            Console.WriteLine("{0}", bundle.ModelUrl);
            using var httpClient = new HttpClient();
            var downloader = new ModelDownloader(httpClient, cacheDirectoryPath);
            string fileName = GetFileNameFromUrl(bundle.ModelUrl);
            string modelPath = await downloader.MayDownloadAsync(fileName, bundle.ModelUrl, bundle.Hash);

            if (settings.Model == "QuartzNet15x5Base-En")
            {
                string inputDirPath = Path.Combine(basePath, "..", "..", "..", "..", "test_data");
                string inputPath = Path.Combine(inputDirPath, "transcript.txt");

                using var recognizer = new SpeechRecognizer(modelPath);
                using var reader = File.OpenText(inputPath);
                string line;
                while ((line = reader.ReadLine()) != null)
                {
                    string[] parts = line.Split("|");
                    string name = parts[0];
                    string targetText = parts[1];
                    string waveFile = Path.Combine(inputDirPath, name);
                    var waveform = WaveFile.ReadWAV(waveFile, 16000);
                    string predictText = recognizer.Recognize(waveform);
                    Console.WriteLine("{0}|{1}|{2}", name, targetText, predictText);
                }
            }
            else if (settings.Model == "vad_marblenet")
            {
                if (settings.Task == "socketaudio")
                {
                    RunSocketAudio(modelPath);
                    return;
                }
                else if (settings.Task == "streamaudio")
                {
                    RunFileStreamAudio(basePath, modelPath);
                    return;
                }

                string inputDirPath = Path.Combine(basePath, "..", "..", "..", "..", "test_data");
                string inputPath = Path.Combine(inputDirPath, "transcript.txt");

                var processor = new MFCCAudioProcessor(
                    sampleRate: 16000,
                    window: WindowFunction.Hann,
                    windowLength: 400,
                    hopLength: 160,
                    fftLength: 512,
                    preNormalize: 0.0,
                    preemph: 0.0,
                    center: false,
                    nMelBands: 64,
                    nMFCC: 64,
                    melMinHz: 0.0,
                    melMaxHz: 0.0,
                    htk: true,
                    melNormalize: MelNorm.None,
                    logOffset: 1e-6,
                    postNormalize: false);
                using var vad = new FrameVAD(modelPath);
                using var reader = File.OpenText(inputPath);
                string line;
                while ((line = reader.ReadLine()) != null)
                {
                    string[] parts = line.Split("|");
                    string name = parts[0];
                    string targetText = parts[1];
                    string waveFile = Path.Combine(inputDirPath, name);
                    var waveform = WaveFile.ReadWAV(waveFile, 16000);
                    var rng = new Random();
                    // for (int i = 0; i < waveform.Length; i++)
                    // {
                    //     waveform[i] = (short)((rng.Next() & 65535) - 32768);
                    // }
                    var predictText = vad.Transcribe(waveform);
                    Console.WriteLine("{0}|{1}|{2}", name, targetText, predictText);
                }
            }
            else
            {
                throw new InvalidDataException();
            }
        }

        private static void RunSocketAudio(string modelPath)
        {
            using var vad = new FrameVAD(modelPath);
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

        private static void RunFileStreamAudio(string basePath, string modelPath)
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
                fMax: 0.0,
                logMels: true,
                melScale: MelScale.HTK,
                melNorm: MelNorm.None);
            var buffer = new AudioFeatureBuffer<short, float>(
                transform,
                hopLength: 160);
            using var vad = new FrameVAD(modelPath);
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
            double recordStartThreshold = 0.5;
            double recordEndThreshold = 0.5;
            bool recording = false;
            int recordStartPad = 32 + 5; // # 50ms and 320ms delay
            int recordEndPad = 5; // # 50ms
            int repeatCount = 0;
            int recordedAudioIndex = 0;
            var recordedAudio = new List<short>();
            int recordedIndex = 0;
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
                        double score = vad.PredictStep(x);
                        scoreSum += score - scores[scoresIndex];
                        scores[scoresIndex] = score;
                        scoresIndex++;
                        if (scoresIndex >= scoresLength) scoresIndex = 0;
                        score = scoreSum / scoresLength;
                        char recordingChar = ' ';
                        recordedAudioIndex += buffer.HopLength;
                        if (!recording)
                        {
                            if (score >= recordStartThreshold)
                            {
                                recording = true;
                                recordingChar = '[';
                                repeatCount = 0;
                                if (recordedAudioIndex - recordStartPad * 160 > 0)
                                {
                                    recordedAudio.RemoveRange(0, recordedAudioIndex - recordStartPad * 160);
                                    recordedAudioIndex = recordStartPad * 160;
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
            string line;
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

        private static float[] ReadBinaryBuffer(string path)
        {
            using var stream = File.Open(path, FileMode.Open);
            using var reader = new BinaryReader(stream, Encoding.UTF8, false);
            int m = reader.ReadInt32();
            int n = reader.ReadInt32();
            var bytes = reader.ReadBytes(m * n * 4);
            return MemoryMarshal.Cast<byte, float>(bytes).ToArray();
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