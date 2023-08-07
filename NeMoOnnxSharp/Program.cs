﻿// Copyright (c) Katsuya Iida.  All Rights Reserved.
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
            var buffer = new AudioFeatureBuffer();
            using var vad = new FrameVAD(modelPath);
            byte[] responseBytes = new byte[1024];
            var audioSignal = new List<short>();
            int c = 0;

            while (true)
            {
                int bytesReceived = stream.Read(responseBytes);
                if (bytesReceived == 0) break;
                if (bytesReceived % 2 != 0)
                {
                    // TODO
                    throw new InvalidDataException();
                }

                var x = MemoryMarshal.Cast<byte, short>(responseBytes.AsSpan(0, bytesReceived)).ToArray();
                for (int offset = 0; offset < x.Length;)
                {
                    int written = buffer.Write(x, offset, x.Length - offset);
                    offset += written;
                    while (buffer.OutputCount >= 16000 + 400)
                    {
                        var y = buffer.OutputCount;
                        Console.Write(".");
                        ++c;
                        if (c % 60 == 0)
                        {
                            c = 0;
                            Console.WriteLine();
                        }
                        buffer.ConsumeOutput(64);
                    }

                }
                
                if (false)
                {
                    audioSignal.AddRange(MemoryMarshal.Cast<byte, short>(responseBytes.AsSpan(0, bytesReceived)).ToArray());
                    if (audioSignal.Count > 16000)
                    {
                        string text = vad.Transcribe(audioSignal.ToArray());
                        Console.WriteLine("text: {0}", text);
                        audioSignal.Clear();
                    }
                }
            }
        }

        private static MemoryStream GetAllAudioStream(string basePath)
        {
            string inputDirPath = Path.Combine(basePath, "..", "..", "..", "..", "test_data");
            string inputPath = Path.Combine(inputDirPath, "transcript.txt");
            using var reader = File.OpenText(inputPath);
            string line;
            var stream = new MemoryStream();
            while ((line = reader.ReadLine()) != null)
            {
                string[] parts = line.Split("|");
                string name = parts[0];
                string waveFile = Path.Combine(inputDirPath, name);
                var waveform = WaveFile.ReadWAV(waveFile, 16000);
                var bytes = MemoryMarshal.Cast<short, byte>(waveform);
                stream.Write(bytes);
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