using System;
using System.IO;
using System.Runtime.InteropServices;

namespace NeMoOnnxSharp.Example
{
    internal class Program
    {
        static void Main(string[] args)
        {
            string appDirPath = AppDomain.CurrentDomain.BaseDirectory;
            string modelPath = Path.Combine(appDirPath, "QuartzNet15x5Base-En.onnx");
            string inputDirPath = Path.Combine(appDirPath, "..", "..", "..", "..", "test_data");
            string inputPath = Path.Combine(inputDirPath, "transcript.txt");
            int sampleRate = 16000;

            using var recognizer = new EncDecCTCModel(modelPath);
            using var reader = File.OpenText(inputPath);
            string? line;
            while ((line = reader.ReadLine()) != null)
            {
                string[] parts = line.Split("|");
                string name = parts[0];
                string targetText = parts[1];
                string waveFile = Path.Combine(inputDirPath, name);
                var audioSignal = WaveFile.ReadWAV(waveFile, sampleRate);
                string predictText = recognizer.Transcribe(audioSignal);
                Console.WriteLine("{0}|{1}|{2}", name, targetText, predictText);
            }
        }
    }
}