using Microsoft.Extensions.Configuration;
using System;
using System.IO;

namespace NeMoOnnxSharp
{
    internal class Program
    {
        static void Main(string[] args)
        {
            IConfiguration config = new ConfigurationBuilder()
                .AddJsonFile("appsettings.json")
                .AddEnvironmentVariables()
                .Build();
            string appDirPath = AppDomain.CurrentDomain.BaseDirectory;
            string cacheDirectoryPath = Path.Combine(appDirPath, "Cache");
            ModelDownloader downloader = new ModelDownloader(null, cacheDirectoryPath);
            string modelPath = downloader.MayDownload();

            var settings = config.GetRequiredSection("Settings").Get<Settings>();
            if (settings.Model == "QuartzNet15x5Base-En")
            {
                string inputDirPath = Path.Combine(appDirPath, "..", "..", "..", "..", "test_data");
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
                    var waveform = WaveFile.ReadWav(waveFile, 16000, true);
                    string predictText = recognizer.Recognize(waveform);
                    Console.WriteLine("{0}|{1}|{2}", name, targetText, predictText);
                }
            }
        }
    }
}