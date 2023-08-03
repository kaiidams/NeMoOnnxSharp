using Microsoft.Extensions.Configuration;
using System;
using System.IO;
using System.Net.Http;
using System.Threading.Tasks;

namespace NeMoOnnxSharp
{
    internal class Program
    {
        private static string AppName = "NeMoOnnxSharp";

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
            string modelPath = await downloader.MayDownloadAsync(bundle.ModelUrl, bundle.Hash);

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
                    var waveform = WaveFile.ReadWav(waveFile, 16000, true);
                    string predictText = recognizer.Recognize(waveform);
                    Console.WriteLine("{0}|{1}|{2}", name, targetText, predictText);
                }
            }
            else
            {
                throw new InvalidDataException();
            }
        }
    }
}