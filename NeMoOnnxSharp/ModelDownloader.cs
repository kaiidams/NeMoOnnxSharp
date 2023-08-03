using System;
using System.Collections.Generic;
using System.IO;
using System.Net.Http;
using System.Security.Cryptography;
using System.Text;
using System.Threading.Tasks;

namespace NeMoOnnxSharp
{
    public class ModelDownloader
    {
        private readonly HttpClient _client;
        private readonly string _cacheDirectoryPath;

        public ModelDownloader(HttpClient client, string cacheDirectoryPath)
        {
            _client = client;
            _cacheDirectoryPath = cacheDirectoryPath;
        }

        private string GetFileChecksum(string path)
        {
            using SHA256 sha256 = SHA256.Create();
            using var stream = File.OpenRead(path);
            var hashValue = sha256.ComputeHash(stream);
            var sb = new StringBuilder();
            foreach (var value in hashValue)
            {
                sb.Append($"{value:x2}");
            }
            return sb.ToString();
        }

        private bool CheckCacheFile(string cacheFilePath, string expectedChecksum)
        {
            if (File.Exists(cacheFilePath))
            {
                if (GetFileChecksum(cacheFilePath) == expectedChecksum)
                {
                    return true;
                }
                File.Delete(cacheFilePath);
            }
            return false;
        }

        public async Task<string> MayDownloadAsync(string url)
        {
            Directory.CreateDirectory(_cacheDirectoryPath);

            string cacheFilePath = Path.Combine(_cacheDirectoryPath, "QuartzNet15x5Base-En.onnx");
            string expectedChecksum = "ee1b72102fd0c5422d088e80f929dbdee7e889d256a4ce1e412cd49916823695";
            if (CheckCacheFile(cacheFilePath, expectedChecksum))
            {
                Console.WriteLine("Cache hit");
                return cacheFilePath;
            }
            using var response = await _client.GetAsync(url);
            using var inputStream = await response.Content.ReadAsStreamAsync();
            using var outputStream = File.OpenWrite(cacheFilePath);
            await inputStream.CopyToAsync(outputStream);
            return cacheFilePath;
        }
    }
}
