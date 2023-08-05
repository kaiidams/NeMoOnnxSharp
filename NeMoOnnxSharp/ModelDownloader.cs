// Copyright (c) Katsuya Iida.  All Rights Reserved.
// See LICENSE in the project root for license information.

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
        private HttpClient _httpClient;
        private string _cacheDirectoryPath;

        public ModelDownloader(HttpClient httpClient, string cacheDirectoryPath)
        {
            _httpClient = httpClient;
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
                string checksum = GetFileChecksum(cacheFilePath);
                if (string.Compare(checksum, expectedChecksum, true) == 0)
                {
                    return true;
                }
                File.Delete(cacheFilePath);
            }
            return false;
        }

        private void ShowProgress(long progress, long? total)
        {
            if (total.HasValue)
            {
                Console.Write("\rDownloading... [{0}/{1} bytes]", progress, total);
            }
            else
            {
                Console.Write("\rDownloading... [{0} bytes]", progress);
            }
        }

        public async Task<string> MayDownloadAsync(string fileName, string url, string sha256)
        {
            Directory.CreateDirectory(_cacheDirectoryPath);

            string cacheFilePath = Path.Combine(_cacheDirectoryPath, fileName);
            if (CheckCacheFile(cacheFilePath, sha256))
            {
                Console.WriteLine("Using cached `{0}'.", fileName);
            }
            else
            {
                await Download(url, cacheFilePath);
                if (!CheckCacheFile(cacheFilePath, sha256))
                {
                    File.Delete(cacheFilePath);
                    throw new InvalidDataException();
                }
            }
            return cacheFilePath;
        }

        private async Task Download(string url, string path)
        {
            using (var response = await _httpClient.GetAsync(url))
            {
                if (!response.IsSuccessStatusCode)
                {
                    throw new InvalidDataException();
                }
                long? contentLength = response.Content.Headers.ContentLength;
                using (var reader = await response.Content.ReadAsStreamAsync())
                {
                    using (var writer = File.OpenWrite(path))
                    {
                        var lastDateTime = DateTime.UtcNow;
                        byte[] buffer = new byte[4096];
                        int bytesRead;
                        while ((bytesRead = await reader.ReadAsync(buffer, 0, buffer.Length)) != 0)
                        {
                            await writer.WriteAsync(buffer, 0, bytesRead);
                            var currentDateTime = DateTime.UtcNow;
                            if ((lastDateTime - currentDateTime).Seconds >= 1)
                            {
                                lastDateTime = currentDateTime;
                                ShowProgress(reader.Position, contentLength);
                            }
                        }
                    }
                }
            }
            Console.WriteLine();
        }
    }
}
