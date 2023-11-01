using NeMoOnnxSharp;
using NeMoOnnxSharp.Models;
using System;
using System.Security.Cryptography;
using System.Text;

namespace NeMoOnnxAndroidApp
{
    public sealed class ModelDownloader : IDisposable
    {
        public class ProgressEventArgs
        {
            public ProgressEventArgs(string fileName, long currentPosition, long? contentLength)
            {
                FileName = fileName;
                CurrentPosition = currentPosition;
                ContentLength = contentLength;
            }

            public string FileName { get; private set; }
            public long CurrentPosition { get; private set; }
            public long? ContentLength { get; private set; }
        }

        private readonly HttpClient _httpClient;
        private readonly string _cacheDirPath;

        private string _language;
        public event EventHandler<ProgressEventArgs>? Progress;

        // Called when the node enters the scene tree for the first time.
        public ModelDownloader(string cacheDirPath)
        {
            _cacheDirPath = cacheDirPath;
            _httpClient = new HttpClient();
            _language = string.Empty;
        }

        /// <summary>
        /// Language of speech models
        /// </summary>
        public string Language
        {
            get { return _language; }
            set
            {
                _language = value;
            }
        }

        /// <summary>
        /// Check if all model files are available locally
        /// </summary>
        public bool ModelFilesAvailable
        {
            get
            {
                foreach (var name in _GetModelList())
                {
                    var info = PretrainedModelInfo.Get(name);
                    string cacheFilePath = _GetCacheFilePathByName(name);
                    if (!_CheckCacheFile(cacheFilePath, info.Hash))
                    {
                        return false;
                    }
                }
                return true;
            }
        }

        public void Dispose()
        {
            _httpClient.Dispose();
        }

        private string _GetFileChecksum(string path)
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

        private bool _CheckCacheFile(string cacheFilePath, string expectedChecksum)
        {
            if (File.Exists(cacheFilePath))
            {
                string checksum = _GetFileChecksum(cacheFilePath);
                if (string.Compare(checksum, expectedChecksum, true) == 0)
                {
                    return true;
                }
                File.Delete(cacheFilePath);
            }
            return false;
        }

        private string _GetCacheFilePathByName(string name)
        {
            var info = PretrainedModelInfo.Get(name);
            var fileName = _GetCacheFileNameFromUrl(info.Location);
            return Path.Combine(_cacheDirPath, fileName);
        }

        private static string _GetCacheFileNameFromUrl(string url)
        {
            int index = url.LastIndexOf('/');
            return url.Substring(index + 1);
        }

        /// <summary>
        /// Download all model files needed for the language.
        /// </summary>
        public async Task MayDownloadAllAsync(
            CancellationToken cancellationToken = default)
        {
            foreach (var name in _GetModelList())
            {
                var info = PretrainedModelInfo.Get(name);
                string modelPath = _GetCacheFilePathByName(name);
                await _MayDownloadAsync(modelPath, info.Location, info.Hash, cancellationToken);
            }
        }

        private async Task _MayDownloadAsync(
            string cacheFilePath, string url, string sha256,
            CancellationToken cancellationToken = default)
        {
            if (!_CheckCacheFile(cacheFilePath, sha256))
            {
                await _DownloadAsync(url, cacheFilePath, cancellationToken);
                if (!_CheckCacheFile(cacheFilePath, sha256))
                {
                    File.Delete(cacheFilePath);
                    throw new InvalidDataException();
                }
            }
        }

        private async Task _DownloadAsync(
            string url, string path,
            CancellationToken cancellationToken = default)
        {
            var cacheFileName = _GetCacheFileNameFromUrl(url);
            using (var response = await _httpClient.GetAsync(
                url, HttpCompletionOption.ResponseHeadersRead, cancellationToken))
            {
                response.EnsureSuccessStatusCode();
                long currentPosition = 0;
                long? contentLength = response.Content.Headers.ContentLength;
                using (var reader = await response.Content.ReadAsStreamAsync(cancellationToken))
                {
                    using (var writer = File.OpenWrite(path))
                    {
                        byte[] buffer = new byte[4096];
                        int bytesRead;
                        while ((bytesRead = await reader.ReadAsync(
                            buffer, 0, buffer.Length, cancellationToken)) != 0)
                        {
                            await writer.WriteAsync(buffer, 0, bytesRead, cancellationToken);
                            currentPosition += bytesRead;
                            if (Progress != null)
                            {
                                Progress(this, new ProgressEventArgs(
                                    cacheFileName,
                                    currentPosition,
                                    contentLength));
                            }
                        }
                    }
                }
            }
        }

        private string[] _GetModelList()
        {
            if (_language == "English")
            {
                return new string[]
                {
                    "vad_marblenet",
                    "stt_en_quartznet15x5",
                    "cmudict-0.7b_nv22.10",
                    "heteronyms-052722",
                    "tts_en_fastpitch",
                    "tts_en_hifigan",
                };
            }
            else if (_language == "German")
            {
                return new string[]
                {
                    "vad_marblenet",
                    "stt_de_quartznet15x5",
                    "tts_de_fastpitch_singleSpeaker_thorstenNeutral_2210",
                    "tts_de_hifigan_singleSpeaker_thorstenNeutral_2210",
                };
            }
            else
            {
                throw new ArgumentException();
            }
        }

        public SpeechConfig BuildSpeechConfig()
        {
            SpeechConfig config;
            if (_language == "English")
            {
                config = new SpeechConfig
                {
                    vad = new EncDecClassificationConfig
                    {
                        modelPath = _GetCacheFilePathByName("vad_marblenet"),
                        labels = EncDecClassificationConfig.VADLabels
                    },
                    asr = new EncDecCTCConfig
                    {
                        modelPath = _GetCacheFilePathByName("stt_en_quartznet15x5"),
                        vocabulary = EncDecCTCConfig.EnglishVocabulary
                    },
                    specGen = new SpectrogramGeneratorConfig
                    {
                        modelPath = _GetCacheFilePathByName("tts_en_fastpitch"),
                        phonemeDictPath = _GetCacheFilePathByName("cmudict-0.7b_nv22.10"),
                        heteronymsPath = _GetCacheFilePathByName("heteronyms-052722"),
                        textTokenizer = "EnglishPhonemesTokenizer"
                    },
                    vocoder = new VocoderConfig
                    {
                        modelPath = _GetCacheFilePathByName("tts_en_hifigan"),
                    },
                };
            }
            else if (_language == "German")
            {
                config = new SpeechConfig
                {
                    vad = new EncDecClassificationConfig
                    {
                        modelPath = _GetCacheFilePathByName("vad_marblenet"),
                        labels = EncDecClassificationConfig.VADLabels
                    },
                    asr = new EncDecCTCConfig
                    {
                        modelPath = _GetCacheFilePathByName("stt_de_quartznet15x5"),
                        vocabulary = EncDecCTCConfig.GermanVocabulary
                    },
                    specGen = new SpectrogramGeneratorConfig
                    {
                        modelPath = _GetCacheFilePathByName("tts_de_fastpitch_singleSpeaker_thorstenNeutral_2210"),
                        textTokenizer = "GermanCharsTokenizer"
                    },
                    vocoder = new VocoderConfig
                    {
                        modelPath = _GetCacheFilePathByName("tts_de_hifigan_singleSpeaker_thorstenNeutral_2210"),
                    },
                };
            }
            else
            {
                throw new ArgumentException();
            }
            return config;
        }
    }
}
