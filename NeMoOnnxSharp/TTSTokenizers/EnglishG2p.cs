// Copyright (c) Katsuya Iida.  All Rights Reserved.
// See LICENSE in the project root for license information.

// A number of implementation details in this file have been translated from the Python scripts of NVIDIA NeMo,
// largely located in the files found in this folder:
//
// https://github.com/NVIDIA/NeMo/blob/main/nemo/collections/tts/g2p/models/en_us_arpabet.py
//
// The origin has the following copyright notice and license:
//
// https://github.com/NVIDIA/NeMo/blob/main/LICENSE
//

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Text.RegularExpressions;

namespace NeMoOnnxSharp.TTSTokenizers
{
    // nemo.collections.tts.torch.g2ps.EnglishG2p

    /// <summary>
    /// English G2P module. This module converts words from grapheme to phoneme representation using phoneme_dict in CMU dict format.
    /// Optionally, it can ignore words which are heteronyms, ambiguous or marked as unchangeable by word_tokenize_func(see code for details).
    /// Ignored words are left unchanged or passed through apply_to_oov_word for handling.
    /// </summary>
    public class EnglishG2p
    {
        private readonly IDictionary<string, string[]> _phonemeDict;
        private readonly HashSet<string> _heteronyms;
        private readonly double _phonemeProbability;
        private readonly Random _random;
        private readonly Regex _alnumRx;
        private readonly bool _ignoreAmbiguousWords;

        /// </summary>
        /// <param name="phonemeDict">Path to file in CMUdict format or dictionary of CMUdict-like entries.</param>
        /// word_tokenize_func: Function for tokenizing text to words.
        /// <param name="heteronyms">Path to file with heteronyms (every line is new word) or list of words.</param>
        /// <param name="phonemeProbability">The probability (0.<var<1.) that each word is phonemized.Defaults to None which is the same as 1.
        /// Note that this code path is only run if the word can be phonemized.For example: If the word does not have an entry in the g2p dict, it will be returned
        /// as characters.If the word has multiple entries and ignore_ambiguous_words is True, it will be returned as characters.
        /// </param>
        public EnglishG2p(
            string phonemeDict,
            string heteronyms,
            bool ignoreAmbiguousWords = true,
            Encoding? encoding = null,
            double phonemeProbability = 0.5)
        {
            encoding = encoding ?? Encoding.GetEncoding("iso-8859-1");
            _phonemeDict = _ParseAsCmuDict(phonemeDict, encoding);
            _heteronyms = new HashSet<string>(_ParseFileByLines(heteronyms, encoding));
            _phonemeProbability = phonemeProbability;
            _random = new Random();
            _alnumRx = new Regex(@"[a-zA-ZÀ-ÿ\d]");
            _ignoreAmbiguousWords = ignoreAmbiguousWords;
        }

        public string[] Parse(string text)
        {
            var words = text.Split(' ');
            var prons = new List<string>();
            foreach (var word in words)
            {
                var wordStr = word;
                var wordByHyphen = wordStr.Split('-');

                var (pron, isHandled) = ParseOneWord(wordStr);
                if (!isHandled && wordByHyphen.Length > 1)
                {
                    pron = new List<string>();
                    foreach (var subWord in wordByHyphen)
                    {
                        var (p, _) = ParseOneWord(subWord);
                        pron.AddRange(p);
                        pron.Add("-");
                    }
                    pron.RemoveAt(pron.Count - 1);
                }
                prons.AddRange(pron);
            }
            return prons.ToArray();
        }

        private (List<string> pron, bool isHandled) ParseOneWord(string word)
        {
            if (_phonemeProbability < 1.0 && _random.NextDouble() > _phonemeProbability)
            {
                return (word.Select(x => x.ToString()).ToList(), true);
            }

            // punctuation or whitespace.
            if (!_alnumRx.IsMatch(word))
            {
                return (word.Select(x => x.ToString()).ToList(), true);
            }

            // heteronyms
            if (_heteronyms != null && _heteronyms.Contains(word))
            {
                return (word.Select(x => x.ToString()).ToList(), true);
            }

            // phoneme dict
            if (_phonemeDict.ContainsKey(word) && (!_ignoreAmbiguousWords || _IsUniqueInPhonemeDict(word)))
            {
                return (_phonemeDict[word][0].Split(" ").ToList(), true);
            }

            return (word.Select(x => x.ToString()).ToList(), false);
        }

        private bool _IsUniqueInPhonemeDict(string word)
        {
            return _phonemeDict[word].Length == 1;
        }

        private static IDictionary<string, string[]> _ParseAsCmuDict(string phonemeDictPath, Encoding encoding)
        {
            var _alt_re = new Regex(@"\([0-9]+\)");
            var g2pDict = new Dictionary<string, string[]>();
            using (var stream = new FileStream(phonemeDictPath, FileMode.Open))
            using (var reader = new StreamReader(stream, encoding))
            {
                string line;
                while ((line = reader.ReadLine()) != null)
                {
                    if (line.Length > 0 && (('A' <= line[0] && line[0] <= 'Z') || line[0] == '\''))
                    {
                        var parts = line.Split("  ");
                        var word = _alt_re.Replace(parts[0], "");
                        word = word.ToLower();

                        var pronunciation = parts[1].Trim();
                        if (g2pDict.ContainsKey(word))
                        {
                            var v = new List<string>(g2pDict[word])
                            {
                                pronunciation
                            };
                            g2pDict[word] = v.ToArray();
                        }
                        else
                        {
                            g2pDict[word] = new string[] { pronunciation };
                        }
                    }
                }
            }
            return g2pDict;
        }

        private static string[] _ParseFileByLines(string p, Encoding encoding)
        {
            var res = new List<string>();
            using (var stream = new FileStream(p, FileMode.Open))
            using (var reader = new StreamReader(stream, encoding))
            {
                string line;
                while ((line = reader.ReadLine()) != null)
                {
                    res.Add(line.TrimEnd());
                }
            }
            return res.ToArray();
        }
    }
}
