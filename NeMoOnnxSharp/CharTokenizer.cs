// Copyright (c) Katsuya Iida.  All Rights Reserved.
// See LICENSE in the project root for license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Text.RegularExpressions;

namespace NeMoOnnxSharp
{
    public class CharTokenizer
    {
        private const string DefaultVocabulary = "_ abcdefghijklmnopqrstuvwxyz'";
        private static readonly Regex MergeRx = new Regex(@"(.)\1+");

        private readonly Regex _vocabRx;
        private readonly IDictionary<char, long> _v2i;
        private readonly string _i2v;

        public CharTokenizer() : this(DefaultVocabulary)
        {
        }

        public CharTokenizer(string characters)
        {
            _vocabRx = new Regex("[^" + characters.Substring(1) + "]");
            _i2v = characters;
            _v2i = new Dictionary<char, long>();
            for (int i = 0; i < _i2v.Length; i++) _v2i[_i2v[i]] = i;
        }

        public long[] Encode(string text)
        {
            string lower = text.ToLower().Trim();
            long[] encoded = new long[lower.Length];
            int j = 0;
            for (int i = 0; i < lower.Length; i++)
            {
                if (_v2i.TryGetValue(lower[i], out encoded[j]))
                {
                    j++;
                }
            }
            return encoded.AsSpan(0, j).ToArray();
        }

        public string Decode(long[] encoded)
        {
            char[] chars = new char[encoded.Length];
            for (int i = 0; i < chars.Length; i++)
            {
                long index = encoded[i];
                if (index < 0 || index >= _i2v.Length) index = 0;
                chars[i] = _i2v[(int)index];
            }
            return new string(chars);
        }

        public string MergeRepeated(string text)
        {
            return MergeRx.Replace(text, @"$1").Replace("_", "");
        }
    }
}