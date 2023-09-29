// Copyright (c) Katsuya Iida.  All Rights Reserved.
// See LICENSE in the project root for license information.

// A number of implementation details in this file have been translated from the Python scripts of NVIDIA NeMo,
// largely located in the files found in this folder:
//
// https://github.com/NVIDIA/NeMo/blob/main/nemo/collections/common/tokenizers/text_to_speech/tokenizer_utils.py
//
// The origin has the following copyright notice and license:
//
// https://github.com/NVIDIA/NeMo/blob/main/LICENSE
//

using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Text;

namespace NeMoOnnxSharp.TTSTokenizers
{
    public static class TokenizerUtils
    {
        private static readonly Dictionary<char, char> SynoGlyph2Ascii;

        static TokenizerUtils()
        {
            Tuple<char, char[]>[] _synoglyphs = {
                new Tuple<char, char[]>('\'', new[] { '’' }),
                new Tuple<char, char[]>('"', new[] { '”', '“' }),
            };

            SynoGlyph2Ascii = new Dictionary<char, char>();
            foreach (var (asc, glyphs) in _synoglyphs)
            {
                foreach (var g in glyphs)
                {
                    SynoGlyph2Ascii[g] = asc;
                }
            }
        }

        public static string EnglishTextPreprocessing(string text, bool lower = true)
        {
            text = new string(
                text.Normalize(NormalizationForm.FormD)
                .Where(ch => CharUnicodeInfo.GetUnicodeCategory(ch) != UnicodeCategory.NonSpacingMark)
                .Select(ch => SynoGlyph2Ascii.ContainsKey(ch) ? SynoGlyph2Ascii[ch] : ch)
                .ToArray());

            if (lower)
            {
                text = text.ToLower();
            }

            return text;
        }

    }
}
