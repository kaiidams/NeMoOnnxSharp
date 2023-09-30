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
using System.Data.SqlTypes;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Text;
using System.Text.RegularExpressions;

namespace NeMoOnnxSharp.TTSTokenizers
{
    public static class TokenizerUtils
    {
        private static readonly Dictionary<char, char> _synoGlyph2Ascii;
        private static readonly Regex _wordsReEn;

        static TokenizerUtils()
        {
            Tuple<char, char[]>[] _synoglyphs = {
                new Tuple<char, char[]>('\'', new[] { '’' }),
                new Tuple<char, char[]>('"', new[] { '”', '“' }),
            };

            _synoGlyph2Ascii = new Dictionary<char, char>();
            foreach (var (asc, glyphs) in _synoglyphs)
            {
                foreach (var g in glyphs)
                {
                    _synoGlyph2Ascii[g] = asc;
                }
            }

            // define char set based on https://en.wikipedia.org/wiki/List_of_Unicode_characters
            var latinAlphabetBasic = "A-Za-z";
            _wordsReEn = new Regex(@$"([{latinAlphabetBasic}]+(?:[{latinAlphabetBasic}\-']*[{latinAlphabetBasic}]+)*)|(\|[^|]*\|)|([^{latinAlphabetBasic}|]+)");
        }

        public static string EnglishTextPreprocessing(string text, bool lower = true)
        {
            text = new string(
                text.Normalize(NormalizationForm.FormD)
                .Where(ch => CharUnicodeInfo.GetUnicodeCategory(ch) != UnicodeCategory.NonSpacingMark)
                .Select(ch => _synoGlyph2Ascii.ContainsKey(ch) ? _synoGlyph2Ascii[ch] : ch)
                .ToArray());

            if (lower)
            {
                text = text.ToLower();
            }
            return text;
        }

        /// <summary>
        /// Process a list of words and attach indicators showing if each word is unchangeable or not. Each word representation
        /// can be one of valid word, any substring starting from | to | (unchangeable word), or punctuation marks including
        /// whitespaces.This function will split unchanged strings by whitespaces and return them as `List[str]`. For example,
        /// 
        /// .. code-block::python
        ///     [
        ///         ('Hello', '', ''),  # valid word
        ///         ('', '', ' '),  # punctuation mark
        ///         ('World', '', ''),  # valid word
        ///         ('', '', ' '),  # punctuation mark
        ///         ('', '|NVIDIA unchanged|', ''),  # unchangeable word
        ///         ('', '', '!')  # punctuation mark
        ///     ]
        ///
        /// will be converted into,
        ///
        /// .. code-block::python
        ///     [
        ///         (["Hello"], false),
        ///         ([" "], false),
        ///         (["World"], false),
        ///         ([" "], false),
        ///         (["NVIDIA", "unchanged"], True),
        ///         (["!"], false)
        ///     ]
        /// </summary>
        /// <param name="words">a list of tuples like `(maybe_word, maybe_without_changes, maybe_punct)` where each element
        /// corresponds to a non-overlapping match of either `_WORDS_RE_EN` or `_WORDS_RE_ANY_LOCALE`.</param>
        /// <param name="isLower">a flag to trigger lowercase all words. By default, it is false.</param>
        /// <returns>a list of tuples like `(a list of words, is_unchanged)`.</returns>
        private static (string[], bool)[] _wordTokenize(MatchCollection words, bool isLower = false)
        {
            var result = new List<(string[], bool)>();
            foreach (Match word in words)
            {
                var maybeWord = word.Groups[0].Value;
                var maybeWithoutChanges = word.Groups[1].Value;
                var maybePunct = word.Groups[2].Value;

                var withoutChanges = false;
                string[] token;
                if (!string.IsNullOrEmpty(maybeWord))
                {
                    if (isLower)
                    {
                        token = new[] { maybeWord.ToLower() };
                    }
                    else
                    {
                        token = new[] { maybeWord };
                    }
                }
                else if (!string.IsNullOrEmpty(maybePunct))
                {
                    token = new[] { maybePunct };
                }
                else if (!string.IsNullOrEmpty(maybeWithoutChanges))
                {
                    withoutChanges = true;
                    token = maybeWithoutChanges.Substring(1, maybeWithoutChanges.Length - 2).Split(' ');
                }
                else
                {
                    throw new InvalidDataException(
                        $"This is not expected. Found empty string: <{word}>. " +
                        $"Please validate your regular expression pattern '_WORDS_RE_EN' or '_WORDS_RE_ANY_LOCALE'."
                    );
                }

                result.Add((token, withoutChanges));
            }
            return result.ToArray();
        }

        public static (string[], bool)[] EnglishWordTokenize(string text)
        {
            var words = _wordsReEn.Matches(text);
            return _wordTokenize(words, isLower: true);
        }
    }
}
