// Copyright (c) Katsuya Iida.  All Rights Reserved.
// See LICENSE in the project root for license information.

// A number of implementation details in this file have been translated from the Python scripts of NVIDIA NeMo,
// largely located in the files found in this folder:
//
// https://github.com/NVIDIA/NeMo/blob/main/nemo/collections/tts/torch/tts_tokenizers.py
//
// The origin has the following copyright notice and license:
//
// https://github.com/NVIDIA/NeMo/blob/main/LICENSE
//

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace NeMoOnnxSharp.TTSTokenizers
{
    // nemo.collections.tts.torch.tts_tokenizers.BaseCharsTokenizer
    public class BaseCharsTokenizer : BaseTokenizer
    {
        public BaseCharsTokenizer(
            string chars,
            bool punct = true,
            string[]? nonDefaultPunctList = null,
            bool apostrophe = true,
            string oov = OOV,
            string sep = "|",  // To be able to distinguish between 2/3 letters codes.
            AddBlankAt addBlankAt = AddBlankAt.None,
            bool padWithSpace = false)
        // object? text_preprocessing_func=lambda text: english_text_preprocessing(text, lower=false),
        {
            _space = 0;
            var tokens = new List<string>();
            tokens.Add(" ");
            tokens.AddRange(chars.Select(ch => ch.ToString()));
            if (apostrophe)
            {
                tokens.Add("'");  // Apostrophe for saving "don't" and "Joe's"
            }

            if (punct)
            {
                if (nonDefaultPunctList != null)
                {
                    tokens.AddRange(nonDefaultPunctList);
                }
                else
                {
                    tokens.AddRange(PunctList);
                }
            }

            tokens.Add(Pad);
            _pad = tokens.Count;
            if (addBlankAt != AddBlankAt.None)
            {
                _blank = tokens.Count;
                tokens.Add(Blank);
            }

            tokens.Add(oov);  // Out Of Vocabulary
            _oov = tokens.Count;

            if (addBlankAt == AddBlankAt.Last)
            {
                throw new NotImplementedException();
            }

            _sep = sep;
            _punct = punct;
            _padWithSpace = padWithSpace;

            _id2token = tokens.ToArray();
            _token2id = new Dictionary<string, int>(
                Enumerable.Range(0, _id2token.Length)
                .Select(i => new KeyValuePair<string, int>(_id2token[i], i)));
            _utilIds = new HashSet<int>() { _pad, _blank, _oov };

            _punct = punct;
        }

        public override int[] Encode(string text)
        {
            var cs = new List<string>();
            var space = _id2token[_space];
            var tokens = Tokens;

            text = TextPreprocessingFunc(text);
            foreach (var c_ in text)
            {
                string c = c_.ToString();

                // Add a whitespace if the current char is a whitespace while the previous char is not a whitespace.
                if (c == space && cs.Count > 0 && cs[cs.Count - 1] != space)
                {
                    cs.Add(c);
                }
                // Add the current char that is an alphanumeric or an apostrophe.
                else if ((char.IsLetterOrDigit(c, 0) || c == "'") && tokens.Contains(c))
                {
                    cs.Add(c);
                }
                // Add a punctuation that has a single char.
                else if (!char.IsLetterOrDigit(c, 0) && _token2id.ContainsKey(c) && _punct)
                {
                    cs.Add(c);
                }
                // Warn about unknown char
                else if (c != space)
                {
                    // Text: [{text}] contains unknown char: [{c}]. Symbol will be skipped.
                }
            }

            // Remove trailing spaces
            if (cs.Count > 0)
            {
                while (cs[cs.Count - 1] == space)
                {
                    cs.RemoveAt(cs.Count - 1);
                }
            }

            if (_padWithSpace)
            {
                cs.Insert(0, space);
                cs.Add(space);
            }
            return cs.Select(c => _token2id[c]).ToArray();
        }

        protected virtual string TextPreprocessingFunc(string text)
        {
            return TokenizerUtils.AnyLocaleTextPreprocessing(text);
        }

        private readonly string[] PunctList =
        {  // Derived from LJSpeech and "/" additionally
            ",", ".", "!", "?", "-",
            ":", ";", "/", "\"", "(",
            ")", "[", "]", "{", "}",
        };

        private readonly bool _punct;
    }
}
