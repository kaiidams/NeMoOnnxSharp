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
    // nemo.collections.tts.torch.tts_tokenizers.EnglishPhonemesTokenizer
    public class EnglishPhonemesTokenizer : BaseTokenizer
    {
        /// <summary>
        /// English phoneme-based tokenizer.
        /// </summary>
        /// <param name="g2p">Grapheme to phoneme module.</param>
        /// <param name="punct">Whether to reserve grapheme for basic punctuation or not.</param>
        /// <param name="nonDefaultPunctList">List of punctuation marks which will be used instead default.</param>
        /// <param name="stresses">Whether to use phonemes codes with stresses (0-2) or not.</param>
        /// <param name="chars">Whether to additionally use chars together with phonemes. It is useful if g2p module can return chars too.</param>
        /// <param name="space">Space token as string.</param>
        /// <param name="silence">Silence token as string (will be disabled if it is None).</param>
        /// <param name="apostrophe">Whether to use apostrophe or not.</param>
        /// <param name="oov">OOV token as string.</param>
        /// <param name="sep">Separation token as string.</param>
        /// <param name="addBlankAt">Add blank to labels in the specified order ("last") or after tokens (any non None),
        ///     if None then no blank in labels.</param>
        /// <param name="padWithSpace">Whether to pad text with spaces at the beginning and at the end or not.
        ///    text_preprocessing_func: Text preprocessing function for correct execution of the tokenizer.
        ///     Basically, it replaces all non-unicode characters with unicode ones.
        ///     Note that lower() function shouldn't be applied here, in case the text contains phonemes (it will be handled by g2p).</param>
        public EnglishPhonemesTokenizer(
            EnglishG2p g2p,
            bool punct = true,
            string[]? nonDefaultPunctList = null,
            bool stresses = false,
            bool chars = false,
            string space = " ",
            string? silence = null,
            bool apostrophe = true,
            string oov = BaseTokenizer.OOV,
            string sep = "|",  // To be able to distinguish between 2/3 letters codes.
            AddBlankAt addBlankAt = AddBlankAt.False,
            bool padWithSpace = false)
        // object? text_preprocessing_func=lambda text: english_text_preprocessing(text, lower=false),
        {
            _phonemeProbability = null;
            _g2p = g2p;
            _space = 0;
            var tokens = new List<string>();
            tokens.Add(space);

            if (silence != null)
            {
                throw new NotImplementedException();
            }

            tokens.AddRange(Consonants);
            var vowels = Vowels;

            if (stresses)
            {
                vowels = vowels.SelectMany(p => Enumerable.Range(0, 3), (p, s) => $"{p}{s}").ToArray();
            }
            tokens.AddRange(vowels);

            if (chars || _phonemeProbability != null)
            {
                if (!chars)
                {
                    // logging.warning(
                    //     "phoneme_probability was not None, characters will be enabled even though "
                    //     "chars was set to False."
                    // );
                }
                tokens.AddRange(AsciiLowercase.Select(ch => ch.ToString()));
            }

            if (apostrophe)
            {
                tokens.Add("'");  // Apostrophe
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
            if (addBlankAt != AddBlankAt.True)
            {
                throw new NotImplementedException();
            }
            tokens.Add(Blank);
            _blank = tokens.Count;
            tokens.Add(oov);  // Out Of Vocabulary
            _oov = tokens.Count;

            _sep = sep;
            _padWithSpace = padWithSpace;

            _id2token = tokens.ToArray();
            _token2id = new Dictionary<string, int>(
                Enumerable.Range(0, _id2token.Length)
                .Select(i => new KeyValuePair<string, int>(_id2token[i], i)));
            _utilIds = new HashSet<int>() { _pad, _blank, _oov };
        }

        public override int[] Encode(string text)
        {
            var g2pText = _g2p.Parse(text);
            return EncodeFromG2p(g2pText);
        }

        public int[] EncodeFromG2p(string[] g2pText)
        {
            var ps = new List<int>();
            ps.Add(0);
            ps.AddRange(g2pText.Select(p => _token2id[p]));
            ps.Add(0);
            return ps.ToArray();
        }

        private readonly string[] PunctList =
        {  // Derived from LJSpeech and "/" additionally
            ",", ".", "!", "?", "-",
            ":", ";", "/", "\"", "(",
            ")", "[", "]", "{", "}",
        };
        private readonly string[] Vowels = {
            "AA", "AE", "AH", "AO", "AW",
            "AY", "EH", "ER", "EY", "IH",
            "IY", "OW", "OY", "UH", "UW",
        };
        private readonly string[] Consonants = {
            "B", "CH", "D", "DH", "F", "G",
            "HH", "JH", "K", "L", "M", "N",
            "NG", "P", "R", "S", "SH", "T",
            "TH", "V", "W", "Y", "Z", "ZH",
        };

        private const string AsciiLowercase = "abcdefghijklmnopqrstuvwxyz";

        private readonly EnglishG2p _g2p;
        private readonly object? _phonemeProbability;
    }
}
