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
using static System.Net.Mime.MediaTypeNames;

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

            _stresses = stresses;
            _punct = punct;
        }

        public override int[] Encode(string text)
        {
            var g2pText = _g2p.Parse(text);
            return EncodeFromG2P(g2pText);
        }

        /// <summary>
        /// Encodes text that has already been run through G2P.
        /// Called for encoding to tokens after text preprocessing and G2P.
        /// </summary>
        /// <param name="g2pText">G2P's output, could be a mixture of phonemes and graphemes,
        ///        e.g. "see OOV" -> ['S', 'IY1', ' ', 'O', 'O', 'V']</param>
        /// <returns></returns>
        public int[] EncodeFromG2P(string[] g2pText)
        {
            var ps = new List<string>();
            var space = _id2token[_space];
            foreach (var _p in g2pText)
            {
                string p = _p;
                // Remove stress
                if (p.Length == 3 && !_stresses)
                {
                    p = p.Substring(0, 2);
                }

                // Add space if last one isn't one
                if (p == space && ps.Count > 0 && ps[ps.Count - 1] != space)
                {
                    ps.Add(p);
                }
                // Add next phoneme or char (if chars=true)
                else if ((char.IsLetterOrDigit(p, 0) || p == "'") && _token2id.ContainsKey(p))
                {
                    ps.Add(p); 
                }
                // Add punct
                else if (_punct && !char.IsLetterOrDigit(p, 0) && _token2id.ContainsKey(p))
                {
                    ps.Add(p);
                }
                else if (p != space)
                {
                    // Unknown char/phoneme
                }
            }

            // Remove trailing spaces
            while (ps.Count > 0 && ps[ps.Count - 1] == space)
            {
                ps.RemoveAt(ps.Count - 1);
            }

            var res = new List<int>();
            if (_padWithSpace)
            {
                res.Add(0);
            }
            res.AddRange(g2pText.Select(p => _token2id[p]));
            if (_padWithSpace)
            {
                res.Add(0);
            }
            return res.ToArray();
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
        private readonly bool _stresses;
        private readonly bool _punct;
    }
}
