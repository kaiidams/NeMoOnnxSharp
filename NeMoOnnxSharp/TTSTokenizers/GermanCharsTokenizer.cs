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
using static NeMoOnnxSharp.TTSTokenizers.BaseTokenizer;

namespace NeMoOnnxSharp.TTSTokenizers
{
    // nemo.collections.tts.torch.tts_tokenizers.EnglishPhonemesTokenizer
    public class GermanCharsTokenizer : BaseCharsTokenizer
    {
        public GermanCharsTokenizer(
            bool padWithSpace = false
        ) : base(
            chars: new string(_CharsetStr),
            punct: true,
            addBlankAt: AddBlankAt.None,
            apostrophe: true,
            padWithSpace: padWithSpace,
            nonDefaultPunctList: _PunctList.Select(c => c.ToString()).ToArray()
        )
        {
        }

        private static readonly char[] _CharsetStr = new char[]
        {
            'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
            'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
            'U', 'V', 'W', 'X', 'Y', 'Z', 'Ä', 'Ö', 'Ü', 'ẞ',
            'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j',
            'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't',
            'u', 'v', 'w', 'x', 'y', 'z', 'ä', 'ö', 'ü', 'ß',
        };

        private static readonly char[] _PunctList = new char[]
        {
            '!', '"', '(', ')', ',', '-', '.', '/', ':', ';',
            '?', '[', ']', '{', '}', '«', '»', '‒', '–', '—',
            '‘', '‚', '“', '„', '‹', '›'
        };
    }
}
