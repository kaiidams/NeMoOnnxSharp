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
using System.Linq;
using System.Text;

namespace NeMoOnnxSharp.TTSTokenizers
{
    // nemo.collections.tts.torch.g2ps.EnglishG2p
    public class EnglishG2p
    {
        public EnglishG2p(
            string phonemeDict,
            string heteronyms,
            double phonemeProbability = 0.5)
        {
        }

        public string[] Parse(string text)
        {
            return text.Split('|');
        }
    }
}
