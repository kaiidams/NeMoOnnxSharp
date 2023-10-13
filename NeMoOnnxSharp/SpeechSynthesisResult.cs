// Copyright (c) Katsuya Iida.  All Rights Reserved.
// See LICENSE in the project root for license information.

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;

namespace NeMoOnnxSharp
{
    public class SpeechSynthesisResult
    {
        public SpeechSynthesisResult()
        {
        }

        public short[]? AudioData { get; set; }
        public int SampleRate { get; set; }
    }
}
