// Copyright (c) Katsuya Iida.  All Rights Reserved.
// See LICENSE in the project root for license information.

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;

namespace NeMoOnnxSharp
{
    public class SpeechRecognitionEventArgs
    {
        public SpeechRecognitionEventArgs(ulong offset, string? text = null, short[]? audio = null)
        {
            Offset = offset;
            Text = text;
            Audio = audio;
        }

        public ulong Offset { get; private set; }
        public string? Text { get; private set; }
        public short[]? Audio { get; private set; }
    }
}
