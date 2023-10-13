// Copyright (c) Katsuya Iida.  All Rights Reserved.
// See LICENSE in the project root for license information.

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Runtime.InteropServices;

namespace NeMoOnnxSharp
{
    public class SpeechConfig
    {
        public SpeechConfig()
        {
            vad = new EncDecClassificationConfig();
            asr = new EncDecCTCConfig();
        }

        public EncDecClassificationConfig vad;
        public EncDecCTCConfig asr;
    }
}
