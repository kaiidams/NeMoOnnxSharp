// Copyright (c) Katsuya Iida.  All Rights Reserved.
// See LICENSE in the project root for license information.

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using NeMoOnnxSharp.Models;

namespace NeMoOnnxSharp
{
    public class SpeechConfig
    {
        public SpeechConfig()
        {
            vad = new EncDecClassificationConfig();
            asr = new EncDecCTCConfig();
            specGen = new SpectrogramGeneratorConfig();
            vocoder = new VocoderConfig();
        }

        public EncDecClassificationConfig vad;
        public EncDecCTCConfig asr;
        public SpectrogramGeneratorConfig specGen;
        public VocoderConfig vocoder;
    }
}
