// Copyright (c) Katsuya Iida.  All Rights Reserved.
// See LICENSE in the project root for license information.

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;

namespace NeMoOnnxSharp
{
    public sealed class SpeechSynthesizer : IDisposable
    {
        private readonly SpectrogramGenerator _specGen;
        private readonly Vocoder _vocoder;

        public SpeechSynthesizer(SpeechConfig config)
        {
            _specGen = new SpectrogramGenerator(config.specGen);
            _vocoder = new Vocoder(config.vocoder);
        }

        public void Dispose()
        {
            _specGen.Dispose();
            _vocoder.Dispose();
        }

        public SpeechSynthesisResult SpeakText(string text)
        {
            var parsed = _specGen.Parse(text);
            var spec = _specGen.GenerateSpectrogram(parsed, pace: 1.0);
            var audio = _vocoder.ConvertSpectrogramToAudio(spec);
            return new SpeechSynthesisResult()
            {
                AudioData = audio,
                SampleRate = _vocoder.SampleRate
            };
        }
    }
}
