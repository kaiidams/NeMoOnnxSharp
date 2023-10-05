// Copyright (c) Katsuya Iida.  All Rights Reserved.
// See LICENSE in the project root for license information.

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;

namespace NeMoOnnxSharp
{
    public class FrameVAD : IDisposable
    {
        private readonly int _sampleRate;
        private readonly int _winLength;
        private readonly AudioFeatureBuffer<short, float> _featureBuffer;
        private readonly EncDecClassificationModel _vad;

        public FrameVAD(EncDecClassificationModel vad)
        {
            _sampleRate = 16000;
            _winLength = 32;
            var transform = new MFCC(
                sampleRate: _sampleRate,
                window: WindowFunction.Hann,
                winLength: 400,
                nFFT: 512,
                nMels: 64,
                nMFCC: 64,
                fMin: 0.0,
                fMax: null,
                logMels: true,
                melScale: MelScale.HTK,
                melNorm: MelNorm.None);
            _featureBuffer = new AudioFeatureBuffer<short, float>(
                transform,
                hopLength: 160);
            _vad = vad;
        }

        public int SampleRate => _sampleRate;

        public void Dispose()
        {
            _vad.Dispose();
        }

        public float[] Transcribe(short[] input, int offset, int count)
        {
            return Transcribe(input.AsSpan(offset, count));
        }

        public float[] Transcribe(Span<short> input)
        {
            int outputPosition = _featureBuffer.OutputPosition;
            var result = new List<float>();
            while (input.Length > 0)
            {
                int written = _featureBuffer.Write(input);
                if (written == 0)
                {
                    throw new InvalidDataException();
                }
                if (_featureBuffer.OutputCount >= _winLength)
                {
                    var logits = _vad.Predict(_featureBuffer.OutputBuffer.AsSpan(0, _featureBuffer.NumOutputChannels * _winLength));
                    double x = Math.Exp(logits[0] - logits[1]);
                    result.Add((float)(1 / (x + 1)));
                    _featureBuffer.ConsumeOutput(_featureBuffer.NumOutputChannels);
                }
                input = input[written..];
            }
            return result.ToArray();
        }
    }
}
