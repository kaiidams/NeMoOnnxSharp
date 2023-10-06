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
        private readonly int _hopLength;
        private int _predictPosition;
        private float[] _predictWindow;
        private readonly AudioFeatureBuffer<short, float> _featureBuffer;
        private readonly EncDecClassificationModel _vad;

        public FrameVAD(EncDecClassificationModel vad)
        {
            _sampleRate = 16000;
            _winLength = 32;
            _hopLength = 1;
            _predictPosition = 0;
            _predictWindow = new float[_winLength / _hopLength];
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
        public int Position {
            get {
                int outputTotalWindow = (_predictWindow.Length - 1) * _hopLength + _winLength;
                int outputPosition = _featureBuffer.OutputPosition;
                outputPosition += _featureBuffer.HopLength * (outputTotalWindow / 2 - _winLength);
                return outputPosition - _featureBuffer.WinLength / 2;
            }
        }

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
            var result = new List<float>();
            while (input.Length > 0)
            {
                int written = _featureBuffer.Write(input);
                if (written == 0)
                {
                    throw new InvalidDataException();
                }
                while (_featureBuffer.OutputCount >= _featureBuffer.NumOutputChannels * _winLength)
                {
                    var logits = _vad.Predict(_featureBuffer.OutputBuffer.AsSpan(0, _featureBuffer.NumOutputChannels * _winLength));
                    double x = Math.Exp(logits[0] - logits[1]);

                    _predictWindow[_predictPosition] = (float)(1 / (x + 1));
                    _predictPosition = (_predictPosition + 1) % _predictWindow.Length;
                    result.Add(_predictWindow.Average());
                    _featureBuffer.ConsumeOutput(_featureBuffer.NumOutputChannels * _hopLength);
                }
                input = input[written..];
            }
            return result.ToArray();
        }
    }
}
