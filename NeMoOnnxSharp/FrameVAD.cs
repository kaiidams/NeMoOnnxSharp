// Copyright (c) Katsuya Iida.  All Rights Reserved.
// See LICENSE in the project root for license information.

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;

namespace NeMoOnnxSharp
{
    public class FrameVAD : IDisposable
    {
        private readonly int _sampleRate;
        private readonly int _modelWinLength;
        private readonly int _modelHopLength;
        private int _predictIndex;
        private float[] _predictWindow;
        private readonly AudioFeatureBuffer<short, float> _featureBuffer;
        private readonly EncDecClassificationModel _vad;

        private FrameVAD(EncDecClassificationModel vad, int smoothingWinLength = 64)
        {
            _sampleRate = 16000;
            _modelWinLength = 32;
            _modelHopLength = 1;
            _predictIndex = 0;
            _predictWindow = new float[smoothingWinLength];
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

        public FrameVAD(string modelPath) : this(
            new EncDecClassificationModel(modelPath))
        {
        }

        public FrameVAD(byte[] model) : this(
            new EncDecClassificationModel(model))
        {
        }

        public int HopLength => _featureBuffer.HopLength * _modelHopLength;

        public int SampleRate => _sampleRate;
        public int PredictionOffset {
            get {
                int outputTotalWindow = (_predictWindow.Length - 1) * _modelHopLength + _modelWinLength;
                int outputPosition = _featureBuffer.OutputPosition;
                outputPosition += _featureBuffer.HopLength * (outputTotalWindow / 2 - _modelWinLength);
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
                while (_featureBuffer.OutputCount >= _featureBuffer.NumOutputChannels * _modelWinLength)
                {
                    var logits = _vad.Predict(_featureBuffer.OutputBuffer.AsSpan(0, _featureBuffer.NumOutputChannels * _modelWinLength));
                    double x = Math.Exp(logits[0] - logits[1]);

                    _predictWindow[_predictIndex] = (float)(1 / (x + 1));
                    _predictIndex = (_predictIndex + 1) % _predictWindow.Length;
                    result.Add(_predictWindow.Average());
                    _featureBuffer.ConsumeOutput(_featureBuffer.NumOutputChannels * _modelHopLength);
                }
                input = input[written..];
            }
            return result.ToArray();
        }
    }
}
