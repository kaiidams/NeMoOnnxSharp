// Copyright (c) Katsuya Iida.  All Rights Reserved.
// See LICENSE in the project root for license information.

using System;

namespace NeMoOnnxSharp
{
    public class AudioToMFCCPreprocessor : IAudioPreprocessor<short, float>
    {
        protected readonly int _sampleRate;
        private readonly bool _center;
        protected readonly int _nWindowSize;
        protected readonly int _nWindowStride;
        private readonly double _preNormalize;
        private readonly IFeaturizer<short, float> _featurizer;

        public AudioToMFCCPreprocessor(
            int sampleRate = 16000,
            double windowSize = 0.02,
            double windowStride = 0.01,
            int? nWindowSize = null,
            int? nWindowStride = null,
            WindowFunction window = WindowFunction.Hann,
            int? nFFT = null,
            double preNormalize = 0.0,
            bool center = true,
            double lowFreq = 0.0,
            double? highFreq = null,
            int nMels = 64,
            int nMFCC = 64,
            int dctType = 2,
            MFCCNorm norm = MFCCNorm.Ortho,
            bool log = true)
        {
            _sampleRate = sampleRate;
            _preNormalize = preNormalize;
            _center = center;
            _nWindowSize = nWindowSize ?? (int)(windowSize * sampleRate);
            _nWindowStride = nWindowStride ?? (int)(windowStride * sampleRate);
            int _nFFT = nFFT ?? (int)Math.Pow(2, Math.Ceiling(Math.Log(_nWindowSize, 2)));
            _featurizer = new MFCC(
                sampleRate: sampleRate,
                window: window,
                winLength: _nWindowSize,
                nFFT: _nFFT,
                fMin: lowFreq,
                fMax: highFreq,
                nMels: nMels,
                nMFCC: nMFCC,
                dctType: dctType,
                mfccNorm: norm,
                logMels: log);
        }

        public float[] GetFeatures(Span<short> input)
        {
            double scale = GetScaleFactor(input);
            int outputLength = GetOutputLength(input.Length);
            int outputStep = _featurizer.OutputLength;
            float[] output = new float[outputStep * outputLength];
            int inputOffset = -(_nWindowSize / 2);
            for (int outputOffset = 0; outputOffset < output.Length; outputOffset += outputStep)
            {
                if (inputOffset > 0 && inputOffset + _nWindowSize <= input.Length)
                {
                    _featurizer.GetFeatures(
                        input.Slice(inputOffset, _nWindowSize),
                        output.AsSpan(outputOffset, outputStep));
                }
                else
                {
                    Span<short> temp = stackalloc short[_nWindowSize];
                    int start = inputOffset;
                    int end = inputOffset + _nWindowSize;
                    int offset = 0;
                    if (start < 0)
                    {
                        offset = -start;
                        start = 0;
                    }
                    if (end >= input.Length)
                    {
                        end = input.Length;
                    }
                    if (end > start)
                    {
                        input.Slice(start, end - start).CopyTo(temp.Slice(offset));
                    }
                    _featurizer.GetFeatures(
                        temp,
                        output.AsSpan(outputOffset, outputStep));
                }
                inputOffset += _nWindowStride;
            }
            return output;
        }

        private int GetOutputLength(int inputLength)
        {
            if (_center)
            {
                return (inputLength + _nWindowStride - 1) / _nWindowStride;
            }
            else
            {
                return (inputLength - _nWindowStride) / _nWindowStride + 1;
            }
        }

        private double GetScaleFactor(Span<short> waveform)
        {
            double scale;
            if (_preNormalize > 0)
            {
                scale = _preNormalize / MaxAbsValue(waveform);
            }
            else
            {
                scale = 1.0 / short.MaxValue;
            }

            return scale;
        }

        private int MaxAbsValue(Span<short> waveform)
        {
            int maxValue = 1;
            for (int i = 0; i < waveform.Length; i++)
            {
                int value = waveform[i];
                if (value < 0) value = -value;
                if (maxValue < value) maxValue = value;
            }
            return maxValue;
        }
    }
}
