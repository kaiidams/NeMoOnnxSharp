// Copyright (c) Katsuya Iida.  All Rights Reserved.
// See LICENSE in the project root for license information.

using System;

namespace NeMoOnnxSharp
{
    public class AudioToMelSpectrogramPreprocessor : IAudioPreprocessor<short, float>
    {
        private enum FrameType
        {
            None,
            Preemph,
            Center,
            CenterPreemph
        }

        private const double FeatureStdOffset = 1e-5;

        private static FrameType GetFrameType(bool center, double preemph)
        {
            if (preemph == 0.0)
            {
                return center ? FrameType.Center : FrameType.None;
            }
            else
            {
                return center ? FrameType.CenterPreemph : FrameType.Preemph;
            }
        }

        protected readonly int _sampleRate;
        protected readonly double[] _window;
        private readonly FrameType _frameType;
        protected readonly int _nWindowStride;
        protected readonly FeatureNormalize _normalize;
        private readonly double _preNormalize;
        protected readonly double _preemph;
        protected readonly double[] _melBands;
        protected readonly int _nFFT;
        protected readonly int _features;
        private readonly MelNorm _melNorm;
        private readonly int _magPower;
        private readonly double _logZeroGuardValue;
        private readonly bool _log;

        public AudioToMelSpectrogramPreprocessor(
            int sampleRate = 16000,
            double windowSize = 0.02,
            double windowStride = 0.01,
            int? nWindowSize = null,
            int? nWindowStride = null,
            WindowFunction window = WindowFunction.Hann,
            FeatureNormalize normalize = FeatureNormalize.PerFeature,
            double preNormalize = 0.0,
            int? nFFT = null,
            double preemph = 0.97,
            bool center = true,
            int features = 64,
            double lowFreq = 0.0,
            double? highFreq = null,
            bool htk = false,
            MelNorm melNorm = MelNorm.Slaney,
            bool log = true,
            double? logZeroGuardValue = null,
            int magPower = 2)
        {
            _sampleRate = sampleRate;
            _preNormalize = preNormalize;
            _preemph = preemph;
            _window = Window.MakeWindow(window, nWindowSize ?? (int)(windowSize * sampleRate));
            _frameType = GetFrameType(center, preemph);
            _nWindowStride = nWindowStride ?? (int)(windowStride * sampleRate);
            _normalize = normalize;
            if (normalize != FeatureNormalize.PerFeature)
            {
                throw new ArgumentException("Only FeatureNormalize.PerFeature is supported");
            }
            _melBands = MelBands.MakeMelBands(
                lowFreq, highFreq ?? sampleRate / 2,
                features,
                htk ? MelScale.HTK : MelScale.Slaney);
            _melNorm = melNorm;
            _nFFT = nFFT ?? (int)Math.Pow(2, Math.Ceiling(Math.Log(_window.Length, 2)));
            _features = features;
            _magPower = magPower;
            _log = log;
            _logZeroGuardValue = logZeroGuardValue ?? Math.Pow(2, -24);
        }

        public float[] GetFeatures(Span<short> input)
        {
            double scale = GetScaleFactor(input);
            int outputStep = _features;
            int outputLength = GetOutputLength(input);
            float[] output = new float[outputStep * outputLength];
            int waveformOffset = 0;
            for (int outputOffset = 0; outputOffset < output.Length; outputOffset += outputStep)
            {
                MelSpectrogramStep(input, waveformOffset, scale, output.AsSpan(outputOffset));
                waveformOffset += _nWindowStride;
            }
            if (_normalize != FeatureNormalize.None)
            {
                NormalizeBatch(output, outputStep);
            }
            return output;
        }

        private int GetOutputLength(Span<short> input)
        {
            if (_frameType == FrameType.Center || _frameType == FrameType.CenterPreemph)
            {
                return (input.Length + _nWindowStride - 1) / _nWindowStride;
            }
            else
            {
                return (input.Length - _window.Length) / _nWindowStride + 1;
            }
        }

        private double GetScaleFactor(Span<short> input)
        {
            double scale;
            if (_preNormalize > 0)
            {
                scale = _preNormalize / MaxAbsValue(input);
            }
            else
            {
                scale = 1.0 / short.MaxValue;
            }

            return scale;
        }

        private int MaxAbsValue(Span<short> input)
        {
            int maxValue = 1;
            for (int i = 0; i < input.Length; i++)
            {
                int value = input[i];
                if (value < 0) value = -value;
                if (maxValue < value) maxValue = value;
            }
            return maxValue;
        }

        public void MelSpectrogramStep(
            Span<short> input, int waveformOffset,
            double scale, Span<float> output)
        {
            Span<double> temp1 = stackalloc double[_nFFT];
            Span<double> temp2 = stackalloc double[_nFFT];
            ReadFrame(input, waveformOffset, scale, temp1);
            FFT.CFFT(temp1, temp2, _nFFT);
            ToMagnitude(temp2, temp1, _nFFT);
            MelBands.ToMelSpectrogram(
                temp2, _melBands, _sampleRate, _nFFT, _features, _melNorm, _log, _logZeroGuardValue, temp1);
            for (int i = 0; i < _features; i++) output[i] = (float)temp1[i];
        }

        protected void ReadFrame(Span<short> input, int offset, double scale, Span<double> frame)
        {
            switch (_frameType)
            {
                case FrameType.None:
                    ReadFrameNone(input, offset, scale, frame);
                    break;
                case FrameType.Preemph:
                    throw new NotImplementedException();
                case FrameType.Center:
                    ReadFrameCenter(input, offset, scale, frame);
                    break;
                case FrameType.CenterPreemph:
                    ReadFrameCenterPreemphasis(input, offset, scale, frame);
                    break;
            }
        }

        private void ReadFrameNone(Span<short> input, int offset, double scale, Span<double> frame)
        {
            for (int i = 0; i < _window.Length; i++)
            {
                frame[i] = input[offset + i] * _window[i] * scale;
            }
            for (int i = _window.Length; i < frame.Length; i++)
            {
                frame[i] = 0.0;
            }
        }

        private void ReadFrameCenter(Span<short> input, int offset, double scale, Span<double> frame)
        {
            int frameOffset = frame.Length / 2 - _window.Length / 2;
            for (int i = 0; i < frameOffset; i++)
            {
                frame[i] = 0;
            }
            int waveformOffset = offset - _window.Length / 2;
            for (int i = 0; i < _window.Length; i++)
            {
                int k = i + waveformOffset;
                double v = (k >= 0 && k < input.Length) ? input[k] : 0;
                frame[i + frameOffset] = scale * v * _window[i];
            }
            for (int i = frameOffset + _window.Length; i < frame.Length; i++)
            {
                frame[i] = 0;
            }
        }

        private void ReadFrameCenterPreemphasis(Span<short> input, int offset, double scale, Span<double> frame)
        {
            int frameOffset = (frame.Length - 1) / 2 - (_window.Length - 1) / 2;
            for (int i = 0; i < frameOffset; i++)
            {
                frame[i] = 0;
            }
            int waveformOffset = offset - (_window.Length - 1) / 2;
            for (int i = 0; i < _window.Length; i++)
            {
                int k = i + waveformOffset;
                double v = (k >= 0 && k < input.Length) ? input[k] : 0;
                k--;
                if (k >= 0 && k < input.Length) v -= _preemph * input[k];
                frame[i + frameOffset] = scale * v * _window[i];
            }
            for (int i = frameOffset + _window.Length; i < frame.Length; i++)
            {
                frame[i] = 0;
            }
        }

        private void ToMagnitude(Span<double> xr, Span<double> xi, int length)
        {
            if (_magPower == 2)
            {
                ToSquareMagnitude(xr, xi, length);
            }
            else if (_magPower == 1)
            {
                ToAbsoluteMagnitude(xr, xi, length);
            }
            else
            {
                throw new NotImplementedException("power must be 1 or 2.");
            }
        }

        private static void ToAbsoluteMagnitude(Span<double> xr, Span<double> xi, int length)
        {
            for (int i = 0; i < length; i++)
            {
                xr[i] = Math.Sqrt(xr[i] * xr[i] + xi[i] * xi[i]);
            }
        }

        private static void ToSquareMagnitude(Span<double> xr, Span<double> xi, int length)
        {
            for (int i = 0; i < length; i++)
            {
                xr[i] = xr[i] * xr[i] + xi[i] * xi[i];
            }
        }

        private void NormalizeBatch(float[] output, int outputStep)
        {
            int melspecLength = output.Length / outputStep;
            for (int i = 0; i < outputStep; i++)
            {
                double sum = 0;
                for (int j = 0; j < melspecLength; j++)
                {
                    double v = output[i + outputStep * j];
                    sum += v;
                }
                float mean = (float)(sum / melspecLength);
                sum = 0;
                for (int j = 0; j < melspecLength; j++)
                {
                    double v = output[i + outputStep * j] - mean;
                    sum += v * v;
                }
                double std = Math.Sqrt(sum / melspecLength);
                float invStd = (float)(1.0 / (FeatureStdOffset + std));

                for (int j = 0; j < melspecLength; j++)
                {
                    float v = output[i + outputStep * j];
                    output[i + outputStep * j] = (v - mean) * invStd;
                }
            }
        }
    }
}
