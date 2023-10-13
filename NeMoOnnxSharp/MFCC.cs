// Copyright (c) Katsuya Iida.  All Rights Reserved.
// See LICENSE in the project root for license information.

using System;

namespace NeMoOnnxSharp
{
    public class MFCC : IFeaturizer<short, float>
    {
        private const double InvMaxShort = 1.0 / short.MaxValue;
        private const double LogOffset = 1e-6;

        protected readonly int _sampleRate;
        protected readonly double[] _window;
        protected readonly double[] _melBands;
        protected readonly int _nFFT;
        protected readonly int _nMels;
        private readonly MelNorm _melNorm;
        private readonly int _power;
        private readonly bool _logMels;
        private readonly int _nMFCC;

        public int SampleRate => _sampleRate;
        public int InputLength => _window.Length;
        public int OutputLength => _nMFCC;

        public MFCC(
            int sampleRate = 16000,
            WindowFunction window = WindowFunction.Hann,
            int? winLength = null,
            int nFFT = 400,
            int power = 2,
            bool normalized = false,
            double fMin = 0.0,
            double? fMax = null,
            int nMels = 128,
            MelNorm melNorm = MelNorm.None,
            MelScale melScale = MelScale.HTK,
            int nMFCC = 40,
            int dctType = 2,
            MFCCNorm mfccNorm = MFCCNorm.Ortho,
            bool logMels = false)
        {
            if (dctType != 2)
            {
                throw new ArgumentException("Only DCT-II is supported");
            }
            if (normalized)
            {
                throw new ArgumentException("Normalizing by magnitude after stft is not supported");
            }
            if (mfccNorm != MFCCNorm.Ortho)
            {
                throw new ArgumentException("Only Ortho is supported for MFCC norm");
            }
            _sampleRate = sampleRate;
            _window = Window.MakeWindow(window, winLength ?? nFFT);
            _melBands = MelBands.MakeMelBands(fMin, fMax ?? sampleRate / 2, nMels, melScale);
            _melNorm = melNorm;
            _nFFT = nFFT;
            _nMels = nMels;
            _power = power;
            _logMels = logMels;
            _nMFCC = nMFCC;
        }

        public void GetFeatures(Span<short> input, Span<float> output)
        {
            Span<double> temp1 = stackalloc double[_nFFT];
            Span<double> temp2 = stackalloc double[_nFFT];
            ReadFrame(input, temp1);
            FFT.CFFT(temp1, temp2, _nFFT);
            ToMagnitude(temp2, temp1);
            MelBands.ToMelSpectrogram(
                temp2, _melBands, _sampleRate, _nFFT, _nMels, _melNorm, true, LogOffset, temp1);
            FFT.DCT2(temp1, temp2, _nMFCC);
            for (int i = 0; i < _nMFCC; i++) output[i] = (float)temp2[i];
        }

        private void ToMagnitude(Span<double> xr, Span<double> xi)
        {
            if (_power == 2)
            {
                ToSquareMagnitude(xr, xi);
            }
            else if (_power == 1)
            {
                ToAbsoluteMagnitude(xr, xi);
            }
            else
            {
                throw new NotImplementedException("power must be 1 or 2.");
            }
        }

        private static void ToAbsoluteMagnitude(Span<double> xr, Span<double> xi)
        {
            for (int i = 0; i < xr.Length; i++)
            {
                xr[i] = Math.Sqrt(xr[i] * xr[i] + xi[i] * xi[i]);
            }
        }

        private static void ToSquareMagnitude(Span<double> xr, Span<double> xi)
        {
            for (int i = 0; i < xr.Length; i++)
            {
                xr[i] = xr[i] * xr[i] + xi[i] * xi[i];
            }
        }

        private void ReadFrame(Span<short> waveform, Span<double> frame)
        {
            int frameOffset = frame.Length / 2 - _window.Length / 2;
            frame.Slice(0, frameOffset).Fill(0);
            for (int i = 0; i < _window.Length; i++)
            {
                frame[i + frameOffset] = InvMaxShort * waveform[i] * _window[i];
            }
            frame.Slice(frameOffset + _window.Length).Fill(0);
        }
    }
}
