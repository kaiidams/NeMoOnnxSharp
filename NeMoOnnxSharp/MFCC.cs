// Copyright (c) Katsuya Iida.  All Rights Reserved.
// See LICENSE in the project root for license information.

using System;

namespace NeMoOnnxSharp
{
    public class MFCC : IFrameTransform<short, float>
    {
        private const double InvMaxShort = 1.0 / short.MaxValue;
        private const double LogOffset = 1e-6;

        protected readonly double _sampleRate;
        protected readonly double[] _window;
        protected readonly double[] _melBands;
        protected readonly int _nFFT;
        protected readonly int _nMels;
        private readonly MelNorm _melNorm;
        private readonly int _power;
        private readonly bool _logMels;
        private readonly int _nMFCC;

        public MFCC(
            int sampleRate = 16000,
            WindowFunction window = WindowFunction.Hann,
            int winLength = 0,
            int nFFT = 400,
            int power = 2,
            bool normalized = false,
            double fMin = 0.0,
            double fMax = 0.0,
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
            if (fMax == 0.0)
            {
                fMax = sampleRate / 2;
            }
            _sampleRate = sampleRate;
            if (winLength == 0) winLength = nFFT;
            _window = Window.MakeWindow(window, winLength);
            _melBands = MelBands.MakeMelBands(fMin, fMax, nMels, melScale);
            _melNorm = melNorm;
            _nFFT = nFFT;
            _nMels = nMels;
            _power = power;
            _logMels = logMels;
            _nMFCC = nMFCC;
        }

        public void Transform(Span<short> input, Span<float> output)
        {
            Span<double> temp1 = stackalloc double[_nFFT];
            Span<double> temp2 = stackalloc double[_nFFT];
            ReadFrame(input, temp1);
            FFT.CFFT(temp1, temp2, _nFFT);
            ToMagnitude(temp2, temp1);
            ToMelSpectrogram(temp2, temp1);
            FFT.DCT2(temp1, temp2, _nMFCC);
            for (int i = 0; i < _nMels; i++) output[i] = (float)temp2[i];
        }

        private void ToSpectrogram(Span<double> input, float[] output, int outputOffset, int outputSize)
        {
            if (_logMels)
            {
                for (int i = 0; i < outputSize; i++)
                {
                    double value = Math.Log(input[i] + LogOffset);
                    output[outputOffset + i] = (float)value;
                }
            }
            else
            {
                for (int i = 0; i < outputSize; i++)
                {
                    output[outputOffset + i] = (float)input[i];
                }
            }
        }

        private void ToMelSpectrogram(Span<double> spec, Span<double> melspec)
        {
            if (!_logMels) throw new NotImplementedException();
            switch (_melNorm)
            {
                case MelNorm.None:
                    ToMelSpectrogramNone(spec, melspec);
                    break;
                case MelNorm.Slaney:
                    ToMelSpectrogramSlaney(spec, melspec);
                    break;
            }
        }

        private void ToMelSpectrogramNone(Span<double> spec, Span<double> melspec)
        {
            for (int i = 0; i < _nMels; i++)
            {
                double startHz = _melBands[i];
                double peakHz = _melBands[i + 1];
                double endHz = _melBands[i + 2];
                double v = 0.0;
                int j = (int)(startHz * _nFFT / _sampleRate) + 1;
                while (true)
                {
                    double hz = j * _sampleRate / _nFFT;
                    if (hz > peakHz)
                        break;
                    double r = (hz - startHz) / (peakHz - startHz);
                    v += spec[j] * r;
                    j++;
                }
                while (true)
                {
                    double hz = j * _sampleRate / _nFFT;
                    if (hz > endHz)
                        break;
                    double r = (endHz - hz) / (endHz - peakHz);
                    v += spec[j] * r;
                    j++;
                }
                melspec[i] = (float)Math.Log(v + LogOffset);
            }
        }

        private void ToMelSpectrogramSlaney(Span<double> spec, Span<double> melspec)
        {
            for (int i = 0; i < _nMels; i++)
            {
                double startHz = _melBands[i];
                double peakHz = _melBands[i + 1];
                double endHz = _melBands[i + 2];
                double v = 0.0;
                int j = (int)(startHz * _nFFT / _sampleRate) + 1;
                while (true)
                {
                    double hz = j * _sampleRate / _nFFT;
                    if (hz > peakHz)
                        break;
                    double r = (hz - startHz) / (peakHz - startHz);
                    v += spec[j] * r * 2 / (endHz - startHz);
                    j++;
                }
                while (true)
                {
                    double hz = j * _sampleRate / _nFFT;
                    if (hz > endHz)
                        break;
                    double r = (endHz - hz) / (endHz - peakHz);
                    v += spec[j] * r * 2 / (endHz - startHz);
                    j++;
                }
                melspec[i] = (float)Math.Log(v + LogOffset);
            }
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
