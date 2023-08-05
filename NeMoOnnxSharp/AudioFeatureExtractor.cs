// Copyright (c) Katsuya Iida.  All Rights Reserved.
// See LICENSE in the project root for license information.

using System;

namespace NeMoOnnxSharp
{
    class AudioFeatureExtractor
    {
        private readonly double[] _window;
        private readonly double[] _melBands;
        private readonly double[] _temp1;
        private readonly double[] _temp2;
        private readonly int _fftLength;
        private readonly int _nMelBands;
        private readonly double _sampleRate;
        private readonly double _logOffset;

        public AudioFeatureExtractor(
            int sampleRate = 16000,
            int stftWindowLength = 400, int stftLength = 512,
            int nMelBands = 64, double melMinHz = 0.0, double melMaxHz = 0.0,
            double logOffset = 1e-6)
        {
            if (melMaxHz == 0.0)
            {
                melMaxHz = sampleRate / 2;
            }
            _sampleRate = sampleRate;
            _window = Window.MakeWindow(WindowFunction.Hann, stftWindowLength);
            _melBands = MakeMelBands(melMinHz, melMaxHz, nMelBands);
            _temp1 = new double[stftLength];
            _temp2 = new double[stftLength];
            _fftLength = stftLength;
            _nMelBands = nMelBands;
            _logOffset = logOffset;
        }

        public void Spectrogram(float[] waveform, int waveformOffset, float[] spec, int specOffset)
        {
            GetFrame(waveform, waveformOffset, _temp1);
            FFT.CFFT(_temp1, _temp2, _fftLength);
            ToMagnitude(_temp2, _temp1, _fftLength);
            int specLength = _fftLength / 2 + 1;
            for (int i = 0; i < specLength; i++)
            {
                float value = (float)(20.0 * Math.Log(_temp2[i] + _logOffset));
                spec[specOffset + i] = value;
            }
        }

        public void MelSpectrogram(float[] waveform, int waveformOffset, float[] melspec, int melspecOffset)
        {
            GetFrame(waveform, waveformOffset, _temp1);
            FFT.CFFT(_temp1, _temp2, _fftLength);
            ToSquareMagnitude(_temp2, _temp1, _fftLength);
            ToMelSpec(_temp2, melspec, melspecOffset);
        }

        public void MelSpectrogram(Span<short> waveform, int waveformOffset, double scale, float[] melspec, int melspecOffset)
        {
            GetFrame(waveform, waveformOffset, scale, _temp1);
            FFT.CFFT(_temp1, _temp2, _fftLength);
            ToSquareMagnitude(_temp2, _temp1, _fftLength);
            ToMelSpec(_temp2, melspec, melspecOffset);
        }

        private void ToMelSpec(double[] spec, float[] melspec, int melspecOffset)
        {
            for (int i = 0; i < _nMelBands; i++)
            {
                double startHz = _melBands[i];
                double peakHz = _melBands[i + 1];
                double endHz = _melBands[i + 2];
                double v = 0.0;
                int j = (int)(startHz * _fftLength / _sampleRate) + 1;
                while (true)
                {
                    double hz = j * _sampleRate / _fftLength;
                    if (hz > peakHz)
                        break;
                    double r = (hz - startHz) / (peakHz - startHz);
                    v += spec[j] * r;
                    j++;
                }
                while (true)
                {
                    double hz = j * _sampleRate / _fftLength;
                    if (hz > endHz)
                        break;
                    double r = (endHz - hz) / (endHz - peakHz);
                    v += spec[j] * r;
                    j++;
                }
                melspec[melspecOffset + i] = (float)Math.Log(v + _logOffset);
            }
        }

        void GetFrame(float[] waveform, int start, double[] frame)
        {
            for (int i = 0; i < _window.Length; i++)
            {
                frame[i] = waveform[start + i] * _window[i];
            }
            for (int i = _window.Length; i < frame.Length; i++)
            {
                frame[i] = 0.0;
            }
        }

        public void GetFrame(Span<short> waveform, int start, double scale, double[] frame)
        {
            int offset = start;
            for (int i = 0; i < _window.Length; i++)
            {
                frame[i] = waveform[offset++] * _window[i] * scale;
                if (offset >= waveform.Length) offset = 0;
            }
            for (int i = _window.Length; i < frame.Length; i++)
            {
                frame[i] = 0.0;
            }
        }

        static void ToMagnitude(double[] xr, double[] xi, int N)
        {
            for (int n = 0; n < N; n++)
            {
                xr[n] = Math.Sqrt(xr[n] * xr[n] + xi[n] * xi[n]);
            }
        }

        static void ToSquareMagnitude(double[] xr, double[] xi, int N)
        {
            for (int n = 0; n < N; n++)
            {
                xr[n] = xr[n] * xr[n] + xi[n] * xi[n];
            }
        }

        static double HzToMel(double hz)
        {
            return 2595 * Math.Log10(1 + hz / 700);
        }

        static double MelToHz(double mel)
        {
            return (Math.Pow(10, mel / 2595) - 1) * 700;
        }

        static double[] MakeMelBands(double melMinHz, double melMaxHz, int nMelBanks)
        {
            double melMin = HzToMel(melMinHz);
            double melMax = HzToMel(melMaxHz);
            double[] melBanks = new double[nMelBanks + 2];
            for (int i = 0; i < nMelBanks + 2; i++)
            {
                double mel = (melMax - melMin) * i / (nMelBanks + 1) + melMin;
                melBanks[i] = MelToHz(mel);
            }
            return melBanks;
        }
    }
}
