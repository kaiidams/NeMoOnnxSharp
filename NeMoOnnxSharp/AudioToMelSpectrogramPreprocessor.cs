﻿using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeMoOnnxSharp
{
    internal class AudioToMelSpectrogramPreprocessor
    {
        private const double InvShortMaxValue = 1.0 / short.MaxValue;

        private static double[] MakeHannWindow(int windowLength)
        {
            double[] window = new double[windowLength];
            for (int i = 0; i < windowLength; i++)
            {
                window[i] = 0.5 * (1 - Math.Cos(2 * Math.PI * i / (windowLength - 1)));
            }
            return window;
        }

        private readonly int _winLength;
        private readonly int _hopWidth;
        private readonly double[] _window;
        private readonly double[] _melBands;
        private readonly double[] _temp1;
        private readonly double[] _temp2;
        private readonly int _fftLength;
        private readonly int _nMelBands;
        private readonly double _sampleRate;
        private readonly double _logOffset;
        private readonly double _stdOffset;
        private readonly double _preemph;

        public AudioToMelSpectrogramPreprocessor(
            int sampleRate = 16000,
            double windowSize = 0.02,
            double windowStride = 0.01,
            int stftLength = 512,
            int nMelBands = 64, double melMinHz = 0.0, double melMaxHz = 0.0,
            double preemph = 0.97)
        {
            if (melMaxHz == 0.0)
            {
                melMaxHz = sampleRate / 2;
            }
            _sampleRate = sampleRate;
            _winLength = (int)(sampleRate * windowSize); // 320
            _hopWidth = (int)(sampleRate * windowStride); // 160
            _window = MakeHannWindow(_winLength);
            _melBands = MakeMelBands(melMinHz, melMaxHz, nMelBands);
            _temp1 = new double[stftLength];
            _temp2 = new double[stftLength];
            _fftLength = stftLength;
            _nMelBands = nMelBands;
            _preemph = preemph;
            _logOffset = Math.Pow(2, -24);
            _stdOffset = 1e-5;
        }

        public float[] Process(short[] waveform)
        {
            int audioSignalLength = waveform.Length / _hopWidth + 1;
            float[] audioSignal = new float[_nMelBands * audioSignalLength]; 
            for (int i = 0; i < audioSignalLength; i++)
            {
                MelSpectrogram(
                    waveform, _hopWidth * i, 
                    audioSignal, i, audioSignalLength);
            }
            Normalize(audioSignal, audioSignalLength);
            return audioSignal;
        }

        private void MelSpectrogram(
            short[] waveform, int waveformPos, 
            float[] melspec, int melspecOffset, int melspecStride)
        {
            GetFrame(waveform, waveformPos, InvShortMaxValue, _temp1);
            CFFT(_temp1, _temp2, _fftLength);
            ToSquareMagnitude(_temp2, _temp1, _fftLength);
            ToMelSpec(_temp2, melspec, melspecOffset, melspecStride);
        }

        private void ToMelSpec(
            double[] spec,
            float[] melspec, int melspecOffset, int melspecStride)
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
                    v += spec[j] * r * 2 / (endHz - startHz);
                    j++;
                }
                while (true)
                {
                    double hz = j * _sampleRate / _fftLength;
                    if (hz > endHz)
                        break;
                    double r = (endHz - hz) / (endHz - peakHz);
                    v += spec[j] * r * 2 / (endHz - startHz);
                    j++;
                }
                melspec[melspecOffset + melspecStride * i] = (float)Math.Log(v + _logOffset);
            }
        }

        private void Normalize(float[] melspec, int melspecStride)
        {
            for (int i = 0; i < _nMelBands; i++)
            {
                double sum = 0;
                for (int j = 0; j < melspecStride; j++)
                {
                    double v = melspec[melspecStride * i + j];
                    sum += v;
                }
                float mean = (float)(sum / melspecStride);
                sum = 0;
                for (int j = 0; j < melspecStride; j++)
                {
                    double v = melspec[melspecStride * i + j] - mean;
                    sum += v * v;
                }
                double std = Math.Sqrt(sum / melspecStride);
                float invStd = (float)(1.0 / (_stdOffset + std));

                for (int j = 0; j < melspecStride; j++)
                {
                    float v = melspec[melspecStride * i + j];
                    melspec[melspecStride * i + j] = (v - mean) * invStd;
                }
            }
        }

        private void GetFrame(short[] waveform, int waveformPos, double scale, double[] frame)
        {
            int winOffset = (_winLength - _fftLength) / 2;
            int waveformOffset = waveformPos - _fftLength / 2;
            for (int i = 0; i < _fftLength; i++)
            {
                int j = i + winOffset;
                if (j >= 0 && j < _winLength)
                {
                    int k = i + waveformOffset;
                    double v = (k >= 0 && k < waveform.Length) ? waveform[k] : 0;
                    k--;
                    if (k >= 0 && k < waveform.Length) v -= _preemph * waveform[k];
                    frame[i] = scale * v * _window[j];
                }
                else
                {
                    frame[i] = 0;
                }
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
            const double minLogHz = 1000.0;  // beginning of log region in Hz
            const double linearMelHz = 200.0 / 3;
            double mel;
            if (hz >= minLogHz)
            {
                // Log region
                const double minLogMel = minLogHz / linearMelHz;
                double logStep = Math.Log(6.4) / 27.0;
                mel = minLogMel + Math.Log(hz / minLogHz) / logStep;
            }
            else
            {
                // Linear region
                mel = hz / linearMelHz;
            }

            return mel;
        }

        static double MelToHz(double mel)
        {
            const double minLogHz = 1000.0;  // beginning of log region in Hz
            const double linearMelHz = 200.0 / 3;
            const double minLogMel = minLogHz / linearMelHz;  // same (Mels)
            double freq;


            if (mel >= minLogMel)
            {
                // Log region
                double logStep = Math.Log(6.4) / 27.0;
                freq = minLogHz * Math.Exp(logStep * (mel - minLogMel));
            }
            else
            {
                // Linear region
                freq = linearMelHz * mel;
            }

            return freq;
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

        static int SwapIndex(int i)
        {
            return (i >> 8) & 0x01
                 | (i >> 6) & 0x02
                 | (i >> 4) & 0x04
                 | (i >> 2) & 0x08
                 | (i) & 0x10
                 | (i << 2) & 0x20
                 | (i << 4) & 0x40
                 | (i << 6) & 0x80
                 | (i << 8) & 0x100;
        }

        public static void CFFT(double[] xr, double[] xi, int N)
        {
            double[] t = xi;
            xi = xr;
            xr = t;
            for (int i = 0; i < N; i++)
            {
                xr[i] = xi[SwapIndex(i)];
            }
            for (int i = 0; i < N; i++)
            {
                xi[i] = 0.0;
            }
            for (int n = 1; n < N; n *= 2)
            {
                for (int j = 0; j < N; j += n * 2)
                {
                    for (int k = 0; k < n; k++)
                    {
                        double ar = Math.Cos(-Math.PI * k / n);
                        double ai = Math.Sin(-Math.PI * k / n);
                        double er = xr[j + k];
                        double ei = xi[j + k];
                        double or = xr[j + k + n];
                        double oi = xi[j + k + n];
                        double aor = ar * or - ai * oi;
                        double aoi = ai * or + ar * oi;
                        xr[j + k] = er + aor;
                        xi[j + k] = ei + aoi;
                        xr[j + k + n] = er - aor;
                        xi[j + k + n] = ei - aoi;
                    }
                }
            }
        }
    }
}
