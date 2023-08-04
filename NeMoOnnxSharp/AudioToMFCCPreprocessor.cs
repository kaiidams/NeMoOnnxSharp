// Copyright (c) Katsuya Iida.  All Rights Reserved.
// See LICENSE in the project root for license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeMoOnnxSharp
{
    internal class AudioToMFCCPreprocessor
    {
        private const double InvShortMaxValue = 1.0 / short.MaxValue;

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

        // AudioToMFCCPreprocessor(
        //   (featurizer): MFCC(
        //     (amplitude_to_DB): AmplitudeToDB()
        //     (MelSpectrogram): MelSpectrogram(
        //       (spectrogram): Spectrogram()
        //       (mel_scale): MelScale()
        //     )
        //   )
        // )
        // {'_target_': 'nemo.collections.asr.modules.AudioToMFCCPreprocessor',
        // 'window_size': 0.025, 'window_stride': 0.01,
        // 'window': 'hann', 'n_mels': 64,
        // 'n_mfcc': 64, 'n_fft': 512}

        public AudioToMFCCPreprocessor(
            int sampleRate = 16000,
            double windowSize = 0.025,
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
            _window = Window.MakeHannWindow(_winLength);
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
            FFT.CFFT(_temp1, _temp2, _fftLength);
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
    }
}
