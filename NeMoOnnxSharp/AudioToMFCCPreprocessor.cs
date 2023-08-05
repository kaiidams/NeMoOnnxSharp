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
        private readonly int _nMels;
        private readonly int _nMFCC;
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
            double windowSize = 0.02,
            double windowStride = 0.01,
            int nFFT = 0,
            int nMels = 64, double fMin = 0.0, double fMax = 0.0,
            int nMFCC = 64,
            double preemph = 0.0)
        {
            _sampleRate = sampleRate;
            if (fMax == 0.0)
            {
                fMax = sampleRate / 2;
            }
            _winLength = (int)(sampleRate * windowSize); // 320
            _hopWidth = (int)(sampleRate * windowStride); // 160
            if (nFFT == 0)
            {
                nFFT = (int)Math.Pow(2, Math.Ceiling(Math.Log2(_winLength)));
            }
            _window = Window.MakeWindow("hann", _winLength);
            _melBands = MakeMelBands(fMin, fMax, nMels);
            _nMFCC = nMFCC;
            _temp1 = new double[nFFT];
            _temp2 = new double[nFFT];
            _fftLength = nFFT;
            _nMels = nMels;
            _preemph = preemph;
            _logOffset = Math.Pow(2, -24);
            _stdOffset = 1e-5;
        }

        public float[] Process(short[] waveform)
        {
            int audioSignalLength = waveform.Length / _hopWidth + 1;
            float[] audioSignal = new float[_nMels * audioSignalLength]; 
            for (int i = 0; i < audioSignalLength; i++)
            {
                MFCC(
                    waveform, _hopWidth * i, 
                    audioSignal, i, audioSignalLength);
            }
            return audioSignal;
        }

        private void MFCC(
            short[] waveform, int waveformPos, 
            float[] mfcc, int melspecOffset, int melspecStride)
        {
            GetFrame(waveform, waveformPos, InvShortMaxValue, _temp1);
            FFT.CFFT(_temp1, _temp2, _fftLength);
            ToSquareMagnitude(_temp2, _temp1, _fftLength);
            ToMelSpec(_temp2, _temp1);
            ToMFCC(_temp1, _temp2);
            for (int i = 0; i < _nMFCC; i++) {
                mfcc[melspecOffset + i * melspecStride] = (float)_temp2[i];
            }
        }

        private void ToMelSpec(
            double[] spec,
            double[] melspec)
        {
            for (int i = 0; i < _nMels; i++)
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
                melspec[i] = (float)Math.Log(v + _logOffset);
            }
        }

        private void ToMFCC(double[] melspec, double[] mfcc)
        {
            FFT.DCT2(melspec, mfcc, _nMFCC);
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
                    if (_preemph > 0)
                    {
                        if (k >= 0 && k < waveform.Length) v -= _preemph * waveform[k];
                    }
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

        static double[] MakeMelBands(double fMin, double fMax, int nMelBanks)
        {
            double melMin = HzToMel(fMin);
            double melMax = HzToMel(fMax);
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
