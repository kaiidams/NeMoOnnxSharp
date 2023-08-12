// Copyright (c) Katsuya Iida.  All Rights Reserved.
// See LICENSE in the project root for license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeMoOnnxSharp
{
    public static class MelBands
    {
        public static double[] MakeMelBands(double melMinHz, double melMaxHz, int nMelBanks, MelScale melScale)
        {
            if (melScale == MelScale.HTK)
            {
                return HTKMelBands.MakeMelBands(melMinHz, melMaxHz, nMelBanks);
            }
            else if (melScale == MelScale.Slaney)
            {
                return SlaneyMelBands.MakeMelBands(melMinHz, melMaxHz, nMelBanks);
            }
            else
            {
                throw new ArgumentException();
            }
        }

        public static void ToMelSpectrogram(
            Span<double> spec, double[] melBands, int sampleRate,
            int nFFT, int nMels,
            MelNorm norm,
            bool log, double logOffset,
            Span<double> melspec)
        {
            if (!log) throw new NotImplementedException();
            switch (norm)
            {
                case MelNorm.None:
                    ToMelSpectrogramNone(spec, melBands, sampleRate, nFFT, nMels, logOffset, melspec);
                    break;
                case MelNorm.Slaney:
                    ToMelSpectrogramSlaney(spec, melBands, sampleRate, nFFT, nMels, logOffset, melspec);
                    break;
            }
        }

        private static void ToMelSpectrogramNone(
            Span<double> spec, double[] melBands, int sampleRate,
            int nFFT, int nMels, double logOffset,
            Span<double> melspec)
        {
            for (int i = 0; i < nMels; i++)
            {
                double startHz = melBands[i];
                double peakHz = melBands[i + 1];
                double endHz = melBands[i + 2];
                double v = 0.0;
                int j = (int)(startHz * nFFT / sampleRate) + 1;
                while (true)
                {
                    double hz = j * sampleRate / nFFT;
                    if (hz > peakHz)
                        break;
                    double r = (hz - startHz) / (peakHz - startHz);
                    v += spec[j] * r;
                    j++;
                }
                while (true)
                {
                    double hz = j * sampleRate / nFFT;
                    if (hz > endHz)
                        break;
                    double r = (endHz - hz) / (endHz - peakHz);
                    v += spec[j] * r;
                    j++;
                }
                melspec[i] = (float)Math.Log(v + logOffset);
            }
        }

        private static void ToMelSpectrogramSlaney(
            Span<double> spec, double[] melBands, int sampleRate,
            int nFFT, int nMels, double logOffset,
            Span<double> melspec)
        {
            for (int i = 0; i < nMels; i++)
            {
                double startHz = melBands[i];
                double peakHz = melBands[i + 1];
                double endHz = melBands[i + 2];
                double v = 0.0;
                int j = (int)(startHz * nFFT / sampleRate) + 1;
                while (true)
                {
                    double hz = j * sampleRate / nFFT;
                    if (hz > peakHz)
                        break;
                    double r = (hz - startHz) / (peakHz - startHz);
                    v += spec[j] * r * 2 / (endHz - startHz);
                    j++;
                }
                while (true)
                {
                    double hz = j * sampleRate / nFFT;
                    if (hz > endHz)
                        break;
                    double r = (endHz - hz) / (endHz - peakHz);
                    v += spec[j] * r * 2 / (endHz - startHz);
                    j++;
                }
                melspec[i] = (float)Math.Log(v + logOffset);
            }
        }
    }
}
