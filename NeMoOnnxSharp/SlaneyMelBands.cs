// Copyright (c) Katsuya Iida.  All Rights Reserved.
// See LICENSE in the project root for license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeMoOnnxSharp
{
    internal static class SlaneyMelBands
    {
        public static double[] MakeMelBands(double melMinHz, double melMaxHz, int nMelBanks)
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

        private static double HzToMel(double hz)
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

        private static double MelToHz(double mel)
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
    }
}
