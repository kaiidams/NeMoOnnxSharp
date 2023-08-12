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
    }
}
