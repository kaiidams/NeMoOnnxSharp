// Copyright (c) Katsuya Iida.  All Rights Reserved.
// See LICENSE in the project root for license information.

using System;

namespace NeMoOnnxSharp
{
    public class MFCCAudioProcessor : AudioProcessor
    {
        public MFCCAudioProcessor(
            int sampleRate = 16000,
            WindowFunction window = WindowFunction.Hann,
            int windowLength = 0,
            int hopLength = 512,
            int fftLength = 2048,
            double preNormalize = 0.0,
            double preemph = 0.0,
            bool center = true,
            int nMelBands = 128,
            double melMinHz = 0.0,
            double melMaxHz = 0.0,
            bool htk = false,
            MelNormalizeType melNormalize = MelNormalizeType.Slaney,
            int power = 2,
            bool logOutput = true,
            double logOffset = 1e-6,
            int nMFCC = 128,
            bool postNormalize = false,
            double postNormalizeOffset = 1e-5) : base(
                sampleRate, window, windowLength, hopLength, fftLength, preNormalize,
                preemph, center, nMelBands, melMinHz, melMaxHz, htk, melNormalize,
                power, logOutput, logOffset, nMFCC, postNormalize, postNormalizeOffset)
        {
        }

        public override void ProcessFrame(Span<short> input, double scale, Span<float> output)
        {
            var x = new float[output.Length];
            MFCCStep(input.ToArray(), 0, scale, x, 0);
            x.AsSpan().CopyTo(output);
        }
    }
}
