// Copyright (c) Katsuya Iida.  All Rights Reserved.
// See LICENSE in the project root for license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace NeMoOnnxSharp
{
    public abstract class ASRModel
    {
        public abstract string Transcribe(Span<short> inputSignal);

        protected float[] TransposeInputSignal(Span<float> inputSignal, int nFeatures)
        {
            var transposedSignal = new float[inputSignal.Length];
            int rows = inputSignal.Length / nFeatures;
            for (int i = 0; i < rows; i++)
            {
                for (int j = 0; j < nFeatures; j++)
                {
                    transposedSignal[j * rows + i] = inputSignal[i * nFeatures + j];
                }
            }
            return transposedSignal;
        }
    }
}