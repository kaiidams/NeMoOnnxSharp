// Copyright (c) Katsuya Iida.  All Rights Reserved.
// See LICENSE in the project root for license information.

using System;
using System.Collections.Generic;
using System.Text;

namespace NeMoOnnxSharp.AudioPreprocessing
{
    public enum WindowFunction
    {
        Bartlett,
        Blackman,
        Hamming,
        Hann,
        Kaiser
    }
}
