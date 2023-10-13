// Copyright (c) Katsuya Iida.  All Rights Reserved.
// See LICENSE in the project root for license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace NeMoOnnxSharp.AudioPreprocessing
{
    public interface IAudioPreprocessor<T1, T2>
    {
        int SampleRate { get; }

        T2[] GetFeatures(Span<T1> input);
    }
}
