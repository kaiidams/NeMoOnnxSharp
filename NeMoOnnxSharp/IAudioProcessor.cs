// Copyright (c) Katsuya Iida.  All Rights Reserved.
// See LICENSE in the project root for license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace NeMoOnnxSharp
{
    public interface IAudioProcessor<T1, T2>
    {
        T2[] Process(Span<T1> input);
    }
}
