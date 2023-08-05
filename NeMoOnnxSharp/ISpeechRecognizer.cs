// Copyright (c) Katsuya Iida.  All Rights Reserved.
// See LICENSE in the project root for license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace NeMoOnnxSharp
{
    public interface ISpeechRecognizer : IDisposable
    {
        string Recognize(short[] waveform);
    }
}
