﻿// Copyright (c) Katsuya Iida.  All Rights Reserved.
// See LICENSE in the project root for license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace NeMoOnnxSharp
{
    public interface IAudioBuffer<T, S>
    {
        public int OutputCount { get; }
        public S[] OutputBuffer { get; }
        public int Write(T[] waveform, int offset, int count);
        public int Write(Span<T> waveform);
        public void ConsumeOutput(int count);
    }
}
