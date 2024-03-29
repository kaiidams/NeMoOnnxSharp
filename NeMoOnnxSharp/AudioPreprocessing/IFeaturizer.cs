﻿// Copyright (c) Katsuya Iida.  All Rights Reserved.
// See LICENSE in the project root for license information.

using System;

namespace NeMoOnnxSharp.AudioPreprocessing
{
    public interface IFeaturizer<T1, T2>
    {
        int SampleRate { get; }
        int InputLength { get; }
        int OutputLength { get; }
        void GetFeatures(Span<T1> input, Span<T2> output);
    }
}