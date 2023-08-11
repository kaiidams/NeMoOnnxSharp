// Copyright (c) Katsuya Iida.  All Rights Reserved.
// See LICENSE in the project root for license information.

using System;

namespace NeMoOnnxSharp
{
    public interface IFrameTransform<T1, T2>
    {
        int InputLength { get; }
        int OutputLength { get; }
        void Transform(Span<T1> input, Span<T2> output);
    }
}