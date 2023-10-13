// Copyright (c) Katsuya Iida.  All Rights Reserved.
// See LICENSE in the project root for license information.

using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using System;
using System.Collections.Generic;
using System.Linq;

namespace NeMoOnnxSharp
{
    public class EncDecClassificationConfig : ModelConfig
    {
        public static readonly string[] SpeechCommandsLabels = new string[]
        {
            "visual", "wow", "learn", "backward", "dog",
            "two", "left", "happy", "nine", "go",
            "up", "bed", "stop", "one", "zero",
            "tree", "seven", "on", "four", "bird",
            "right", "eight", "no", "six", "forward",
            "house", "marvin", "sheila", "five", "off",
            "three", "down", "cat", "follow", "yes"
        };
        public static readonly string[] VADLabels = new string[]
        {
            "background",
            "speech"
        };

        public string[]? labels;
    }
}
