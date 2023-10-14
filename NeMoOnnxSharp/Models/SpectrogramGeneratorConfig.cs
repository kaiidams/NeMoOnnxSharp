// Copyright (c) Katsuya Iida.  All Rights Reserved.
// See LICENSE in the project root for license information.

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;

namespace NeMoOnnxSharp.Models
{
    public class SpectrogramGeneratorConfig : ModelConfig
    {
        public string? phonemeDictPath;
        public string? heteronymsPath;
        public string? textTokenizer;
    }
}
