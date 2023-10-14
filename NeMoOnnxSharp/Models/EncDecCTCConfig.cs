﻿// Copyright (c) Katsuya Iida.  All Rights Reserved.
// See LICENSE in the project root for license information.

using System;
using System.Collections.Generic;
using System.Linq;

namespace NeMoOnnxSharp.Models
{
    public class EncDecCTCConfig : ModelConfig
    {
        public const string EnglishVocabulary = " abcdefghijklmnopqrstuvwxyz'_";
        public const string GermanVocabulary = " abcdefghijklmnopqrstuvwxyzäöüß_";

        public string? vocabulary;
    }
}
