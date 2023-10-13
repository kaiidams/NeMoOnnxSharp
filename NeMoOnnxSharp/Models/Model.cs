// Copyright (c) Katsuya Iida.  All Rights Reserved.
// See LICENSE in the project root for license information.

using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;

namespace NeMoOnnxSharp.Models
{
    public abstract class Model
    {
        protected readonly InferenceSession _inferSess;

        protected Model(ModelConfig config)
        {
            if (config.model != null)
            {
                _inferSess = new InferenceSession(config.model);
            }
            else if (config.modelPath != null)
            {
                _inferSess = new InferenceSession(config.modelPath);
            }
            else
            {
                throw new InvalidDataException();
            }
        }
    }
}
