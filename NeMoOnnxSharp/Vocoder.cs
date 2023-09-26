// Copyright (c) Katsuya Iida.  All Rights Reserved.
// See LICENSE in the project root for license information.

using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using NeMoOnnxSharp.TextTokenizers;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace NeMoOnnxSharp
{
    public class Vocoder : IDisposable
    {
        private readonly InferenceSession _inferSess;

        private Vocoder(InferenceSession inferSess)
        {
            _inferSess = inferSess;
        }

        public Vocoder(string modelPath)
            : this(new InferenceSession(modelPath))
        {
        }

        public Vocoder(byte[] model)
            : this(new InferenceSession(model))
        {
        }

        public void Dispose()
        {
            _inferSess.Dispose();
        }

        public float[] ConvertSpectrogramToAudio(float[] spec)
        {
            var container = new List<NamedOnnxValue>();
            var specData = new DenseTensor<float>(
                spec,
                new int[3] { 1, 80, spec.Length / 80 });
            container.Add(NamedOnnxValue.CreateFromTensor("spec", specData));
            float[] audio;
            using (var res = _inferSess.Run(container, new string[] { "audio" }))
            {
                var audioTensor = res.First().AsTensor<float>();
                audio = audioTensor.ToArray();
            }
            return audio;
        }
    }
}
