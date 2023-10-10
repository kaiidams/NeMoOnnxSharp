// Copyright (c) Katsuya Iida.  All Rights Reserved.
// See LICENSE in the project root for license information.

using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using System;
using System.Collections.Generic;
using System.Linq;

namespace NeMoOnnxSharp
{
    public sealed class Vocoder : IDisposable
    {
        private readonly InferenceSession _inferSess;
        private readonly int _nfilt;
        private readonly int _sampleRate;
        private Vocoder(InferenceSession inferSess)
        {
            _inferSess = inferSess;
            _nfilt = 80;
            _sampleRate = 22050;
        }

        public Vocoder(string modelPath)
            : this(new InferenceSession(modelPath))
        {
        }

        public Vocoder(byte[] model)
            : this(new InferenceSession(model))
        {
        }

        public int SampleRate { get { return _sampleRate; } }

        public void Dispose()
        {
            _inferSess.Dispose();
        }

        public short[] ConvertSpectrogramToAudio(float[] spec)
        {
            var container = new List<NamedOnnxValue>();
            var specData = new DenseTensor<float>(
                spec,
                new int[3] { 1, _nfilt, spec.Length / _nfilt });
            container.Add(NamedOnnxValue.CreateFromTensor("spec", specData));
            float[] audio;
            using (var res = _inferSess.Run(container, new string[] { "audio" }))
            {
                var audioTensor = res.First().AsTensor<float>();
                audio = audioTensor.ToArray();
            }
            return audio.Select(x => (short)(x * short.MaxValue)).ToArray();
        }
    }
}
