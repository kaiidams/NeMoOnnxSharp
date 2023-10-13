// Copyright (c) Katsuya Iida.  All Rights Reserved.
// See LICENSE in the project root for license information.

using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using NeMoOnnxSharp.TTSTokenizers;
using System;
using System.Collections.Generic;
using System.Linq;

namespace NeMoOnnxSharp
{
    public sealed class SpectrogramGenerator : Model, IDisposable
    {
        private readonly BaseTokenizer _tokenizer;

        public SpectrogramGenerator(SpectrogramGeneratorConfig config) : base(config)
        {
            if (config.phonemeDictPath == null) throw new ArgumentNullException();
            if (config.heteronymsPath == null) throw new ArgumentNullException();
            var g2p = new EnglishG2p(
                phonemeDict: config.phonemeDictPath,
                heteronyms: config.heteronymsPath,
                phonemeProbability: 1.0);
            _tokenizer = new EnglishPhonemesTokenizer(
                g2p,
                punct: true,
                stresses: true,
                chars: true,
                apostrophe: true,
                padWithSpace: true,
                addBlankAt: BaseTokenizer.AddBlankAt.True);
        }

        public void Dispose()
        {
            _inferSess.Dispose();
        }

        public int[] Parse(string strInput, bool normalize = true)
        {
            if (normalize)
            {
                strInput = _NormalizeText(strInput);
            }
            var encoded = _tokenizer.Encode(strInput);
            return encoded;
        }

        private string _NormalizeText(string strInput)
        {
            return strInput;
        }

        public float[] GenerateSpectrogram(int[] tokens, double pace = 1.0)
        {
            var container = new List<NamedOnnxValue>();
            var textData = new DenseTensor<long>(
                tokens.Select(p => (long)p).ToArray(),
                new int[2] { 1, tokens.Length });
            container.Add(NamedOnnxValue.CreateFromTensor("text", textData));
            var paceData = new DenseTensor<float>(
                new float[] { (float)pace },
                new int[2] { 1, 1 });
            container.Add(NamedOnnxValue.CreateFromTensor("pace", paceData));
            var pitchData = new DenseTensor<float>(
                Enumerable.Range(0, tokens.Length).Select(i => 0.0f).ToArray(),
                new int[2] { 1, tokens.Length });
            container.Add(NamedOnnxValue.CreateFromTensor("pitch", pitchData));
            float[] spec;
            using (var res = _inferSess.Run(container, new string[] { "spect" }))
            {
                var pitchPredictedData = res.First().AsTensor<float>();
                spec = pitchPredictedData.ToArray();
            }
            return spec;
        }
    }
}
