// Copyright (c) Katsuya Iida.  All Rights Reserved.
// See LICENSE in the project root for license information.

using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using NeMoOnnxSharp.TTSTokenizers;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace NeMoOnnxSharp
{
    public class SpectrogramGenerator : IDisposable
    {
        private readonly BaseTokenizer _tokenizer;
        private readonly InferenceSession _inferSess;

        private SpectrogramGenerator(InferenceSession inferSess, string phonemeDict, string heteronyms)
        {
            var g2p = new EnglishG2p(
                phonemeDict: phonemeDict,
                heteronyms: heteronyms,
                phonemeProbability: 1.0);
            _tokenizer = new EnglishPhonemesTokenizer(
                g2p,
                punct: true,
                stresses: true,
                chars: true,
                apostrophe: true,
                padWithSpace: true,
                addBlankAt: BaseTokenizer.AddBlankAt.True);
            _inferSess = inferSess;
        }

        public SpectrogramGenerator(string modelPath, string phonemeDict, string heteronyms)
            : this(new InferenceSession(modelPath), phonemeDict, heteronyms)
        {
        }

        public SpectrogramGenerator(byte[] model, string phonemeDict, string heteronyms)
            : this (new InferenceSession(model), phonemeDict, heteronyms)
        {
        }

        public void Dispose()
        {
            _inferSess.Dispose();
        }

        public int[] Parse(string strText, bool normalize = true)
        {
            var encoded = _tokenizer.Encode(strText);
            return encoded;
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
            using (var res = _inferSess.Run(container, new string[] { "pitch_predicted" }))
            {
                var pitchPredictedData = res.First().AsTensor<float>();
                container[2] = NamedOnnxValue.CreateFromTensor("pitch", pitchPredictedData);
            }

            float[] spec;
            using (var res = _inferSess.Run(container, new string[] { "spect" }))
            {
                var spect = res.First().AsTensor<float>();
                spec = spect.ToArray();
            }
            // text pitch  pace spect num_frames durs_predicted
            // log_durs_predicted pitch_predicted
            return spec;
        }
    }
}
