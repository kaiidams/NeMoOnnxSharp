// Copyright (c) Katsuya Iida.  All Rights Reserved.
// See LICENSE in the project root for license information.

using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using NeMoOnnxSharp.TextTokenizers;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using static System.Net.Mime.MediaTypeNames;

namespace NeMoOnnxSharp
{
    public class SpectrogramGenerator : IDisposable
    {
        private readonly BaseTokenizer _tokenizer;
        private readonly InferenceSession _inferSess;

        private SpectrogramGenerator(InferenceSession inferSess)
        {
            var g2p = new EnglishG2p(
                phonemeDict: "../../scripts/tts_dataset_files/cmudict-0.7b_nv22.10",
                heteronyms: "../../scripts/tts_dataset_files/heteronyms-052722",
                phonemeProbability: 0.5);
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

        public SpectrogramGenerator(string modelPath)
            : this(new InferenceSession(modelPath))
        {
        }

        public SpectrogramGenerator(byte[] model)
            : this(new InferenceSession(model))
        {
        }

        public void Dispose()
        {
            _inferSess.Dispose();
        }

        public int[] Parse(string text)
        {
            var encoded = _tokenizer.Encode("Hello World!");
            return encoded;
        }

        public float[] GenerateSpectrogram(int[] tokens)
        {
            foreach (var x in _inferSess.InputNames)
            {
                Console.WriteLine(x);
            }
            foreach (var x in _inferSess.OutputNames)
            {
                Console.WriteLine(x);
            }
            var container = new List<NamedOnnxValue>();
            var textData = new DenseTensor<long>(
                new long[] { 0, 1, 2, 3, 4, 5, 6 },
                new int[2] { 1, 7 });
            container.Add(NamedOnnxValue.CreateFromTensor("text", textData));
            var paceData = new DenseTensor<float>(
                new float[] { 1.0f },
                new int[2] { 1, 1 });
            container.Add(NamedOnnxValue.CreateFromTensor("pace", paceData));
            var pitchData = new DenseTensor<float>(
                new float[] { 0, 1, 2, 3, 4, 5, 6 },
                new int[2] { 1, 7 });
            container.Add(NamedOnnxValue.CreateFromTensor("pitch", pitchData));
            using (var res = _inferSess.Run(container, new string[] { "spect" }))
            {
                var spect = res.First();
            }
            // _inferSess.InputMetadata
            // text pitch  pace spect num_frames durs_predicted
            // log_durs_predicted pitch_predicted
            return Array.Empty<float>();
        }
    }
}
