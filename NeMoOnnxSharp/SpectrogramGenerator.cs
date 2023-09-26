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

        public int[] Parse(string strText, bool normalize = true)
        {
            var encoded = _tokenizer.Encode(strText);
            return encoded;
        }

        public float[] GenerateSpectrogram(int[] tokens, double pace = 1.0)
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
                tokens.Select(p => (long)p).ToArray(),
                new int[2] { 1, tokens.Length });
            container.Add(NamedOnnxValue.CreateFromTensor("text", textData));
            var paceData = new DenseTensor<float>(
                new float[] { (float)pace },
                new int[2] { 1, 1 });
            container.Add(NamedOnnxValue.CreateFromTensor("pace", paceData));
            var pitchData = new DenseTensor<float>(
                Enumerable.Range(0, tokens.Length).Select(i => (float)Pitch[i]).ToArray(),
                new int[2] { 1, tokens.Length });
            container.Add(NamedOnnxValue.CreateFromTensor("pitch", pitchData));
            float[] spec;
            using (var res = _inferSess.Run(container, new string[] { "spect" }))
            {
                var spect = res.First().AsTensor<float>();
                spec = spect.ToArray();
            }
            // _inferSess.InputMetadata
            // text pitch  pace spect num_frames durs_predicted
            // log_durs_predicted pitch_predicted
            return spec;
        }

        private static readonly double[] Pitch = {
             0.0033,  0.5483,  0.5461,  0.4986,  0.0033, -0.0690, -0.0242, -0.6999,
            -0.8962, -0.7284, -0.4535, -0.3336, -0.1561,  0.0033,  0.0033,  0.0033,
             0.0648,  0.5356,  0.3349, -0.2566, -0.0302, -0.3782, -0.7125, -0.6549,
            -0.1496, -0.4956, -0.5527, -0.6449, -0.5915, -0.3172, -0.0253,  0.2901,
             0.2966,  0.1652,  0.0667,  0.7135,  0.5498,  0.2913, -0.0606, -0.0361,
            -0.6641, -0.2825,  0.0033,  0.0033, -0.0485,  0.4449,  0.2335,  0.0774,
             0.0359,  0.5133,  0.9142,  0.8634,  0.4422, -0.1363, -0.3429, -0.2477,
            -0.1410, -0.0121,  0.0033,  0.2908,  0.4933,  0.6256,  0.1013, -0.2046,
            -0.3443, -0.1680, -0.1048, -0.0646, -0.0322,  0.0269,  0.0881, -0.0237,
            -0.0855,  0.0207, -0.0691, -0.1584, -0.4479, -0.6376, -0.7185, -0.0590,
            -0.6801, -0.8423, -0.9083, -0.8783, -0.5256, -0.1679,  0.4350,  1.0480,
             1.1230,  0.4760, -0.1293, -0.1669,  0.0023, -0.3808, -0.6546, -0.7691,
            -0.5113, -0.3765, -0.4005, -0.1967,  0.3357,  0.3811,  0.1452, -0.1938,
            -0.1909, -0.0684,  0.2460,  0.0568,  0.0428, -0.0156, -0.2643, -0.7842,
            -0.2525, -0.1598, -0.2141, -0.0720, -0.0664,  0.1902, -0.3387, -0.4038,
            -0.1317,  0.0217,  0.2929,  0.3183,  0.1739,  0.2973,  0.4624,  0.3315,
            -0.2851, -0.1798, -0.2390,  0.2239,  0.3466,  0.5441,  0.5055,  0.6241,
             0.3064,  0.1307, -0.0866, -0.2154, -0.3030, -0.3270,  0.5362,  0.4793,
             0.5178, -0.1232, -0.4456, -0.3816, -0.2474, -0.1951, -0.3448, -0.3251,
             0.0462
        };
    }
}
