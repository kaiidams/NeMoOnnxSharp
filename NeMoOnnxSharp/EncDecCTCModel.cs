// Copyright (c) Katsuya Iida.  All Rights Reserved.
// See LICENSE in the project root for license information.

using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Text.RegularExpressions;
using System.Threading.Tasks;

namespace NeMoOnnxSharp
{
    public class EncDecCTCModel : ASRModel, IDisposable
    {
        private const string Vocabulary = " abcdefghijklmnopqrstuvwxyz'_";

        private readonly IAudioPreprocessor<short, float> _processor;
        private readonly CharTokenizer _tokenizer;
        private readonly InferenceSession _inferSess;
        private readonly int _features;

        private EncDecCTCModel(InferenceSession inferSess)
        {
            _features = 64;
            _processor = new AudioToMelSpectrogramPreprocessor(
                sampleRate: 16000,
                window: WindowFunction.Hann,
                windowSize: 0.02,
                windowStride: 0.01,
                nFFT: 512,
                features: _features);
            _tokenizer = new CharTokenizer(Vocabulary);
            _inferSess = inferSess;
        }

        public EncDecCTCModel(string modelPath)
            : this(new InferenceSession(modelPath))
        {
        }

        public EncDecCTCModel(byte[] model)
            : this(new InferenceSession(model))
        {
        }

        public void Dispose()
        {
            _inferSess.Dispose();
        }

        public override string Transcribe(short[] inputSignal)
        {
            string text = string.Empty;
            var processedSignal = _processor.GetFeatures(inputSignal);
            processedSignal = TransposeInputSignal(processedSignal, _features);
            var container = new List<NamedOnnxValue>();
            var audioSignalData = new DenseTensor<float>(
                processedSignal,
                new int[3] { 1, _features, processedSignal.Length / _features });
            container.Add(NamedOnnxValue.CreateFromTensor("audio_signal", audioSignalData));
            using (var res = _inferSess.Run(container, new string[] { "logprobs" }))
            {
                var logprobs = res.First();
                long[] preds = ArgMax(logprobs.AsTensor<float>());
                text = _tokenizer.Decode(preds);
                text = _tokenizer.MergeRepeated(text);
            }
            return text;
        }

        private long[] ArgMax(Tensor<float> logprobs)
        {
            long[] preds = new long[logprobs.Dimensions[1]];
            for (int l = 0; l < preds.Length; l++)
            {
                int k = -1;
                float m = float.MinValue;
                for (int j = 0; j < logprobs.Dimensions[2]; j++)
                {
                    if (m < logprobs[0, l, j])
                    {
                        k = j;
                        m = logprobs[0, l, j];
                    }
                }
                preds[l] = k;
            }

            return preds;
        }
    }
}
