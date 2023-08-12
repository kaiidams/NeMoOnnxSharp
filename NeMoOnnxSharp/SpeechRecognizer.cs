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
    public class SpeechRecognizer : ISpeechRecognizer
    {
        private const string Vocabulary = " abcdefghijklmnopqrstuvwxyz'_";

        private readonly IAudioPreprocessor<short, float> _processor;
        private readonly CharTokenizer _tokenizer;
        private readonly InferenceSession _inferSess;
        private readonly int _nMelBands;

        private SpeechRecognizer()
        {
            _nMelBands = 64;
            _processor = new AudioToMelSpectrogramPreprocessor(
                sampleRate: 16000,
                window: WindowFunction.Hann,
                windowLength: 400,
                hopLength: 160,
                fftLength: 512,
                preNormalize: 0.0,
                preemph: 0.97,
                center: true,
                nMelBands: 64,
                melMinHz: 0.0,
                melMaxHz: 0.0,
                htk: false,
                melNormalize: MelNorm.Slaney,
                logOffset: Math.Pow(2, -24),
                postNormalize: true,
                postNormalizeOffset: 1e-5);
            _tokenizer = new CharTokenizer(Vocabulary);
        }

        public SpeechRecognizer(string modelPath) : this()
        {
            _inferSess = new InferenceSession(modelPath);
        }

        public SpeechRecognizer(byte[] model) : this()
        {
            _inferSess = new InferenceSession(model);
        }

        public void Dispose()
        {
            _inferSess.Dispose();
        }

        public string Recognize(short[] waveform)
        {
            string text = string.Empty;
            var audioSignal = _processor.GetFeatures(waveform);
            audioSignal = Transpose(audioSignal, _nMelBands);
            var container = new List<NamedOnnxValue>();
            var audioSignalData = new DenseTensor<float>(
                audioSignal,
                new int[3] { 1, _nMelBands, audioSignal.Length / _nMelBands });
            container.Add(NamedOnnxValue.CreateFromTensor("audio_signal", audioSignalData));
            using (var res = _inferSess.Run(container, new string[] { "logprobs" }))
            {
                foreach (var score in res)
                {
                    long[] preds = ArgMax(score.AsTensor<float>());
                    text = _tokenizer.Decode(preds);
                    text = _tokenizer.MergeRepeated(text);
                }
            }
            return text;
        }

        private float[] Transpose(float[] x, int cols)
        {
            var y = new float[x.Length];
            int rows = x.Length / cols;
            for (int i = 0; i < rows; i++)
            {
                for (int j = 0; j < cols; j++)
                {
                    y[j * rows + i] = x[i * cols + j];  
                }
            }
            return y;
        }

        private long[] ArgMax(Tensor<float> score)
        {
            long[] preds = new long[score.Dimensions[1]];
            for (int l = 0; l < preds.Length; l++)
            {
                int k = -1;
                float m = -10000.0f;
                for (int j = 0; j < score.Dimensions[2]; j++)
                {
                    if (m < score[0, l, j])
                    {
                        k = j;
                        m = score[0, l, j];
                    }
                }
                preds[l] = k;
            }

            return preds;
        }
    }
}
