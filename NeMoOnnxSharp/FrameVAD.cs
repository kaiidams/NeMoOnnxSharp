// Copyright (c) Katsuya Iida.  All Rights Reserved.
// See LICENSE in the project root for license information.

using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace NeMoOnnxSharp
{
    public class FrameVAD : IDisposable
    {
        private readonly IAudioPreprocessor<short, float> _processor;
        private readonly InferenceSession _inferSess;
        private readonly int _nMelBands;
        private readonly string[] _labels;

        private FrameVAD(InferenceSession inferSess)
        {
            _nMelBands = 64;
            _processor = new AudioToMFCCPreprocessor(
                sampleRate: 16000,
                window: WindowFunction.Hann,
                windowSize: 0.025,
                windowStride: 0.01,
                nFFT: 512,
                //preNormalize: 0.8,
                nMels: 64,
                nMFCC: 64);
            _labels = new string[]
            {
                "background",
                "speech"
            };
            _inferSess = inferSess;
        }

        public FrameVAD(string modelPath) : this(new InferenceSession(modelPath))
        {
        }

        public FrameVAD(byte[] model) : this(new InferenceSession(model))
        {
        }

        public void Dispose()
        {
            _inferSess.Dispose();
        }

        public string Transcribe(short[] waveform)
        {
            string text = string.Empty;
            int windowLength = (int)(16000 * 0.5);
            int stepSize = (int)(16000 * 0.025);
            for (int j = 0; j + windowLength < waveform.Length; j += stepSize)
            {
                var waveform2 = waveform.AsSpan(j, windowLength).ToArray();
                var processedSignal = _processor.GetFeatures(waveform2);
                processedSignal = Transpose(processedSignal, _nMelBands);
                var container = new List<NamedOnnxValue>();
                var audioSignalData = new DenseTensor<float>(
                    processedSignal,
                    new int[3] { 1, _nMelBands, processedSignal.Length / _nMelBands });
                container.Add(NamedOnnxValue.CreateFromTensor("audio_signal", audioSignalData));
                using (var res = _inferSess.Run(container, new string[] { "logits" }))
                {
                    var scoreTensor = res.First();
                    float[] scores = scoreTensor.AsTensor<float>().ToArray();
                    int score = (int)(10 / (1 + Math.Exp(scores[0] - scores[1])));
                    text += (scores[0] > scores[1]) ? "." : "X";
                }
            }
            return text;
        }

        public double PredictStep(Span<float> processedSignal)
        {
            var transposedProcessedSignal = Transpose(processedSignal, _nMelBands);
            var container = new List<NamedOnnxValue>();
            var audioSignalData = new DenseTensor<float>(
                transposedProcessedSignal,
                new int[3] { 1, _nMelBands, transposedProcessedSignal.Length / _nMelBands });
            container.Add(NamedOnnxValue.CreateFromTensor("audio_signal", audioSignalData));
            double score;
            using (var res = _inferSess.Run(container, new string[] { "logits" }))
            {
                var scoreTensor = res.First();
                float[] scores = scoreTensor.AsTensor<float>().ToArray();
                score = 1.0 / (1.0 + Math.Exp(scores[0] - scores[1]));
            }
            return score;
        }

        private float[] Transpose(Span<float> x, int cols)
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
    }
}
