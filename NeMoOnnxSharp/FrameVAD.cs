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
        private readonly AudioProcessor _processor;
        private readonly InferenceSession _inferSess;
        private readonly int _nMelBands;
        private readonly string[] _labels;

        private FrameVAD()
        {
            _nMelBands = 64;
            _processor = new AudioProcessor(
                sampleRate: 16000,
                window: WindowFunction.Hann,
                windowLength: 400,
                hopLength: 160,
                fftLength: 512,
                //preNormalize: 0.8,
                preemph: 0.0,
                center: true,
                nMelBands: 64,
                melMinHz: 0.0,
                melMaxHz: 0.0,
                htk: true,
                melNormalize: MelNormalizeType.None,
                nMFCC: 64,
                logOffset: 1e-6,
                postNormalize: false);
            _labels = new string[]
            {
                "background",
                "speech"
            };
        }

        public FrameVAD(string modelPath) : this()
        {
            _inferSess = new InferenceSession(modelPath);
        }

        public FrameVAD(byte[] model) : this()
        {
            _inferSess = new InferenceSession(model);
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
                var processedSignal = _processor.MFCC(waveform2);
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

        public string TranscribeStep(float[] processedSignal)
        {
            processedSignal = Transpose(processedSignal, _nMelBands);
            var container = new List<NamedOnnxValue>();
            var audioSignalData = new DenseTensor<float>(
                processedSignal,
                new int[3] { 1, _nMelBands, processedSignal.Length / _nMelBands });
            container.Add(NamedOnnxValue.CreateFromTensor("audio_signal", audioSignalData));
            string text;
            using (var res = _inferSess.Run(container, new string[] { "logits" }))
            {
                var scoreTensor = res.First();
                float[] scores = scoreTensor.AsTensor<float>().ToArray();
                int score = (int)(10 / (1 + Math.Exp(scores[0] - scores[1])));
                text = (scores[0] > scores[1]) ? _labels[0] : _labels[1];
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
    }
}
