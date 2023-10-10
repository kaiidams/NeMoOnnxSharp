// Copyright (c) Katsuya Iida.  All Rights Reserved.
// See LICENSE in the project root for license information.

using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using System;
using System.Collections.Generic;
using System.Linq;

namespace NeMoOnnxSharp
{
    public sealed class EncDecClassificationModel : ASRModel, IDisposable
    {
        private static readonly string[] SpeechCommandsLabels = new string[]
        {
            "visual", "wow", "learn", "backward", "dog",
            "two", "left", "happy", "nine", "go",
            "up", "bed", "stop", "one", "zero",
            "tree", "seven", "on", "four", "bird",
            "right", "eight", "no", "six", "forward",
            "house", "marvin", "sheila", "five", "off",
            "three", "down", "cat", "follow", "yes"
        };
        private static readonly string[] VADLabels = new string[]
        {
            "background",
            "speech"
        };

        private readonly IAudioPreprocessor<short, float> _preProcessor;
        private readonly InferenceSession _inferSess;
        private readonly int _nMelBands;
        private readonly string[] _labels;

        public IAudioPreprocessor<short, float> PreProcessor => _preProcessor;

        private EncDecClassificationModel(InferenceSession inferSess, bool speechCommands)
        {
            _nMelBands = 64;
            _preProcessor = new AudioToMFCCPreprocessor(
                sampleRate: 16000,
                window: WindowFunction.Hann,
                windowSize: 0.025,
                windowStride: 0.01,
                nFFT: 512,
                //preNormalize: 0.8,
                nMels: 64,
                nMFCC: 64);
            _labels = speechCommands ? SpeechCommandsLabels : VADLabels;
            _inferSess = inferSess;
        }

        public EncDecClassificationModel(string modelPath, bool speechCommands = false)
            : this(new InferenceSession(modelPath), speechCommands)
        {
        }

        public EncDecClassificationModel(byte[] model, bool speechCommands = false)
            : this(new InferenceSession(model), speechCommands)
        {
        }

        public void Dispose()
        {
            _inferSess.Dispose();
        }

        public override string Transcribe(Span<short> inputSignal)
        {
            string text = string.Empty;
            var processedSignal = _preProcessor.GetFeatures(inputSignal);
            processedSignal = TransposeInputSignal(processedSignal, _nMelBands);
            var container = new List<NamedOnnxValue>();
            var audioSignalData = new DenseTensor<float>(
                processedSignal,
                new int[3] { 1, _nMelBands, processedSignal.Length / _nMelBands });
            container.Add(NamedOnnxValue.CreateFromTensor("audio_signal", audioSignalData));
            using (var res = _inferSess.Run(container, new string[] { "logits" }))
            {
                var scoreTensor = res.First();
                long pred = ArgMax(scoreTensor.AsTensor<float>());
                text = _labels[pred];
            }
            return text;
        }

        public float[] Predict(Span<float> processedSignal)
        {
            var transposedProcessedSignal = TransposeInputSignal(processedSignal, _nMelBands);
            var container = new List<NamedOnnxValue>();
            var audioSignalData = new DenseTensor<float>(
                transposedProcessedSignal,
                new int[3] { 1, _nMelBands, transposedProcessedSignal.Length / _nMelBands });
            container.Add(NamedOnnxValue.CreateFromTensor("audio_signal", audioSignalData));
            float[] logits;
            using (var res = _inferSess.Run(container, new string[] { "logits" }))
            {
                var logitsTensor = res.First();
                logits = logitsTensor.AsTensor<float>().ToArray();
            }
            return logits;
        }

        private long ArgMax(Tensor<float> score)
        {
            int k = -1;
            float m = float.MinValue;
            for (int j = 0; j < score.Dimensions[1]; j++)
            {
                if (m < score[0, j])
                {
                    k = j;
                    m = score[0, j];
                }
            }
            return k;
        }
    }
}
