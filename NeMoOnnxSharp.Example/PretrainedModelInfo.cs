// Copyright (c) Katsuya Iida.  All Rights Reserved.
// See LICENSE in the project root for license information.

using System;

namespace NeMoOnnxSharp.Example
{
    internal class PretrainedModelInfo
    {
        private static PretrainedModelInfo[]? _modelList = null;

        public static PretrainedModelInfo[] ModelList
        {
            get
            {
                if (_modelList == null)
                {
                    _modelList = CreateModelList();
                }
                return _modelList;
            }
        }

        private static PretrainedModelInfo[] CreateModelList()
        {
            return new PretrainedModelInfo[]
            {
                new PretrainedModelInfo(
                    "stt_en_quartznet15x5",
                    "https://github.com/kaiidams/NeMoOnnxSharp/releases/download/v1.1/stt_en_quartznet15x5.onnx",
                    "dde27f0528e92c05f7bc220a9be4a7bb99927da0a3a25db8f2f861e3559da90d"
                ),
                new PretrainedModelInfo(
                    "QuartzNet15x5Base-En",
                    "https://github.com/kaiidams/NeMoOnnxSharp/releases/download/v1.1/QuartzNet15x5Base-En.onnx",
                    "ee1b72102fd0c5422d088e80f929dbdee7e889d256a4ce1e412cd49916823695"
                ),
                new PretrainedModelInfo(
                    "vad_marblenet",
                    "https://github.com/kaiidams/NeMoOnnxSharp/releases/download/v1.1/vad_marblenet.onnx",
                    "edaf8a7bb62e4335f97aa70d1a447ccbd3942b58b870e08a20c0408a0fb106e0"
                ),
                new PretrainedModelInfo(
                    "commandrecognition_en_matchboxnet3x1x64_v2",
                    "https://github.com/kaiidams/NeMoOnnxSharp/releases/download/v1.1/commandrecognition_en_matchboxnet3x1x64_v2.onnx",
                    "a0c5e4d14e83d3b6afdaf239265a390c2ca513bcdedf3d295bc1f9f97f19868a"
                ),
                new PretrainedModelInfo(
                    "cmudict-0.7b_nv22.10",
                    "https://github.com/kaiidams/NeMoOnnxSharp/releases/download/v1.2/cmudict-0.7b_nv22.10",
                    "d330f3a3554d4c7ff8ef7bfc0c338ed74831d5f54109508fb829bdd82173608b"
                ),
                new PretrainedModelInfo(
                    "heteronyms-052722",
                    "https://github.com/kaiidams/NeMoOnnxSharp/releases/download/v1.2/heteronyms-052722",
                    "b701909aedf753172eff223950f8859cd4b9b4c80199cf0a6e9ac4a307c8f8ec"
                ),
                new PretrainedModelInfo(
                    "tts_en_fastpitch",
                    "https://github.com/kaiidams/NeMoOnnxSharp/releases/download/v1.2/tts_en_fastpitch.onnx",
                    "a297174dea1084bd34d1af1a8447bc07f6c8aab7a4fea312c610eba6bc3d0eac"
                ),
                new PretrainedModelInfo(
                    "tts_en_hifigan",
                    "https://github.com/kaiidams/NeMoOnnxSharp/releases/download/v1.2/tts_en_hifigan.onnx",
                    "54501000b9de86b724931478b5bb8911e1b6ca6e293f68e9e10f60351f1949a3"
                )
            };
        }

        public static PretrainedModelInfo Get(string pretrainedModelName)
        {
            foreach (var info in ModelList)
            {
                if (pretrainedModelName == info.PretrainedModelName)
                {
                    return info;
                }
            }

            throw new IndexOutOfRangeException();
        }

        public string PretrainedModelName { get; private set; }
        public string Location { get; private set; }
        public string Hash { get; private set; }

        public PretrainedModelInfo(
            string pretrainedModelName,
            string location,
            string hash)
        {
            PretrainedModelName = pretrainedModelName;
            Location = location;
            Hash = hash;
        }
    }
}