// Copyright (c) Katsuya Iida.  All Rights Reserved.
// See LICENSE in the project root for license information.

using System;

namespace NeMoOnnxSharp.Example
{
    internal class PretrainedModelInfo
    {
        public static PretrainedModelInfo GetInfo(string pretrainedModelName) 
        {
            if (pretrainedModelName == "stt_en_quartznet15x5")
            {
                return new PretrainedModelInfo(
                    "stt_en_quartznet15x5",
                    "https://github.com/kaiidams/NeMoOnnxSharp/releases/download/v1.1.0.pre1/stt_en_quartznet15x5.onnx",
                    "dde27f0528e92c05f7bc220a9be4a7bb99927da0a3a25db8f2f861e3559da90d"
                );
            }

            if (pretrainedModelName == "QuartzNet15x5Base-En")
            {
                return new PretrainedModelInfo(
                    "QuartzNet15x5Base-En",
                    "https://github.com/kaiidams/NeMoOnnxSharp/releases/download/v1.1.0.pre1/QuartzNet15x5Base-En.onnx",
                    "ee1b72102fd0c5422d088e80f929dbdee7e889d256a4ce1e412cd49916823695"
                );
            }

            if (pretrainedModelName == "vad_marblenet")
            {
                return new PretrainedModelInfo(
                    "vad_marblenet",
                    "https://github.com/kaiidams/NeMoOnnxSharp/releases/download/v1.1.0.pre1/vad_marblenet.onnx",
                    "edaf8a7bb62e4335f97aa70d1a447ccbd3942b58b870e08a20c0408a0fb106e0"
                );
            }

            if (pretrainedModelName == "commandrecognition_en_matchboxnet3x1x64_v2")
            {
                return new PretrainedModelInfo(
                    "commandrecognition_en_matchboxnet3x1x64_v2",
                    "https://github.com/kaiidams/NeMoOnnxSharp/releases/download/v1.1.0.pre1/commandrecognition_en_matchboxnet3x1x64_v2.onnx",
                    "a0c5e4d14e83d3b6afdaf239265a390c2ca513bcdedf3d295bc1f9f97f19868a"
                );
            }

            throw new ArgumentException();
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