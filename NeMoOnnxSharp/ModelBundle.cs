// Copyright (c) Katsuya Iida.  All Rights Reserved.
// See LICENSE in the project root for license information.

using System;

namespace NeMoOnnxSharp
{
    public class ModelBundle
    {
        public static ModelBundle GetBundle(string name) 
        {
            if (name == "QuartzNet15x5Base-En")
            {
                return new ModelBundle(
                    "https://github.com/kaiidams/NeMoOnnxSharp/releases/download/v1.1.0.pre1/QuartzNet15x5Base-En.onnx",
                    "ee1b72102fd0c5422d088e80f929dbdee7e889d256a4ce1e412cd49916823695"
                );
            }

            if (name == "vad_marblenet")
            {
                return new ModelBundle(
                    "https://github.com/kaiidams/NeMoOnnxSharp/releases/download/v1.1.0.pre1/vad_marblenet.onnx",
                    "edaf8a7bb62e4335f97aa70d1a447ccbd3942b58b870e08a20c0408a0fb106e0"
                );
            }
            throw new ArgumentException();
        }

        public string ModelUrl { get; private set; }
        public string Hash { get; private set; }

        public ModelBundle(
            string modelUrl,
            string hash)
        {
            ModelUrl = modelUrl;
            Hash = hash;
        }
    }
}