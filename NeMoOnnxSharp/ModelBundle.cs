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
                    "https://github.com/kaiidams/NeMoOnnxSharp/raw/main/NeMoOnnxSharp/QuartzNet15x5Base-En.onnx",
                    "ee1b72102fd0c5422d088e80f929dbdee7e889d256a4ce1e412cd49916823695"
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