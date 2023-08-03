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
                    "https://github.com/kaiidams/NeMoOnnxSharp/raw/main/NeMoOnnxSharp/QuartzNet15x5Base-En.onnx"
                );
            }
            throw new ArgumentException();
        }

        public string ModelUrl { get; private set; }

        public ModelBundle(
            string modelUrl)
        {
            ModelUrl = modelUrl;
        }
    }
}