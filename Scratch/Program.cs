using System.Collections.Generic;
using NeMoOnnxSharp.TextTokenizers;

namespace NeMoOnnxSharp.Scratch
{
    class Program
    {
        static void Main(string[] args)
        {
            var g2p = new EnglishG2p(
                phonemeDict: "../../scripts/tts_dataset_files/cmudict-0.7b_nv22.10",
                heteronyms: "../../scripts/tts_dataset_files/heteronyms-052722",
                phonemeProbability: 0.5);
            var tokenizer = new EnglishPhonemesTokenizer(
                g2p,
                punct: true,
                stresses: true,
                chars: true,
                apostrophe: true,
                padWithSpace: true,
                addBlankAt: BaseTokenizer.AddBlankAt.True);
            var encoded = tokenizer.Encode("Hello World!");
            var decoded = tokenizer.Decode(encoded);

            foreach (var x in encoded)
            {
                Console.WriteLine("{0}", x);
            }
            Console.WriteLine("decoded: {0}", decoded);
            Console.WriteLine("end");
        }
    }
}