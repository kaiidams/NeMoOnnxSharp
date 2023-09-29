using System.Collections.Generic;
using NeMoOnnxSharp.TTSTokenizers;

namespace NeMoOnnxSharp.Scratch
{
    class Program
    {
        private const string ParsedText =
            "K|AA1|N|SH|AH0|S| |o|f| |i|t|s| |S|P|IH1|R|IH0|CH|UW2|AH0|"
            + "L| |a|n|d| |M|AO1|R|AH0|L| |h|e|r|i|t|a|g|e|,| |t|h|e| |Y|"
            + "UW1|N|Y|AH0|N| |i|s| |F|AW1|N|D|IH0|D| |o|n| |t|h|e| |IH2|"
            + "N|D|IH0|V|IH1|S|IH0|B|AH0|L|,| |Y|UW2|N|AH0|V|ER1|S|AH0|L|"
            + " |V|AE1|L|Y|UW0|Z| |o|f| |h|u|m|a|n| |D|IH1|G|N|AH0|T|IY0|"
            + ",| |F|R|IY1|D|AH0|M|,| |IH0|K|W|AA1|L|AH0|T|IY0| |a|n|d| |"
            + "S|AA2|L|AH0|D|EH1|R|AH0|T|IY0";


        static void Main(string[] args)
        {
            var g2p = new EnglishG2P(
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

            var specGen = new SpectrogramGenerator(@"C:\Users\kaiida\AppData\Local\NeMoOnnxSharp\Cache\tts_en_fastpitch.onnx");
            var parsed = specGen.Parse(ParsedText);
            var spec = specGen.GenerateSpectrogram(parsed, pace: 1.0);

            var vocoder = new Vocoder(@"C:\Users\kaiida\AppData\Local\NeMoOnnxSharp\Cache\tts_en_hifigan.onnx");
            var audio = vocoder.ConvertSpectrogramToAudio(spec);
            var x = audio.Select(x => (short)(x * 32765)).ToArray();
            WaveFile.WriteWAV(@"C:\Users\kaiida\source\Repos\kaiidams\NeMoOnnxSharp\test.wav", x, 22050);
        }
    }
}