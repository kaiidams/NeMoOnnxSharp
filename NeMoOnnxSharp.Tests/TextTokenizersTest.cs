using Microsoft.VisualStudio.TestTools.UnitTesting;
using NeMoOnnxSharp.TextTokenizers;
using System;
using System.Diagnostics;
using System.IO;
using System.Reflection;
using System.Runtime.InteropServices;

namespace NeMoOnnxSharp.Tests
{
    [TestClass]
    public class TextTokenizersTest
    {
        private string[] ExpectedTokens =
        {
            " ", "B", "CH", "D", "DH", "F", "G", "HH", "JH", "K", "L", "M",
            "N", "NG", "P", "R", "S", "SH", "T", "TH", "V", "W", "Y", "Z", "ZH",
            "AA0", "AA1", "AA2", "AE0", "AE1", "AE2", "AH0", "AH1", "AH2", "AO0",
            "AO1", "AO2", "AW0", "AW1", "AW2", "AY0", "AY1", "AY2", "EH0", "EH1",
            "EH2", "ER0", "ER1", "ER2", "EY0", "EY1", "EY2", "IH0", "IH1", "IH2",
            "IY0", "IY1", "IY2", "OW0", "OW1", "OW2", "OY0", "OY1", "OY2", "UH0",
            "UH1", "UH2", "UW0", "UW1", "UW2", "a", "b", "c", "d", "e", "f", "g",
            "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t", "u",
            "v", "w", "x", "y", "z", "'", ",", ".", "!", "?", "-", ":", ";", "/",
            "\"", "(", ")", "[", "]", "{", "}", "<pad>", "<blank>", "<oov>"
        };

        [TestMethod]
        public void Test1()
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
            CollectionAssert.AreEquivalent(ExpectedTokens, tokenizer.Tokens);
        }
    }
}