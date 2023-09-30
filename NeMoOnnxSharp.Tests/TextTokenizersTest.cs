using Microsoft.VisualStudio.TestTools.UnitTesting;
using NeMoOnnxSharp.TTSTokenizers;
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
        private readonly static string[] ExpectedTokens =
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

        private const string NormalizedText =
            "Conscious of its spiritual and moral heritage, the Union is "
            + "founded on the indivisible, universal values of human dignity, "
            + "freedom, equality and solidarity";

        private const string ParsedText =
            "K|AA1|N|SH|AH0|S| |o|f| |i|t|s| |S|P|IH1|R|IH0|CH|UW2|AH0|"
            + "L| |a|n|d| |M|AO1|R|AH0|L| |h|e|r|i|t|a|g|e|,| |t|h|e| |Y|"
            + "UW1|N|Y|AH0|N| |i|s| |F|AW1|N|D|IH0|D| |o|n| |t|h|e| |IH2|"
            + "N|D|IH0|V|IH1|S|IH0|B|AH0|L|,| |Y|UW2|N|AH0|V|ER1|S|AH0|L|"
            + " |V|AE1|L|Y|UW0|Z| |o|f| |h|u|m|a|n| |D|IH1|G|N|AH0|T|IY0|"
            + ",| |F|R|IY1|D|AH0|M|,| |IH0|K|W|AA1|L|AH0|T|IY0| |a|n|d| |"
            + "S|AA2|L|AH0|D|EH1|R|AH0|T|IY0";

        private readonly static int[] ExpectedParsed =
        {
            0,  9, 26, 12, 17, 31, 16,  0, 84, 75,  0, 78, 89, 88,  0, 16, 14, 53,
            15, 52,  2, 69, 31, 10,  0, 70, 83, 73,  0, 11, 35, 15, 31, 10,  0, 77,
            74, 87, 78, 89, 70, 76, 74, 97,  0, 89, 77, 74,  0, 22, 68, 12, 22, 31,
            12,  0, 78, 88,  0,  5, 38, 12,  3, 52,  3,  0, 84, 83,  0, 89, 77, 74,
            0, 54, 12,  3, 52, 20, 53, 16, 52,  1, 31, 10, 97,  0, 22, 69, 12, 31,
            20, 47, 16, 31, 10,  0, 20, 29, 10, 22, 67, 23,  0, 84, 75,  0, 77, 90,
            82, 70, 83,  0,  3, 53,  6, 12, 31, 18, 55, 97,  0,  5, 15, 56,  3, 31,
            11, 97,  0, 52,  9, 21, 26, 10, 31, 18, 55,  0, 70, 83, 73,  0, 16, 27,
            10, 31,  3, 44, 15, 31, 18, 55,  0
        };

        [TestInitialize]
        public void Initialize()
        {
            _g2p = new EnglishG2p(
                phonemeDict: "../../scripts/tts_dataset_files/cmudict-0.7b_nv22.10",
                heteronyms: "../../scripts/tts_dataset_files/heteronyms-052722",
                phonemeProbability: 0.5);
            _tokenizer = new EnglishPhonemesTokenizer(
                _g2p,
                punct: true,
                stresses: true,
                chars: true,
                apostrophe: true,
                padWithSpace: true,
                addBlankAt: BaseTokenizer.AddBlankAt.True);
        }

        [TestMethod]
        public void Test1()
        {
            CollectionAssert.AreEquivalent(ExpectedTokens, _tokenizer.Tokens);
        }

        [TestMethod]
        public void Test2()
        {
            var parsed = _tokenizer.EncodeFromG2p(ParsedText.Split('|'));
            CollectionAssert.AreEquivalent(ExpectedParsed, parsed);
        }

        private EnglishG2p? _g2p;
        private EnglishPhonemesTokenizer? _tokenizer;

    }
}