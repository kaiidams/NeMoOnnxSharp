using Microsoft.VisualStudio.TestTools.UnitTesting;
using NeMoOnnxSharp.TTSTokenizers;
using System;
using System.Diagnostics;
using System.IO;

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

        private const string SampleText =
            "You've read the book “Operating Systems Design and Implementation, 3rd edition”. Did you?";
        private const string NormalizedSampleText =
            "You've read the book “Operating Systems Design and Implementation, third edition”. Did you?";
        private const string SamplePronText =
            "Y|UW1|V| |r|e|a|d| |t|h|e| |B|UH1|K| |“|o|p|e|r|a|t|i|n|g| |"
            + "S|IH1|S|T|AH0|M|Z| |D|IH0|Z|AY1|N| |a|n|d| |IH2|M|P|L|AH0|"
            + "M|EH0|N|T|EY1|SH|AH0|N|,| |TH|ER1|D| |e|d|i|t|i|o|n|”|.| |"
            + "d|i|d| |Y|UW1|?";

        private readonly static int[] SampleParsed =
        {
             0,  22,  68,  20,   0,  87,  74,  70,  73,   0,  89,  77,  74,
             0,   1,  65,   9,   0, 105,  84,  85,  74,  87,  70,  89,  78,
            83,  76,   0,  16,  53,  16,  18,  31,  11,  23,   0,   3,  52,
            23,  41,  12,   0,  70,  83,  73,   0,  54,  11,  14,  10,  31,
            11,  43,  12,  18,  50,  17,  31,  12,  97,   0,  19,  47,   3,
             0,  74,  73,  78,  89,  78,  84,  83, 105,  98,   0,  73,  78,
            73,   0,  22,  68, 100,   0
        };

        [TestInitialize]
        public void Initialize()
        {
            string appDirPath = AppDomain.CurrentDomain.BaseDirectory;
            _g2p = new EnglishG2p(
                phonemeDict: Path.Combine(appDirPath, "Data", "cmudict-test"),
                heteronyms: Path.Combine(appDirPath, "Data", "heteronyms-test"),
                phonemeProbability: 1.0);
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
        public void TestTokenizerVocab()
        {
            Assert.IsNotNull(_tokenizer);
            CollectionAssert.AreEquivalent(ExpectedTokens, _tokenizer.Tokens);
        }

        [TestMethod]
        public void TestEnglishG2p()
        {
            Assert.IsNotNull(_g2p);
            var pron = string.Join("|", _g2p.Parse(NormalizedSampleText));
            Assert.AreEqual(SamplePronText, pron);
        }

        [TestMethod]
        public void TestEnglishEncode()
        {
            Assert.IsNotNull(_tokenizer);
            var parsed = _tokenizer.Encode(NormalizedSampleText);
            CollectionAssert.AreEquivalent(SampleParsed, parsed);
        }

        private EnglishG2p? _g2p;
        private EnglishPhonemesTokenizer? _tokenizer;

    }
}