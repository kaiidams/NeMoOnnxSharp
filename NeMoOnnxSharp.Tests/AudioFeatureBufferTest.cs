using Microsoft.VisualStudio.TestTools.UnitTesting;
using NeMoOnnxSharp.AudioPreprocessing;
using NuGet.Frameworks;
using System;
using System.Diagnostics;
using System.IO;
using System.Reflection;
using System.Runtime.InteropServices;
using System.Security.Cryptography;

namespace NeMoOnnxSharp.Tests
{
    [TestClass]
    public class AudioFeatureBufferTest
    {
        private AudioFeatureBuffer<short, float>? _buffer;

        [TestInitialize] 
        public void Initialize()
        {
            int sampleRate = 16000;
            var transform = new MFCC(
                sampleRate: sampleRate,
                window: WindowFunction.Hann,
                winLength: 400,
                nFFT: 512,
                nMels: 64,
                nMFCC: 64,
                fMin: 0.0,
                fMax: 0.0,
                logMels: true,
                melScale: MelScale.HTK,
                melNorm: MelNorm.None);
            _buffer = new AudioFeatureBuffer<short, float>(
                transform,
                hopLength: 160);
        }

        [TestMethod]
        public void Test1()
        {
            Assert.IsNotNull(_buffer);
            int written;
            Assert.AreEqual(0, _buffer.OutputCount);
            written = _buffer.Write(new short[399]);
            Assert.AreEqual(399, written);
            Assert.AreEqual(0, _buffer.OutputCount);
            written = _buffer.Write(new short[1]);
            Assert.AreEqual(1, written);
            Assert.AreEqual(64, _buffer.OutputCount);
            _buffer.ConsumeOutput(64);
            Assert.AreEqual(0, _buffer.OutputCount);
            written = _buffer.Write(new short[160 * 3]);
            Assert.AreEqual(160 * 3, written);
            Assert.AreEqual(64 * 3, _buffer.OutputCount);
            written = _buffer.Write(new short[480]);
            Assert.AreEqual(480, written);
            Assert.AreEqual(64 * 6, _buffer.OutputCount);
        }

        [TestMethod]
        public void Test2()
        {
            Assert.IsNotNull(_buffer);
            int totalWritten = 0;
            int totalOutput = 0;
            var rng = new Random();
            for (int i = 0; i < 1000; i++)
            {
                int n = rng.Next(1024);
                int written = _buffer.Write(new short[n]);
                Assert.AreEqual(0, _buffer.OutputCount % 64);
                totalWritten += written;
                totalOutput += _buffer.OutputCount;
                if (totalWritten < 400)
                {
                    Assert.AreEqual(0, totalOutput);
                }
                else
                {
                    int m = (totalWritten - 400) / 160 + 1;
                    Assert.AreEqual(m * 64, totalOutput);
                }
                _buffer.ConsumeOutput(_buffer.OutputCount);
            }
        }
    }
}