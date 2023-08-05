using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using System.Diagnostics;
using System.IO;
using System.Reflection;
using System.Runtime.InteropServices;

namespace NeMoOnnxSharp.Tests
{
    [TestClass]
    public class WaveFileTest
    {
        private const int SampleRate = 16000;
        private const string SampleWAVSpeech1File = "61-70968-0000.wav";
        private const int SampleWAVSpeech1Length = 78480;
        private const string SampleWAVSpeech2File = "61-70968-0000-mod.wav";
        private const int SampleWAVSpeech2Length = 78480 / 2;
        private const string TempFile = "temp.wav";

        [TestMethod]
        public void Test1()
        {
            string appDirPath = AppDomain.CurrentDomain.BaseDirectory;
            string waveFile = Path.Combine(appDirPath, "Data", SampleWAVSpeech1File);
            var waveform = WaveFile.ReadWAV(waveFile, SampleRate);
            Assert.AreEqual(waveform.Length, SampleWAVSpeech1Length);

            WaveFile.WriteWAV(TempFile, waveform, SampleRate);
            var waveform2 = WaveFile.ReadWAV(TempFile, SampleRate);
            Assert.IsTrue(IsArraysEqual(waveform, waveform2));
        }

        [TestMethod]
        public void Test2()
        {
            string appDirPath = AppDomain.CurrentDomain.BaseDirectory;
            string waveFile = Path.Combine(appDirPath, "Data", SampleWAVSpeech2File);
            var waveform = WaveFile.ReadWAV(waveFile, SampleRate);
            Assert.AreEqual(waveform.Length, SampleWAVSpeech2Length);

            byte[] bytes = WaveFile.GetWAVBytes(waveform, SampleRate);
            Assert.AreEqual(bytes.Length, SampleWAVSpeech2Length * 2 + 44);
        }

        private bool IsArraysEqual<T>(T[] x, T[] y)
        {
            if (x.Length != y.Length) return false;
            for (int i = 0; i < x.Length; i++)
            {
                if (!x[i].Equals(y[i])) return false;
            }
            return true;
        }
    }
}