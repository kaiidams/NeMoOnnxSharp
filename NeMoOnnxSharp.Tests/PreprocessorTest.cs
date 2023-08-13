using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using System.Diagnostics;
using System.IO;
using System.Reflection;
using System.Runtime.InteropServices;

namespace NeMoOnnxSharp.Tests
{
    [TestClass]
    public class PreprocessorTest
    {
        private const int SampleRate = 16000;
        private const string SampleWAVSpeechFile = "61-70968-0000.wav";

        private static float[] ReadData(string file)
        {
            string appDirPath = AppDomain.CurrentDomain.BaseDirectory;
            string path = Path.Combine(appDirPath, "Data", file);
            var bytes = File.ReadAllBytes(path);
            return MemoryMarshal.Cast<byte, float>(bytes).ToArray();
        }

        private static void AssertMSE(string path, float[] x, double threshold = 1e-3)
        {
            var truth = ReadData(path);
            double mse = MSE(truth, x);
            Console.WriteLine("MSE: {0}", mse);
            Assert.IsTrue(mse < threshold);
        }

        private static double MSE(float[] a, float[] b)
        {
            if (a.Length != b.Length) throw new ArgumentException();
            int len = Math.Min(a.Length, b.Length);
            double err = 0.0;
            for (int i = 0; i < len; i++)
            {
                double diff = a[i] - b[i];
                err += diff * diff;
            }
            return err / len;
        }

        short[]? audioSignal;

        [TestInitialize]
        public void Initialize()
        {
            string appDirPath = AppDomain.CurrentDomain.BaseDirectory;
            string waveFile = Path.Combine(appDirPath, "Data", SampleWAVSpeechFile);
            audioSignal = WaveFile.ReadWAV(waveFile, SampleRate);
        }

        [TestMethod]
        public void TestMelSpectrogram()
        {
            var preprocessor = new AudioToMelSpectrogramPreprocessor(
                sampleRate: 16000,
                window: WindowFunction.Hann,
                windowSize: 0.02,
                windowStride: 0.01,
                nFFT: 512,
                features: 64);
            var x = preprocessor.GetFeatures(audioSignal);
            // NeMo pads the result to 16 time staps.
            var y = new float[((x.Length / 64 + 15) / 16) * 16 * 64];
            Array.Copy(x, y, x.Length);
            AssertMSE("mel_spectrogram.bin", y);
        }

        [TestMethod]
        public void TestMFCC()
        {
            var preprocessor = new AudioToMFCCPreprocessor(
                sampleRate: 16000,
                windowSize: 0.025,
                windowStride: 0.01,
                //preNormalize: 0.8,
                window: WindowFunction.Hann,
                nMels: 64,
                nMFCC: 64,
                nFFT: 512);
            var processedSignal = preprocessor.GetFeatures(audioSignal);
            AssertMSE("mfcc.bin", processedSignal, threshold: 1e-2);
        }
    }
}