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

        private static double MSE(double[] a, double[] b)
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

        short[]? waveform;

        [TestInitialize]
        public void TestInitialize()
        {
            string appDirPath = AppDomain.CurrentDomain.BaseDirectory;
            string waveFile = Path.Combine(appDirPath, "Data", SampleWAVSpeechFile);
            waveform = WaveFile.ReadWAV(waveFile, SampleRate);
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
            var x = preprocessor.GetFeatures(waveform);
            AssertMSE("melspectrogram.bin", x);
        }

        [TestMethod]
        public void TestMFCC()
        {
            var preprocessor = new AudioToMFCCPreprocessor(
                sampleRate: 16000,
                window: WindowFunction.Hann,
                windowSize: 0.025,
                windowStride: 0.01,
                nFFT: 512,
                //preNormalize: 0.8,
                nMels: 64,
                nMFCC: 64);
            var x = preprocessor.GetFeatures(waveform);
        }

        [TestMethod]
        public void TestReadFrame()
        {
            int windowLength = 5;
            int fftLength = 9;
            var preprocessor = new AudioToMFCCPreprocessor(
                nWindowSize: windowLength,
                nFFT: fftLength);

            MethodInfo? methodInfo1 = typeof(AudioToMFCCPreprocessor).GetMethod(
                "ReadFrameCenter", BindingFlags.NonPublic | BindingFlags.Instance);
            MethodInfo? methodInfo2 = typeof(AudioToMFCCPreprocessor).GetMethod(
                "ReadFrameCenterPreemphasis", BindingFlags.NonPublic | BindingFlags.Instance);
            Assert.IsNotNull(methodInfo1);
            Assert.IsNotNull(methodInfo2);
            var rng = new Random();
            short[] waveform = new short[1200];
            double[] frame1 = new double[fftLength];
            double[] frame2 = new double[fftLength];
            for (int i = 0; i < waveform.Length; i++) waveform[i] = (short)rng.Next(short.MinValue, short.MaxValue);
#if true
            for (int i = 0; i < 100; i++)
            {
                int offset = rng.Next(waveform.Length);
                double scale = rng.NextDouble();
                object[] parameters1 = { waveform, offset, scale, frame1 };
                methodInfo1.Invoke(preprocessor, parameters1);
                object[] parameters2 = { waveform, offset, scale, frame2 };
                methodInfo2.Invoke(preprocessor, parameters2);
                double error = MSE(frame1, frame2);
                Assert.IsTrue(error == 0);
            }
#else
            for (int j = 0; j < 5; j++)
            {
                var stopWatch = new Stopwatch();
                stopWatch.Start();
                for (int i = 0; i < 1000000; i++)
                {
                    int offset = rng.Next(waveform.Length);
                    double scale = rng.NextDouble();
                    object[] parameters1 = { waveform, offset, scale, frame1 };
                    methodInfo1.Invoke(processor, parameters1);
                }
                stopWatch.Stop();
                Console.WriteLine(stopWatch.Elapsed);

                stopWatch = new Stopwatch();
                stopWatch.Start();
                for (int i = 0; i < 1000000; i++)
                {
                    int offset = rng.Next(waveform.Length);
                    double scale = rng.NextDouble();
                    object[] parameters2 = { waveform, offset, scale, frame1 };
                    methodInfo2.Invoke(processor, parameters2);
                }
                stopWatch.Stop();
                Console.WriteLine(stopWatch.Elapsed);
            }
#endif
        }

        private void AssertMSE(string path, float[] x, double threshold = 1e-3)
        {
            var truth = ReadData(path);
            double mse = MSE(truth, x);
            Console.WriteLine("MSE: {0}", mse);
            Assert.IsTrue(mse < threshold);
        }
    }
}