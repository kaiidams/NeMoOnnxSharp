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

        short[] waveform;
        AudioToMFCCPreprocessor processor;

        public PreprocessorTest()
        {
            string appDirPath = AppDomain.CurrentDomain.BaseDirectory;
            string waveFile = Path.Combine(appDirPath, "Data", SampleWAVSpeechFile);
            waveform = WaveFile.ReadWAV(waveFile, SampleRate);
            processor = new AudioToMFCCPreprocessor(
                sampleRate: SampleRate,
                window: WindowFunction.Hann,
                windowLength: 400,
                hopLength: 160,
                fftLength: 512,
                //preNormalize: 0.8,
                preemph: 0.0,
                center: false,
                nMelBands: 64,
                melMinHz: 0.0,
                melMaxHz: 0.0,
                htk: true,
                melNormalize: MelNorm.None,
                logOffset: 1e-6,
                postNormalize: false);
        }

        [TestMethod]
        public void TestSpectrogram()
        {
            var x = processor.Spectrogram(waveform);
            AssertMSE("spectrogram.bin", x);
        }

        [TestMethod]
        public void TestMelSpectrogram()
        {
            var x = processor.MelSpectrogram(waveform);
            AssertMSE("melspectrogram.bin", x);
        }

        [TestMethod]
        public void TestMFCC()
        {
            var processor = new AudioToMFCCPreprocessor(
                sampleRate: SampleRate,
                window: WindowFunction.Hann,
                windowLength: 400,
                hopLength: 160,
                fftLength: 512,
                //preNormalize: 0.8,
                preemph: 0.0,
                center: true,
                nMelBands: 64,
                melMinHz: 0.0,
                melMaxHz: 0.0,
                htk: true,
                melNormalize: MelNorm.None,
                nMFCC: 64,
                logOffset: 1e-6,
                postNormalize: false);
            var x = processor.MFCC(waveform);
        }

        [TestMethod]
        public void TestReadFrame()
        {
            int windowLength = 5;
            int fftLength = 9;
            var processor = new AudioToMFCCPreprocessor(
                windowLength: windowLength,
                fftLength: fftLength,
                preemph: 0.0);

            MethodInfo methodInfo1 = typeof(AudioToMFCCPreprocessor).GetMethod(
                "ReadFrameCenter", BindingFlags.NonPublic | BindingFlags.Instance);
            MethodInfo methodInfo2 = typeof(AudioToMFCCPreprocessor).GetMethod(
                "ReadFrameCenterPreemphasis", BindingFlags.NonPublic | BindingFlags.Instance);

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
                methodInfo1.Invoke(processor, parameters1);
                object[] parameters2 = { waveform, offset, scale, frame2 };
                methodInfo2.Invoke(processor, parameters2);
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