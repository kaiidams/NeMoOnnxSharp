using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using System.IO;
using System.Runtime.InteropServices;

namespace NeMoOnnxSharp.Tests
{
    [TestClass]
    public class FFTTest
    {
        private static void CFFTRef(double[] xr, double[] xi, int N)
        {
            double[] yr = new double[N];
            double[] yi = new double[N];
            for (int i = 0; i < N; i++)
            {
                double vr = 0.0;
                double vi = 0.0;
                for (int k = 0; k < N; k++)
                {
                    vr += Math.Cos(-2 * Math.PI * k * i / N) * xr[k];
                    vi += Math.Sin(-2 * Math.PI * k * i / N) * xr[k];
                }
                yr[i] = vr;
                yi[i] = vi;
            }
            for (int i = 0; i < N; i++)
            {
                xr[i] = yr[i];
                xi[i] = yi[i];
            }
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

        [TestMethod]
        public void TestCFFT()
        {
            var rng = new Random();
            for (int N = 256; N <= 2048; N *= 2)
            {
                var xr0 = new double[N];
                var xi0 = new double[N];
                var xr1 = new double[N];
                var xi1 = new double[N];
                for (int i = 0; i < 10; i++)
                {
                    for (int j = 0; j < N; j++)
                    {
                        xr0[j] = rng.NextDouble();
                        xi0[j] = rng.NextDouble();
                        xr1[j] = xr0[j];
                        xi1[j] = rng.NextDouble();
                    }
                    CFFTRef(xr0, xi0, N);
                    FFT.CFFT(xr1, xi1, N);
                    double error = MSE(xr0, xi1);
                    Assert.IsTrue(error < 1e-20);
                    error = MSE(xi0, xr1);
                    Assert.IsTrue(error < 1e-20);
                }
            }
        }
    }
}