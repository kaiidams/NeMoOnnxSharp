using System;
using System.Collections.Generic;
using System.Text;

namespace NeMoOnnxSharp
{
    public static class FFT
    {
        private static int SwapIndex(int i)
        {
            return (i >> 8) & 0x01
                 | (i >> 6) & 0x02
                 | (i >> 4) & 0x04
                 | (i >> 2) & 0x08
                 | (i) & 0x10
                 | (i << 2) & 0x20
                 | (i << 4) & 0x40
                 | (i << 6) & 0x80
                 | (i << 8) & 0x100;
        }

        public static void CFFT(double[] xr, double[] xi, int N)
        {
            double[] t = xi;
            xi = xr;
            xr = t;
            for (int i = 0; i < N; i++)
            {
                xr[i] = xi[SwapIndex(i)];
            }
            for (int i = 0; i < N; i++)
            {
                xi[i] = 0.0;
            }
            for (int n = 1; n < N; n *= 2)
            {
                for (int j = 0; j < N; j += n * 2)
                {
                    for (int k = 0; k < n; k++)
                    {
                        double ar = Math.Cos(-Math.PI * k / n);
                        double ai = Math.Sin(-Math.PI * k / n);
                        double er = xr[j + k];
                        double ei = xi[j + k];
                        double or = xr[j + k + n];
                        double oi = xi[j + k + n];
                        double aor = ar * or - ai * oi;
                        double aoi = ai * or + ar * oi;
                        xr[j + k] = er + aor;
                        xi[j + k] = ei + aoi;
                        xr[j + k + n] = er - aor;
                        xi[j + k + n] = ei - aoi;
                        //Console.WriteLine("{0} {1}", j + k, j + k + n);
                    }
                }
            }
        }

#if false
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
#endif
    }
}
