// Copyright (c) Katsuya Iida.  All Rights Reserved.
// See LICENSE in the project root for license information.

using System;
using System.Collections.Generic;
using System.Text;

namespace NeMoOnnxSharp
{
    public static class FFT
    {
        public static void CFFT(Span<double> xr, Span<double> xi, int N)
        {
            Span<double> t = xi;
            xi = xr;
            xr = t;
            Swap(xr, xi, N);
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

        public static void DCT2(Span<double> xr, Span<double> xi, int N)
        {
            // TODO Implement more efficiently.
            for (int i = 0; i < N; i++)
            {
                double s = 0;
                for (int j = 0; j < N; j++)
                {
                    s += xr[j] * Math.Cos(Math.PI * (j + 0.5) * i / N);
                }
                xi[i] = i == 0 ? s / Math.Sqrt(N) : s / Math.Sqrt(N / 2);
            }
        }

        private static void Swap(Span<double> xr, Span<double> xi, int N)
        {
            if (N == 256)
            {
                Swap256(xr, xi);
            }
            else if (N == 512)
            {
                Swap512(xr, xi);
            }
            else if (N == 1024)
            {
                Swap1024(xr, xi);
            }
            else if (N == 2048)
            {
                Swap2048(xr, xi);
            }
            else
            {
                throw new ArgumentException("Only 256, 512, 1024 or 2048 is supported for N");
            }
            for (int i = 0; i < N; i++)
            {
                xi[i] = 0.0;
            }
        }

        private static void Swap256(Span<double> xr, Span<double> xi)
        {
            for (int i = 0; i < 256; i++)
            {
                int j = ((i >> 7) & 0x01)
                 + ((i >> 5) & 0x02)
                 + ((i >> 3) & 0x04)
                 + ((i >> 1) & 0x08)
                 + ((i << 1) & 0x10)
                 + ((i << 3) & 0x20)
                 + ((i << 5) & 0x40)
                 + ((i << 7) & 0x80);
                xr[i] = xi[j];
            }
        }

        private static void Swap512(Span<double> xr, Span<double> xi)
        {
            for (int i = 0; i < 512; i++)
            {
                int j = ((i >> 8) & 0x01)
                 + ((i >> 6) & 0x02)
                 + ((i >> 4) & 0x04)
                 + ((i >> 2) & 0x08)
                 + ((i) & 0x10)
                 + ((i << 2) & 0x20)
                 + ((i << 4) & 0x40)
                 + ((i << 6) & 0x80)
                 + ((i << 8) & 0x100);
                xr[i] = xi[j];
            }
        }

        private static void Swap1024(Span<double> xr, Span<double> xi)
        {
            for (int i = 0; i < 1024; i++)
            {
                int j = ((i >> 9) & 0x01)
                 + ((i >> 7) & 0x02)
                 + ((i >> 5) & 0x04)
                 + ((i >> 3) & 0x08)
                 + ((i >> 1) & 0x10)
                 + ((i << 1) & 0x20)
                 + ((i << 3) & 0x40)
                 + ((i << 5) & 0x80)
                 + ((i << 7) & 0x100)
                 + ((i << 9) & 0x200);
                xr[i] = xi[j];
            }
        }

        private static void Swap2048(Span<double> xr, Span<double> xi)
        {
            for (int i = 0; i < 2048; i++)
            {
                int j = ((i >> 10) & 0x01)
                 + ((i >> 8) & 0x02)
                 + ((i >> 6) & 0x04)
                 + ((i >> 4) & 0x08)
                 + ((i >> 2) & 0x10)
                 + ((i) & 0x20)
                 + ((i << 2) & 0x40)
                 + ((i << 4) & 0x80)
                 + ((i << 6) & 0x100)
                 + ((i << 8) & 0x200)
                 + ((i << 10) & 0x400);
                xr[i] = xi[j];
            }
        }
    }
}
