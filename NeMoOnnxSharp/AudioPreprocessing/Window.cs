// Copyright (c) Katsuya Iida.  All Rights Reserved.
// See LICENSE in the project root for license information.

using System;
using System.Collections.Generic;
using System.Text;

namespace NeMoOnnxSharp.AudioPreprocessing
{
    public static class Window
    {
        public static double[] MakeWindow(WindowFunction function, int length)
        {
            if (function == WindowFunction.Hann)
            {
                return MakeHannWindow(length);
            }
            else
            {
                throw new ArgumentException("Unknown windows name");
            }
        }

        private static double[] MakeHannWindow(int length)
        {
            double[] window = new double[length];
            for (int i = 0; i < length; i++)
            {
                window[i] = 0.5 * (1 - Math.Cos(2 * Math.PI * i / (length - 1)));
            }
            return window;
        }
    }
}
