using System;
using System.Collections.Generic;
using System.Text;

namespace NeMoOnnxSharp
{
    public static class Window
    {
        public static double[] MakeHannWindow(int windowLength)
        {
            double[] window = new double[windowLength];
            for (int i = 0; i < windowLength; i++)
            {
                window[i] = 0.5 * (1 - Math.Cos(2 * Math.PI * i / (windowLength - 1)));
            }
            return window;
        }
    }
}
