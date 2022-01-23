using System;
using System.Collections.Generic;
using System.IO;
using System.Runtime.InteropServices;
using System.Text;

namespace NeMoOnnxSharp
{
    public class WaveFile
    {
        public static short[] ReadWav(string waveFile)
        {
            using var stream = File.OpenRead(waveFile);
            using var reader = new BinaryReader(stream);
            string fourCC = new string(reader.ReadChars(4));
            if (fourCC != "RIFF")
                throw new InvalidDataException();
            int chunkLen = reader.ReadInt32();
            fourCC = new string(reader.ReadChars(4));
            if (fourCC != "WAVE")
                throw new InvalidDataException();
            while (true)
            {
                fourCC = new string(reader.ReadChars(4));
                chunkLen = reader.ReadInt32();
                byte[] byteData = reader.ReadBytes(chunkLen);
                if (fourCC == "data")
                {
                    return MemoryMarshal.Cast<byte, short>(byteData).ToArray();
                }
            }
        }
    }
}
