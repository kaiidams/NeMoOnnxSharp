using System;
using System.Collections.Generic;
using System.IO;
using System.Runtime.InteropServices;
using System.Text;

namespace NeMoOnnxSharp
{
    public static class WaveFile
    {
        /// <summary>
        /// Load WAV file as a short array.
        /// </summary>
        /// <param name="path"></param>
        /// <param name="rate"></param>
        /// <param name="mono"></param>
        /// <returns></returns>
        /// <exception cref="InvalidDataException"></exception>
        public static short[] ReadWav(string path, int rate, bool mono)
        {
            using var stream = File.OpenRead(path);
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
                if (fourCC == "fmt ")
                {
                    if (chunkLen < 16) throw new InvalidDataException();
                    short formatTag = reader.ReadInt16();
                    if (formatTag != 1) throw new InvalidDataException("Only PCM format is supported");
                    short numChannels = reader.ReadInt16();
                    if (numChannels != (mono ? 1 : 2)) throw new NotSupportedException();
                    int fileRate = reader.ReadInt32();
                    if (fileRate != rate) throw new NotSupportedException();
                    int avgBytesPerSec = reader.ReadInt32();
                    short blockAlign = reader.ReadInt16();
                    short bitsPerSample = reader.ReadInt16();
                    if (avgBytesPerSec * 8 != fileRate * bitsPerSample * numChannels)
                    {
                        throw new InvalidDataException();
                    }
                    if (chunkLen > 16)
                    {
                        byte[] byteData = reader.ReadBytes(chunkLen - 16);
                    }
                }
                else
                {
                    byte[] byteData = reader.ReadBytes(chunkLen);
                    if (fourCC == "data")
                    {
                        return MemoryMarshal.Cast<byte, short>(byteData).ToArray();
                    }
                }
            }
        }
    }
}
