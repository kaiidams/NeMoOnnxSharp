// Copyright (c) Katsuya Iida.  All Rights Reserved.
// See LICENSE in the project root for license information.

using System;
using System.Collections.Generic;
using System.IO;
using System.Runtime.InteropServices;
using System.Text;

namespace NeMoOnnxSharp
{
    /// <summary>
    /// A static class to read and write WAV files.
    /// </summary>
    public static class WaveFile
    {
        /// <summary>
        /// Load a WAV file as a short array. The result is resampled
        /// with the target sampling rate and Multi-channel audio
        /// is converted to mono.
        /// </summary>
        /// <param name="path">File to read.</param>
        /// <param name="rate">the target sampling rate</param>
        /// <returns>Waveform data.</returns>
        /// <exception cref="InvalidDataException"></exception>
        public static short[] ReadWAV(string path, int rate)
        {
            using (var stream = File.OpenRead(path))
            using (var reader = new BinaryReader(stream, Encoding.ASCII))
            {
                int originalRate;
                short originalNumChannels;
                var waveform = ReadWAV(reader, out originalRate, out originalNumChannels);
                return PostProcess(waveform, originalRate, originalNumChannels, rate);
            }
        }

        /// <summary>
        /// Save a short array as a WAV file.
        /// </summary>
        /// <param name="path">File to write.</param>
        /// <param name="waveform">Waveform data.</param>
        /// <param name="rate"></param>
        /// <returns></returns>
        public static void WriteWAV(string path, short[] waveform, int rate)
        {
            short numChannels = 1;
            using (var stream = File.OpenWrite(path))
            {
                WriteWAV(stream, waveform, rate, numChannels);
            }
        }

        /// <summary>
        /// Encode a short array into a byte array in WAV format.
        /// </summary>
        /// <param name="waveform">Waveform data.</param>
        /// <param name="rate"></param>
        /// <returns>A byte array in WAV format</returns>
        public static byte[] GetWAVBytes(short[] waveform, int rate)
        {
            byte[] data;
            short numChannels = 1;
            using (var stream = new MemoryStream())
            {
                WriteWAV(stream, waveform, rate, numChannels);
                data = stream.ToArray();
            }
            return data;
        }

        private static short[] ReadWAV(BinaryReader reader, out int rate, out short numChannels)
        {
            rate = 0;
            numChannels = 0;
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
                    numChannels = reader.ReadInt16();
                    rate = reader.ReadInt32();
                    int avgBytesPerSec = reader.ReadInt32();
                    short blockAlign = reader.ReadInt16();
                    short bitsPerSample = reader.ReadInt16();
                    if (avgBytesPerSec * 8 != rate * bitsPerSample * numChannels || blockAlign * 8 != bitsPerSample * numChannels)
                    {
                        throw new InvalidDataException();
                    }
                    if (chunkLen > 16)
                    {
                        reader.ReadBytes(chunkLen - 16);
                    }
                }
                else
                {
                    if (rate == 0)
                    {
                        throw new InvalidDataException();
                    }
                    byte[] byteData = reader.ReadBytes(chunkLen);
                    if (fourCC == "data")
                    {
                        return MemoryMarshal.Cast<byte, short>(byteData).ToArray();
                    }
                }
            }
        }

        private static void WriteWAV(Stream stream, short[] waveform, int rate, short numChannels)
        {
            using (var writer = new BinaryWriter(stream, Encoding.ASCII))
            {
                WriteWAV(writer, waveform, rate, numChannels);
            }
        }

        private static void WriteWAV(BinaryWriter writer, short[] waveform, int rate, short numChannels)
        {
            short formatTag = 1; // PCM
            short bitsPerSample = 16;
            int avgBytesPerSec = rate * bitsPerSample * numChannels / 8;
            short blockAlign = (short)(numChannels * bitsPerSample / 8);

            string fourCC = "RIFF";
            writer.Write(fourCC.ToCharArray());
            int chunkLen = 36 + waveform.Length * (bitsPerSample / 8);
            writer.Write(chunkLen);

            fourCC = "WAVE";
            writer.Write(fourCC.ToCharArray());

            fourCC = "fmt ";
            chunkLen = 16;

            writer.Write(fourCC.ToCharArray());
            writer.Write(chunkLen);
            writer.Write(formatTag);
            writer.Write(numChannels);
            writer.Write(rate);
            writer.Write(avgBytesPerSec);
            writer.Write(blockAlign);
            writer.Write(bitsPerSample);

            fourCC = "data";
            chunkLen = waveform.Length * (bitsPerSample / 8);

            writer.Write(fourCC.ToCharArray());
            writer.Write(chunkLen);
            var waveformBytes = MemoryMarshal.Cast<short, byte>(waveform);
            writer.Write(waveformBytes.ToArray());
        }

        private static short[] PostProcess(short[] waveform, int sourceRate, int sourceNumChannels, int targetRate)
        {
            waveform = ToMono(waveform, sourceNumChannels);
            waveform = Resample(waveform, sourceRate, targetRate);
            return waveform;
        }

        private static short[] Resample(short[] waveform, int sourceRate, int targetRate)
        {
            if (sourceRate == targetRate) return waveform;
            if (waveform.Length == 0) return Array.Empty<short>();
            long targetLength = (waveform.LongLength - 1) * targetRate / sourceRate + 1;
            short[] result = new short[targetLength];
            for (long i = 0; i < result.LongLength; i++)
            {
                result[i] = waveform[i * sourceRate / targetRate];
            }
            return result;
        }

        private static short[] ToMono(short[] waveform, int numChannels)
        {
            if (numChannels == 1) return waveform;
            int length = waveform.Length / numChannels;
            short[] result = new short[length];
            for (int i = 0; i < length; i++)
            {
                int value = 0;
                for (int j = 0; j < numChannels; j++)
                {
                    value += waveform[i * numChannels + j];
                }
                result[i] = (short)(value / numChannels);
            }
            return result;
        }
    }
}
