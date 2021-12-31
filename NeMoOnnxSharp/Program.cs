using NAudio.Wave;
using NeMoOnnxSharp;
using System.Runtime.InteropServices;

// See https://aka.ms/new-console-template for more information
Console.WriteLine("Hello, World!");
string waveFile = "test.wav";

short[] waveData;
float[] audioSignal;
using (var reader = new WaveFileReader(waveFile))
using (var writer = new MemoryStream())
{
    int k = 0;
    while (true)
    {
        byte[] buffer = new byte[4096];
        int readBytes = reader.Read(buffer, 0, buffer.Length);
        if (readBytes == 0) break;
        writer.Write(buffer, 0, readBytes);
        k += readBytes;
    }
    var byteData = writer.ToArray();
    waveData = MemoryMarshal.Cast<byte, short>(byteData).ToArray();
}
int m = 0;
for (int i = 0; i < waveData.Length; i++)
{
    var x = Math.Abs(waveData[i]);
    if (m < x) m = x;
}
double s = short.MaxValue * 0.8 / m;
for (int i = 0; i < waveData.Length; i++)
{
    double x = waveData[i];
    waveData[i] = (short)(x * s);
}

var preprocessor = new AudioToMelSpectrogramPreprocessor();
audioSignal = preprocessor.Process(waveData);
Console.WriteLine(audioSignal.Length / 64);
using (var writer = File.OpenWrite("test.bin"))
{
    byte[] data = MemoryMarshal.Cast<float, byte>(audioSignal).ToArray();
    writer.Write(data, 0, data.Length);
}