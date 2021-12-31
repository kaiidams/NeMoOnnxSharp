using NAudio.Wave;
using NeMoOnnxSharp;
using System.Runtime.InteropServices;

// See https://aka.ms/new-console-template for more information
string waveFile = "test.wav";

short[] ReadWav(string waveFile)
{
    short[] waveData;
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
    return waveData;
}

string appDirPath = AppDomain.CurrentDomain.BaseDirectory;
string modelPath = Path.Combine(appDirPath, "QuartzNet15x5Base-En.onnx");
var recognizer = new SpeechRecognizer(modelPath);

for (int i = 0; i < 30; i++)
{
    var waveform = ReadWav(string.Format(@"C:\Users\kaiida\Desktop\hoge\test{0}.wav", i));
    string text2 = recognizer.Recognize(waveform);
    Console.WriteLine("{0} {1}", i, text2);
}

var waveData = ReadWav(waveFile); 
float[] audioSignal;
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

string text = recognizer.Recognize(waveData);

if (false)
{
    var preprocessor = new AudioToMelSpectrogramPreprocessor();
    audioSignal = preprocessor.Process(waveData);
    Console.WriteLine(audioSignal.Length / 64);
    using (var writer = File.OpenWrite("test.bin"))
    {
        byte[] data = MemoryMarshal.Cast<float, byte>(audioSignal).ToArray();
        writer.Write(data, 0, data.Length);
    }
}
