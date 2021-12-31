using NAudio.Wave;
using NeMoOnnxSharp;
using System.Runtime.InteropServices;

short[] ReadWav(string waveFile)
{
    short[] waveData;
    using (var reader = new WaveFileReader(waveFile))
    using (var writer = new MemoryStream())
    {
        while (true)
        {
            byte[] buffer = new byte[4096];
            int readBytes = reader.Read(buffer, 0, buffer.Length);
            if (readBytes == 0) break;
            writer.Write(buffer, 0, readBytes);
        }
        var byteData = writer.ToArray();
        waveData = MemoryMarshal.Cast<byte, short>(byteData).ToArray();
    }
    return waveData;
}

string waveFile = "test.wav";
string appDirPath = AppDomain.CurrentDomain.BaseDirectory;
string modelPath = Path.Combine(appDirPath, "QuartzNet15x5Base-En.onnx");
using (var recognizer = new SpeechRecognizer(modelPath))
{
    var waveform = ReadWav(waveFile);
    string text = recognizer.Recognize(waveform);
    Console.WriteLine("Recognized: {0}", text);
}
