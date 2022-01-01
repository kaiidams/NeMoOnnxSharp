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

string appDirPath = AppDomain.CurrentDomain.BaseDirectory;
string modelPath = Path.Combine(appDirPath, "QuartzNet15x5Base-En.onnx");
string inputDirPath = @"..\..\..\..\test_data";
string inputPath = Path.Combine(inputDirPath, "transcript.txt");

using var recognizer = new SpeechRecognizer(modelPath);
using var reader = File.OpenText(inputPath);
string line;
while ((line = reader.ReadLine()) != null)
{
    string[] parts = line.Split("|");
    string name = parts[0];
    string targetText = parts[1];
    string waveFile = Path.Combine(inputDirPath, name);
    var waveform = ReadWav(waveFile);
    string predictText = recognizer.Recognize(waveform);
    Console.WriteLine("{0}|{1}|{2}", name, targetText, predictText);
}