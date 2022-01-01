using NeMoOnnxSharp;
using System.Runtime.InteropServices;

short[] ReadWav(string waveFile)
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