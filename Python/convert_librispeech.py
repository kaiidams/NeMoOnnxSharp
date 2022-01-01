import os
import sys
import soundfile as sf
import librosa

id1, id2 = 61, 70968
input_dir = os.path.join(sys.argv[1], "test-clean", str(id1), str(id2))
output_dir = os.path.join("..", "test_data")
transcript_file = os.path.join(input_dir, "%d-%d.trans.txt" % (id1, id2))
output_file = os.path.join(output_dir, "transcript.txt")
sample_rate = 16000

os.makedirs(output_dir, exist_ok=True)
with open(transcript_file, 'rt') as f:
    with open(output_file, 'wt') as outf:
        for line in f:
            name, _, text = line.rstrip('\r\n').partition(" ")
            text = text.lower()
            audio_file = os.path.join(input_dir, name + ".flac")
            wav_file = os.path.join(output_dir, name + ".wav")
            x, orig_sample_rate = sf.read(audio_file)
            assert x.ndim == 1
            x = librosa.resample(x, orig_sample_rate, sample_rate)
            print("Writing %s..." % (wav_file,))
            outf.write("%s.wav|%s\n" % (name, text))
            sf.write(wav_file, x, samplerate=sample_rate, subtype="PCM_16")
