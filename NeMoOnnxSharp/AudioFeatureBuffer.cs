// Copyright (c) Katsuya Iida.  All Rights Reserved.
// See LICENSE in the project root for license information.

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeMoOnnxSharp
{
    internal class AudioFeatureBuffer
    {
        private readonly AudioProcessor _processor;
        private readonly int _stftHopLength;
        private readonly int _stftWindowLength;
        private readonly int _nMelBands;
        private readonly double _audioScale;

        private readonly short[] _waveformBuffer;
        private int _waveformCount;
        private readonly float[] _outputBuffer;
        private int _outputCount;

        public AudioFeatureBuffer(int stftHopLength = 160, int stftWindowLength = 400, int nMelBands = 64)
        {
#if false
            _processor = new AudioProcessor(
                sampleRate: 16000,
                window: WindowFunction.Hann,
                windowLength: 400,
                hopLength: 160,
                fftLength: 512,
                preNormalize: 0.8,
                preemph: 0.0,
                center: false,
                nMelBands: 64,
                melMinHz: 0.0,
                melMaxHz: 0.0,
                htk: true,
                melNormalize: MelNormalizeType.None,
                logOffset: 1e-6,
                postNormalize: false);
#else
            _processor = new AudioProcessor(
                sampleRate: 16000,
                window: WindowFunction.Hann,
                windowLength: 400,
                hopLength: 160,
                fftLength: 512,
                preNormalize: 0.0,
                preemph: 0.0,
                center: false,
                nMelBands: 64,
                nMFCC: 64,
                melMinHz: 0.0,
                melMaxHz: 0.0,
                htk: true,
                melNormalize: MelNormalizeType.None,
                logOffset: 1e-6,
                postNormalize: false);
#endif
            _stftHopLength = stftHopLength;
            _stftWindowLength = stftWindowLength;
            _nMelBands = nMelBands;
            // _audioScale = 0.5 / short.MaxValue;
            _audioScale = 1.0 / short.MaxValue;

            _waveformBuffer = new short[2 * _stftHopLength + _stftWindowLength];
            _waveformCount = 0;
            _outputBuffer = new float[_nMelBands * (_stftWindowLength + _stftHopLength)];
            _outputCount = 0;
        }

        public int OutputCount { get { return _outputCount; } }
        public float[] OutputBuffer { get { return _outputBuffer; } }

        public int Write(short[] waveform, int offset, int count)
        {
            int written = 0;

            if (_waveformCount > 0)
            {
                int needed = ((_waveformCount - 1) / _stftHopLength) * _stftHopLength + _stftWindowLength - _waveformCount;
                written = Math.Min(needed, count);

                Array.Copy(waveform, offset, _waveformBuffer, _waveformCount, written);
                _waveformCount += written;

                int wavebufferOffset = 0;
                while (wavebufferOffset + _stftWindowLength < _waveformCount)
                {
                    _processor.MFCCStep(_waveformBuffer, wavebufferOffset, _audioScale, _outputBuffer, _outputCount);
                    _outputCount += _nMelBands;
                    wavebufferOffset += _stftHopLength;
                }

                if (written < needed)
                {
                    Array.Copy(_waveformBuffer, wavebufferOffset, _waveformBuffer, 0, _waveformCount - wavebufferOffset);
                    _waveformCount -= wavebufferOffset;
                    return written;
                }

                _waveformCount = 0;
                written -= _stftWindowLength - _stftHopLength;
            }

            while (written + _stftWindowLength < count)
            {
                if (_outputCount + _nMelBands >= _outputBuffer.Length)
                {
                    return written;
                }
                _processor.MFCCStep(waveform, offset + written, _audioScale, _outputBuffer, _outputCount);
                _outputCount += _nMelBands;
                written += _stftHopLength;
            }

            Array.Copy(waveform, offset + written, _waveformBuffer, 0, count - written);
            _waveformCount = count - written;
            written = count;
            return written;
        }

        public void ConsumeOutput(int count)
        {
            Array.Copy(_outputBuffer, count, _outputBuffer, 0, _outputCount - count);
            _outputCount -= count;
        }
    }
}
