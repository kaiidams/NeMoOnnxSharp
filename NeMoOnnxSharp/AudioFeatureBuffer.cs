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
    internal class AudioFeatureBuffer<T, S> : IAudioFeatureBuffer<T, S>
    {
        private readonly IFrameTransform<T, S> _processor;
        private readonly int _numInputChannels;
        private readonly int _numOutputChannels;
        private readonly int _hopLength;
        private readonly int _windowLength;
        private readonly double _audioScale;
        private readonly T[] _waveformBuffer;
        private int _waveformCount;
        private readonly S[] _outputBuffer;
        private int _outputCount;

        public AudioFeatureBuffer(
            IFrameTransform<T, S> processor,
            int hopLength = 160, int windowLength = 400,
            double audioScale = 1.0,
            int numOutputChannels = 64, int numOutputFrames = 1000)
        {
            _processor = processor;
            _hopLength = hopLength;
            _windowLength = windowLength;
            _audioScale = audioScale;
            _numInputChannels = 1;
            _numOutputChannels = numOutputChannels;
            _waveformBuffer = new T[2 * _hopLength + _windowLength];
            _waveformCount = 0;
            _outputBuffer = new S[_numOutputChannels * numOutputFrames];
            _outputCount = 0;
        }

        public int NumInputChannels => _numInputChannels;

        public int NumOutputChannels => _numOutputChannels;

        public int HopLength => _hopLength;

        public int WindowLength => _windowLength;

        public int OutputCount { get { return _outputCount; } }
        public S[] OutputBuffer { get { return _outputBuffer; } }

        public int Write(Span<T> waveform)
        {
            var x = waveform.ToArray();
            return Write(x, 0, x.Length);
        }

        public int Write(T[] waveform, int offset, int count)
        {
            int written = 0;

            if (_waveformCount > 0)
            {
                int needed = ((_waveformCount - 1) / _hopLength) * _hopLength + _windowLength - _waveformCount;
                written = Math.Min(needed, count);

                Array.Copy(waveform, offset, _waveformBuffer, _waveformCount, written);
                _waveformCount += written;

                int wavebufferOffset = 0;
                while (wavebufferOffset + _windowLength < _waveformCount)
                {
                    _processor.Transform(
                        _waveformBuffer.AsSpan(wavebufferOffset, _numInputChannels * _windowLength),
                        _outputBuffer.AsSpan(_outputCount, _numOutputChannels));
                    _outputCount += _numOutputChannels;
                    wavebufferOffset += _hopLength;
                }

                if (written < needed)
                {
                    Array.Copy(_waveformBuffer, wavebufferOffset, _waveformBuffer, 0, _waveformCount - wavebufferOffset);
                    _waveformCount -= wavebufferOffset;
                    return written;
                }

                _waveformCount = 0;
                written -= _windowLength - _hopLength;
            }

            while (written + _windowLength < count)
            {
                if (_outputCount + _numOutputChannels >= _outputBuffer.Length)
                {
                    return written;
                }
                _processor.Transform(
                    waveform.AsSpan(offset + written, _numInputChannels * _windowLength),
                    _outputBuffer.AsSpan(_outputCount, _numOutputChannels));
                _outputCount += _numOutputChannels;
                written += _hopLength;
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
