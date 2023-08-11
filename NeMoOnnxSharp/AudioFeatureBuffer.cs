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
    internal class AudioFeatureBuffer<T1, T2> : IAudioFeatureBuffer<T1, T2>
    {
        private readonly IFrameTransform<T1, T2> _transform;
        private readonly int _numInputChannels;
        private readonly int _numOutputChannels;
        private readonly int _hopLength;
        private readonly int _winLength;
        private readonly T1[] _waveformBuffer;
        private int _waveformCount;
        private readonly T2[] _outputBuffer;
        private int _outputCount;

        public AudioFeatureBuffer(
            IFrameTransform<T1, T2> transform,
            int hopLength = 160,
            int numOutputFrames = 1000)
        {
            _transform = transform;
            _hopLength = hopLength;
            _winLength = transform.InputLength;
            _numInputChannels = 1;
            _numOutputChannels = transform.OutputLength;
            _waveformBuffer = new T1[2 * _hopLength + _winLength];
            _waveformCount = 0;
            _outputBuffer = new T2[_numOutputChannels * numOutputFrames];
            _outputCount = 0;
        }

        public int NumInputChannels => _numInputChannels;
        public int NumOutputChannels => _numOutputChannels;
        public int HopLength => _hopLength;
        public int WinLength => _winLength;
        public int OutputCount => _outputCount;
        public T2[] OutputBuffer => _outputBuffer;

        public int Write(T1[] waveform, int offset, int count)
        {
            int written = 0;

            if (_waveformCount > 0)
            {
                int needed = ((_waveformCount - 1) / _hopLength) * _hopLength + _winLength - _waveformCount;
                written = Math.Min(needed, count);

                Array.Copy(waveform, offset, _waveformBuffer, _waveformCount, written);
                _waveformCount += written;

                int wavebufferOffset = 0;
                while (wavebufferOffset + _winLength < _waveformCount)
                {
                    _transform.Transform(
                        _waveformBuffer.AsSpan(wavebufferOffset, _numInputChannels * _winLength),
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
                written -= _winLength - _hopLength;
            }

            while (written + _winLength < count)
            {
                if (_outputCount + _numOutputChannels >= _outputBuffer.Length)
                {
                    return written;
                }
                _transform.Transform(
                    waveform.AsSpan(offset + written, _numInputChannels * _winLength),
                    _outputBuffer.AsSpan(_outputCount, _numOutputChannels));
                _outputCount += _numOutputChannels;
                written += _hopLength;
            }

            Array.Copy(waveform, offset + written, _waveformBuffer, 0, count - written);
            _waveformCount = count - written;
            written = count;
            return written;
        }

        public int Write(Span<T1> waveform)
        {
            int written = 0;

            if (_waveformCount > 0)
            {
                int needed = ((_waveformCount - 1) / _hopLength) * _hopLength + _winLength - _waveformCount;
                written = Math.Min(needed, waveform.Length);

                waveform.Slice(0, written).CopyTo(_waveformBuffer.AsSpan(_waveformCount, written));
                _waveformCount += written;

                int wavebufferOffset = 0;
                while (wavebufferOffset + _winLength < _waveformCount)
                {
                    _transform.Transform(
                        _waveformBuffer.AsSpan(wavebufferOffset, _numInputChannels * _winLength),
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
                written -= _winLength - _hopLength;
            }

            while (written + _winLength < waveform.Length)
            {
                if (_outputCount + _numOutputChannels >= _outputBuffer.Length)
                {
                    return written;
                }
                _transform.Transform(
                    waveform.Slice(written, _numInputChannels * _winLength),
                    _outputBuffer.AsSpan(_outputCount, _numOutputChannels));
                _outputCount += _numOutputChannels;
                written += _hopLength;
            }

            waveform.Slice(written, waveform.Length - written).CopyTo(_waveformBuffer);
            _waveformCount = waveform.Length - written;
            written = waveform.Length;
            return written;
        }

        public void ConsumeOutput(int count)
        {
            Array.Copy(_outputBuffer, count, _outputBuffer, 0, _outputCount - count);
            _outputCount -= count;
        }
    }
}
