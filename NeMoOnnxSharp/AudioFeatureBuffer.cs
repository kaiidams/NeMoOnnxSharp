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
    public class AudioFeatureBuffer<T1, T2> : IAudioFeatureBuffer<T1, T2>
    {
        private readonly IFeaturizer<T1, T2> _transform;
        private readonly int _numInputChannels;
        private readonly int _numOutputChannels;
        private readonly int _hopLength;
        private readonly int _winLength;
        private readonly T1[] _inputBuffer;
        private int _inputCount;
        private readonly T2[] _outputBuffer;
        private int _outputCount;

        public int NumInputChannels => _numInputChannels;
        public int NumOutputChannels => _numOutputChannels;
        public int HopLength => _hopLength;
        public int WinLength => _winLength;
        public int OutputCount => _outputCount;
        public T2[] OutputBuffer => _outputBuffer;

        public AudioFeatureBuffer(
            IFeaturizer<T1, T2> transform,
            int hopLength,
            int numOutputFrames = 100)
        {
            _transform = transform;
            _hopLength = hopLength;
            _winLength = transform.InputLength;
            _numInputChannels = 1;
            _numOutputChannels = transform.OutputLength;
            _inputBuffer = new T1[(_winLength / _hopLength) * _hopLength + _winLength];
            _inputCount = 0;
            _outputBuffer = new T2[_numOutputChannels * numOutputFrames];
            _outputCount = 0;
        }

        public int Write(T1[] input, int offset, int count)
        {
            return Write(input.AsSpan(offset, count));
        }

        public int Write(Span<T1> input)
        {
            int written = 0;

            if (_inputCount > 0)
            {
                // Here _inputCount < _winLength. Copy n elements where
                //   0 < _inputCount <= 160  ->  n = _winLength - _inputCount
                // 160 < _inputCount <= 320  ->  n = _hopLength + _winLength - _inputCount
                // 320 < _inputCount <  400  ->  n = 2 * _hopLength + _winLength - _inputCount
                int needed = ((_inputCount - 1) / _hopLength) * _hopLength + _winLength - _inputCount;
                written = Math.Min(needed, input.Length);

                input.Slice(0, written).CopyTo(_inputBuffer.AsSpan(_inputCount, written));
                _inputCount += written;

                int inputBufferOffset = 0;
                while (inputBufferOffset + _winLength <= _inputCount)
                {
                    _transform.GetFeatures(
                        _inputBuffer.AsSpan(inputBufferOffset, _numInputChannels * _winLength),
                        _outputBuffer.AsSpan(_outputCount, _numOutputChannels));
                    _outputCount += _numOutputChannels;
                    inputBufferOffset += _hopLength;
                }

                if (written < needed)
                {
                    Array.Copy(_inputBuffer, inputBufferOffset, _inputBuffer, 0, _inputCount - inputBufferOffset);
                    _inputCount -= inputBufferOffset;
                    return written;
                }

                _inputCount = 0;
                written -= _winLength - _hopLength;
            }

            while (written + _winLength <= input.Length)
            {
                if (_outputCount + _numOutputChannels >= _outputBuffer.Length)
                {
                    return written;
                }
                _transform.GetFeatures(
                    input.Slice(written, _numInputChannels * _winLength),
                    _outputBuffer.AsSpan(_outputCount, _numOutputChannels));
                _outputCount += _numOutputChannels;
                written += _hopLength;
            }

            input.Slice(written).CopyTo(_inputBuffer);
            _inputCount = input.Length - written;
            written = input.Length;
            return written;
        }

        public void ConsumeOutput(int count)
        {
            Array.Copy(_outputBuffer, count, _outputBuffer, 0, _outputCount - count);
            _outputCount -= count;
        }
    }
}
