// Copyright (c) Katsuya Iida.  All Rights Reserved.
// See LICENSE in the project root for license information.

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;

namespace NeMoOnnxSharp
{
    public class SpeechRecognizer : IDisposable
    {
        private int _AudioBufferIncrease = 5 * 16000;
        public delegate void SpeechStart(long position);
        public delegate void SpeechEnd(long position, short[] audioSignal, string? transcript);

        private readonly FrameVAD _frameVad;
        int _audioBufferSize;
        int _audioBufferIndex;
        long _currentPosition;
        byte[] _audioBuffer;
        bool _isSpeech;

        private SpeechRecognizer(FrameVAD frameVad)
        {
            _frameVad = frameVad;
            _currentPosition = 0;
            _audioBufferSize = sizeof(short) * _frameVad.SampleRate * 2; // 2sec
            _audioBufferIndex = 0;
            _audioBuffer = new byte[_audioBufferSize];
            _isSpeech = false;
            OnSpeechEnd = null;
            OnSpeechStart = null;
        }

        public SpeechRecognizer(string modelPath) : this(
            new FrameVAD(modelPath))
        {
        }

        public SpeechRecognizer(byte[] model) : this(
            new FrameVAD(model))
        {
        }

        public int SampleRate => _frameVad.SampleRate;
        public SpeechEnd? OnSpeechEnd { get; set; }
        public SpeechStart? OnSpeechStart { get; set; }

        public void Dispose()
        {
            _frameVad.Dispose();
        }

        public void Transcribe(byte[] input, int offset, int count)
        {
            Transcribe(input.AsSpan(offset, count));
        }

        public void Transcribe(Span<byte> input)
        {
            while (input.Length > 0)
            {
                int len = input.Length;
                if (_isSpeech)
                {
                    if (len > _audioBuffer.Length - _audioBufferIndex)
                    {
                        var tmp = new byte[_audioBuffer.Length + _AudioBufferIncrease];
                        Array.Copy(_audioBuffer, tmp, _audioBufferIndex);
                        _audioBuffer = tmp;
                    }
                }
                else
                {
                    if (_audioBufferIndex >= _audioBuffer.Length)
                    {
                        _audioBufferIndex = 0;
                    }
                    len = Math.Min(_audioBuffer.Length - _audioBufferIndex, len);
                }
                input.Slice(0, len).CopyTo(_audioBuffer.AsSpan(_audioBufferIndex, len));
                input = input.Slice(len);
                len = (len / sizeof(short)) * sizeof(short);
                var audioSignal = MemoryMarshal.Cast<byte, short>(_audioBuffer.AsSpan(_audioBufferIndex, len));
                _Transcribe(audioSignal);
            }
        }

        private void _Transcribe(Span<short> audioSignal)
        {
            var pos = -_frameVad.Position;
            var result = _frameVad.Transcribe(audioSignal);
            foreach (var prob in result)
            {
                if (_isSpeech)
                {
                    if (prob < 0.3)
                    {
                        _isSpeech = false;
                        if (OnSpeechEnd != null)
                        {
                            var audio = _audioBuffer.AsSpan(0, _audioBufferIndex + sizeof(short) * audioSignal.Length);
                            var x = MemoryMarshal.Cast<byte, short>(audio).ToArray();
                            OnSpeechEnd(_currentPosition + pos, x, null);
                        }
                        _ResetAudioBuffer();
                    }
                }
                else
                {
                    if (prob >= 0.7)
                    {
                        _isSpeech = true;
                        if (OnSpeechStart != null) OnSpeechStart(_currentPosition + pos);
                        _ChangeAudioBufferForSpeech(pos, audioSignal);
                    }
                }
                pos += 160;
            }
            _currentPosition += audioSignal.Length;
        }

        private void _ResetAudioBuffer()
        {
            _audioBufferIndex = 0;
            _audioBuffer = new byte[_audioBufferSize];
        }

        private void _ChangeAudioBufferForSpeech(int pos, Span<short> audioSignal)
        {
            if (_audioBufferIndex + sizeof(short) * pos >= 0)
            {
                Array.Copy(
                    _audioBuffer, _audioBufferIndex + sizeof(short) * pos,
                    _audioBuffer, 0,
                    -sizeof(short) * (pos + audioSignal.Length));
                _audioBufferIndex = -sizeof(short) * (pos + audioSignal.Length);
            }
            else if (_audioBufferIndex + sizeof(short) * pos + _audioBuffer.Length >= _audioBufferIndex + sizeof(short) * audioSignal.Length)
            {
                var tmp = new byte[_audioBuffer.Length + _AudioBufferIncrease];
                Array.Copy(
                    _audioBuffer, _audioBufferIndex + sizeof(short) * pos + _audioBuffer.Length,
                    tmp, 0,
                    -(_audioBufferIndex + sizeof(short) * pos));
                Array.Copy(
                    _audioBuffer, 0,
                    tmp, -(_audioBufferIndex + sizeof(short) * pos),
                    _audioBufferIndex + sizeof(short) * audioSignal.Length);
                _audioBufferIndex = _audioBufferIndex + audioSignal.Length;
            }
            else
            {
                var tmp = new byte[_audioBuffer.Length + _AudioBufferIncrease];
                Array.Copy(
                    _audioBuffer, _audioBufferIndex + sizeof(short) * audioSignal.Length,
                    tmp, 0,
                    _audioBuffer.Length - (_audioBufferIndex + sizeof(short) * audioSignal.Length));
                Array.Copy(
                    _audioBuffer, 0,
                    tmp, _audioBuffer.Length - (_audioBufferIndex + sizeof(short) * audioSignal.Length),
                    _audioBufferIndex + audioSignal.Length);
                _audioBufferIndex = _audioBuffer.Length;
            }
        }
    }
}
