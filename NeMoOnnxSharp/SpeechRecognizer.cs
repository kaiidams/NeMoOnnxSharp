// Copyright (c) Katsuya Iida.  All Rights Reserved.
// See LICENSE in the project root for license information.

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Reflection;
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
        private readonly EncDecCTCModel _asrModel;
        int _audioBufferSize;
        int _audioBufferIndex;
        long _currentPosition;
        byte[] _audioBuffer;
        bool _isSpeech;

        private SpeechRecognizer(FrameVAD frameVad, EncDecCTCModel asrModel)
        {
            _frameVad = frameVad;
            _asrModel = asrModel;
            _currentPosition = 0;
            _audioBufferSize = sizeof(short) * _frameVad.SampleRate * 2; // 2sec
            _audioBufferIndex = 0;
            _audioBuffer = new byte[_audioBufferSize];
            _isSpeech = false;
            OnSpeechEnd = null;
            OnSpeechStart = null;
        }

        public SpeechRecognizer(string vadModelPath, string asrModelPath) : this(
            new FrameVAD(vadModelPath), new EncDecCTCModel(asrModelPath))
        {
        }

        public SpeechRecognizer(byte[] vadModel, byte[] asrModel) : this(
            new FrameVAD(vadModel), new EncDecCTCModel(asrModel))
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
                _audioBufferIndex += len;
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
                            string predictText = _asrModel.Transcribe(x);
                            OnSpeechEnd(_currentPosition + pos, x, predictText);
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
                        int audioSignalLength2 = audioSignal.Length * sizeof (short);
                        int pos2 = pos * sizeof(short);
                        _ChangeAudioBufferForSpeech(pos2, audioSignalLength2);
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

        private void _ChangeAudioBufferForSpeech(int pos2, int audioSignalLength2)
        {
            if (_audioBufferIndex + pos2 >= 0)
            {
                Array.Copy(
                    _audioBuffer, _audioBufferIndex + pos2,
                    _audioBuffer, 0,
                    -(pos2 + audioSignalLength2));
                _audioBufferIndex = -(pos2 + audioSignalLength2);
            }
            else if (_audioBufferIndex + pos2 + _audioBuffer.Length >= _audioBufferIndex + audioSignalLength2)
            {
                var tmp = new byte[_audioBuffer.Length + _AudioBufferIncrease];
                Array.Copy(
                    _audioBuffer, _audioBufferIndex + pos2 + _audioBuffer.Length,
                    tmp, 0,
                    -(_audioBufferIndex + pos2));
                Array.Copy(
                    _audioBuffer, 0,
                    tmp, -(_audioBufferIndex + pos2),
                    _audioBufferIndex + audioSignalLength2);
                _audioBufferIndex = _audioBufferIndex + audioSignalLength2;
            }
            else
            {
                var tmp = new byte[_audioBuffer.Length + _AudioBufferIncrease];
                Array.Copy(
                    _audioBuffer, _audioBufferIndex + audioSignalLength2,
                    tmp, 0,
                    _audioBuffer.Length - (_audioBufferIndex + audioSignalLength2));
                Array.Copy(
                    _audioBuffer, 0,
                    tmp, _audioBuffer.Length - (_audioBufferIndex + audioSignalLength2),
                    _audioBufferIndex + audioSignalLength2);
                _audioBufferIndex = _audioBuffer.Length;
            }
        }
    }
}
