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
        public delegate void SpeechStart(long position);
        public delegate void SpeechEnd(long position, short[] audioSignal, string? transcript);

        private readonly FrameVAD _frameVad;
        private readonly EncDecCTCModel _asrModel;
        private readonly int _audioBufferIncrease;
        private readonly int _audioBufferSize;
        int _audioBufferIndex;
        long _currentPosition;
        byte[] _audioBuffer;
        bool _isSpeech;
        private readonly float _speechStartThreadhold;
        private readonly float _speechEndThreadhold;

        private SpeechRecognizer(FrameVAD frameVad, EncDecCTCModel asrModel)
        {
            _frameVad = frameVad;
            _asrModel = asrModel;
            _currentPosition = 0;
            _audioBufferIndex = 0;
            _audioBufferSize = sizeof(short) * _frameVad.SampleRate * 2; // 2sec
            _audioBufferIncrease = sizeof(short) * 5 * _frameVad.SampleRate; // 10sec
            _audioBuffer = new byte[_audioBufferSize];
            _isSpeech = false;
            OnSpeechEnd = null;
            OnSpeechStart = null;
            _speechStartThreadhold = 0.7f;
            _speechEndThreadhold = 0.3f;
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
                        var tmp = new byte[_audioBuffer.Length + _audioBufferIncrease];
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
                int len2 = (len / sizeof(short)) * sizeof(short);
                var audioSignal = MemoryMarshal.Cast<byte, short>(_audioBuffer.AsSpan(_audioBufferIndex, len2));
                _audioBufferIndex += len;
                _currentPosition += audioSignal.Length;
                _Transcribe(audioSignal);
            }
        }

        private void _Transcribe(Span<short> audioSignal)
        {
            var pos = -(audioSignal.Length + _frameVad.Position);
            var result = _frameVad.Transcribe(audioSignal);
            foreach (var prob in result)
            {
                if (_isSpeech)
                {
                    if (prob < _speechEndThreadhold)
                    {
                        _isSpeech = false;
                        int posBytes = pos * sizeof(short);
                        if (OnSpeechEnd != null)
                        {
                            var audio = _audioBuffer.AsSpan(0, _audioBufferIndex + posBytes);
                            var x = MemoryMarshal.Cast<byte, short>(audio).ToArray();
                            string predictText = _asrModel.Transcribe(x);
                            OnSpeechEnd(_currentPosition + pos, x, predictText);
                        }
                        _ResetAudioBuffer(posBytes);
                    }
                }
                else
                {
                    if (prob >= _speechStartThreadhold)
                    {
                        _isSpeech = true;
                        if (OnSpeechStart != null) OnSpeechStart(_currentPosition + pos);
                        int pos2 = pos * sizeof(short);
                        _ChangeAudioBufferForSpeech(pos2);
                    }
                }
                pos += _frameVad.HopLength;
            }
        }

        private void _ResetAudioBuffer(int posBytes)
        {
            var tmp = new byte[_audioBufferSize];
            Array.Copy(
                _audioBuffer, _audioBufferIndex + posBytes,
                tmp, 0,
                -posBytes);
            _audioBuffer = tmp;
            _audioBufferIndex = -posBytes;
        }

        private void _ChangeAudioBufferForSpeech(int posBytes)
        {
            int audioBufferStart = _audioBufferIndex + posBytes;
            int audioBufferEnd = _audioBufferIndex;
            if (audioBufferStart >= 0)
            {
                Array.Copy(
                    _audioBuffer, audioBufferStart,
                    _audioBuffer, 0,
                    audioBufferEnd - audioBufferStart);
                _audioBufferIndex = audioBufferEnd - audioBufferStart;
            }
            else if (audioBufferStart + _audioBuffer.Length >= audioBufferEnd)
            {
                var tmp = new byte[_audioBuffer.Length + _audioBufferIncrease];
                Array.Copy(
                    _audioBuffer, audioBufferStart + _audioBuffer.Length,
                    tmp, 0,
                    -audioBufferStart);
                Array.Copy(
                    _audioBuffer, 0,
                    tmp, -audioBufferStart,
                    audioBufferEnd);
                _audioBuffer = tmp;
                _audioBufferIndex = audioBufferEnd - audioBufferStart;
            }
            else
            {
                var tmp = new byte[_audioBuffer.Length + _audioBufferIncrease];
                Array.Copy(
                    _audioBuffer, audioBufferEnd,
                    tmp, 0,
                    _audioBuffer.Length - audioBufferEnd);
                Array.Copy(
                    _audioBuffer, 0,
                    tmp, _audioBuffer.Length - audioBufferEnd,
                    audioBufferEnd);
                _audioBuffer = tmp;
                _audioBufferIndex = _audioBuffer.Length;
            }
        }
    }
}
