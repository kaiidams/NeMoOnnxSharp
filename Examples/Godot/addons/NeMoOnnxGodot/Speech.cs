using Godot;
using NeMoOnnxSharp;
using NeMoOnnxSharp.Models;
using System;
using System.Linq;

[GlobalClass]
public partial class Speech : Node
{
    [Signal]
    public delegate void DownloadEndEventHandler(bool success);

    [Signal]
    public delegate void SpeakEndEventHandler();

    [Signal]
    public delegate void SpeechStartDetectedEventHandler();

    [Signal]
    public delegate void SpeechEndDetectedEventHandler();

    [Signal]
    public delegate void RecognizedEventHandler(string text);

    private const int _AudioChunkSize = 4096;

    // Model downloading
    private string _language;
    int _loadingIndex = -1;
	private string[] _modelNames;
	private HttpRequest _httpRequest;
	private SpeechRecognizer _recognizer;
	private SpeechSynthesizer _synthesizer;

	// AudioBus
	private bool _transcribing;
	private bool _speaking;
	private AudioStreamPlayer _microphone;
	private AudioStreamPlayer _speaker;
	private AudioEffectCapture _capture;
	private AudioStreamGeneratorPlayback _playback;
	private short[] _waveData;
	private int _waveIndex;

	/// <summary>
	/// Language of speech. <c>English</c> or <c>German</c>
	/// </summary>
	[Export]
	public string Language
	{
		get
		{
			return _language;
		}
		set
		{
			_ChangeLanguage(value);
		}
	}

	[Export]
	public string ModelPath { get; set; }

    public bool IsTranscribing { get { return _transcribing; } }

    public bool IsSpeaking { get { return _speaking; } }


    // Called when the node enters the scene tree for the first time.
    public override void _Ready()
	{
		_SetupNetwork();
		_SetupAudioBus();
	}

	private void _SetupNetwork()
	{
		_httpRequest = new HttpRequest();
		AddChild(_httpRequest);
		_httpRequest.Connect("request_completed", new Callable(this, "HttpRequestCompleted"));
	}

	private void _SetupAudioBus()
	{
		int idx = AudioServer.BusCount;
		AudioServer.AddBus(idx);
		AudioServer.SetBusMute(idx, true);

		_capture = new AudioEffectCapture();
		AudioServer.AddBusEffect(idx, _capture);

		_microphone = new AudioStreamPlayer();
		_microphone.Stream = new AudioStreamMicrophone();
		_microphone.Bus = AudioServer.GetBusName(idx);
		AddChild(_microphone);
		_microphone.Stop();

		_speaker = new AudioStreamPlayer();
		_speaker.Stream = new AudioStreamGenerator();

		_speaker.Bus = AudioServer.GetBusName(0);
		AddChild(_speaker);
	}

	public override void _ExitTree()
	{
		_CleanupSpeech();
		_CleanupNetwork();
		_CleanupAudioBus();
	}

	private void _CleanupSpeech()
	{
		if (_recognizer != null)
		{
			_recognizer.Dispose();
			_recognizer = null;
		}
		if (_synthesizer != null)
		{
			_synthesizer.Dispose();
			_synthesizer = null;
		}
	}

	private void _CleanupNetwork()
	{
		if (_httpRequest != null)
		{
			RemoveChild(_httpRequest);
			_httpRequest.Dispose();
			_httpRequest = null;
		}
	}

	private void _CleanupAudioBus()
	{
		if (_microphone != null)
		{
			RemoveChild(_microphone);
			_microphone.Dispose();
			_microphone = null;
		}
		if (_speaker != null)
		{
			RemoveChild(_speaker);
			_speaker.Dispose();
			_speaker = null;
		}
	}

	/// <summary>
	/// Check if all model files are available locally
	/// </summary>
	/// <returns></returns>
	public bool CheckAllModelFiles()
	{
		if (_language == null || _modelNames == null)
		{
			throw new InvalidOperationException("Language must be set.");
		}
		foreach (var name in _modelNames)
		{
			var info = PretrainedModelInfo.Get(name);
			if (!_CheckCacheFile(_GetCachePathFromUrl(info.Location), info.Hash))
			{
				return false;
			}
		}
		return true;
	}

	/// <summary>
	/// Start downloading all model files needed for the language.
	/// </summary>
	public void DownloadAllModelFiles()
	{
		if (_language == null || _modelNames == null)
        {
            throw new InvalidOperationException("Language must be set.");
        }
        if (_loadingIndex >= 0)
		{
			throw new InvalidOperationException("Already start downloading.");
		}
		_DownloadNextModel();
    }

    private void _DownloadNextModel()
    {
        while (true)
        {
            _loadingIndex++;
            if (_loadingIndex >= _modelNames.Length)
            {
                EmitSignal(SignalName.DownloadEnd, true);
                return;
            }
            string name = _modelNames[_loadingIndex];
            var info = PretrainedModelInfo.Get(name);
            if (!_CheckCacheFile(_GetCachePathFromUrl(info.Location), info.Hash))
            {
                _httpRequest.Request(info.Location);
                return;
            }
        }
    }   
	
	/// <summary>
    /// Get the status of downloading
    /// </summary>
    /// <returns></returns>
    public DownloadStatus GetDownloadStatus()
    {
        if (_loadingIndex >= 0 && _loadingIndex < _modelNames.Length)
        {
            var result = new DownloadStatus();
            result.FileName = _modelNames[_loadingIndex];
            result.FileSize = _httpRequest.GetBodySize();
            int curSize = _httpRequest.GetDownloadedBytes();
            result.Percent = (int)(100.0 * curSize / result.FileSize);
            return result;
        }
        return null;
    }

    /// <summary>
    /// Load models
    /// </summary>
    public void LoadAllModels()
	{
		if (!CheckAllModelFiles())
		{
			throw new InvalidOperationException();
		}

		var config = _GetSpeechConfig();

		_recognizer = new SpeechRecognizer(config);
		_recognizer.SpeechStartDetected += (s, e) =>
		{
			EmitSignal(SignalName.SpeechStartDetected);
		};
		_recognizer.SpeechEndDetected += (s, e) =>
		{
			EmitSignal(SignalName.SpeechEndDetected);
		};
		_recognizer.Recognized += (s, e) =>
		{
            EmitSignal(SignalName.Recognized, e.Text);
		};

		_synthesizer = new SpeechSynthesizer(config);
	}

	/// <summary>
	/// Start transcribing
	/// </summary>
    public void StartTranscribe()
    {
		if (!_transcribing)
		{
			_transcribing = true;
			_microphone.Play();
		}
	}

	public void StopTranscribe()
	{
		if (_transcribing)
		{
			_transcribing = false;
			_microphone.Stop();
		}
	}

    public void SpeakText(string text)
    {
        var result = _synthesizer.SpeakText(text);
        _waveData = result.AudioData;
        _waveIndex = 0;
        _speaker.Play();
        _playback = _speaker.GetStreamPlayback() as AudioStreamGeneratorPlayback;
        (_speaker.Stream as AudioStreamGenerator).MixRate = result.SampleRate;
        _FillBuffer();
        _speaking = true;
    }

	public void CancelSpeak()
	{
        if (_speaking)
        {
            _speaking = false;
            _waveIndex = 0;
            _waveData = null;
        }
    }

    private void _FillBuffer()
	{
		if (_speaking)
		{
			if (_waveIndex >= _waveData.Length)
			{
				_speaking = false;
				_waveIndex = 0;
				_waveData = null;
                EmitSignal(SignalName.SpeakEnd);
				return;
			}

			var toFill = Math.Min(_waveData.Length - _waveIndex, _playback.GetFramesAvailable());
			toFill = Math.Min(toFill, _AudioChunkSize);

			if (toFill > 0)
			{
				var buffer = new Vector2[toFill];
				for (int i = 0; i < buffer.Length; i++)
				{
					buffer[i] = (Vector2.One / short.MaxValue) * _waveData[_waveIndex + i];
				}
				_playback.PushBuffer(buffer);
				_waveIndex += toFill;
			}
		}
	}

	private void _CaptureBuffer()
	{
		if (_transcribing)
		{
			int avail = _capture.GetFramesAvailable();
			if (avail > 0)
			{
				var buffer = _capture.GetBuffer(avail);
				// captured audio from Godot is 48kHz, but we need 16kHz.
				// Downsample here.
				var input = new short[buffer.Length / 3];
				for (int i = 0; i < input.Length; i++)
				{
					input[i] = (short)((buffer[i * 3].X + buffer[i * 3].Y) * (short.MaxValue / 2.0));
				}
				_recognizer.Write(input);
			}
		}
	}

	public void HttpRequestCompleted(
		int result, int responseCode, string[] headers, byte[] body)
	{
		if (result == 0)
		{
			string name = _modelNames[_loadingIndex];
			var info = PretrainedModelInfo.Get(name);
			var file = FileAccess.Open(_GetCachePathFromUrl(info.Location), FileAccess.ModeFlags.Write);
			file.StoreBuffer(body);
			file.Close();

            _DownloadNextModel();
        }
		else
		{
            _loadingIndex = -1;
            EmitSignal(SignalName.DownloadEnd, false);
		}
	}

	// Called every frame. 'delta' is the elapsed time since the previous frame.
	public override void _Process(double delta)
	{
		_FillBuffer();
		_CaptureBuffer();
	}

	private void _ChangeLanguage(string language)
	{
		if (_transcribing)
		{
			_transcribing = false;
			_microphone.Stop();
		}
		if (_speaking)
		{
			_speaking = false;
            _waveIndex = 0;
            _waveData = null;
        }

        _language = language;

		_modelNames = _GetModelList();
		if (CheckAllModelFiles())
		{
			_loadingIndex = _modelNames.Length;
		}
		else
		{
			_loadingIndex = -1;
		}
	}

	private void _SetStatusText(string text)
	{
	}

	private string[] _GetModelList()
	{
		if (_language == "English")
		{
			return new string[]
			{
				"vad_marblenet",
				"stt_en_quartznet15x5",
				"cmudict-0.7b_nv22.10",
				"heteronyms-052722",
				"tts_en_fastpitch",
				"tts_en_hifigan",
			};
		}
		else if (_language == "German")
		{
			return new string[]
			{
				"vad_marblenet",
				"stt_de_quartznet15x5",
				"tts_de_fastpitch_singleSpeaker_thorstenNeutral_2210",
				"tts_de_hifigan_singleSpeaker_thorstenNeutral_2210",
			};
		}
		else
		{
			throw new ArgumentException();
		}
	}

	private SpeechConfig _GetSpeechConfig()
	{
		SpeechConfig config;
		if (_language == "English")
		{
			config = new SpeechConfig
			{
				vad = new EncDecClassificationConfig
				{
					modelPath = _GetModelGlobalPath("vad_marblenet"),
					labels = EncDecClassificationConfig.VADLabels
				},
				asr = new EncDecCTCConfig
				{
					modelPath = _GetModelGlobalPath("stt_en_quartznet15x5"),
					vocabulary = EncDecCTCConfig.EnglishVocabulary
				},
				specGen = new SpectrogramGeneratorConfig
				{
					modelPath = _GetModelGlobalPath("tts_en_fastpitch"),
					phonemeDictPath = _GetModelGlobalPath("cmudict-0.7b_nv22.10"),
					heteronymsPath = _GetModelGlobalPath("heteronyms-052722"),
					textTokenizer = "EnglishPhonemesTokenizer"
				},
				vocoder = new VocoderConfig
				{
					modelPath = _GetModelGlobalPath("tts_en_hifigan"),
				},
			};
		}
		else if (_language == "German")
		{
			config = new SpeechConfig
			{
				vad = new EncDecClassificationConfig
				{
					modelPath = _GetModelGlobalPath("vad_marblenet"),
					labels = EncDecClassificationConfig.VADLabels
				},
				asr = new EncDecCTCConfig
				{
					modelPath = _GetModelGlobalPath("stt_de_quartznet15x5"),
					vocabulary = EncDecCTCConfig.GermanVocabulary
				},
				specGen = new SpectrogramGeneratorConfig
				{
					modelPath = _GetModelGlobalPath("tts_de_fastpitch_singleSpeaker_thorstenNeutral_2210"),
					textTokenizer = "GermanCharsTokenizer"
				},
				vocoder = new VocoderConfig
				{
					modelPath = _GetModelGlobalPath("tts_de_hifigan_singleSpeaker_thorstenNeutral_2210"),
				},
			};
		}
		else
		{
			throw new ArgumentException();
		}
		return config;
	}

	private string _GetModelGlobalPath(string name)
	{
		var info = PretrainedModelInfo.Get(name);
		string path = _GetCachePathFromUrl(info.Location);
		return ProjectSettings.GlobalizePath(path);
	}

	private static bool _CheckCacheFile(string cacheFilePath, string expectedChecksum)
	{
		if (FileAccess.FileExists(cacheFilePath))
		{
			string checksum = FileAccess.GetSha256(cacheFilePath);
			if (string.Compare(checksum, expectedChecksum, true) == 0)
			{
				return true;
			}
		}
		return false;
	}

	private static string _GetCachePathFromUrl(string url)
	{
		int index = url.LastIndexOf('/');
		return "user://" + url.Substring(index + 1);
	}
}
