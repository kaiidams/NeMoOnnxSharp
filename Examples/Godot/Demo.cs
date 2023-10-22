using Godot;
using NeMoOnnxSharp;
using NeMoOnnxSharp.Models;
using System;
using System.Linq;

public partial class Demo : Node2D
{
	private const int AudioChunkSize = 4096;

	// User interface
	private TextEdit _textEdit;
	private Label _label;
	private Button _downloadButton;
	private Button _transcribeButton;
	private Button _speakButton;
	private Button _stopButton;
	private Container _buttons;
	private MenuButton _languageMenu;

	// Model downloading
	int _loadingIndex = -1;
	private string _language;
	private string[] _modelNames;
	private HttpRequest _httpRequest;
	private SpeechRecognizer _recognizer;
	private SpeechSynthesizer _synthesizer;

	private bool _transcribing;
	private bool _speaking;
	private AudioStreamPlayer _microphone;
	private AudioStreamPlayer _speaker;
	private AudioEffectCapture _capture;
	private AudioStreamGeneratorPlayback _playback;
	private short[] _waveData;
	private int _waveIndex;

	// Called when the node enters the scene tree for the first time.
	public override void _Ready()
	{
		_SetupComponents();
		_SetupNetwork();
		_SetupAudioBus();
		_ChangeLanguage("English");
	}

	private void _SetupComponents()
	{
		_label = GetNode<Label>("CanvasLayer/VBoxContainer/StatusLabel");
		_transcribeButton = GetNode<Button>("CanvasLayer/VBoxContainer/Buttons/TranscriptButton");
		_transcribeButton.Connect("pressed", new Callable(this, "OnTranscriptClick"));
		_speakButton = GetNode<Button>("CanvasLayer/VBoxContainer/Buttons/SpeakButton");
		_speakButton.Connect("pressed", new Callable(this, "OnSpeakClick"));
		_stopButton = GetNode<Button>("CanvasLayer/VBoxContainer/Buttons/StopButton");
		_stopButton.Connect("pressed", new Callable(this, "OnStopClick"));
		_textEdit = GetNode<TextEdit>("CanvasLayer/VBoxContainer/TextEdit");
		_downloadButton = GetNode<Button>("CanvasLayer/VBoxContainer/DownloadButton");
		_downloadButton.Connect("pressed", new Callable(this, "OnDownloadClick"));
		_buttons = GetNode<Container>("CanvasLayer/VBoxContainer/Buttons");
		_languageMenu = GetNode<MenuButton>("CanvasLayer/VBoxContainer/GridContainer/MenuButton");
		_languageMenu.GetPopup().Connect("index_pressed", new Callable(this, "OnLanguageMenu"));
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
		_DisposeSpeech();
	}

	private void UpdateButtons()
	{
		_transcribeButton.Disabled = _transcribing;
		_speakButton.Disabled = _speaking;
		_stopButton.Disabled = !_speaking && !_transcribing;
	}

	public void OnLanguageMenu(int index)
	{
		if (index == 0)
		{
			_ChangeLanguage("English");
		}
		else if (index == 1)
		{
			_ChangeLanguage("German");
		}
		else
		{
			throw new ArgumentException();
		}
	}

	public void OnDownloadClick()
	{
		_downloadButton.Disabled = true;
		_loadingIndex = -1;
		_DownloadNextModel();
	}

	public void OnTranscriptClick()
	{
		if (!_transcribing)
		{
			_transcribing = true;
			_microphone.Play();
			_SetStatusText("Silent");
			UpdateButtons();
		}
	}

	public void OnSpeakClick()
	{
		if (_synthesizer != null)
		{
			string text = _textEdit.Text;
			if (!string.IsNullOrWhiteSpace(text))
			{
				try
				{
					var result = _synthesizer.SpeakText(text);
					_waveData = result.AudioData.ToList().ToArray();
					_waveIndex = 0;
					GD.Print(string.Format("AudioData.Length={0}", result.AudioData.Length));
					_speaker.Play();
					_playback = _speaker.GetStreamPlayback() as AudioStreamGeneratorPlayback;
					(_speaker.Stream as AudioStreamGenerator).MixRate = result.SampleRate;
					_FillBuffer();
				}
				catch (Exception ex)
				{
					GD.Print(ex.ToString());
				}

				_speaking = true;
				UpdateButtons();
			}
		}
	}

	public void OnStopClick()
	{
		if (_transcribing)
		{
			_transcribing = false;
			_microphone.Stop();
		}
		if (_speaking)
		{
			_speaking = false;
		}
		_SetStatusText("");
		UpdateButtons();
	}

	private void _FillBuffer()
	{
		if (_speaking)
		{
			if (_waveIndex >= _waveData.Length)
			{
				_speaking = false;
				UpdateButtons();
				return;
			}

			var toFill = Math.Min(_waveData.Length - _waveIndex, _playback.GetFramesAvailable());
			toFill = Math.Min(toFill, AudioChunkSize);

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
			_downloadButton.Disabled = false;
			_SetStatusText("Error downloading models");
		}
	}

	// Called every frame. 'delta' is the elapsed time since the previous frame.
	public override void _Process(double delta)
	{
		_FillBuffer();
		_CaptureBuffer();

		if (_loadingIndex >= 0 && _loadingIndex < _modelNames.Length)
		{
			int totalSize = _httpRequest.GetBodySize();
			int curSize = _httpRequest.GetDownloadedBytes();
			string name = _modelNames[_loadingIndex];
			int kb = totalSize / 1024;
			int percent = (int)(100.0 * curSize / totalSize);
			_label.Text = string.Format("Loading {0} {1}% ({1}KB)", name, percent, kb);
		}
	}

	private void _ChangeLanguage(string language)
	{
		GD.Print(string.Format("Language={0}", language));

		if (_transcribing)
		{
			_transcribing = false;
			_microphone.Stop();
		}
		if (_speaking)
		{
			_speaking = false;
		}

		_languageMenu.Text = language;
		_language = language;

		_modelNames = _GetModelList();
		if (_CheckAllCacheFile())
		{
			_loadingIndex = _modelNames.Length;
			_ModelsDownloaded();
		}
		else
		{
			_loadingIndex = -1;
			_downloadButton.Show();
			_buttons.Hide();
		}
	}

	private void _DisposeSpeech()
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

	private void _DownloadNextModel()
	{
		while (true)
		{
			_loadingIndex++;
			if (_loadingIndex >= _modelNames.Length)
			{
				_ModelsDownloaded();
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

	private void _ModelsDownloaded()
	{
		try
		{
			var config = _GetSpeechConfig();

			_recognizer = new SpeechRecognizer(config);
			_recognizer.SpeechStartDetected += (s, e) =>
			{
				_SetStatusText("Speaking");
			};
			_recognizer.SpeechEndDetected += (s, e) =>
			{
				_SetStatusText("Silent");
			};
			_recognizer.Recognized += (s, e) =>
			{
				_textEdit.Text = e.Text;
			};
			GD.Print("SpeechRecognizer ready");

			_synthesizer = new SpeechSynthesizer(config);
			GD.Print("SpeechSynthesizer ready");

			if (_language == "English")
			{
				_textEdit.Text = "The quick brown fox jumps over the lazy dog.";
			}
			else if (_language == "German")
			{
				_textEdit.Text = "Franz jagt im komplett verwahrlosten Taxi quer durch Bayern.";
			}

			_downloadButton.Hide();
			_buttons.Show();
			UpdateButtons();
			_SetStatusText("");
		}
		catch (Exception)
		{
			_downloadButton.Disabled = false;
			_SetStatusText("Error building models");
		}
	}

	private void _SetStatusText(string text)
	{
		_label.Text = text;
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

	private bool _CheckAllCacheFile()
	{
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
