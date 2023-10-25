using Godot;
using System;

public partial class Demo : Node2D
{
	// User interface
	private TextEdit _textEdit;
	private Label _label;
	private Button _downloadButton;
	private Button _transcribeButton;
	private Button _speakButton;
	private Button _stopButton;
	private Container _buttons;
	private MenuButton _languageMenu;

	private Speech _speech;

	// Called when the node enters the scene tree for the first time.
	public override void _Ready()
	{
		_SetupComponents();
		_SetupSpeech();
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

	private void _SetupSpeech()
	{
		_speech = GetNode<Speech>("Speech");
        _speech.SpeakEnd += () =>
        {
            UpdateButtons();
        };
        _speech.SpeechStartDetected += () =>
        {
            _SetStatusText("Speech");
        };
        _speech.SpeechEndDetected += () =>
        {
            _SetStatusText("Silent");
        };
        _speech.Recognized += (string text) =>
        {
			_textEdit.Text = text;
        };
		_speech.DownloadEnd += (bool success) =>
		{
            _downloadButton.Disabled = false;
            _SetStatusText("Error downloading models");
        };
    }

    public override void _ExitTree()
	{
	}

	private void UpdateButtons()
	{
		_transcribeButton.Disabled = _speech.IsTranscribing;
		_speakButton.Disabled = _speech.IsSpeaking;
		_stopButton.Disabled = !_speech.IsSpeaking && !_speech.IsTranscribing;
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
		_speech.DownloadAllModelFiles();
	}

	public void OnTranscriptClick()
	{
		if (!_speech.IsTranscribing)
		{
			_speech.StartTranscribe();
			_SetStatusText("Silent");
			UpdateButtons();
		}
	}

	public void OnSpeakClick()
	{
		string text = _textEdit.Text;
		if (!string.IsNullOrWhiteSpace(text))
		{
			try
			{
				_speech.SpeakText(text);
			}
			catch (Exception ex)
			{
				GD.Print(ex.ToString());
			}

			UpdateButtons();
		}
	}

	public void OnStopClick()
	{
		_speech.StopTranscribe();
		_speech.CancelSpeak();
		_SetStatusText("");
		UpdateButtons();
	}

	public void HttpRequestCompleted(
		int result, int responseCode, string[] headers, byte[] body)
	{
		if (result == 0)
		{			
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
		var status = _speech.GetDownloadStatus();
		if (status != null)
		{
			_label.Text = string.Format("Loading {0} {1}% ({1}KB)", status.FileName, status.Percent, status.FileSize);
		}
	}

	private void _ChangeLanguage(string language)
	{
		GD.Print(string.Format("Language={0}", language));
		_speech.Language = language;
		if (_speech.CheckAllModelFiles())
		{
			_ModelsDownloaded();
		}
		else
		{
			_downloadButton.Show();
			_buttons.Hide();
		}
	}

	private void _ModelsDownloaded()
	{
		try
		{
			_speech.LoadAllModels();
			GD.Print("SpeechSynthesizer ready");

			if (_speech.Language == "English")
			{
				_textEdit.Text = "The quick brown fox jumps over the lazy dog.";
			}
			else if (_speech.Language == "German")
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
}
