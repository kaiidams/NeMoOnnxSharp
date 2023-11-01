using System;
using Android.App;
using Android.OS;
using Android.Runtime;
using Android.Views;
using System.IO;
using Android.Media;
using NeMoOnnxSharp;
using System.Threading.Tasks;
using Android;
using Android.Content.PM;

#nullable disable
namespace NeMoOnnxAndroidApp
{
    [Activity(Label = "@string/app_name", MainLauncher = true)]
    public class MainActivity : Activity
    {
        private const int AudioBufferLength = 4096; // 256 msec
        private const int RecordAudioPermission = 1;

        private static string[] LanguageList =
        {
            "English",
            "German"
        };

        private Button _downloadButton;
        private Button _startRecordButton;
        private Button _stopRecordButton;
        private Button _startPlayButton;
        private Button _stopPlayButton;
        private TextView _statusText;
        private EditText _inputTextEditText;

        private ModelDownloader _modelDownloader;
        private SpeechRecognizer _speechRecognizer;
        private SpeechSynthesizer _speechSynthesizer;
        private Thread _downloadThread;
        private CancellationTokenSource _downloadCancellationToken;
        private DateTime _lastProgressUpdateTime;

        private bool _IsRecording => _recordingThread != null;
        private bool _IsPlaying => _playingThread != null;
        private Thread _recordingThread;
        private Thread _playingThread;
        private AudioRecord _audioRecorder;
        private ProgressBar _progressBar;
        private Spinner _spinner;

        protected override void OnCreate(Bundle savedInstanceState)
        {
            base.OnCreate(savedInstanceState);

            // Set our view from the "main" layout resource
            SetContentView(Resource.Layout.activity_main);

            _spinner = FindViewById<Spinner>(Resource.Id.language);
            var adapter = ArrayAdapter.CreateFromResource(
                this,
                Resource.Array.language,
                Android.Resource.Layout.SimpleSpinnerItem
            );
            adapter.SetDropDownViewResource(Android.Resource.Layout.SimpleSpinnerDropDownItem);
            _spinner.Adapter = adapter;
            _spinner.ItemSelected += _SpinnerItemSelected;

            _statusText = FindViewById<TextView>(Resource.Id.status);

            _downloadButton = FindViewById<Button>(Resource.Id.download_models);
            _downloadButton.Click += _DownloadClick;
            _startRecordButton = FindViewById<Button>(Resource.Id.start_recording);
            _startRecordButton.Click += _StartRecordingClick;
            _stopRecordButton = FindViewById<Button>(Resource.Id.stop_recording);
            _stopRecordButton.Click += _StopRecordingClick;
            _startPlayButton = FindViewById<Button>(Resource.Id.start_playing);
            _startPlayButton.Click += _StartPlayingClick;
            _stopPlayButton = FindViewById<Button>(Resource.Id.stop_playing);
            _stopPlayButton.Click += _StopPlayingClick;

            _inputTextEditText = FindViewById<EditText>(Resource.Id.input_text);

            string cacheDirPath = CacheDir.AbsolutePath;
            _modelDownloader = new ModelDownloader(cacheDirPath);
            _speechRecognizer = null;
            _speechSynthesizer = null;
            _UpdateSpeechButtons();
            _progressBar = FindViewById<ProgressBar>(Resource.Id.download_progress);
            _lastProgressUpdateTime = DateTime.Now;
            _modelDownloader.Progress += _DownloadProgress;
        }

        private void _DownloadProgress(object sender, ModelDownloader.ProgressEventArgs e)
        {
            var currentTime = DateTime.Now;
            if ((currentTime - _lastProgressUpdateTime).TotalMilliseconds >= 100)
            {
                _lastProgressUpdateTime = currentTime;
                this.RunOnUiThread(() =>
                {
                    _statusText.Text = string.Format(GetString(Resource.String.status_downloading), e.FileName);
                    _progressBar.Max = (int)e.ContentLength;
                    _progressBar.Progress = (int)e.CurrentPosition;
                    _progressBar.Min = 0;
                });
            }
        }

        protected override void OnStop()
        {
            _CleanSpeech();
            _modelDownloader.Dispose();
            _modelDownloader = null;
        }

        private void _CleanSpeech()
        {
            _StopRecording();
            _StopPlaying();
            if (_speechRecognizer != null)
            {
                _speechRecognizer.Dispose();
                _speechRecognizer = null;
            }
            if (_speechSynthesizer != null)
            {
                _speechSynthesizer.Dispose();
                _speechSynthesizer = null;
            }
        }

        private void _SpinnerItemSelected(object sender, EventArgs e)
        {
            int index = _spinner.SelectedItemPosition;
            string language = LanguageList[index];
            string text;
            if (language == "German")
            {
                text = GetString(Resource.String.default_input_text_de);
            }
            else
            {
                text = GetString(Resource.String.default_input_text_en);
            }
            _CleanSpeech();
            _inputTextEditText.Text = text;
            _UpdateSpeechButtons();
            _modelDownloader.Language = language;
            _PrepareSpeech(false);
        }

        private void _PrepareSpeech(bool download)
        {
            _downloadThread = new Thread(async () =>
            {
                SpeechRecognizer speechRecognizer = null;
                SpeechSynthesizer speechSynthesizer = null;
                try
                {
                    if (download)
                    {
                        _downloadCancellationToken = new CancellationTokenSource();
                        await _modelDownloader.MayDownloadAllAsync(_downloadCancellationToken.Token);
                    }
                    RunOnUiThread(() =>
                    {
                        if (_downloadThread != null)
                        {
                            _statusText.Text = GetString(Resource.String.status_preparing);
                            _downloadButton.Visibility = ViewStates.Invisible;
                            _progressBar.Progress = 0;
                            _downloadButton.Enabled = false;
                        }
                    });
                    if (_modelDownloader.ModelFilesAvailable)
                    {
                        var config = _modelDownloader.BuildSpeechConfig();
                        speechRecognizer = new SpeechRecognizer(config);
                        speechSynthesizer = new SpeechSynthesizer(config);
                    }
                }
                catch (Exception)
                {
                }
                RunOnUiThread(() =>
                {
                    _SpeechPrepared(speechRecognizer, speechSynthesizer);
                });
            });
            _downloadThread.Start();
            _UpdateDownloadButton();
        }

        private void _SpeechPrepared(
            SpeechRecognizer speechRecognizer,
            SpeechSynthesizer speechSynthesizer)
        {
            if (_downloadThread == null)
            {
                return;
            }

            _downloadThread.Join();
            _downloadThread = null;
            _downloadCancellationToken = null;

            if (speechRecognizer == null && speechSynthesizer == null)
            {
                _statusText.Text = GetString(Resource.String.status_need_download);
            }
            else
            {
                if (speechRecognizer != null)
                {
                    _speechRecognizer = speechRecognizer;
                    _speechRecognizer.SpeechStartDetected += (s, e) =>
                    {
                        RunOnUiThread(() =>
                        {
                            _statusText.Text = GetString(Resource.String.status_speech);
                        });
                    };
                    _speechRecognizer.SpeechEndDetected += (s, e) =>
                    {
                        RunOnUiThread(() =>
                        {
                            _statusText.Text = GetString(Resource.String.status_background);
                        });
                    };
                    _speechRecognizer.Recognized += (s, e) =>
                    {
                        RunOnUiThread(() =>
                        {
                            _inputTextEditText.Text = e.Text;
                        });
                    };
                }
                if (speechSynthesizer != null)
                {
                    _speechSynthesizer = speechSynthesizer;
                }
                _statusText.Text = GetString(Resource.String.status_ready);
            }
            _progressBar.Progress = 0;
            _UpdateDownloadButton();
            _UpdateSpeechButtons();
        }

        private void _DownloadClick(object sender, EventArgs e)
        {
            if (_downloadThread == null)
            {
                _PrepareSpeech(true);
            }
            else
            {
                _downloadCancellationToken.Cancel();
                _downloadThread.Join();
                _downloadThread = null;
                _downloadCancellationToken = null;
                _statusText.Text = GetString(Resource.String.status_canceled);
                _progressBar.Progress = 0;
                _UpdateDownloadButton();
            }
        }

        private void _UpdateDownloadButton()
        {
            if (_speechRecognizer == null && _speechSynthesizer == null)
            {
                if (_downloadThread != null)
                {
                    _downloadButton.Text = GetString(Resource.String.cancel_downloading);
                }
                else
                {
                    _downloadButton.Text = GetString(Resource.String.download_models);
                }
                _downloadButton.Enabled = true;
                _downloadButton.Visibility = ViewStates.Visible;
            }
            else
            {
                _downloadButton.Visibility = ViewStates.Invisible;
            }
        }

        protected override void OnPause()
        {
            base.OnPause();
            _StopRecording();
            _StopPlaying();
            _UpdateSpeechButtons();
        }

        private void _UpdateSpeechButtons()
        {
            _startRecordButton.Enabled = _speechRecognizer != null && !_IsRecording;
            _stopRecordButton.Enabled = _speechRecognizer != null && _IsRecording;
            _startPlayButton.Enabled = _speechSynthesizer != null && !_IsPlaying;
            _stopPlayButton.Enabled = _speechSynthesizer != null && _IsPlaying;
        }

        private void _StartRecordingClick(object sender, EventArgs e)
        {
            if (CheckSelfPermission(Manifest.Permission.RecordAudio) != Permission.Granted)
            {
                RequestPermissions(
                   new[] { Manifest.Permission.RecordAudio },
                   RecordAudioPermission);
            }
            else
            {
                _StartRecording();
                _UpdateSpeechButtons();
            }
        }

        public override void OnRequestPermissionsResult(
            int requestCode,
            string[] permissions,
            Permission[] grantResults)
        {
            switch (requestCode)
            {
                case RecordAudioPermission:
                    if (grantResults.Length > 0 && grantResults[0] == Permission.Granted)
                    {
                        _StartRecording();
                        _UpdateSpeechButtons();
                    }
                    else
                    {
                        Toast.MakeText(
                            this,
                            Resource.String.audio_recording_permission_denied,
                            ToastLength.Long).Show();
                    }
                    break;
            }
        }

        private void _StartRecording()
        {
            _audioRecorder = new AudioRecord(
                AudioSource.Mic,
                _speechRecognizer.SampleRate,
                ChannelIn.Mono,
                Encoding.Pcm16bit,
                AudioBufferLength * sizeof(short)
            );
            _audioRecorder.StartRecording();

            _recordingThread = new Thread(() =>
            {
                var audioBuffer = new byte[AudioBufferLength * sizeof(short)];
                while (true)
                {
                    try
                    {
                        // Keep reading the buffer while audio input is available.
                        int read = _audioRecorder.Read(audioBuffer, 0, audioBuffer.Length);
                        // Write out the audio file.
                        if (read == 0)
                        {
                            break;
                        }
                        _speechRecognizer.Write(audioBuffer, 0, read);
                    }
                    catch (Exception ex)
                    {
                        Console.Out.WriteLine(ex.Message);
                        break;
                    }
                }
            });
            _recordingThread.Start();
        }

        private void _StopRecordingClick(object sender, EventArgs e)
        {
            _StopRecording();
            _UpdateSpeechButtons();
        }

        private void _StopRecording()
        {
            if (_IsRecording)
            {
                _audioRecorder.Stop();
                _recordingThread.Join();
                _audioRecorder = null;
                _recordingThread = null;
            }
        }

        private void _StartPlayingClick(object sender, EventArgs e)
        {
            int OutputBufferSizeInBytes = 10 * 1024;

            string text = _inputTextEditText.Text;

            _playingThread = new Thread(() =>
            {
                var result = _speechSynthesizer.SpeakText(text);

                var audioTrack = (
                    new AudioTrack.Builder()
                    .SetAudioAttributes(new AudioAttributes.Builder()
                            .SetUsage(AudioUsageKind.Assistant)
                            .SetContentType(AudioContentType.Speech)
                            .Build())
                    .SetAudioFormat(new AudioFormat.Builder()
                            .SetEncoding(Encoding.Pcm16bit)
                            .SetSampleRate(result.SampleRate)
                            .SetChannelMask(ChannelOut.Mono)
                            .Build())
                    .SetBufferSizeInBytes(OutputBufferSizeInBytes)
                    .Build());
                audioTrack.Play();

                var y = result.AudioData;
                for (int i = 0; i < y.Length && _IsPlaying;)
                {
                    int bytesToWrite = Math.Min(y.Length - i, 4096);
                    int bytesWritten = audioTrack.Write(y, i, bytesToWrite);
                    if (bytesWritten < 0) break;
                    i += bytesWritten;
                }

                RunOnUiThread(() =>
                {
                    _StopPlaying();
                    _UpdateSpeechButtons();
                });
            });

            _playingThread.Start();
            _UpdateSpeechButtons();
        }

        private void _StopPlayingClick(object sender, EventArgs e)
        {
            _StopPlaying();
            _UpdateSpeechButtons();
        }

        private void _StopPlaying()
        {
            if (_IsPlaying)
            {
                _playingThread.Join();
                _playingThread = null;
            }
        }
	}
}
