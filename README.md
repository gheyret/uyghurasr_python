# Uyghur Automatic Speech Recognition (ASR)

This project recognizes Uyghur speech from an audio file using an ONNX-based model.

## Requirements

- Python 3.9+
- pip

Install the required Python packages:

```bash
pip install numpy librosa pydub onnx onnxruntime
```

## Download the model

Download the model file and save it in the same folder as the scripts:

- [uyghur_asr.onnx](https://github.com/gheyret/uyghurasr_python/releases/download/tunji/uyghur_asr.onnx)

The file should be placed in the project root next to:

- `uyghurasr.py`
- `uyghurasrgui.py`
- `test5.wav`

## Command-line usage

From the project directory, run:

```bash
python uyghurasr.py test5.wav
```

You can replace `test5.wav` with any audio file path you want to recognize:

```bash
python uyghurasr.py path/to/your_audio.wav
```

The script reads the audio file, preprocesses it, runs the model, and prints the recognized text.

## Example

```bash
python uyghurasr.py test5.wav
```

This uses the sample audio file included in the repository.

## GUI version

A simple GUI version is also included:

```bash
python uyghurasrgui.py
```

Then:

1. Click the file selection button.
2. Choose an audio file.
3. Click the recognition button.
4. The recognized text will be displayed in the window.

## Notes

- The model file must be named exactly `uyghur_asr.onnx`.
- The script expects an audio file path as a command-line argument.
- The sample audio file `test5.wav` is provided for testing.
- If you use a different audio format, make sure the file is readable by `pydub`.

## Troubleshooting

If you get an import error, install the missing package with `pip install ...`.

If the model is not found, ensure `uyghur_asr.onnx` is downloaded and stored in the same folder as the Python scripts.

If the script fails to read the audio file, confirm the file exists and is a valid audio file.
