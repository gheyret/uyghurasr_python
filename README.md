# Uyghur Automatic Speech Recognition (ASR)

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue)](https://www.python.org/)
[![ONNX](https://img.shields.io/badge/Model-ONNX-orange)](https://onnx.ai/)

Uyghur Automatic Speech Recognition (ASR) — this project uses an ONNX model to transcribe Uyghur speech from audio files.

Uyghurche Awaz Tonush (ASR) — bu proyekt ONNX modeli yakiya awazni Uyghurche téxstqa ötküzüsh üchün ishlitilidu.

## Overview / Kisa qisa

This project loads a pre-trained ONNX speech recognition model and converts an input audio file into text.

Bu proyekt aldin yasalghan ONNX awaz tanush modelini yükley p, kirish awazini téxstqa ötküzidiken.

## Features / Xususiyatlar

- Recognizes Uyghur speech from WAV and other compatible audio files
- Works with an ONNX inference runtime
- Includes both command-line and GUI versions
- Simple sample audio is provided for testing

- WAV we bashqa mos awaz höjjetlerinden Uyghurche sözni tanushqa qadir
- ONNX inference runtime bilen ishlitilidu
- Buyruq satiri hem GUI nüsxisi bar
- Sinash üchün namuna awaz höjjiti bérilgen

## Requirements / Talaplar

- Python 3.9+
- pip
- Internet connection to download the ONNX model

- Python 3.9+
- pip
- ONNX modelini yuklash üchün internet

## Installation / O'rnatish

Clone or download the repository, then install the required packages:

Repozni klon qilin yaki yuklap, andin talap bolghan paketlarni ornatiliw:

```bash
pip install numpy librosa pydub onnx onnxruntime
```

If `pip` is not available, make sure Python and pip are correctly installed on your system.

Agar `pip` yoq bolsa, Python we pip to'g'ri ornatilghanliqini tekshürüng.

## Download the model / Modelni yuklash

Download the model file below and save it in the same folder as the Python scripts:

Ashqida modelni yuklap, Python skriptliri bilen bir öyge saqlang:

- [uyghur_asr.onnx](https://github.com/gheyret/uyghurasr_python/releases/download/tunji/uyghur_asr.onnx)

The file must be named exactly:

Faylning ismi aniq shunda bolush kérek:

```text
uyghur_asr.onnx
```

It should be placed next to these files:

Bu fayl bu fayllarning yanida turush kérek:

- `uyghurasr.py`
- `uyghurasrgui.py`
- `test5.wav`

## Command-line usage / Buyruq satiri ishlitish

Run the script with a WAV file path:

Skriptni awaz fayli bilen ishlitish:

```bash
python uyghurasr.py test5.wav
```

You can replace `test5.wav` with your own file:

`test5.wav` ornigha öz awaz höjjitingizni qoyush mumkin:

```bash
python uyghurasr.py path/to/your_audio.wav
```

The script will:

Skript quyidagilarni qilidu:

1. Load the audio file
2. Preprocess the signal
3. Run inference with the ONNX model
4. Print the recognized text

1. Awaz höjjiti ni yükleydu
2. Signaldin emeliy tayyarlaydighan qisimni ishlitidu
3. ONNX model bilen tanush yürütidu
4. Tanulghan téxstni ekranga chiqaridu

## Example / Misal

```bash
python uyghurasr.py test5.wav
```

This uses the sample audio included in the repository.

Bu repo ichidiki namuna awazdan foydalanidu.

## GUI version / GUI nüsxisi

A desktop GUI is also included:

Ish stoli üchün GUI ham bérilgen:

```bash
python uyghurasrgui.py
```

Then:

Andin:

1. Click the file selection button
2. Choose an audio file
3. Press the recognition button
4. View the transcription result in the text area

1. Fayl tallash tugmasini bassaq
2. Awaz höjjiti ni tallaymiz
3. Tanush tugmasini bassaq
4. Netijeni tekislik ichkide körüp bolamiz

## Project structure / Proyekt qurulushi

```text
uyghurasr_python/
├── README.md
├── test5.wav
├── uyghurasr.py
├── uyghurasrgui.py
├── uyghur_asr.onnx   # downloaded model
└── ...
```

## Notes / Eslentiler

- The model file must be named exactly `uyghur_asr.onnx`
- The script expects a valid audio input file path
- Supported input is usually WAV or other formats readable by `pydub`
- For best results, use clear speech with minimal background noise

- Model fayli `uyghur_asr.onnx` dep atalmish bolushi kérek
- Skript to'g'ri awaz fayli yolini talap qilidu
- Qabul qilinidighan awaz formatlari asasen WAV yaki `pydub` oqiy alidighan formatlardur
- Eng yaxshi netije üchün tozashqiliq awaz ishlitilsun, arqa ta'wir az bolsun

## Troubleshooting / Muamni çözish

If you see an import error, install the missing dependency:

Agar import xatosi chiqsa, qaysi paket yoq bolsa ornatiliw:

```bash
pip install <package_name>
```

Examples / Misallar:

```bash
pip install numpy
pip install librosa
pip install onnxruntime
```

If the model is not found, make sure it is downloaded and saved in the same folder as the Python files.

Agar model topulmas bolsa, yuklab, Python fayllari bilen bir öyge saqlang.

If the audio file cannot be read, verify that the file exists and is a valid audio file.

Agar awaz höjjiti oqulmas bolsa, faylning barmu yoqini tekshürüp, awaz formati to'g'ri bolushini tekshürüng.

## License / Litsenziya

This project is provided as-is for research, experimentation, and educational use.

Bu proyekt tadqiqot, sinash we bilim öyrenish maqsidida bérilgen.

## Contact / Bog'lanish

If you have questions or want to contribute, please open an issue or contact the repository maintainer.

Sualingiz bolsa yaki hissa qoshuqchisiz, issue aching yaki proyekt mualliriga murojat qiling.
