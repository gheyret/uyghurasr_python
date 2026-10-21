# Uyghurche Awaz Tonush (ASR)

## Uyghurche

### Ishlitish usuli:

Bu proyekt ONNX modelini ishlitip, awazni Uyghurche téxstqa ötküzidiken. Modelni yuklap, proyekttiki skriptlar bilen bir öyge saqlang.

### Talaplar:

- Python 3.9+
- pip
- Internet bolushi kérek (modelni yuklash üchün)

### O'rnatish:

Repozni klon qiling yaki yuklang, andin kerék paketlarni ornatiliw:

```bash
pip install numpy librosa pydub onnx onnxruntime
```

### Modelni yuklash:

Töwendiki model faylini yuklang ve proyekttiki bashqa skriptlar bilen bir öyge saqlang:

- [uyghur_asr.onnx](https://github.com/gheyret/uyghurasr_python/releases/download/tunji/uyghur_asr.onnx)

Modelning ismi aniq shunda bolushi kérek:

```text
uyghur_asr.onnx
```

Bu fayl quyidagilar bilen bir öyge turush kérek:

- `uyghurasr.py`
- `uyghurasrgui.py`
- `test5.wav`

### Buyruq satiri bilen ishlitish:

Proyekt papkisi ichide quyidagicha ishlitishingiz mumkin:

```bash
python uyghurasr.py test5.wav
```

Agar öz awaz höjjitiningizni ishlitishni xahlasangiz, `test5.wav` ornigha öz fayliningizni qoyush mumkin:

```bash
python uyghurasr.py path/to/your_audio.wav
```

Skript quyidagilarni qilidu:

1. Awaz höjjitini yükleydu
2. Awazni qayta ishlash (preprocess) qilidu
3. ONNX model bilen tanush qilishni yürütidu
4. Tanulghan téxstni ekranga chiqaridu

### Misal:

```bash
python uyghurasr.py test5.wav
```

Bu namuna awaz höjjiti bilen ishlitilidu, repo ichide bérilgen `test5.wav` fayli.

### GUI nüsxisi:

Ish stoli üchün GUI ham bérilgen:

```bash
python uyghurasrgui.py
```

Andin:

1. Fayl tallash tugmasini bassaq
2. Awaz höjjitini tallaymiz
3. Tanush tugmasini bassaq
4. Netijeni tekislik ichide körüp bolamiz

### Eslentiler:

- Model fayli `uyghur_asr.onnx` dep atalsun
- Skript to'g'ri awaz fayli yolini talap qilidu
- Awaz formati asasen WAV yaki `pydub` oqiy alidighan format bolushi kérek
- Eng yaxshi netije üchün tozashqiliq awazdan foydalaning

### Muamni çözish:

Agar import xatosi chiqsa, qaysi paket yoq bolsa ornatiliw:

```bash
pip install <package_name>
```

Misal:

```bash
pip install numpy
pip install librosa
pip install onnxruntime
```

Agar model topulmas bolsa, modelni yuklap ve Python fayllari bilen bir öyge saqlang.

Agar awaz höjjiti oqulmas bolsa, faylning barliqi hem awaz formati to'g'ri bolushini tekshürüng.

---

## English

### Using:

This project uses an ONNX model to recognize Uyghur speech and convert it into text. Download the model and keep it in the same folder as the project scripts.

### Requirements:

- Python 3.9+
- pip
- Internet connection to download the model

### Installation:

Clone or download the repository, then install the required packages:

```bash
pip install numpy librosa pydub onnx onnxruntime
```

### Download the model:

Download the model file below and save it in the same directory as the scripts:

- [uyghur_asr.onnx](https://github.com/gheyret/uyghurasr_python/releases/download/tunji/uyghur_asr.onnx)

The file must be named exactly:

```text
uyghur_asr.onnx
```

It should be placed alongside:

- `uyghurasr.py`
- `uyghurasrgui.py`
- `test5.wav`

### Command-line usage:

From the project folder, run:

```bash
python uyghurasr.py test5.wav
```

To use your own audio file, replace `test5.wav` with your file path:

```bash
python uyghurasr.py path/to/your_audio.wav
```

The script will:

1. Load the audio file
2. Preprocess the audio signal
3. Run inference with the ONNX model
4. Print the recognized text

### Example:

```bash
python uyghurasr.py test5.wav
```

This uses the sample audio file included in the repository.

### GUI version:

A desktop GUI version is also included:

```bash
python uyghurasrgui.py
```

Then:

1. Click the file selection button
2. Choose an audio file
3. Click the recognition button
4. View the result in the text area

### Notes:

- The model file must be named exactly `uyghur_asr.onnx`
- The script expects a valid audio file path
- Input is usually WAV or another format supported by `pydub`
- Clear speech with minimal background noise gives better results

### Troubleshooting:

If you encounter an import error, install the missing dependencies:

```bash
pip install <package_name>
```

Examples:

```bash
pip install numpy
pip install librosa
pip install onnxruntime
```

If the model is not found, download it and save it in the same folder as the Python scripts.

If the audio file cannot be read, verify that the file exists and is a valid audio file.

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

## License / Litsenziya

This project is provided as-is for research, experimentation, and educational purposes.

Bu proyekt tadqiqot, sinash ve bilim öyrenish maqsidida bérilgen.
