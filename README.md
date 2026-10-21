# Uyghurche Awaz Tonush (ASR)

## Uyghurche

### Qollash usuli

Bu loyiha ONNX modelini ishlitip, awazni Uyghurche tekstqa aylitidu. Modelni yükläp, loyiha skriptlari bilen bir öyge saqlang.

### Talaplar

- Python 3.9+
- pip
- Internet (modelni yüklash üchün)

### O'rnatish

Repozni klon qiling yaki yuklang, andin kerakli paketlarni o'rnating:

```bash
pip install numpy librosa pydub onnx onnxruntime
```

### Modelni yüklash

Töwendiki model faylini yükläp, loyiha papkisi ichide saqlang:

- [uyghur_asr.onnx](https://github.com/gheyret/uyghurasr_python/releases/download/tunji/uyghur_asr.onnx)

Modelning ismi aniq shu bolushi shart:

```text
uyghur_asr.onnx
```

Bu fayl quyidagilar bilen bir öyde turushi kerak:

- `uyghurasr.py`
- `uyghurasrgui.py`
- `test5.wav`

### Buyruq satiri bilen ishlash

Loyiha papkisi ichide quyidagiche ishlitishingiz mumkin:

```bash
python uyghurasr.py test5.wav
```

Agar öz awaz faylingizni ishlitishni xahishingiz bolsa, `test5.wav` ornigha öz faylingizni qoyishingiz mumkin:

```bash
python uyghurasr.py path/to/your_audio.wav
```

Skript quyidagilarni qilidu:

1. Awaz faylini yüklitidu
2. Awazni qayta ishlitidu (preprocess)
3. ONNX model bilan xulosa chiqaridu
4. Tanulghan tekstni ekranda chiqaridu

### Misal

```bash
python uyghurasr.py test5.wav
```

Bu repo ichide bérilgen namuna awaz fayli bilen ishlitilidu.

### GUI usuli

Ish stoli üchün GUI ham bérilgen:

```bash
python uyghurasrgui.py
```

Andin:

1. Fayl tanlash tugmasini bosing
2. Awaz faylini tanlang
3. Tanush tugmasini bosing
4. Natijeni matn maydani ichide köring

### Eslitishler

- Model fayli `uyghur_asr.onnx` deb atalsun
- Skript to'g'ri awaz fayli yolini talap qilidu
- Awaz formati asasen WAV yaki `pydub` oqiy alidighan format bolushi kerak
- Yaxshi natije üchün toza awazdan, arqa tovush az bolghan sadashtan foydalaning

### Xatoliklarni tuzatish

Agar import xatosi chiqsa, kerakli paketni o'rnating:

```bash
pip install <package_name>
```

Misallar:

```bash
pip install numpy
pip install librosa
pip install onnxruntime
```

Agar model topulmas bolsa, uni yükläp, Python fayllari bilen bir öyge saqlang.

Agar awaz fayli oqulmas bolsa, faylning mavjudligini va awaz formatining to'g'riligini tekshürüng.

---

## English

### Using

This project uses an ONNX model to recognize Uyghur speech and convert it into text. Download the model and keep it in the same folder as the project scripts.

### Requirements

- Python 3.9+
- pip
- Internet access to download the model

### Installation

Clone or download the repository, then install the required packages:

```bash
pip install numpy librosa pydub onnx onnxruntime
```

### Download the model

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

### Command-line usage

From the project folder, run:

```bash
python uyghurasr.py test5.wav
```

To use your own audio file, replace `test5.wav` with your own file path:

```bash
python uyghurasr.py path/to/your_audio.wav
```

The script will:

1. Load the audio file
2. Preprocess the audio signal
3. Run inference with the ONNX model
4. Print the recognized text

### Example

```bash
python uyghurasr.py test5.wav
```

This uses the sample audio file included in the repository.

### GUI usage

A desktop GUI version is also included:

```bash
python uyghurasrgui.py
```

Then:

1. Click the file selection button
2. Choose an audio file
3. Click the recognition button
4. View the result in the text area

### Notes

- The model file must be named exactly `uyghur_asr.onnx`
- The script expects a valid audio file path
- Input is usually WAV or another format supported by `pydub`
- Clear speech with minimal background noise gives better results

### Troubleshooting

If you encounter an import error, install the missing package:

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

## Proyekt qurulishi

```text
uyghurasr_python/
├── README.md
├── test5.wav
├── uyghurasr.py
├── uyghurasrgui.py
├── uyghur_asr.onnx   # yüklän model
└── ...
```

## Litsenziya

Bu loyiha tadqiqot, sinash va bilim öyrinish maqsidida berilgen.

This project is provided as-is for research, experimentation, and educational use.
