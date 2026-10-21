# Uyghurche Awaz Tonush (ASR)

## Uyghurche

### Ishlitish

Bu pirojékt ONNX modélini ishlitip Uyghurche nutuqni tonup, uni tékistke aylandurup béridu. Modélni chüshürüp, pirojékt höjjetliri bar yerge saqlang.

### Telep

- Python 3.9+
- pip
- Modélni chüshürüsh üchün Intérnétqa ulinish

### Ornitish

chüshürüp, kéreklik bolaqlarni orniting:

```bash
pip install numpy librosa pydub onnx onnxruntime
```

### Modélni chüshürüsh

Töwendiki modél höjjitini chüshürüp, kod bilen oxshash yerge saqlang:

- [uyghur_asr.onnx](https://github.com/gheyret/uyghurasr_python/releases/download/tunji/uyghur_asr.onnx)

höjjetning nami mundaq bolushi shert:

```text
uyghur_asr.onnx
```

U töwendikiler bilen bir yerge qoyulushi kérek:

- `uyghurasr.py`
- `uyghurasrgui.py`
- `test5.wav`

### Buyruq qurida ishlitish

Buyruq qurigha töwendikidek buyruq yézing:

```bash
python uyghurasr.py test5.wav
```

Özingizning awaz höjjitingizni ishlitish üchün `test5.wav` ni öz özingizning awaz höjjiti bilen almashturung:

```bash
python uyghurasr.py path/to/your_audio.wav
```

Kod töwendikidek meshghulatni élip baridu:

1. Awaz höjjitini yükleydu
2. Awaz höjjitini aldin’ala bir terep qilidu
3. ONNX modéli bilen tonuydu
4. Tonulghan tékistni chiqiridu

### Misal

```bash
python uyghurasr.py test5.wav
```

### GUI ishlitish

Kompyutér üstel’üsti (desktop) GUI nusxisimu bar:

```bash
python uyghurasrgui.py
```

Andin:

1. höjjet tallash topchisi bésing
2. Bir awaz höjjitini tallang
3. Tonush topchisi bésing
4. Netije tékist rayonida körünidu

### Ésinglarda tutung

- Modél höjjitining nami toghra `uyghur_asr.onnx` bolushi shert
- Kodqa toghra choqum toghra bolghan awaz höjjitini yoli bilen körsitip bérish kérek
- Awaz höjjitining formati WAV yaki `pydub` qollaydighan bashqa formatlar da bolushi kérek
- Éniq nutuq we az fon shawqunliri téximu yaxshi netije béridu

### Mesile hel qilish

Eger import xataliqi chiqsa, kem bolaqlarni orniting:

```bash
pip install <package_name>
```

Misallar:

```bash
pip install numpy
pip install librosa
pip install onnxruntime
```

Eger modél tépilmisa, uni chüshürüp, Python kodi bilen oxshash yerde saqlang.

Eger awaz höjjitini oquyalmisa, höjjet barmu, formati toghrimu tekshürüng.

## Pirojéktning qurulmisi

```text
uyghurasr_python/
├── README.md
├── test5.wav
├── uyghurasr.py
├── uyghurasrgui.py
├── uyghur_asr.onnx   
└── ...
```

Bu pirojékt tetqiqat, sinaq we ögitish üchün teminlendi

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
