import os
import queue
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import numpy as np
import onnx
import onnxruntime
import librosa
from pydub import AudioSegment

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "uyghur_asr.onnx")
SAMPLE_RATE = 22050
CHUNK_SIZE = SAMPLE_RATE * 30   # 30 sekunt
OVERLAP = SAMPLE_RATE           # 1 sekont

# ---------------------------------------------------------------- Vocab
class Uyghur:
    def __init__(self):
        self.uyghur_latin = "abcdefghijklmnopqrstuvwxyz éöü’"
        self._vocab_list = [self.pad_char, self.sos_char, self.eos_char] + list(self.uyghur_latin)
        self._vocab2idx = {v: i for i, v in enumerate(self._vocab_list)}

    def vocab_to_idx(self, vocab):
        return self._vocab2idx[vocab]

    def idx_to_vocab(self, idx):
        return self._vocab_list[idx]

    @property
    def pad_idx(self):
        return self.vocab_to_idx(self.pad_char)

    @property
    def pad_char(self):
        return "<pad>"

    @property
    def sos_char(self):
        return "<sos>"

    @property
    def eos_char(self):
        return "<eos>"


# ---------------------------------------------------------------- ASR
class UyghurASR:
    def __init__(self):
        self.uyghur_latin = Uyghur()
        model = onnx.load(MODEL_PATH)
        self.sess = onnxruntime.InferenceSession(model.SerializeToString())

    @staticmethod
    def load_preprocess(audio_name):
        mono = AudioSegment.from_file(audio_name).split_to_mono()[0]
        mono = mono.set_frame_rate(SAMPLE_RATE)
        return librosa.util.buf_to_float(mono.get_array_of_samples(), n_bytes=mono.frame_width)

    def greedy_decode(self, results):
        max_yps = np.argmax(results[0], axis=1)
        pred, last = [], None
        for i in range(max_yps.shape[1]):
            char = int(max_yps[0][i])
            if char != self.uyghur_latin.pad_idx and char != last:
                pred.append(self.uyghur_latin.idx_to_vocab(char))
            last = char
        return "".join(pred)

    def chunk_starts(self, n):
        if n <= CHUNK_SIZE:
            return [0]
        starts, s = [], 0
        while s < n:
            starts.append(s)
            s += CHUNK_SIZE - OVERLAP
        return starts

    def run_chunk(self, chunk):
        return self.greedy_decode(self.sess.run(["output"], {"input": chunk}))


# ---------------------------------------------------------------- GUI
class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Uyghurche Awaz Tonush")
        self.geometry("640x480")
        self.minsize(480, 360)

        self.asr = None
        self.audio_path = None
        self.q = queue.Queue()

        # ---- yuqiri qisim: tuyuqlar
        top = ttk.Frame(self, padding=8)
        top.pack(fill="x")
        self.btn_select = ttk.Button(top, text="Talla", width=10, command=self.select_file)
        self.btn_select.pack(side="left")
        self.btn_run = ttk.Button(top, text="Tonu", width=10, command=self.start_recognition)
        self.btn_run.pack(side="left", padx=(6, 0))
        self.btn_copy = ttk.Button(top, text="Köchür", width=10, command=self.copy_text)
        self.btn_copy.pack(side="left", padx=(6, 0))
        self.lbl_file = ttk.Label(top, text="Höjjet tallanmidi", anchor="w")
        self.lbl_file.pack(side="left", fill="x", expand=True, padx=10)

        # ---- progress
        mid = ttk.Frame(self, padding=(8, 0))
        mid.pack(fill="x")
        self.progress = ttk.Progressbar(mid, mode="determinate", maximum=1, value=0)
        self.progress.pack(fill="x")
        self.lbl_status = ttk.Label(mid, text="Teyyar")
        self.lbl_status.pack(anchor="w", pady=(2, 0))

        # ---- netije
        bottom = ttk.Frame(self, padding=8)
        bottom.pack(fill="both", expand=True)
        self.text = tk.Text(bottom, wrap="word", font=("Segoe UI", 12))
        sb = ttk.Scrollbar(bottom, orient="vertical", command=self.text.yview)
        self.text.configure(yscrollcommand=sb.set)
        self.text.pack(side="left", fill="both", expand=True)
        sb.pack(side="right", fill="y")

        self.after(100, self.poll_queue)

    # ---- tallash
    def select_file(self):
        path = filedialog.askopenfilename(
            title="Awaz höjjiti tallang",
            filetypes=[("Awaz höjjiti", "*.wav *.ogg *.mp3 *.flac *.m4a *.opus"),
                       ("Hemme höjjet", "*.*")],
        )
        if path:
            self.audio_path = path
            self.lbl_file.config(text=os.path.basename(path))
            self.lbl_status.config(text="Teyyar")
            self.progress.config(value=0)

    # ---- tonush
    def start_recognition(self):
        if not self.audio_path:
            messagebox.showwarning("Agahlandurush", "Aldi bilen awaz höjjiti tallang!")
            return
        self.set_busy(True)
        self.text.delete("1.0", "end")
        self.progress.config(value=0, maximum=1)
        threading.Thread(target=self.worker, args=(self.audio_path,), daemon=True).start()

    def worker(self, path):
        try:
            if self.asr is None:
                self.q.put(("status", "Model yüklewatidu..."))
                self.asr = UyghurASR()

            self.q.put(("status", "Awaz oqulwatidu..."))
            audio = self.asr.load_preprocess(path)
            minutes = len(audio) / (SAMPLE_RATE * 60)
            starts = self.asr.chunk_starts(len(audio))
            total = len(starts)
            self.q.put(("total", total))
            self.q.put(("status", f"Uzunluqi: {minutes:.2f} minut"))

            for i, s in enumerate(starts, 1):
                self.q.put(("status", f"Tonuwatidu... Parche {i}/{total}"))
                txt = self.asr.run_chunk(audio[s:s + CHUNK_SIZE])
                self.q.put(("text", txt))
                self.q.put(("progress", i))

            self.q.put(("done", None))
        except Exception as e:
            self.q.put(("error", str(e)))

    # ---- GUI yangilash (asosliq thread)
    def poll_queue(self):
        try:
            while True:
                kind, val = self.q.get_nowait()
                if kind == "status":
                    self.lbl_status.config(text=val)
                elif kind == "total":
                    self.progress.config(maximum=val, value=0)
                elif kind == "progress":
                    self.progress.config(value=val)
                elif kind == "text":
                    self.text.insert("end", val + "\n")
                    self.text.see("end")
                elif kind == "done":
                    self.lbl_status.config(text="Tamamlandi")
                    self.set_busy(False)
                elif kind == "error":
                    self.lbl_status.config(text="Xataliq")
                    self.set_busy(False)
                    messagebox.showerror("Xataliq", val)
        except queue.Empty:
            pass
        self.after(100, self.poll_queue)

    def copy_text(self):
        content = self.text.get("1.0", "end").strip()
        if not content:
            self.lbl_status.config(text="Köchürgüdek tekist yoq")
            return
        self.clipboard_clear()
        self.clipboard_append(content)
        self.update()  # clipboard ni sistemigha saqlash
        self.lbl_status.config(text="Köchürüldi")

    def set_busy(self, busy):
        state = "disabled" if busy else "normal"
        self.btn_select.config(state=state)
        self.btn_run.config(state=state)


if __name__ == "__main__":
    App().mainloop()