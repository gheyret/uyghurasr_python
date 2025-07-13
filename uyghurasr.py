import onnx
import onnxruntime
import numpy as np
import librosa
from pydub import AudioSegment

class Uyghur():
    def __init__(self):
        self.uyghur_latin = "abcdefghijklmnopqrstuvwxyz éöü’" # $ for padding char. index must be 0
        self._vocab_list = [self.pad_char, self.sos_char,self.eos_char] + list(self.uyghur_latin)
        self._vocab2idx = {v: idx for idx, v in enumerate(self._vocab_list)}

    def vocab_to_idx(self, vocab):
        return self._vocab2idx[vocab]

    def idx_to_vocab(self, idx):
        return self._vocab_list[idx]

    def vocab_list(self):
        return self._vocab_list

    @property
    def vocab_size(self):
        return len(self._vocab_list)

    @property
    def pad_idx(self):
        return self.vocab_to_idx(self.pad_char)

    @property
    def sos_idx(self):
        return self.vocab_to_idx(self.sos_char)

    @property
    def eos_idx(self):
        return self.vocab_to_idx(self.eos_char)

    @property
    def pad_char(self):
        return "<pad>"

    @property
    def sos_char(self):
        return "<sos>"

    @property
    def eos_char(self):
        return "<eos>"

class UyghurASR():
    def __init__(self):
        self.sample_rate = 22050
        self.fft_len     = 1024
        self.uyghur_latin = Uyghur()
        model = onnx.load("uyghur_asr.onnx")
        self.sess = onnxruntime.InferenceSession(model.SerializeToString())

    def load_prepocess(self,audio_name):
        mono = AudioSegment.from_file(audio_name).split_to_mono()[0] 
        audio = librosa.util.buf_to_float(mono.get_array_of_samples(),n_bytes=mono.frame_width)
        return audio    

    def recognize(self,audio_name):    
        input = self.load_prepocess(audio_name)
        results = self.sess.run(["output"], {"input": input})

        max_yps = np.argmax(results[0],axis=1)
        pred = []
        last = None
        count = max_yps.shape[1]
        for i in range(count):
            char = int(max_yps[0][i])
            if char != self.uyghur_latin.pad_idx:
                if char != last:
                    pred.append(self.uyghur_latin.idx_to_vocab(char))
            last = char
        return "".join(pred)

if __name__ == '__main__':
    urecognizer = UyghurASR()
    txt = urecognizer.recognize(R"test5.wav")
    print(txt)
