import jax
import jax.numpy as jnp
import numpy as np
import librosa
import soundfile as sf
import threading
import queue
import random
import os
import pyloudnorm as pyln
from typing import List, Tuple

# --- CONFIGURAÇÃO DSP ---
SAMPLE_RATE = 44100
N_FFT = 2048
HOP_LENGTH = 512
CHUNK_DURATION = 3.0
CHUNK_SAMPLES = int(CHUNK_DURATION * SAMPLE_RATE)
TARGET_LUFS = -24.0

class AudioLoader:
    def __init__(self, data_dir: str, batch_size: int = 4, queue_size: int = 20):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.track_folders = self._scan_musdb(data_dir)
        self.meter = pyln.Meter(SAMPLE_RATE)
        
        if not self.track_folders:
            print(f"CRITICAL WARNING: No folders found in {data_dir}.")
        else:
            print(f"Dataset Index: {len(self.track_folders)} songs ready.")
        
        self.queue = queue.Queue(maxsize=queue_size)
        self.running = False
        self.worker_thread = None

    def _scan_musdb(self, directory: str) -> List[str]:
        valid_folders = []
        if not os.path.exists(directory): return []
        for root, dirs, files in os.walk(directory):
            if 'mixture.wav' in files and 'vocals.wav' in files:
                valid_folders.append(root)
        return valid_folders

    def _normalize_lufs(self, audio: np.ndarray) -> np.ndarray:
        try:
            loudness = self.meter.integrated_loudness(audio)
            if loudness == -float('inf'): return audio
            delta = TARGET_LUFS - loudness
            if delta > 20: delta = 20
            if delta < -20: delta = -20
            gain = 10.0 ** (delta / 20.0)
            return np.clip(audio * gain, -1.0, 1.0)
        except:
            return audio

    def _load_chunk_pair(self, folder_path: str) -> Tuple[np.ndarray, np.ndarray]:
        mix_path = os.path.join(folder_path, 'mixture.wav')
        vox_path = os.path.join(folder_path, 'vocals.wav')
        try:
            info = sf.info(mix_path)
            total = info.frames
            start = 0 if total < CHUNK_SAMPLES else random.randint(0, total - CHUNK_SAMPLES)
            
            mix, _ = sf.read(mix_path, start=start, frames=CHUNK_SAMPLES, dtype='float32', always_2d=True)
            vox, _ = sf.read(vox_path, start=start, frames=CHUNK_SAMPLES, dtype='float32', always_2d=True)
            
            if mix.shape[1] == 1: mix = np.tile(mix, (1, 2))
            if vox.shape[1] == 1: vox = np.tile(vox, (1, 2))
            
            if mix.shape[0] < CHUNK_SAMPLES:
                pad = CHUNK_SAMPLES - mix.shape[0]
                mix = np.pad(mix, ((0, pad), (0, 0)))
                vox = np.pad(vox, ((0, pad), (0, 0)))
            else:
                mix = mix[:CHUNK_SAMPLES]
                vox = vox[:CHUNK_SAMPLES]
            return mix, vox
        except Exception as e:
            print(f"Error: {e}")
            return np.zeros((CHUNK_SAMPLES, 2)), np.zeros((CHUNK_SAMPLES, 2))

    def _augment_physics(self, mixture: np.ndarray) -> np.ndarray:
        if random.random() < 0.5:
            shift = random.randint(-10, 10)
            mixture = np.roll(mixture, shift, axis=0)
        if random.random() < 0.3:
            mixture = np.tanh(mixture * 1.5)
        return mixture

    def _compute_spectrogram(self, audio: np.ndarray) -> np.ndarray:
        # Input Audio: (Samples, 2)
        audio_T = audio.T # (2, Samples)
        specs = []
        for ch in range(audio_T.shape[0]):
            stft = librosa.stft(audio_T[ch], n_fft=N_FFT, hop_length=HOP_LENGTH).T
            # (Time, Freq, 2) -> Real/Imag
            specs.append(np.stack([stft.real, stft.imag], axis=-1))
        
        # CORREÇÃO ESTÉREO: Stack em vez de Mean
        # Resultado: (Time, Freq, 4) onde 4 = [L_Re, L_Im, R_Re, R_Im]
        spec_stereo = np.concatenate(specs, axis=-1)
        return spec_stereo.astype(np.float32)

    def _worker(self):
        while self.running:
            batch_mix, batch_tgt = [], []
            for _ in range(self.batch_size):
                if not self.track_folders:
                    batch_mix.append(np.zeros((128, 128, 4)))
                    batch_tgt.append(np.zeros((128, 128, 4)))
                    continue
                
                folder = random.choice(self.track_folders)
                mix_wav, vox_wav = self._load_chunk_pair(folder)
                
                mix_wav = self._normalize_lufs(mix_wav)
                vox_wav = self._normalize_lufs(vox_wav)
                mix_wav_aug = self._augment_physics(mix_wav.copy())
                
                spec_mix = self._compute_spectrogram(mix_wav_aug)
                spec_tgt = self._compute_spectrogram(vox_wav)
                
                spec_mix = spec_mix[:128, :128, :]
                spec_tgt = spec_tgt[:128, :128, :]
                
                batch_mix.append(spec_mix)
                batch_tgt.append(spec_tgt)
            try:
                self.queue.put((np.stack(batch_mix), np.stack(batch_tgt)), timeout=1)
            except queue.Full: continue

    def start(self):
        self.running = True
        self.worker_thread = threading.Thread(target=self._worker, daemon=True)
        self.worker_thread.start()
        print("Data Loader Started (Stereo Mode).")

    def stop(self):
        self.running = False
        if self.worker_thread: self.worker_thread.join()
    
    def get_batch(self):
        mix, tgt = self.queue.get()
        return jnp.array(mix), jnp.array(tgt)