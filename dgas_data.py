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

# --- CONFIGURAÇÃO DSP OTIMIZADA ---
SAMPLE_RATE = 44100
N_FFT = 2048
HOP_LENGTH = 512
# Reduzido para 1.5s para aliviar o CPU e aumentar velocidade de carregamento
CHUNK_DURATION = 1.5  
CHUNK_SAMPLES = int(CHUNK_DURATION * SAMPLE_RATE)
TARGET_LUFS = -24.0
NUM_WORKERS = 8  # Fixo para máxima performance

class AudioLoader:
    def __init__(self, data_dir: str, batch_size: int = 4, queue_size: int = 30):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.track_folders = self._scan_musdb(data_dir)
        # Meter reiniciado em cada worker para evitar race conditions, não aqui
        
        if not self.track_folders:
            print(f"CRITICAL WARNING: No folders found in {data_dir}.")
        else:
            print(f"Dataset Index: {len(self.track_folders)} songs ready.")
        
        self.queue = queue.Queue(maxsize=queue_size)
        self.running = False
        self.workers = []

    def _scan_musdb(self, directory: str) -> List[str]:
        valid_folders = []
        if not os.path.exists(directory): return []
        for root, dirs, files in os.walk(directory):
            if 'mixture.wav' in files and 'vocals.wav' in files:
                valid_folders.append(root)
        return valid_folders

    def _normalize_lufs(self, audio: np.ndarray, meter) -> np.ndarray:
        try:
            loudness = meter.integrated_loudness(audio)
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
            return mix, vox
        except:
            return np.zeros((CHUNK_SAMPLES, 2)), np.zeros((CHUNK_SAMPLES, 2))

    def _augment_physics(self, mixture: np.ndarray) -> np.ndarray:
        if random.random() < 0.5:
            shift = random.randint(-10, 10)
            mixture = np.roll(mixture, shift, axis=0)
        return mixture

    def _compute_spectrogram(self, audio: np.ndarray) -> np.ndarray:
        audio_T = audio.T 
        specs = []
        for ch in range(audio_T.shape[0]):
            stft = librosa.stft(audio_T[ch], n_fft=N_FFT, hop_length=HOP_LENGTH).T
            specs.append(np.stack([stft.real, stft.imag], axis=-1))
        
        spec_stereo = np.concatenate(specs, axis=-1)
        return spec_stereo.astype(np.float32)

    def _worker(self):
        # Meter local para thread safety
        local_meter = pyln.Meter(SAMPLE_RATE)
        while self.running:
            batch_mix, batch_tgt = [], []
            for _ in range(self.batch_size):
                if not self.track_folders:
                    time.sleep(0.1)
                    continue
                
                folder = random.choice(self.track_folders)
                mix_wav, vox_wav = self._load_chunk_pair(folder)
                
                # Otimização: Normalização rápida
                mix_wav = self._normalize_lufs(mix_wav, local_meter)
                vox_wav = self._normalize_lufs(vox_wav, local_meter)
                mix_wav_aug = self._augment_physics(mix_wav.copy())
                
                spec_mix = self._compute_spectrogram(mix_wav_aug)
                spec_tgt = self._compute_spectrogram(vox_wav)
                
                # Recorte para 128 frames (compatível com o modelo)
                spec_mix = spec_mix[:128, :, :]
                spec_tgt = spec_tgt[:128, :, :]
                
                batch_mix.append(spec_mix)
                batch_tgt.append(spec_tgt)
            try:
                # Timeout curto para verificar self.running frequentemente
                self.queue.put((np.stack(batch_mix), np.stack(batch_tgt)), timeout=1)
            except queue.Full: continue

    def start(self):
        self.running = True
        print(f"🚀 TURBO LOADER: Launching {NUM_WORKERS} workers...")
        for _ in range(NUM_WORKERS):
            t = threading.Thread(target=self._worker, daemon=True)
            t.start()
            self.workers.append(t)

    def stop(self):
        self.running = False
        for t in self.workers: t.join(timeout=1.0)
    
    def get_batch(self):
        mix, tgt = self.queue.get()
        return jnp.array(mix), jnp.array(tgt)