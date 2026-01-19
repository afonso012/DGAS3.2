import time
import numpy as np
import librosa
import soundfile as sf
import threading
import queue
import random
import os
import pyloudnorm as pyln
from typing import List, Tuple

# Configurações básicas
SAMPLE_RATE = 44100
CHUNK_DURATION = 1.5  
CHUNK_SAMPLES = int(CHUNK_DURATION * SAMPLE_RATE)
TARGET_LUFS = -24.0
NUM_WORKERS = 8 

class AudioLoader:
    def __init__(self, data_dir: str, batch_size: int = 32, queue_size: int = 40):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.track_folders = self._scan_musdb(data_dir)
        
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
            # Limitar ganho para evitar explosões
            gain = 10.0 ** (np.clip(delta, -20, 20) / 20.0)
            return np.clip(audio * gain, -1.0, 1.0)
        except:
            return audio

    def _load_chunk_pair(self, folder_path: str) -> Tuple[np.ndarray, np.ndarray]:
        mix_path = os.path.join(folder_path, 'mixture.wav')
        vox_path = os.path.join(folder_path, 'vocals.wav')
        try:
            # Ler apenas a info para saber o tamanho
            info = sf.info(mix_path)
            if info.frames < CHUNK_SAMPLES:
                start = 0
            else:
                start = random.randint(0, info.frames - CHUNK_SAMPLES)
            
            # Ler apenas o pedaço necessário (IO rápido)
            mix, _ = sf.read(mix_path, start=start, frames=CHUNK_SAMPLES, dtype='float32', always_2d=True)
            vox, _ = sf.read(vox_path, start=start, frames=CHUNK_SAMPLES, dtype='float32', always_2d=True)
            
            # Pad se for pequeno demais
            if mix.shape[0] < CHUNK_SAMPLES:
                pad = CHUNK_SAMPLES - mix.shape[0]
                mix = np.pad(mix, ((0, pad), (0, 0)))
                vox = np.pad(vox, ((0, pad), (0, 0)))
            
            return mix, vox
        except:
            return np.zeros((CHUNK_SAMPLES, 2)), np.zeros((CHUNK_SAMPLES, 2))

    def _augment_physics(self, mixture: np.ndarray) -> np.ndarray:
        if random.random() < 0.5:
            shift = random.randint(-50, 50) # Shift pequeno em samples
            mixture = np.roll(mixture, shift, axis=0)
        return mixture

    def _worker(self):
        local_meter = pyln.Meter(SAMPLE_RATE)
        while self.running:
            batch_mix, batch_tgt = [], []
            for _ in range(self.batch_size):
                if not self.track_folders: continue
                
                folder = random.choice(self.track_folders)
                mix_wav, vox_wav = self._load_chunk_pair(folder)
                
                # Normalizar
                mix_wav = self._normalize_lufs(mix_wav, local_meter)
                vox_wav = self._normalize_lufs(vox_wav, local_meter)
                
                # Augmentação leve
                mix_wav = self._augment_physics(mix_wav)
                
                # Transpor para (Channels, Samples) para o JAX gostar
                batch_mix.append(mix_wav.T)
                batch_tgt.append(vox_wav.T)
            
            try:
                # Envia AUDIO CRU (Raw waveform)
                self.queue.put((np.stack(batch_mix), np.stack(batch_tgt)), timeout=1)
            except queue.Full:
                continue

    def start(self):
        self.running = True
        print(f"🚀 GPU MODE LOADER: Launching {NUM_WORKERS} workers (Raw Audio)...")
        for _ in range(NUM_WORKERS):
            t = threading.Thread(target=self._worker, daemon=True)
            t.start()
            self.workers.append(t)

    def stop(self):
        self.running = False
        for t in self.workers: t.join(timeout=1.0)
    
    def get_batch(self):
        mix, tgt = self.queue.get()
        return mix, tgt  # Retorna numpy array, conversão JAX feita no loop