import time
import numpy as np
import librosa
import soundfile as sf
import threading
import queue
import random
import os
import pyloudnorm as pyln

# Configurações básicas
SAMPLE_RATE = 44100
CHUNK_DURATION = 1.5  
CHUNK_SAMPLES = int(CHUNK_DURATION * SAMPLE_RATE)
TARGET_LUFS = -24.0
NUM_WORKERS = 8 

class AudioLoader:
    def __init__(self, data_dir: str, batch_size: int = 16, queue_size: int = 40):
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

    def _scan_musdb(self, directory: str):
        valid_folders = []
        if not os.path.exists(directory): return []
        for root, dirs, files in os.walk(directory):
            if 'mixture.wav' in files and 'vocals.wav' in files:
                valid_folders.append(root)
        return valid_folders

    def _load_chunk_pair(self, folder_path):
        try:
            mix_path = os.path.join(folder_path, 'mixture.wav')
            vox_path = os.path.join(folder_path, 'vocals.wav')
            info = sf.info(mix_path)
            start = random.randint(0, max(0, info.frames - CHUNK_SAMPLES))
            mix, _ = sf.read(mix_path, start=start, frames=CHUNK_SAMPLES, dtype='float32', always_2d=True)
            vox, _ = sf.read(vox_path, start=start, frames=CHUNK_SAMPLES, dtype='float32', always_2d=True)
            if mix.shape[0] < CHUNK_SAMPLES:
                mix = np.pad(mix, ((0, CHUNK_SAMPLES - mix.shape[0]), (0, 0)))
                vox = np.pad(vox, ((0, CHUNK_SAMPLES - vox.shape[0]), (0, 0)))
            return mix, vox
        except: return np.zeros((CHUNK_SAMPLES, 2)), np.zeros((CHUNK_SAMPLES, 2))

    def _worker(self):
        while self.running:
            batch_mix, batch_tgt = [], []
            for _ in range(self.batch_size):
                if not self.track_folders: continue
                folder = random.choice(self.track_folders)
                m, v = self._load_chunk_pair(folder)
                # Augmentation simples: Roll
                if random.random() < 0.5:
                    sh = random.randint(-100, 100)
                    m = np.roll(m, sh, axis=0)
                batch_mix.append(m.T)
                batch_tgt.append(v.T)
            try: self.queue.put((np.stack(batch_mix), np.stack(batch_tgt)), timeout=1)
            except: continue

    def start(self):
        self.running = True
        print(f"🚀 RAW AUDIO LOADER: {NUM_WORKERS} workers...")
        for _ in range(NUM_WORKERS):
            t = threading.Thread(target=self._worker, daemon=True)
            t.start()
            self.workers.append(t)
    def stop(self):
        self.running = False
    def get_batch(self):
        return self.queue.get()