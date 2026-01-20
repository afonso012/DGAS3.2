import numpy as np
import soundfile as sf
import threading
import queue
import random
import os
import pyloudnorm as pyln # AGORA A SER USADO!

SAMPLE_RATE = 44100
CHUNK_DURATION = 1.5  
CHUNK_SAMPLES = int(CHUNK_DURATION * SAMPLE_RATE)
NUM_WORKERS = 8 
TARGET_LUFS = -24.0

class AudioLoader:
    def __init__(self, data_dir: str, batch_size: int = 16):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.track_folders = self._scan_musdb(data_dir)
        self.queue = queue.Queue(maxsize=40)
        self.running = False
        # Meter para calcular loudness
        self.meter = pyln.Meter(SAMPLE_RATE)
        print(f"Dataset: {len(self.track_folders)} faixas.")

    def _scan_musdb(self, directory: str):
        valid = []
        if not os.path.exists(directory): return []
        for root, _, files in os.walk(directory):
            if 'mixture.wav' in files and 'vocals.wav' in files:
                valid.append(root)
        return valid

    def _normalize(self, audio):
        # Medir Loudness e normalizar
        try:
            # Pyloudnorm espera (Samples, Channels)
            loudness = self.meter.integrated_loudness(audio)
            # Proteção contra silêncio infinito
            if loudness == -float('inf'): return audio
            return pyln.normalize.loudness(audio, loudness, TARGET_LUFS)
        except:
            return audio # Fallback

    def _load_chunk_pair(self, folder_path):
        try:
            mix_path = os.path.join(folder_path, 'mixture.wav')
            vox_path = os.path.join(folder_path, 'vocals.wav')
            info = sf.info(mix_path)
            
            if info.frames <= CHUNK_SAMPLES: start = 0
            else: start = random.randint(0, info.frames - CHUNK_SAMPLES)
            
            # Load com shape (Samples, Channels)
            mix, _ = sf.read(mix_path, start=start, frames=CHUNK_SAMPLES, dtype='float32', always_2d=True)
            vox, _ = sf.read(vox_path, start=start, frames=CHUNK_SAMPLES, dtype='float32', always_2d=True)
            
            # Pad
            if mix.shape[0] < CHUNK_SAMPLES:
                pad = CHUNK_SAMPLES - mix.shape[0]
                mix = np.pad(mix, ((0, pad), (0, 0)))
                vox = np.pad(vox, ((0, pad), (0, 0)))
            
            # Check Stereo
            if mix.shape[1] == 1:
                mix = np.concatenate([mix, mix], axis=1)
                vox = np.concatenate([vox, vox], axis=1)
            elif mix.shape[1] > 2:
                mix, vox = mix[:, :2], vox[:, :2]
            
            # --- NORMALIZAÇÃO CRÍTICA ---
            # Normalizar Mixture (Input) para -24 LUFS
            # Aplicar mesmo ganho ao Vocal (Target) para manter a relação relativa
            try:
                mix_loudness = self.meter.integrated_loudness(mix)
                if mix_loudness > -70: # Só normalizar se houver sinal
                    delta = TARGET_LUFS - mix_loudness
                    gain = 10.0 ** (delta / 20.0)
                    mix = mix * gain
                    vox = vox * gain
            except: pass # Se falhar, usa original
                
            return mix, vox
        except:
            return np.zeros((CHUNK_SAMPLES, 2)), np.zeros((CHUNK_SAMPLES, 2))

    def _worker(self):
        while self.running:
            batch_m, batch_v = [], []
            for _ in range(self.batch_size):
                if not self.track_folders: continue
                folder = random.choice(self.track_folders)
                m, v = self._load_chunk_pair(folder)
                batch_m.append(m.T)
                batch_v.append(v.T)
            try: self.queue.put((np.stack(batch_m), np.stack(batch_v)), timeout=1)
            except: continue

    def start(self):
        self.running = True
        for _ in range(NUM_WORKERS):
            t = threading.Thread(target=self._worker, daemon=True)
            t.start()
    def stop(self): self.running = False
    def get_batch(self): return self.queue.get()