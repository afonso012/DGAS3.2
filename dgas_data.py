import numpy as np
import soundfile as sf
import threading
import queue
import random
import os

SAMPLE_RATE = 44100
CHUNK_DURATION = 1.5  
CHUNK_SAMPLES = int(CHUNK_DURATION * SAMPLE_RATE)
NUM_WORKERS = 8 

class AudioLoader:
    def __init__(self, data_dir: str, batch_size: int = 16):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.track_folders = self._scan_musdb(data_dir)
        self.queue = queue.Queue(maxsize=100)
        self.running = False
        print(f"Dataset: {len(self.track_folders)} faixas detectadas em {data_dir}")

    def _scan_musdb(self, directory: str):
        valid = []
        if not os.path.exists(directory): 
            print(f"❌ ERRO: Diretoria {directory} não existe!")
            return []
        for root, _, files in os.walk(directory):
            if 'mixture.wav' in files and 'vocals.wav' in files:
                valid.append(root)
        if not valid:
            print("❌ AVISO: Nenhuma pasta com mixture.wav e vocals.wav encontrada!")
        return valid

    def _normalize_peak(self, mix, vox):
        max_mix = np.max(np.abs(mix))
        max_vox = np.max(np.abs(vox))
        
        # FILTRO DE SILÊNCIO DUPLO
        # 1. Se a mistura for silêncio, ignorar.
        if max_mix < 0.05: 
            return None, None
            

        scale = 0.95 / (max_mix + 1e-8)
        return mix * scale, vox * scale

    def _load_chunk_pair(self, folder_path):
        try:
            mix_path = os.path.join(folder_path, 'mixture.wav')
            vox_path = os.path.join(folder_path, 'vocals.wav')
            info = sf.info(mix_path)
            
            for _ in range(10): # Tentar mais vezes encontrar uma parte com voz
                if info.frames <= CHUNK_SAMPLES: start = 0
                else: start = random.randint(0, info.frames - CHUNK_SAMPLES)
                
                mix, _ = sf.read(mix_path, start=start, frames=CHUNK_SAMPLES, dtype='float32', always_2d=True)
                vox, _ = sf.read(vox_path, start=start, frames=CHUNK_SAMPLES, dtype='float32', always_2d=True)
                
                # Padding se necessário
                if mix.shape[0] < CHUNK_SAMPLES:
                    pad = CHUNK_SAMPLES - mix.shape[0]
                    mix = np.pad(mix, ((0, pad), (0, 0)))
                    vox = np.pad(vox, ((0, pad), (0, 0)))
                
                # Stereo Check
                if mix.shape[1] == 1:
                    mix = np.concatenate([mix, mix], axis=1)
                    vox = np.concatenate([vox, vox], axis=1)
                elif mix.shape[1] > 2:
                    mix, vox = mix[:, :2], vox[:, :2]
                
                res_mix, res_vox = self._normalize_peak(mix, vox)
                if res_mix is not None:
                    return res_mix, res_vox
            
            return None, None # Falhou em encontrar voz
        except Exception as e:
            print(f"Erro a ler ficheiro {folder_path}: {e}")
            return None, None

    def _worker(self):
        while self.running:
            batch_m, batch_v = [], []
            while len(batch_m) < self.batch_size:
                if not self.track_folders: break
                folder = random.choice(self.track_folders)
                m, v = self._load_chunk_pair(folder)
                
                if m is not None: # Só adiciona se for válido (não silêncio)
                    batch_m.append(m.T)
                    batch_v.append(v.T)
            
            if len(batch_m) == self.batch_size:
                try: self.queue.put((np.stack(batch_m), np.stack(batch_v)), timeout=1)
                except: continue

    def start(self):
        self.running = True
        for _ in range(NUM_WORKERS):
            t = threading.Thread(target=self._worker, daemon=True)
            t.start()
    def stop(self): self.running = False
    def get_batch(self): return self.queue.get()