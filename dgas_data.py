import numpy as np
import soundfile as sf
import threading
import queue
import random
import os

# --- CONFIGURAÇÕES FÍSICAS (SOTA) ---
SAMPLE_RATE = 44100
# GEOMETRIA PERFEITA: 128 frames x 512 hop = 65536 amostras (~1.48s)
# Isto elimina o erro de alinhamento com a STFT
CHUNK_SAMPLES = 65536 
NUM_WORKERS = 8 

class AudioLoader:
    def __init__(self, data_dir: str, batch_size: int = 16, train_mode: bool = True):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.train_mode = train_mode # Permite desligar augments na validação
        self.track_folders = self._scan_musdb(data_dir)
        self.queue = queue.Queue(maxsize=100)
        self.running = False
        print(f"Dataset SOTA: {len(self.track_folders)} faixas. Chunk: {CHUNK_SAMPLES}")

    def _scan_musdb(self, directory: str):
        valid = []
        if not os.path.exists(directory): 
            print(f"❌ ERRO: Diretoria {directory} não existe!")
            return []
        for root, _, files in os.walk(directory):
            if 'mixture.wav' in files and 'vocals.wav' in files:
                valid.append(root)
        if not valid:
            print("❌ AVISO: Nenhuma pasta válida encontrada!")
        return valid

    def _sota_normalization(self, mix, vox):
        """
        Engenharia de Normalização Profissional:
        1. Reflect Padding: Continuidade matemática nas bordas.
        2. RMS Gating: Rejeição estatística de silêncio/ruído.
        3. Gain Clamping: Impede explosão de ruído de fundo.
        """
        # 1. PADDING REFLECTIVO (C^0 Continuity)
        if mix.shape[0] < CHUNK_SAMPLES:
            pad_len = CHUNK_SAMPLES - mix.shape[0]
            # Espelha o áudio para evitar o "degrau" de silêncio
            mix = np.pad(mix, ((0, pad_len), (0, 0)), mode='reflect')
            vox = np.pad(vox, ((0, pad_len), (0, 0)), mode='reflect')
        
        # Crop de segurança (Garante tamanho exato)
        mix = mix[:CHUNK_SAMPLES, :]
        vox = vox[:CHUNK_SAMPLES, :]

        # 2. RMS GATING (Filtro de Energia)
        # Calcula a energia média quadrática
        rms_mix = np.sqrt(np.mean(mix**2))
        if rms_mix < 0.01: # -40dB Threshold
            return None, None

        # 3. NORMALIZAÇÃO ACOPLADA COM CLAMP
        peak = np.max(np.abs(mix))
        target_peak = 0.95
        
        # Calcula ganho necessário para atingir 0.95
        gain = target_peak / (peak + 1e-8)
        
        # CLAMP: Nunca amplificar mais que 10x (+20dB)
        # Isto impede que sussurros virem ruído branco
        gain = min(gain, 10.0) 
        
        return mix * gain, vox * gain

    def _load_chunk_pair(self, folder_path):
        try:
            mix_path = os.path.join(folder_path, 'mixture.wav')
            vox_path = os.path.join(folder_path, 'vocals.wav')
            info = sf.info(mix_path)
            
            for _ in range(15): # Tentativas aumentadas para encontrar segmento válido
                if info.frames <= CHUNK_SAMPLES: start = 0
                else: start = random.randint(0, info.frames - CHUNK_SAMPLES)
                
                # Leitura Física
                mix, _ = sf.read(mix_path, start=start, frames=CHUNK_SAMPLES, dtype='float32', always_2d=True)
                vox, _ = sf.read(vox_path, start=start, frames=CHUNK_SAMPLES, dtype='float32', always_2d=True)
                
                # Correção Estéreo (Mono -> Stereo Duplicado)
                if mix.shape[1] == 1:
                    mix = np.concatenate([mix, mix], axis=1)
                    vox = np.concatenate([vox, vox], axis=1)
                elif mix.shape[1] > 2:
                    mix, vox = mix[:, :2], vox[:, :2]

                # --- AUGMENTATION FÍSICA (Apenas Treino) ---
                if self.train_mode:
                    # 1. Extração do Instrumental (Assumindo alinhamento de fase)
                    inst = mix - vox
                    
                    # 2. DYNAMIC REMIXING
                    # Cria uma nova mistura com balanço diferente
                    g_vox = random.uniform(0.75, 1.25)
                    g_inst = random.uniform(0.75, 1.25)
                    
                    # Recalcula alvos
                    aug_vox = vox * g_vox
                    aug_inst = inst * g_inst
                    aug_mix = aug_vox + aug_inst # Nova mistura sintética
                    
                    # 3. CHANNEL SWAPPING (Invariância Espacial)
                    if random.random() < 0.5:
                        aug_mix = np.flip(aug_mix, axis=1) # Troca L <-> R
                        aug_vox = np.flip(aug_vox, axis=1)
                    
                    mix_proc, vox_proc = aug_mix, aug_vox
                else:
                    mix_proc, vox_proc = mix, vox
                
                # --- NORMALIZAÇÃO FINAL ---
                res_mix, res_vox = self._sota_normalization(mix_proc, vox_proc)
                
                if res_mix is not None:
                    # Transposição Final: (Samples, Channels) -> (Channels, Samples)
                    return res_mix.T, res_vox.T
            
            return None, None
        except Exception as e:
            print(f"Erro ficheiro {folder_path}: {e}")
            return None, None

    def _worker(self):
        while self.running:
            batch_m, batch_v = [], []
            while len(batch_m) < self.batch_size:
                if not self.track_folders: break
                folder = random.choice(self.track_folders)
                m, v = self._load_chunk_pair(folder)
                if m is not None:
                    batch_m.append(m)
                    batch_v.append(v)
            
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