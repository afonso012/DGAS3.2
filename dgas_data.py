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
from typing import List, Tuple, Optional

# --- CONFIGURAÇÃO DSP (Igual ao Train) ---
SAMPLE_RATE = 44100
N_FFT = 2048
HOP_LENGTH = 512
CHUNK_DURATION = 3.0
CHUNK_SAMPLES = int(CHUNK_DURATION * SAMPLE_RATE)
TARGET_LUFS = -24.0  # Padrão Broadcast (EBU R 128)

class AudioLoader:
    """
    Carregador Profissional DGAS (MUSDB18 Compatible).
    Lê pares (Mistura, Stem) e normaliza loudness.
    """
    def __init__(self, data_dir: str, batch_size: int = 4, queue_size: int = 20):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.track_folders = self._scan_musdb(data_dir)
        
        # Medidor LUFS
        self.meter = pyln.Meter(SAMPLE_RATE)
        
        if not self.track_folders:
            print(f"CRITICAL WARNING: No song folders found in {data_dir}. Check paths.")
        else:
            print(f"Dataset Index: Found {len(self.track_folders)} songs.")
        
        self.queue = queue.Queue(maxsize=queue_size)
        self.running = False
        self.worker_thread = None

    def _scan_musdb(self, directory: str) -> List[str]:
        """
        Procura pastas do MUSDB18 que contenham 'mixture.wav' e 'vocals.wav'.
        """
        valid_folders = []
        if not os.path.exists(directory):
            return []
            
        for root, dirs, files in os.walk(directory):
            # Verifica se é uma pasta de música válida
            if 'mixture.wav' in files and 'vocals.wav' in files:
                valid_folders.append(root)
        
        return valid_folders

    def _normalize_lufs(self, audio: np.ndarray) -> np.ndarray:
        """Aplica Normalização EBU R 128 para estabilizar o treino."""
        try:
            # Medir loudness (Integrado)
            loudness = self.meter.integrated_loudness(audio)
            
            # Proteger contra silêncio absoluto (-inf)
            if loudness == -float('inf'):
                return audio
                
            # Calcular ganho necessário
            delta = TARGET_LUFS - loudness
            
            # Limitar ganho para evitar explosão de ruído em silêncios (+- 20dB max)
            if delta > 20: delta = 20
            if delta < -20: delta = -20
                
            gain = 10.0 ** (delta / 20.0)
            normalized_audio = audio * gain
            
            # Hard Clip limiter de segurança (-1.0 a 1.0)
            return np.clip(normalized_audio, -1.0, 1.0)
            
        except Exception:
            return audio

    def _load_chunk_pair(self, folder_path: str) -> Tuple[np.ndarray, np.ndarray]:
        """Lê o mesmo intervalo de tempo da mistura e da vocal."""
        mix_path = os.path.join(folder_path, 'mixture.wav')
        vox_path = os.path.join(folder_path, 'vocals.wav')
        
        try:
            # Ler metadados do mixture para saber a duração
            info = sf.info(mix_path)
            total_samples = info.frames
            
            # Definir ponto de início aleatório
            if total_samples < CHUNK_SAMPLES:
                start = 0
            else:
                start = random.randint(0, total_samples - CHUNK_SAMPLES)
                
            # Ler ambos os ficheiros sincronizados
            # always_2d=True garante que mono venha como (N, 1)
            mix, _ = sf.read(mix_path, start=start, frames=CHUNK_SAMPLES, dtype='float32', always_2d=True)
            vox, _ = sf.read(vox_path, start=start, frames=CHUNK_SAMPLES, dtype='float32', always_2d=True)
            
            # Garantir stereo (N, 2) se for mono
            if mix.shape[1] == 1: mix = np.tile(mix, (1, 2))
            if vox.shape[1] == 1: vox = np.tile(vox, (1, 2))
            
            # Padding com zeros se o ficheiro for menor que 3s
            if mix.shape[0] < CHUNK_SAMPLES:
                pad_len = CHUNK_SAMPLES - mix.shape[0]
                mix = np.pad(mix, ((0, pad_len), (0, 0)))
                vox = np.pad(vox, ((0, pad_len), (0, 0)))
            else:
                mix = mix[:CHUNK_SAMPLES]
                vox = vox[:CHUNK_SAMPLES]
                
            return mix, vox
            
        except Exception as e:
            print(f"Error reading {folder_path}: {e}")
            return np.zeros((CHUNK_SAMPLES, 2)), np.zeros((CHUNK_SAMPLES, 2))

    def _augment_physics(self, mixture: np.ndarray) -> np.ndarray:
        """Simula degradações físicas (Fita, Vinil, Saturação)."""
        # 1. Phase Jitter (Wow/Flutter)
        if random.random() < 0.5:
            shift = random.randint(-10, 10)
            mixture = np.roll(mixture, shift, axis=0)
            
        # 2. Soft Clipping Analógico
        if random.random() < 0.3:
            # Tanh simula a curva de saturação de transístores/válvulas
            mixture = np.tanh(mixture * 1.5)
            
        return mixture

    def _compute_spectrogram(self, audio: np.ndarray) -> np.ndarray:
        # Transpose para (Channels, Samples) para o Librosa
        audio_T = audio.T
        specs = []
        for ch in range(audio_T.shape[0]):
            stft = librosa.stft(audio_T[ch], n_fft=N_FFT, hop_length=HOP_LENGTH)
            stft = stft.T # (Time, Freq)
            # Stack Real/Imag: (Time, Freq, 2)
            complex_view = np.stack([stft.real, stft.imag], axis=-1)
            specs.append(complex_view)
        
        # Média Stereo para Input Mono (Roadmap v3.2 Standard)
        # Processar Stereo completo duplicaria a VRAM necessária.
        # Para "God Tier" com VRAM limitada, Mono Spec de alta resolução é preferível.
        spec_avg = np.mean(specs, axis=0) 
        return spec_avg.astype(np.float32)

    def _worker(self):
        while self.running:
            batch_mix = []
            batch_target = []
            
            for _ in range(self.batch_size):
                if not self.track_folders:
                    # Dummy fallback
                    batch_mix.append(np.zeros((128, 128, 2)))
                    batch_target.append(np.zeros((128, 128, 2)))
                    continue

                folder = random.choice(self.track_folders)
                mix_wav, vox_wav = self._load_chunk_pair(folder)
                
                # --- PROCESSAMENTO CRÍTICO ---
                # 1. Normalizar Loudness (Estabilidade)
                mix_wav = self._normalize_lufs(mix_wav)
                vox_wav = self._normalize_lufs(vox_wav)
                
                # 2. Augmentação Física (Só na mistura)
                mix_wav_aug = self._augment_physics(mix_wav.copy())
                
                # 3. STFT
                spec_mix = self._compute_spectrogram(mix_wav_aug)
                spec_tgt = self._compute_spectrogram(vox_wav)
                
                # 4. Crop Seguro (Garantir dimensões fixas para o Batch)
                # O crop temporal (128) define a janela de contexto da rede
                spec_mix = spec_mix[:128, :128, :]
                spec_tgt = spec_tgt[:128, :128, :]
                
                batch_mix.append(spec_mix)
                batch_target.append(spec_tgt)
            
            try:
                # Colocar na fila
                self.queue.put((np.stack(batch_mix), np.stack(batch_target)), timeout=1)
            except queue.Full:
                continue

    def start(self):
        self.running = True
        self.worker_thread = threading.Thread(target=self._worker, daemon=True)
        self.worker_thread.start()
        print(f"DGAS Data Engine Started. Target: {TARGET_LUFS} LUFS")

    def stop(self):
        self.running = False
        if self.worker_thread: self.worker_thread.join()
    
    def get_batch(self):
        mix, tgt = self.queue.get()
        return jnp.array(mix), jnp.array(tgt)