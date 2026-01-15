import jax
import jax.numpy as jnp
import numpy as np
import librosa
import soundfile as sf
import threading
import queue
import random
import os
from typing import List, Tuple, Optional

# --- CONFIGURAÇÃO DE DSP ---
# Devem bater certo com o dgas_train.py
SAMPLE_RATE = 44100
N_FFT = 2048
HOP_LENGTH = 512
CHUNK_DURATION = 3.0  # Segundos de áudio por amostra de treino
CHUNK_SAMPLES = int(CHUNK_DURATION * SAMPLE_RATE)

class AudioLoader:
    """
    Carregador de Áudio Assíncrono para Treino de Alta Performance.
    Lê ficheiros do disco, processa DSP e coloca em fila para a GPU.
    """
    def __init__(self, data_dir: str, batch_size: int = 4, queue_size: int = 20):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.file_list = self._scan_files(data_dir)
        
        if not self.file_list:
            print(f"WARNING: No audio files found in {data_dir}. Using dummy mode.")
        
        # Fila thread-safe para passar dados para o JAX
        self.queue = queue.Queue(maxsize=queue_size)
        self.running = False
        self.worker_thread = None

    def _scan_files(self, directory: str) -> List[str]:
        """Encontra todos os ficheiros wav/mp3/flac recursivamente."""
        files = []
        valid_exts = ('.wav', '.mp3', '.flac', '.ogg', '.stem.m4a')
        if not os.path.exists(directory):
            return []
            
        for root, _, filenames in os.walk(directory):
            for filename in filenames:
                if filename.lower().endswith(valid_exts):
                    files.append(os.path.join(root, filename))
        return files

    def _load_chunk(self, filepath: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Lê um pedaço aleatório do ficheiro de áudio.
        Retorna (Mistura, Target).
        """
        try:
            # Obter duração total sem ler o ficheiro todo
            info = sf.info(filepath)
            total_samples = info.frames
            
            if total_samples < CHUNK_SAMPLES:
                # Se for muito curto, fazer padding ou ignorar
                # Aqui fazemos loop simples
                start = 0
                frames_to_read = total_samples
            else:
                start = random.randint(0, total_samples - CHUNK_SAMPLES)
                frames_to_read = CHUNK_SAMPLES
                
            audio, _ = sf.read(filepath, start=start, frames=frames_to_read, dtype='float32')
            
            # Garantir estéreo e tamanho fixo
            if len(audio.shape) == 1: # Mono -> Stereo
                audio = np.stack([audio, audio], axis=-1)
            
            # Padding se necessário (para clips curtos)
            if audio.shape[0] < CHUNK_SAMPLES:
                pad_len = CHUNK_SAMPLES - audio.shape[0]
                audio = np.pad(audio, ((0, pad_len), (0, 0)))
            else:
                audio = audio[:CHUNK_SAMPLES, :]
            
            # --- SIMULAÇÃO DE SEPARAÇÃO (AUTOSUPERVISIONADO) ---
            # Em dados reais (MusDB18), carregaríamos 'mixture.wav' e 'vocals.wav'.
            # Para dados brutos sem stems, usamos uma estratégia auto-supervisionada:
            # Mistura = Audio Original + Ruído
            # Target = Audio Original (Denoising/Restoration Task)
            
            target = audio
            
            # Gerar "Mistura" degradada artificialmente
            noise = np.random.normal(0, 0.05, audio.shape).astype(np.float32)
            mixture = target + noise
            
            return mixture, target
            
        except Exception as e:
            print(f"Error loading {filepath}: {e}")
            return np.zeros((CHUNK_SAMPLES, 2)), np.zeros((CHUNK_SAMPLES, 2))

    def _augment_physics(self, mixture: np.ndarray) -> np.ndarray:
        """
        Aplica Augmentação Física (CPU/Numpy).
        """
        # 1. Phase Jitter (Simulação de Wow/Flutter)
        # Shift temporal aleatório muito subtil
        if random.random() < 0.5:
            shift = random.randint(-10, 10)
            mixture = np.roll(mixture, shift, axis=0)
            
        # 2. Soft Clipping (Saturação)
        if random.random() < 0.3:
            mixture = np.tanh(mixture * 1.5)
            
        return mixture

    def _compute_spectrogram(self, audio: np.ndarray) -> np.ndarray:
        """
        Converte Time-Domain -> Complex Spectrogram.
        Retorna shape (Time, Freq, Channels, 2) onde o último 2 é (Real, Imag).
        """
        # Audio input: (Samples, Channels)
        # Transpor para librosa: (Channels, Samples)
        audio_T = audio.T
        
        specs = []
        for ch in range(audio_T.shape[0]):
            stft = librosa.stft(audio_T[ch], n_fft=N_FFT, hop_length=HOP_LENGTH)
            # stft shape: (Freq, Time Frames) -> Transpor para (Time, Freq)
            stft = stft.T
            # Separar Real e Imag
            # Shape final: (Time, Freq, 2)
            complex_view = np.stack([stft.real, stft.imag], axis=-1)
            specs.append(complex_view)
            
        # Stack Channels: (Time, Freq, Channels, 2) -> (Time, Freq, 2) se for Mono
        # O modelo espera (Time, Freq, Channels) mas Channels está implícito na dimensão complexa?
        # Vamos verificar o dgas_model.py. 
        # O modelo espera `mixture_spec` como (Time, Freq, 2) para processamento canal a canal via vmap?
        # O treino faz: B, T, F, C.
        
        # Vamos retornar (Time, Freq, Channels) onde Channels é na verdade 2 * Stereo = 4 dimensões?
        # NÃO. O dgas_model processa 2 dimensões (Real/Imag).
        # Se temos estéreo, processamos cada canal independentemente ou concatenados?
        # Para simplificar o data loader inicial: Retornamos apenas o Canal Esquerdo (Mono) ou média.
        # OU melhor: Retornamos (Batch, Time, Freq, 2) onde 2 é Real/Imag de UM canal.
        # Para Stereo real, o Batch size duplicaria.
        
        # Vamos assumir MONO mixdown para começar a validar o pipeline real.
        spec_mono = specs[0] # Canal Esquerdo
        return spec_mono.astype(np.float32)

    def _worker(self):
        while self.running:
            batch_mix = []
            batch_target = []
            
            for _ in range(self.batch_size):
                # Escolher ficheiro aleatório
                if not self.file_list:
                    # Fallback dummy se não houver ficheiros
                    dummy_mix = np.random.randn(128, 128, 2).astype(np.float32)
                    dummy_tgt = np.random.randn(128, 128, 2).astype(np.float32)
                    batch_mix.append(dummy_mix)
                    batch_target.append(dummy_tgt)
                    continue

                fpath = random.choice(self.file_list)
                mix_wav, tgt_wav = self._load_chunk(fpath)
                
                # Augmentação
                mix_wav = self._augment_physics(mix_wav)
                
                # STFT
                # Nota: O tamanho exato do tempo depende do CHUNK_DURATION.
                # 3s @ 44.1k / 512 hop ~= 258 frames.
                # Precisamos de garantir tamanho fixo para o JAX (ex: cortar ou pad).
                spec_mix = self._compute_spectrogram(mix_wav)
                spec_tgt = self._compute_spectrogram(tgt_wav)
                
                # Crop para garantir 128x128 ou o que o modelo espera
                # O modelo atual aceita qualquer tamanho, mas para batching precisamos de igualdade.
                # Vamos fixar Time=128, Freq=1025 (padrão 2048 FFT).
                # Para testes rápidos, vamos fazer crop drástico:
                spec_mix = spec_mix[:128, :128, :]
                spec_tgt = spec_tgt[:128, :128, :]
                
                batch_mix.append(spec_mix)
                batch_target.append(spec_tgt)
            
            # Stack e converter para JAX Array (na thread principal depois)
            batch_mix_np = np.stack(batch_mix)
            batch_target_np = np.stack(batch_target)
            
            try:
                self.queue.put((batch_mix_np, batch_target_np), timeout=1)
            except queue.Full:
                continue

    def start(self):
        self.running = True
        self.worker_thread = threading.Thread(target=self._worker, daemon=True)
        self.worker_thread.start()
        print(f"Data Loader started. Scanning {self.data_dir}...")

    def stop(self):
        self.running = False
        if self.worker_thread:
            self.worker_thread.join()

    def get_batch(self):
        # Bloqueia até ter dados
        mix, tgt = self.queue.get()
        # Converter para JAX aqui é seguro? Sim, JAX array creation é rápida.
        return jnp.array(mix), jnp.array(tgt)

# --- TESTE UNITÁRIO DO DATA LOADER ---
if __name__ == "__main__":
    # Cria uma pasta dummy e um ficheiro wav dummy para testar
    os.makedirs("dummy_data", exist_ok=True)
    dummy_audio = np.random.uniform(-1, 1, (44100*5, 2))
    sf.write("dummy_data/test.wav", dummy_audio, 44100)
    
    loader = AudioLoader("dummy_data", batch_size=4)
    loader.start()
    
    try:
        print("Waiting for batch...")
        b_mix, b_tgt = loader.get_batch()
        print(f"Batch received!")
        print(f"Mix Shape: {b_mix.shape}") # Esperado: (4, 128, 128, 2)
        print(f"Target Shape: {b_tgt.shape}")
        
    finally:
        loader.stop()
        # Limpar dummy
        os.remove("dummy_data/test.wav")
        os.rmdir("dummy_data")