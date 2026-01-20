import jax
import jax.numpy as jnp
import equinox as eqx
import numpy as np
import librosa
import soundfile as sf
import os
from tqdm import tqdm
from dgas_model import Generator, Discriminator

# Configuração Global
jax.config.update("jax_platform_name", "gpu") 

CONFIG = {
    "CHECKPOINT": "checkpoints/dgas_test.eqx", 
    "N_FFT": 2048,
    "HOP_LENGTH": 512,
    "CHUNK_SIZE": 1.5,   # IMPORTANTE: 1.5s igual ao treino
    "OVERLAP": 0.5,
    "ODE_STEPS": 32, 
    "TARGET_SR": 44100
}

def load_models():
    print(f"--- A Carregar Checkpoint: {CONFIG['CHECKPOINT']} ---")
    key = jax.random.PRNGKey(0)
    k1, k2 = jax.random.split(key)
    gen = Generator(key=k1)
    models = (gen, Discriminator(key=k2))
    try:
        models = eqx.tree_deserialise_leaves(CONFIG['CHECKPOINT'], models)
        return models[0]
    except FileNotFoundError:
        raise FileNotFoundError(f"Erro: Checkpoint não encontrado em {CONFIG['CHECKPOINT']}")

# --- STFT/ISTFT HÍBRIDO (JAX CPU -> JAX GPU) ---

# Definimos a função normalmente (sem decorador aqui)
def cpu_stft_jax_impl(audio):
    # audio: (1, 2, N) -> Batch, Channels, Samples
    window = jnp.hanning(CONFIG["N_FFT"])
    
    f, t, Zxx = jax.scipy.signal.stft(
        audio, 
        fs=44100, 
        window=window, 
        nperseg=CONFIG["N_FFT"], 
        noverlap=CONFIG["N_FFT"] - CONFIG["HOP_LENGTH"]
    )
    
    # Converter para formato do modelo: (B, 4, F, T) -> [L_real, L_imag, R_real, R_imag]
    spec = jnp.stack([Zxx.real, Zxx.imag], axis=-1) # (1, 2, F, T, 2)
    spec = jnp.transpose(spec, (0, 1, 4, 2, 3))     # (1, 2, 2, F, T)
    
    B, C, _, F, T = spec.shape
    return spec.reshape(B, C * 2, F, T)

# Aplicamos o JIT com backend CPU explicitamente
cpu_stft_jax = jax.jit(cpu_stft_jax_impl, backend='cpu')


def cpu_istft_jax_impl(spec):
    # spec: (1, 4, F, T)
    B, _, F, T = spec.shape
    spec = spec.reshape(B, 2, 2, F, T)
    
    Zxx = spec[:, :, 0] + 1j * spec[:, :, 1] # (1, 2, F, T)
    
    window = jnp.hanning(CONFIG["N_FFT"])
    _, audio = jax.scipy.signal.istft(
        Zxx, 
        fs=44100, 
        window=window, 
        nperseg=CONFIG["N_FFT"], 
        noverlap=CONFIG["N_FFT"] - CONFIG["HOP_LENGTH"]
    )
    return audio # (1, 2, N)

# Aplicamos o JIT com backend CPU explicitamente
cpu_istft_jax = jax.jit(cpu_istft_jax_impl, backend='cpu')


def get_log_coords(F, T):
    times = jnp.linspace(0, 1, T)
    linear_freqs = jnp.linspace(0, 1, F)
    log_freqs = jnp.log1p(linear_freqs * 10.0) / jnp.log1p(10.0)
    return log_freqs, times

# --- MODEL INFERENCE (GPU) ---
@eqx.filter_jit
def predict_spectrogram(model, mix_spec, key, steps):
    # mix_spec: (1, 4, F, T)
    cond = jax.vmap(model.encoder)(mix_spec) 
    x = jax.random.normal(key, mix_spec.shape)
    dt = 1.0 / steps
    
    B, C, F, T = mix_spec.shape
    freqs, times = get_log_coords(F, T)
    grid_f, grid_t = jnp.meshgrid(freqs, times, indexing='ij')
    f_flat, t_flat = grid_f.flatten(), grid_t.flatten()
    
    def get_velocity_single(t_curr, x_curr, cond_curr):
        # x_curr: (4, F, T)
        # Transpose para (F, T, 4) para alinhar com grid_f/grid_t
        x_flat = jnp.transpose(x_curr, (1, 2, 0)).reshape(-1, 4)
        
        def field_point(f_val, t_val, x_val_i):
            return model.field(t_curr, jnp.array([t_val, f_val]), x_val_i, cond_curr)
            
        v_flat = jax.vmap(field_point)(f_flat, t_flat, x_flat)
        # Reconstrói (F, T, 4) -> Transpose (4, F, T)
        return jnp.transpose(v_flat.reshape(F, T, 4), (2, 0, 1))

    def loop_body(i, curr_x):
        t = i / steps
        d1 = jax.vmap(get_velocity_single, in_axes=(None, 0, 0))(t, curr_x, cond)
        x_tilde = curr_x + d1 * dt
        d2 = jax.vmap(get_velocity_single, in_axes=(None, 0, 0))(t + dt, x_tilde, cond)
        curr_x = curr_x + (d1 + d2) * 0.5 * dt
        return curr_x

    final_spec = jax.lax.fori_loop(0, steps, loop_body, x)
    return final_spec

def process_file(file_path, model):
    print(f"\n>>> A processar: {file_path}")
    
    try:
        audio, sr = librosa.load(file_path, sr=CONFIG["TARGET_SR"], mono=False)
    except Exception as e:
        print(f"Erro: {e}")
        return

    if audio.ndim == 1: audio = np.stack([audio, audio])
    
    # 1. Normalização Peak 0.95 (Crucial)
    original_peak = np.max(np.abs(audio))
    if original_peak > 0:
        audio = audio * (0.95 / original_peak)
        print(f"Normalização: {original_peak:.4f} -> 0.95")

    total_samples = audio.shape[1]
    chunk_samples = int(CONFIG["CHUNK_SIZE"] * sr)
    hop_samples = int(chunk_samples * (1 - CONFIG["OVERLAP"]))
    
    output_buffer = np.zeros_like(audio)
    weight_buffer = np.zeros(total_samples)
    window = np.hanning(chunk_samples)
    
    key = jax.random.PRNGKey(42)
    
    print("A iniciar inferência...")
    # Tenta obter o dispositivo CPU explicitamente
    try:
        cpu_device = jax.devices("cpu")[0]
        gpu_device = jax.devices("gpu")[0]
    except:
        print("Aviso: Falha ao detetar dispositivos CPU/GPU específicos. A usar padrão.")
        cpu_device = None
        gpu_device = None

    for i in tqdm(range(0, total_samples - chunk_samples + 1, hop_samples)):
        chunk = audio[:, i : i + chunk_samples]
        
        # Preparar chunk (1, 2, N)
        chunk_jax = jnp.array(chunk)[None, ...]
        
        # 1. STFT (JAX on CPU)
        spec_jax = cpu_stft_jax(chunk_jax)
        
        # Mover para GPU
        if gpu_device:
            spec_gpu = jax.device_put(spec_jax, gpu_device)
        else:
            spec_gpu = spec_jax # Fallback

        # 2. Inferência (GPU)
        key, subkey = jax.random.split(key)
        rec_spec_gpu = predict_spectrogram(model, spec_gpu, subkey, CONFIG["ODE_STEPS"])
        
        # 3. ISTFT (JAX on CPU)
        if cpu_device:
            rec_spec_cpu = jax.device_put(rec_spec_gpu, cpu_device)
        else:
            rec_spec_cpu = rec_spec_gpu # Fallback
            
        rec_audio_jax = cpu_istft_jax(rec_spec_cpu)
        
        rec_audio = np.array(rec_audio_jax[0])
        
        valid_len = min(rec_audio.shape[1], chunk_samples)
        output_buffer[:, i : i + valid_len] += rec_audio[:, :valid_len] * window[:valid_len]
        weight_buffer[i : i + valid_len] += window[:valid_len]

    weight_buffer[weight_buffer < 1e-8] = 1.0
    output_buffer /= weight_buffer
    
    out_name = file_path.rsplit('.', 1)[0] + "_DGAS_OUT.wav"
    sf.write(out_name, output_buffer.T, sr)
    print(f"✅ Salvo: {out_name}")

if __name__ == "__main__":
    generator = load_models()
    process_file("mixture.wav", generator)