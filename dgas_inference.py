import jax
import jax.numpy as jnp
import equinox as eqx
import numpy as np
import librosa
import soundfile as sf
import os
from tqdm import tqdm
from dgas_model import Generator, Discriminator

CONFIG = {
    "CHECKPOINT": "checkpoints/dgas_latest.eqx", 
    "N_FFT": 2048,
    "HOP_LENGTH": 512,
    "CHUNK_SIZE": 5.0,
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
        raise FileNotFoundError(f"Erro: Checkpoint não encontrado!")

# --- JAX SIGNAL PROCESSING (GPU) ---
def gpu_stft_inference(audio):
    # Shape Input: (B=1, Channels, Samples)
    window = jnp.hanning(CONFIG["N_FFT"])
    f, t, Zxx = jax.scipy.signal.stft(audio, fs=44100, window=window, nperseg=CONFIG["N_FFT"], noverlap=CONFIG["N_FFT"] - CONFIG["HOP_LENGTH"])
    Zxx = jnp.transpose(Zxx, (0, 1, 2, 3))
    # Output: (B, C, F, T, 2) -> (B, 2*C, F, T)
    spec = jnp.stack([Zxx.real, Zxx.imag], axis=-1)
    B, C, F, T, _ = spec.shape
    return jnp.transpose(spec, (0, 1, 4, 2, 3)).reshape(B, C * 2, F, T)

def gpu_istft_inference(spec):
    # Input: (B, 4, F, T) -> De volta para audio
    # Reconstruir complexos
    # L: canais 0 (real), 1 (imag) | R: canais 2 (real), 3 (imag)
    l_complex = spec[:, 0] + 1j * spec[:, 1]
    r_complex = spec[:, 2] + 1j * spec[:, 3]
    
    window = jnp.hanning(CONFIG["N_FFT"])
    # jax.scipy.signal.istft existe nas versões recentes
    _, audio_l = jax.scipy.signal.istft(l_complex, fs=44100, window=window, nperseg=CONFIG["N_FFT"], noverlap=CONFIG["N_FFT"] - CONFIG["HOP_LENGTH"])
    _, audio_r = jax.scipy.signal.istft(r_complex, fs=44100, window=window, nperseg=CONFIG["N_FFT"], noverlap=CONFIG["N_FFT"] - CONFIG["HOP_LENGTH"])
    
    return jnp.stack([audio_l, audio_r], axis=1)

def get_log_coords(F, T):
    times = jnp.linspace(0, 1, T)
    linear_freqs = jnp.linspace(0, 1, F)
    log_freqs = jnp.log1p(linear_freqs * 10.0) / jnp.log1p(10.0)
    return log_freqs, times

@eqx.filter_jit
def predict_chunk(model, audio_chunk, key, steps):
    # 1. GPU STFT
    mix_spec = gpu_stft_inference(audio_chunk) # (1, 4, F, T)
    
    cond = jax.vmap(model.encoder)(mix_spec) 
    x = jax.random.normal(key, mix_spec.shape)
    dt = 1.0 / steps
    
    B, C, F, T = mix_spec.shape
    freqs, times = get_log_coords(F, T)
    grid_f, grid_t = jnp.meshgrid(freqs, times, indexing='ij')
    f_flat, t_flat = grid_f.flatten(), grid_t.flatten()
    
    def get_velocity_single(t_curr, x_curr, cond_curr):
        x_flat = jnp.transpose(x_curr, (1, 2, 0)).reshape(-1, 4)
        def field_point(f_val, t_val, x_val_i):
            return model.field(t_curr, jnp.array([t_val, f_val]), x_val_i, cond_curr)
        v_flat = jax.vmap(field_point)(f_flat, t_flat, x_flat)
        return jnp.transpose(v_flat.reshape(F, T, 4), (2, 0, 1))

    def loop_body(i, curr_x):
        t = i / steps
        d1 = jax.vmap(get_velocity_single, in_axes=(None, 0, 0))(t, curr_x, cond)
        x_tilde = curr_x + d1 * dt
        d2 = jax.vmap(get_velocity_single, in_axes=(None, 0, 0))(t + dt, x_tilde, cond)
        curr_x = curr_x + (d1 + d2) * 0.5 * dt
        return curr_x

    final_spec = jax.lax.fori_loop(0, steps, loop_body, x)
    
    # 2. GPU ISTFT
    return gpu_istft_inference(final_spec)

def process_file(file_path, model):
    print(f"\n>>> A processar: {file_path}")
    audio, sr = librosa.load(file_path, sr=CONFIG["TARGET_SR"], mono=False)
    if audio.ndim == 1: audio = np.stack([audio, audio])
    
    total_samples = audio.shape[1]
    chunk_samples = int(CONFIG["CHUNK_SIZE"] * sr)
    hop_samples = int(chunk_samples * (1 - CONFIG["OVERLAP"]))
    
    # Prepara buffers
    output_buffer = np.zeros_like(audio)
    weight_buffer = np.zeros(total_samples)
    window = np.hanning(chunk_samples)
    
    key = jax.random.PRNGKey(42)
    
    for i in tqdm(range(0, total_samples - chunk_samples + 1, hop_samples)):
        chunk = audio[:, i : i + chunk_samples]
        
        # Envia para GPU como JAX Array (B=1, C, Samples)
        chunk_jax = jnp.array(chunk)[None, ...]
        
        key, subkey = jax.random.split(key)
        
        # O modelo faz STFT -> Flow -> ISTFT tudo na GPU
        rec_audio_jax = predict_chunk(model, chunk_jax, subkey, CONFIG["ODE_STEPS"])
        
        # Traz de volta para CPU só o áudio final
        rec_audio = np.array(rec_audio_jax[0])
        
        valid_len = min(rec_audio.shape[1], chunk_samples)
        output_buffer[:, i : i + valid_len] += rec_audio[:, :valid_len] * window[:valid_len]
        weight_buffer[i : i + valid_len] += window[:valid_len]

    weight_buffer[weight_buffer < 1e-8] = 1.0
    output_buffer /= weight_buffer
    
    out_name = file_path.rsplit('.', 1)[0] + "_DGAS_A100.wav"
    sf.write(out_name, output_buffer.T, sr)
    print(f"✅ Salvo: {out_name}")

if __name__ == "__main__":
    generator = load_models()
    # process_file("input.wav", generator)