import jax
import jax.numpy as jnp
import equinox as eqx
import numpy as np
import librosa
import soundfile as sf
import os
from tqdm import tqdm
from dgas_model import Generator, Discriminator

jax.config.update("jax_platform_name", "gpu") 

CONFIG = {
    # Garante que aponta para o checkpoint do treino novo
    "CHECKPOINT": "checkpoints/dgas_latest.eqx", 
    "N_FFT": 2048,
    "HOP_LENGTH": 512,
    "CHUNK_SIZE": 1.5,
    "OVERLAP": 0.5,
    "ODE_STEPS": 64, 
    "TARGET_SR": 44100,
    # --- O SEGREDO DO SUCESSO ---
    "SIGNAL_SCALE": 5.0  # Tem de ser IGUAL ao treino
}

def load_models():
    print(f"--- A Carregar Checkpoint: {CONFIG['CHECKPOINT']} ---")
    key = jax.random.PRNGKey(42)
    k1, k2 = jax.random.split(key)
    gen = Generator(key=k1)
    if not os.path.exists(CONFIG['CHECKPOINT']):
        raise FileNotFoundError(f"Erro: {CONFIG['CHECKPOINT']} não existe.")
    try:
        models = (gen, Discriminator(key=k2))
        models = eqx.tree_deserialise_leaves(CONFIG['CHECKPOINT'], models)
        return models[0]
    except Exception as e:
        raise RuntimeError(f"Erro checkpoint: {e}")

# STFT com BOOST
def cpu_stft_jax_impl(audio):
    window = jnp.hanning(CONFIG["N_FFT"])
    f, t, Zxx = jax.scipy.signal.stft(audio, fs=44100, window=window, nperseg=CONFIG["N_FFT"], noverlap=CONFIG["N_FFT"] - CONFIG["HOP_LENGTH"])
    spec = jnp.stack([Zxx.real, Zxx.imag], axis=-1)
    spec = jnp.transpose(spec, (0, 1, 4, 2, 3))
    B, C, _, F, T = spec.shape
    spec = spec.reshape(B, C * 2, F, T)
    

    return spec * CONFIG["SIGNAL_SCALE"]

cpu_stft_jax = jax.jit(cpu_stft_jax_impl, backend='cpu')


def cpu_istft_jax_impl(spec):

    spec = spec / CONFIG["SIGNAL_SCALE"]
    
    B, _, F, T = spec.shape
    spec = spec.reshape(B, 2, 2, F, T)
    Zxx = spec[:, :, 0] + 1j * spec[:, :, 1]
    window = jnp.hanning(CONFIG["N_FFT"])
    _, audio = jax.scipy.signal.istft(Zxx, fs=44100, window=window, nperseg=CONFIG["N_FFT"], noverlap=CONFIG["N_FFT"] - CONFIG["HOP_LENGTH"])
    return audio

cpu_istft_jax = jax.jit(cpu_istft_jax_impl, backend='cpu')

def get_log_coords(F, T):
    times = jnp.linspace(0, 1, T)
    linear_freqs = jnp.linspace(0, 1, F)
    log_freqs = jnp.log1p(linear_freqs * 10.0) / jnp.log1p(10.0)
    return log_freqs, times

@eqx.filter_jit
def predict_spectrogram(model, mix_spec, key, steps):
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
    return final_spec

def process_file(file_path, model):
    print(f"\n>>> A processar (Boost x{CONFIG['SIGNAL_SCALE']}): {file_path}")
    try:
        audio, sr = librosa.load(file_path, sr=CONFIG["TARGET_SR"], mono=False)
    except Exception as e:
        print(f"Erro: {e}")
        return
    if audio.ndim == 1: audio = np.stack([audio, audio])

    total_samples = audio.shape[1]
    chunk_samples = int(CONFIG["CHUNK_SIZE"] * sr)
    hop_samples = int(chunk_samples * (1 - CONFIG["OVERLAP"]))
    output_buffer = np.zeros_like(audio)
    weight_buffer = np.zeros(total_samples)
    window = np.hanning(chunk_samples)
    key = jax.random.PRNGKey(42)

    try:
        cpu_dev = jax.devices("cpu")[0]
        gpu_dev = jax.devices("gpu")[0]
    except:
        cpu_dev, gpu_dev = None, None

    print("A iniciar inferência...")
    for i in tqdm(range(0, total_samples - chunk_samples + 1, hop_samples)):
        chunk = audio[:, i : i + chunk_samples]
        chunk_peak = np.max(np.abs(chunk))
        
        
        if chunk_peak < 0.02:
            valid_len = min(chunk.shape[1], chunk_samples)
            weight_buffer[i : i + valid_len] += window[:valid_len]
            continue
            
     
        scale_factor = 0.95 / chunk_peak
        chunk_norm = chunk * scale_factor
        
        chunk_jax = jnp.array(chunk_norm)[None, ...]
        spec_jax = cpu_stft_jax(chunk_jax) 
        
        if gpu_dev: spec_gpu = jax.device_put(spec_jax, gpu_dev)
        else: spec_gpu = spec_jax
        
        key, subkey = jax.random.split(key)
        rec_spec_gpu = predict_spectrogram(model, spec_gpu, subkey, CONFIG["ODE_STEPS"])
        
        if cpu_dev: rec_spec_cpu = jax.device_put(rec_spec_gpu, cpu_dev)
        else: rec_spec_cpu = rec_spec_gpu
        rec_audio_jax = cpu_istft_jax(rec_spec_cpu)
        rec_audio = np.array(rec_audio_jax[0])

        rec_audio = rec_audio / scale_factor
        
        valid_len = min(rec_audio.shape[1], chunk_samples)
        output_buffer[:, i : i + valid_len] += rec_audio[:, :valid_len]
        weight_buffer[i : i + valid_len] += window[:valid_len]

    weight_buffer[weight_buffer < 1e-8] = 1.0
    output_buffer /= weight_buffer
    
    out_name = file_path.rsplit('.', 1)[0] + "_DGAS_BOOST.wav"
    sf.write(out_name, output_buffer.T, sr)
    print(f"✅ Salvo: {out_name}")

if __name__ == "__main__":
    generator = load_models()
    if os.path.exists("mixture.wav"):
        process_file("mixture.wav", generator)
    else:
        print("Erro: mixture.wav não encontrado.")