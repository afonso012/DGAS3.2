import jax
import jax.numpy as jnp
import equinox as eqx
import numpy as np
import librosa
import soundfile as sf
import scipy.signal
import os
from tqdm import tqdm
from dgas_model import Generator, Discriminator

CONFIG = {
    "CHECKPOINT": "checkpoints/dgas_test.eqx", 
    "N_FFT": 2048,
    "HOP_LENGTH": 512,
    "CHUNK_SIZE": 1.5,
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

# --- CPU SIGNAL PROCESSING (Scipy) ---
# Movemos isto para CPU para evitar o crash do cuFFT na A100
def cpu_stft(audio):
    # audio shape: (2, N)
    f, t, Zxx = scipy.signal.stft(
        audio, 
        fs=CONFIG["TARGET_SR"], 
        window='hann', 
        nperseg=CONFIG["N_FFT"], 
        noverlap=CONFIG["N_FFT"] - CONFIG["HOP_LENGTH"],
        boundary='zeros',
        padded=True
    )
    # Zxx shape: (2, F, T) -> complexo
    # Converter para formato do modelo: (1, 4, F, T) -> [L_real, L_imag, R_real, R_imag]
    spec_l = Zxx[0]
    spec_r = Zxx[1]
    spec = np.stack([spec_l.real, spec_l.imag, spec_r.real, spec_r.imag], axis=0)
    return spec[None, ...] # Adicionar Batch dim: (1, 4, F, T)

def cpu_istft(spec_jax):
    # spec_jax: (1, 4, F, T) -> vindo da GPU
    spec = np.array(spec_jax[0]) # Converter para Numpy CPU
    
    # Reconstruir complexos
    l_complex = spec[0] + 1j * spec[1]
    r_complex = spec[2] + 1j * spec[3]
    
    _, audio_l = scipy.signal.istft(l_complex, fs=CONFIG["TARGET_SR"], window='hann', nperseg=CONFIG["N_FFT"], noverlap=CONFIG["N_FFT"] - CONFIG["HOP_LENGTH"])
    _, audio_r = scipy.signal.istft(r_complex, fs=CONFIG["TARGET_SR"], window='hann', nperseg=CONFIG["N_FFT"], noverlap=CONFIG["N_FFT"] - CONFIG["HOP_LENGTH"])
    
    return np.stack([audio_l, audio_r], axis=0)

def get_log_coords(F, T):
    times = jnp.linspace(0, 1, T)
    linear_freqs = jnp.linspace(0, 1, F)
    log_freqs = jnp.log1p(linear_freqs * 10.0) / jnp.log1p(10.0)
    return log_freqs, times

# --- GPU MODEL INFERENCE (Apenas Neural Net) ---
@eqx.filter_jit
def predict_spectrogram(model, mix_spec, key, steps):
    # mix_spec já entra como (1, 4, F, T)
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
    print(f"\n>>> A processar: {file_path}")
    
    # 1. Carregar Audio
    try:
        audio, sr = librosa.load(file_path, sr=CONFIG["TARGET_SR"], mono=False)
    except Exception as e:
        print(f"Erro ao abrir ficheiro: {e}")
        return

    if audio.ndim == 1:
        print("Aviso: Audio Mono detetado. A converter para Stereo...")
        audio = np.stack([audio, audio])
    
    # 2. Normalização de Pico (ESSENCIAL)
    original_peak = np.max(np.abs(audio))
    if original_peak > 0:
        scale_factor = 0.95 / original_peak
        audio = audio * scale_factor
        print(f"Normalização aplicada: Peak {original_peak:.4f} -> 0.95")
    else:
        print("Aviso: Audio silencioso detetado.")

    total_samples = audio.shape[1]
    chunk_samples = int(CONFIG["CHUNK_SIZE"] * sr)
    hop_samples = int(chunk_samples * (1 - CONFIG["OVERLAP"]))
    
    output_buffer = np.zeros_like(audio)
    weight_buffer = np.zeros(total_samples)
    window = np.hanning(chunk_samples)
    
    key = jax.random.PRNGKey(42)
    
    print("A iniciar inferência (STFT CPU -> GPU Model -> ISTFT CPU)...")
    for i in tqdm(range(0, total_samples - chunk_samples + 1, hop_samples)):
        chunk = audio[:, i : i + chunk_samples]
        
        # 1. CPU STFT
        spec_cpu = cpu_stft(chunk) # Retorna numpy array
        
        # 2. Envia para GPU
        spec_jax = jnp.array(spec_cpu)
        
        # 3. Inferência Neural (Flow Matching)
        key, subkey = jax.random.split(key)
        rec_spec_jax = predict_spectrogram(model, spec_jax, subkey, CONFIG["ODE_STEPS"])
        
        # 4. CPU ISTFT
        rec_audio = cpu_istft(rec_spec_jax)
        
        # Overlap-Add
        # Verifica tamanhos (pode haver pequenas diferenças de rounding no ISTFT)
        valid_len = min(rec_audio.shape[1], chunk_samples)
        output_buffer[:, i : i + valid_len] += rec_audio[:, :valid_len] * window[:valid_len]
        weight_buffer[i : i + valid_len] += window[:valid_len]

    weight_buffer[weight_buffer < 1e-8] = 1.0
    output_buffer /= weight_buffer
    
    out_name = file_path.rsplit('.', 1)[0] + "_DGAS_OUT.wav"
    sf.write(out_name, output_buffer.T, sr)
    print(f"✅ Sucesso! Salvo em: {out_name}")

if __name__ == "__main__":
    generator = load_models()
    process_file("mixture.wav", generator)