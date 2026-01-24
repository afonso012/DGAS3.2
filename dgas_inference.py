import jax
import jax.numpy as jnp
import equinox as eqx
import numpy as np
import librosa
import soundfile as sf
import os
from tqdm import tqdm
from dgas_model import Generator, Discriminator

# --- CONFIGURAÇÃO PRODUCTION (SOTA + TTA + Batching) ---
CONFIG = {
    "CHECKPOINT": "checkpoints/dgas_latest.eqx",
    "CHUNK_SIZE": 65536, 
    "N_FFT": 2048,
    "HOP_LENGTH": 512,
    "ODE_STEPS": 32,
    "OVERLAP": 0.25, 
    "TARGET_SR": 44100,
    
    # Batch & TTA (Ajusta se tiveres pouca VRAM)
    "INFERENCE_BATCH_SIZE": 4, 
    "USE_TTA": True,
}

# ==============================================================================
# 1. CARREGAMENTO DO MOTOR
# ==============================================================================

def load_models():
    print(f"--- A Carregar Motor PRODUCTION: {CONFIG['CHECKPOINT']} ---")
    key = jax.random.PRNGKey(42)
    gen = Generator(key=key)
    disc = Discriminator(key=key) 
    
    if not os.path.exists(CONFIG['CHECKPOINT']):
        fallback = "checkpoints/dgas_latest.eqx"
        if os.path.exists(fallback):
            print(f"⚠️ EMA não encontrado. A usar fallback: {fallback}")
            CONFIG['CHECKPOINT'] = fallback
        else:
            raise FileNotFoundError(f"Erro: Checkpoint não encontrado.")
    
    try:
        (gen, _) = eqx.tree_deserialise_leaves(CONFIG['CHECKPOINT'], (gen, disc))
        print("✅ Motor carregado com sucesso.")
    except Exception:
        print("⚠️ Aviso: Estrutura parcial. A carregar apenas Gerador...")
        gen = eqx.tree_deserialise_leaves(CONFIG['CHECKPOINT'], gen)
            
    return gen

# ==============================================================================
# 2. ENGENHARIA DE SINAL (LOG-AWARE + SAFETY CLAMP)
# ==============================================================================

@jax.jit
def stft_log_preprocess(audio):
    """Áudio -> Log Spectrogram (Batch Aware)"""
    window = jnp.hanning(CONFIG["N_FFT"])
    
    def single_stft(a):
        f, t, Zxx = jax.scipy.signal.stft(
            a, fs=44100, window=window, 
            nperseg=CONFIG["N_FFT"], 
            noverlap=CONFIG["N_FFT"] - CONFIG["HOP_LENGTH"]
        )
        return Zxx
        
    B, C, T = audio.shape
    audio_flat = audio.reshape(B*C, T)
    Zxx_flat = jax.vmap(single_stft)(audio_flat) 
    
    _, F, T_spec = Zxx_flat.shape
    Zxx = Zxx_flat.reshape(B, C, F, T_spec)
    
    mag = jnp.abs(Zxx)
    phase = jnp.angle(Zxx)
    mag_log = jnp.log1p(mag * 1000.0) * 0.1
    
    spec = jnp.stack([mag_log * jnp.cos(phase), mag_log * jnp.sin(phase)], axis=-1)
    spec = jnp.transpose(spec, (0, 1, 4, 2, 3)).reshape(B, C * 2, F, T_spec)
    return spec

@jax.jit
def istft_log_postprocess(spec):
    """Log Spectrogram -> Áudio (Batch Aware + Safety Clamp)"""
    B, C2, F, T = spec.shape
    
    l_re, l_im = spec[:, 0], spec[:, 1]
    r_re, r_im = spec[:, 2], spec[:, 3]
    
    def recover_complex(re, im):
        mag = jnp.sqrt(re**2 + im**2)
        phase = jnp.arctan2(im, re)
        
        # --- A CORREÇÃO MÁGICA ---
        # Tal como no treino, impedimos a rede de prever > 1.0
        # Isto evita que o expm1 dispare para 22.0 (clipping)
        limit = 0.7
        mag = limit * jnp.tanh(mag / limit) # Substitui o jnp.clip
        
        mag_linear = jnp.expm1(mag * 10.0) / 1000.0
        return mag_linear * jnp.exp(1j * phase)

    Z_l = recover_complex(l_re, l_im)
    Z_r = recover_complex(r_re, r_im)
    
    window = jnp.hanning(CONFIG["N_FFT"])
    def single_istft(z):
        return jax.scipy.signal.istft(z, fs=44100, window=window, 
                                     nperseg=CONFIG["N_FFT"], 
                                     noverlap=CONFIG["N_FFT"] - CONFIG["HOP_LENGTH"])[1]
    
    Z_concat = jnp.concatenate([Z_l, Z_r], axis=0)
    wav_concat = jax.vmap(single_istft)(Z_concat)
    
    wav_l = wav_concat[:B]
    wav_r = wav_concat[B:]
    
    return jnp.stack([wav_l, wav_r], axis=1)

# ==============================================================================
# 3. SOLVER ODE & TTA
# ==============================================================================

def get_log_coords(B, F, T):
    times = jnp.linspace(0, 1, T)
    freqs_log = jnp.logspace(jnp.log10(1e-3), jnp.log10(1.0), F)
    freqs_log = (freqs_log - 1e-3) / (1.0 - 1e-3)
    grid_f, grid_t = jnp.meshgrid(freqs_log, times, indexing='ij')
    return grid_f.flatten(), grid_t.flatten()

@eqx.filter_jit
def predict_batch(model, mix_spec, key):
    # TTA Logic
    if CONFIG["USE_TTA"]:
        mix_spec_flip = jnp.stack([mix_spec[:, 2], mix_spec[:, 3], mix_spec[:, 0], mix_spec[:, 1]], axis=1)
        full_input = jnp.concatenate([mix_spec, mix_spec_flip], axis=0)
    else:
        full_input = mix_spec

    # Inference
    cond_grids = jax.vmap(model.encoder)(full_input)
    x = jax.random.normal(key, full_input.shape)
    
    B_total, _, F, T = full_input.shape
    ff, tt = get_log_coords(B_total, F, T)
    
    def get_velocity(t_scalar, x_curr):
        x_flat = jnp.transpose(x_curr, (0, 2, 3, 1)).reshape(B_total, -1, 4)
        def predict_pixel(ti, xi_flat, zi_grids):
            return jax.vmap(lambda f, t_val, x_val: model.field(ti, jnp.array([t_val, f]), x_val, zi_grids))(ff, tt, xi_flat)
        v_flat = jax.vmap(predict_pixel)(t_scalar * jnp.ones((B_total,)), x_flat, cond_grids)
        return jnp.transpose(v_flat.reshape(B_total, F, T, 4), (0, 3, 1, 2))

    # Solver Heun
    steps = CONFIG["ODE_STEPS"]
    dt = 1.0 / steps
    
    def loop_body(i, curr_x):
        t = i * dt
        v1 = get_velocity(t, curr_x)
        x_tilde = curr_x + v1 * dt
        v2 = get_velocity(t + dt, x_tilde)
        return curr_x + 0.5 * dt * (v1 + v2)

    final_spec = jax.lax.fori_loop(0, steps, loop_body, x)
    
    # TTA Merge
    if CONFIG["USE_TTA"]:
        B_orig = mix_spec.shape[0]
        res_normal = final_spec[:B_orig]
        res_flip = final_spec[B_orig:]
        res_flip_back = jnp.stack([res_flip[:, 2], res_flip[:, 3], res_flip[:, 0], res_flip[:, 1]], axis=1)
        final_spec = (res_normal + res_flip_back) * 0.5

    return istft_log_postprocess(final_spec)

# ==============================================================================
# 4. PROCESSAMENTO OTIMIZADO
# ==============================================================================

def process_file(file_path, model):
    print(f"\n>>> A processar: {file_path}")
    
    try:
        audio, sr = librosa.load(file_path, sr=CONFIG["TARGET_SR"], mono=False)
    except Exception as e:
        print(f"❌ Erro: {e}")
        return
        
    if audio.ndim == 1: audio = np.stack([audio, audio])
    if audio.shape[1] < CONFIG["CHUNK_SIZE"]:
        pad = CONFIG["CHUNK_SIZE"] - audio.shape[1]
        audio = np.pad(audio, ((0, 0), (0, pad)))

    total_samples = audio.shape[1]
    chunk_size = CONFIG["CHUNK_SIZE"]
    hop_size = int(chunk_size * (1 - CONFIG["OVERLAP"]))
    
    output_buffer = np.zeros_like(audio)
    weight_buffer = np.zeros(total_samples)
    window = np.hanning(chunk_size)
    
    starts = list(range(0, total_samples - chunk_size + 1, hop_size))
    batches = [starts[i:i + CONFIG["INFERENCE_BATCH_SIZE"]] for i in range(0, len(starts), CONFIG["INFERENCE_BATCH_SIZE"])]
    
    print(f"Chunks: {len(starts)} | Batches: {len(batches)}")
    
    key = jax.random.PRNGKey(0)
    
    for batch_indices in tqdm(batches):
        batch_audio = []
        valid_indices = []
        scales = []
        
        for i in batch_indices:
            chunk = audio[:, i : i + chunk_size]
            peak = np.max(np.abs(chunk))
            
            if peak < 0.01:
                weight_buffer[i : i + chunk_size] += window
                continue
                
            scale = 0.95 / (peak + 1e-8)
            chunk_norm = chunk * scale
            
            batch_audio.append(chunk_norm)
            valid_indices.append(i)
            scales.append(scale)
            
        if not batch_audio: continue
        
        batch_tensor = jnp.array(np.stack(batch_audio)) 
        spec_input = stft_log_preprocess(batch_tensor)
        
        key, subkey = jax.random.split(key)
        rec_batch_jax = predict_batch(model, spec_input, subkey)
        rec_batch = np.array(rec_batch_jax) 
        
        for rec_chunk, scale, start_sample in zip(rec_batch, scales, valid_indices):
            rec_chunk = rec_chunk / scale
            output_buffer[:, start_sample : start_sample + chunk_size] += rec_chunk * window
            weight_buffer[start_sample : start_sample + chunk_size] += window

    weight_buffer[weight_buffer < 1e-8] = 1.0
    output_buffer /= weight_buffer
    
    # Clip final para garantir que está no range de áudio válido
    output_buffer = np.clip(output_buffer, -1.0, 1.0)
    
    suffix = "_SOTA_TTA.wav" if CONFIG["USE_TTA"] else "_SOTA.wav"
    out_name = file_path.rsplit('.', 1)[0] + suffix
    sf.write(out_name, output_buffer.T, CONFIG["TARGET_SR"])
    print(f"✅ Fim: {out_name}")

if __name__ == "__main__":
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    try:
        gen = load_models()
        if os.path.exists("mixture.wav"):
            process_file("mixture.wav", gen)
        else:
            print("Cria 'mixture.wav' para testar.")
    except KeyboardInterrupt:
        print("\nStop.")
    except Exception as e:
        print(f"Erro: {e}")