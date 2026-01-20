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
    "ODE_STEPS": 32, # Heun precisa de mais steps para brilhar
    "TARGET_SR": 44100
}

def load_models():
    print(f"--- A Carregar Checkpoint: {CONFIG['CHECKPOINT']} ---")
    key = jax.random.PRNGKey(0)
    k1, k2 = jax.random.split(key)
    
    gen = Generator(key=k1)
    disc = Discriminator(key=k2)
    models = (gen, disc)
    
    try:
        models = eqx.tree_deserialise_leaves(CONFIG['CHECKPOINT'], models)
        print(">>> Pesos carregados com SUCESSO.")
        gen = models[0]
        return gen.field, gen.encoder
    except FileNotFoundError:
        raise FileNotFoundError(f"Erro: Checkpoint não encontrado!")

@eqx.filter_jit
def predict_step(model, encoder, mix_spec, key, steps):
    # mix_spec: (1, 4, F, T)
    cond = encoder(mix_spec)
    x = jax.random.normal(key, mix_spec.shape)
    dt = 1.0 / steps
    
    B, C, F, T = mix_spec.shape
    freqs = jnp.linspace(0, 1, F)
    times = jnp.linspace(0, 1, T)
    grid_f, grid_t = jnp.meshgrid(freqs, times, indexing='ij')
    
    # Flatten grids
    f_flat, t_flat = grid_f.flatten(), grid_t.flatten()
    
    # Função auxiliar para calcular o campo vetorial
    def get_velocity(t_curr, x_curr):
        # x_curr: (1, 4, F, T) -> Precisa ser transformado para (F*T, 4) para o vmap
        # Flatten espacialmente preservando canais no último eixo
        x_flat = jnp.transpose(x_curr, (0, 2, 3, 1)).reshape(-1, 4) # (F*T, 4)
        
        # O vmap agora itera sobre (f, t, x_val)
        def field_point(f_val, t_val, x_val_i):
            # Passamos x_val_i (4,) para o modelo junto com t e pos
            return model(t_curr, jnp.array([t_val, f_val]), x_val_i, cond[0])
            
        # Vmap sobre todos os pixels do espectrograma
        v_flat = jax.vmap(field_point)(f_flat, t_flat, x_flat)
        
        # Reshape de volta para (1, 4, F, T)
        return jnp.transpose(v_flat.reshape(1, F, T, 4), (0, 3, 1, 2))

    def loop_body(i, curr_x):
        t = i / steps
        
        # === Solver HEUN (2nd Order) ===
        # 1. Predict (Euler)
        d1 = get_velocity(t, curr_x)
        x_tilde = curr_x + d1 * dt
        
        # 2. Correct (Trapezoidal Rule)
        d2 = get_velocity(t + dt, x_tilde)
        curr_x = curr_x + (d1 + d2) * 0.5 * dt
        
        return curr_x

    return jax.lax.fori_loop(0, steps, loop_body, x)

def spectrogram_to_audio(spec_chunk):
    l_re, l_im = spec_chunk[0], spec_chunk[1]
    r_re, r_im = spec_chunk[2], spec_chunk[3]
    l_complex = l_re + 1j * l_im
    r_complex = r_re + 1j * r_im
    audio_l = librosa.istft(np.array(l_complex), hop_length=CONFIG["HOP_LENGTH"], n_fft=CONFIG["N_FFT"])
    audio_r = librosa.istft(np.array(r_complex), hop_length=CONFIG["HOP_LENGTH"], n_fft=CONFIG["N_FFT"])
    return np.stack([audio_l, audio_r])

def process_file(file_path, model, encoder):
    print(f"\n>>> A processar: {file_path}")
    audio, sr = librosa.load(file_path, sr=CONFIG["TARGET_SR"], mono=False)
    if audio.ndim == 1: audio = np.stack([audio, audio])
    
    total_samples = audio.shape[1]
    chunk_samples = int(CONFIG["CHUNK_SIZE"] * sr)
    hop_samples = int(chunk_samples * (1 - CONFIG["OVERLAP"]))
    
    output_buffer = np.zeros_like(audio)
    weight_buffer = np.zeros(total_samples)
    window = np.hanning(chunk_samples)
    
    key = jax.random.PRNGKey(42)
    
    for i in tqdm(range(0, total_samples - chunk_samples + 1, hop_samples)):
        chunk = audio[:, i : i + chunk_samples]
        l_stft = librosa.stft(chunk[0], n_fft=CONFIG["N_FFT"], hop_length=CONFIG["HOP_LENGTH"])
        r_stft = librosa.stft(chunk[1], n_fft=CONFIG["N_FFT"], hop_length=CONFIG["HOP_LENGTH"])
        
        spec_input = np.stack([l_stft.real, l_stft.imag, r_stft.real, r_stft.imag])
        spec_input_jax = jnp.array(spec_input)[None, ...] 
        
        key, subkey = jax.random.split(key)
        predicted_spec = predict_step(model, encoder, spec_input_jax, subkey, CONFIG["ODE_STEPS"])
        predicted_spec = np.array(predicted_spec[0]) 
        
        rec_audio = spectrogram_to_audio(predicted_spec)
        valid_len = min(rec_audio.shape[1], chunk_samples)
        
        output_buffer[:, i : i + valid_len] += rec_audio[:, :valid_len] * window[:valid_len]
        weight_buffer[i : i + valid_len] += window[:valid_len]

    weight_buffer[weight_buffer < 1e-8] = 1.0
    output_buffer /= weight_buffer
    
    out_name = file_path.rsplit('.', 1)[0] + "_DGAS_FIXED.wav"
    sf.write(out_name, output_buffer.T, sr)
    print(f"✅ Salvo (Fixed): {out_name}")

if __name__ == "__main__":
    model, encoder = load_models()
    # Adicionar loop de input aqui se desejado