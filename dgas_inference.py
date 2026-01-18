import jax
import jax.numpy as jnp
import equinox as eqx
import numpy as np
import librosa
import soundfile as sf
import os
from tqdm import tqdm
from dgas_model import DGASField, LatentEncoder, MultiPeriodDiscriminator

# === CONFIGURAÇÃO DE INFERÊNCIA ===
CONFIG = {
    "CHECKPOINT": "dgas_stereo_step_20000.eqx",  # O teu checkpoint atual
    "N_FFT": 1024,
    "HOP_LENGTH": 256,
    "CHUNK_SIZE": 5.0,    # Processar 5 segundos de cada vez (Save RAM)
    "OVERLAP": 0.5,       # Sobreposição para cross-fade suave
    "ODE_STEPS": 16,      # Qualidade vs Velocidade (16 é bom, 32 é estúdio, 64 é exagero)
    "TARGET_SR": 44100
}

def load_models():
    print(f"--- A Carregar Checkpoint: {CONFIG['CHECKPOINT']} ---")
    key = jax.random.PRNGKey(0)
    k1, k2, k3 = jax.random.split(key, 3)
    
    # Inicializar estrutura idêntica ao treino
    # Nota: Precisamos do Discriminador aqui apenas para carregar os pesos corretamente,
    # depois descartamo-lo.
    models = (
        DGASField(k1), 
        LatentEncoder(k2, input_channels=4), 
        MultiPeriodDiscriminator(k3)
    )
    
    try:
        models = eqx.tree_deserialise_leaves(CONFIG['CHECKPOINT'], models)
        print(">>> Pesos carregados com SUCESSO.")
        return models[0], models[1] # Retorna apenas Gerador e Encoder
    except FileNotFoundError:
        raise FileNotFoundError(f"Erro: O ficheiro {CONFIG['CHECKPOINT']} não existe!")

@eqx.filter_jit
def predict_step(model, encoder, mix_spec, key, steps):
    """
    Resolve a ODE (Ordinary Differential Equation) do Rectified Flow.
    Transforma Ruído -> Voz Limpa condicionado na Mistura.
    """
    # 1. Obter condicionamento da mistura (Latent Representation)
    cond = encoder(mix_spec)
    
    # 2. Preparar estado inicial (Ruído Gaussiano)
    x = jax.random.normal(key, mix_spec.shape)
    
    # 3. Solver de Euler (Traçar a linha reta do ruído ao som)
    dt = 1.0 / steps
    
    def loop_body(i, curr_x):
        t = i / steps
        t_batch = jnp.ones((1,)) * t
        # O modelo prevê a velocidade (vector field)
        v = model(t_batch, curr_x, cond)
        return curr_x + v * dt

    # Executar o loop da ODE
    final_x = jax.lax.fori_loop(0, steps, loop_body, x)
    
    return final_x

def spectrogram_to_audio(spec_chunk):
    """Reconstrução Estéreo: (4, F, T) -> (2, Samples)"""
    # Desempacotar canais: L_Re, L_Im, R_Re, R_Im
    l_re, l_im = spec_chunk[0], spec_chunk[1]
    r_re, r_im = spec_chunk[2], spec_chunk[3]
    
    # Reconstruir complexos
    l_complex = l_re + 1j * l_im
    r_complex = r_re + 1j * r_im
    
    # iSTFT
    audio_l = librosa.istft(np.array(l_complex), hop_length=CONFIG["HOP_LENGTH"], n_fft=CONFIG["N_FFT"])
    audio_r = librosa.istft(np.array(r_complex), hop_length=CONFIG["HOP_LENGTH"], n_fft=CONFIG["N_FFT"])
    
    return np.stack([audio_l, audio_r])

def process_file(file_path, model, encoder):
    print(f"\n>>> A processar: {file_path}")
    
    # 1. Carregar Áudio
    audio, sr = librosa.load(file_path, sr=CONFIG["TARGET_SR"], mono=False)
    
    # Garantir Estéreo
    if audio.ndim == 1:
        audio = np.stack([audio, audio])
    
    # Normalizar
    peak = np.max(np.abs(audio))
    audio = audio / (peak + 1e-8)
    
    total_samples = audio.shape[1]
    chunk_samples = int(CONFIG["CHUNK_SIZE"] * sr)
    overlap_samples = int(chunk_samples * CONFIG["OVERLAP"])
    hop_samples = chunk_samples - overlap_samples
    
    # Buffers para reconstrução (Overlap-Add)
    output_buffer = np.zeros_like(audio)
    weight_buffer = np.zeros(total_samples)
    
    # Janela de Hanning para suavizar as bordas dos chunks
    window = np.hanning(chunk_samples)
    
    # Key para geração aleatória
    key = jax.random.PRNGKey(42)
    
    # Loop de Chunks
    num_chunks = (total_samples - overlap_samples) // hop_samples + 1
    
    print(f"Dividido em {num_chunks} chunks de {CONFIG['CHUNK_SIZE']}s...")
    
    for i in tqdm(range(0, total_samples - chunk_samples + 1, hop_samples)):
        # Cortar
        chunk = audio[:, i : i + chunk_samples]
        
        # STFT
        l_stft = librosa.stft(chunk[0], n_fft=CONFIG["N_FFT"], hop_length=CONFIG["HOP_LENGTH"])
        r_stft = librosa.stft(chunk[1], n_fft=CONFIG["N_FFT"], hop_length=CONFIG["HOP_LENGTH"])
        
        # Preparar Tensor (4 canais: L_Re, L_Im, R_Re, R_Im)
        spec_input = np.stack([
            l_stft.real, l_stft.imag,
            r_stft.real, r_stft.imag
        ]) # Shape: (4, F, T)
        
        # Converter para JAX
        spec_input_jax = jnp.array(spec_input)
        
        # === INFERÊNCIA (A Magia Acontece Aqui) ===
        key, subkey = jax.random.split(key)
        predicted_spec = predict_step(model, encoder, spec_input_jax, subkey, CONFIG["ODE_STEPS"])
        predicted_spec = np.array(predicted_spec)
        # ==========================================
        
        # Reconstruir Áudio
        rec_audio = spectrogram_to_audio(predicted_spec)
        
        # Ajustar tamanho (iSTFT pode variar ligeiramente)
        valid_len = min(rec_audio.shape[1], chunk_samples)
        rec_audio = rec_audio[:, :valid_len]
        
        # Overlap-Add com Janela
        # Precisamos expandir a janela para estéreo (2, N)
        win_stack = np.stack([window[:valid_len], window[:valid_len]])
        
        output_buffer[:, i : i + valid_len] += rec_audio * win_stack
        weight_buffer[i : i + valid_len] += window[:valid_len]

    # Normalizar pelo peso das janelas (para não aumentar volume nas sobreposições)
    weight_buffer[weight_buffer < 1e-8] = 1.0
    output_buffer /= weight_buffer
    
    # Gravar resultado
    out_name = file_path.rsplit('.', 1)[0] + "_VOCALS_DGAS_20k.wav"
    sf.write(out_name, output_buffer.T, sr)
    print(f"\n✅ Concluído! Salvo em: {out_name}")

def main():
    print("=== DGAS 3.2: INFERENCE SYSTEM (GOD TIER) ===")
    
    # 1. Carregar Modelo
    model, encoder = load_models()
    
    # 2. Interface Simples
    while True:
        print("\n" + "="*40)
        path = input("Arrasta uma música para aqui (ou 'q' para sair): ").strip().replace("'", "").strip()
        
        if path.lower() == 'q':
            break
            
        if not os.path.exists(path):
            print("❌ Ficheiro não encontrado. Tenta outra vez.")
            continue
            
        try:
            process_file(path, model, encoder)
        except Exception as e:
            print(f"❌ Erro fatal: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()