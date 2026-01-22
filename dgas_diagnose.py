import jax
import jax.numpy as jnp
import equinox as eqx
import numpy as np
import librosa
import os
from dgas_model import Generator, Discriminator

# --- CONFIGURAÇÃO DE DIAGNÓSTICO ---
jax.config.update("jax_platform_name", "gpu")
CHECKPOINT = "checkpoints/dgas_latest.eqx"
FILE_PATH = "mixture.wav"
SIGNAL_SCALE = 5.0  # <--- OBRIGATÓRIO PARA VALIDAR O NOVO TREINO

def analyze_tensor(name, tensor):
    data = np.array(tensor)
    print(f"[{name}] Shape: {data.shape}")
    print(f"   Mean: {np.mean(data):.4f} | Std: {np.std(data):.4f}")
    print(f"   Min:  {np.min(data):.4f} | Max: {np.max(data):.4f}")
    
    # Análise de Magnitude para o Boost
    if "Spectrogram" in name:
        peak = np.max(np.abs(data))
        if peak < 1.0:
            print(f"   ⚠️  ALERTA: Sinal muito fraco ({peak:.4f}). O Boost não está a funcionar!")
        elif peak > 5.0:
            print(f"   ✅ SUCESSO: Sinal forte detetado ({peak:.4f}). O Boost está ativo.")

def main():
    print(f"=== DGAS DIAGNOSTIC TOOL (Boost x{SIGNAL_SCALE}) ===")
    
    # 1. Carregar Modelo
    print("\n1. A carregar modelo...")
    key = jax.random.PRNGKey(42)
    k1, k2 = jax.random.split(key)
    gen = Generator(key=k1)
    
    if not os.path.exists(CHECKPOINT):
        print(f"❌ Erro: Checkpoint {CHECKPOINT} não encontrado.")
        # Não abortamos, vamos testar a matemática do input à mesma
    else:
        try:
            models = (gen, Discriminator(key=k2))
            models = eqx.tree_deserialise_leaves(CHECKPOINT, models)
            gen = models[0]
            print("✅ Modelo carregado.")
        except Exception as e:
            print(f"❌ Erro ao carregar pesos: {e}")

    # 2. Carregar Audio
    print("\n2. A carregar áudio...")
    try:
        audio, sr = librosa.load(FILE_PATH, sr=44100, mono=False, duration=1.5)
        if audio.ndim == 1: audio = np.stack([audio, audio])
        
        # Simula a normalização do dgas_data.py
        peak = np.max(np.abs(audio))
        scale = 0.95 / (peak + 1e-8)
        audio_norm = audio * scale
        analyze_tensor("Audio Normalized (0.95 peak)", audio_norm)
        
    except Exception as e:
        print(f"❌ Erro no áudio: {e}")
        return

    # 3. Simular Pipeline com BOOST
    print("\n3. A simular entrada com SIGNAL_SCALE...")
    
    # STFT
    window = jnp.hanning(2048)
    chunk_jax = jnp.array(audio_norm)[None, ...]
    
    f, t, Zxx = jax.scipy.signal.stft(chunk_jax, fs=44100, window=window, nperseg=2048, noverlap=1536)
    spec = jnp.stack([Zxx.real, Zxx.imag], axis=-1)
    spec = jnp.transpose(spec, (0, 1, 4, 2, 3)).reshape(1, 4, spec.shape[2], spec.shape[3])
    
    # --- APLICAR O BOOST (AQUI ESTÁ A PROVA) ---
    spec_boosted = spec * SIGNAL_SCALE
    
    analyze_tensor(f"Spectrogram Input (Boost x{SIGNAL_SCALE})", spec_boosted)
    
    # Teste Rápido do Encoder (se o modelo carregou)
    if os.path.exists(CHECKPOINT):
        print("\n4. Teste de Reação do Modelo...")
        cond = gen.encoder(spec_boosted[0])
        analyze_tensor("Encoder Output", cond)
        
        # O encoder deve reagir com valores fortes se o input for forte
        if np.std(np.array(cond)) < 0.01:
             print("   ⚠️  AVISO: O Encoder parece 'adormecido' (Std baixo). Pode precisar de mais treino.")
        else:
             print("   ✅ O Encoder está a reagir bem ao sinal boostado.")

    print("\n=== FIM DO DIAGNÓSTICO ===")

if __name__ == "__main__":
    main()