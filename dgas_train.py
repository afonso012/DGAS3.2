import jax
import jax.numpy as jnp
import optax
import equinox as eqx
import time
import os
import signal
import sys
from dgas_model import DGASModel, Generator, Discriminator
from dgas_data import AudioLoader

# --- CONFIGURAÇÃO ---
CONFIG = {
    "DATA_DIR": "/workspace/musdb18hq/train",
    "BATCH_SIZE": 32,          # Tenta 32 ou 48 na A100
    "LEARNING_RATE": 1e-4,
    "STEPS": 1000000,
    "N_FFT": 2048,
    "HOP_LENGTH": 512
}

# --- FUNÇÃO STFT NA GPU (JAX) ---
def gpu_stft(audio):
    # Entrada: (Batch, Channels, Samples)
    # Hann Window criada na GPU
    window = jnp.hanning(CONFIG["N_FFT"])
    
    # Função STFT do JAX
    # Nota: output do stft é (..., Frequencias, Tempo)
    f, t, Zxx = jax.scipy.signal.stft(
        audio, 
        fs=44100, 
        window=window, 
        nperseg=CONFIG["N_FFT"], 
        noverlap=CONFIG["N_FFT"] - CONFIG["HOP_LENGTH"]
    )
    
    # Ajuste de forma para: (Batch, Channels, Freq, Time)
    # E separar Real/Imag para o canal final
    Zxx = jnp.transpose(Zxx, (0, 1, 2, 3)) # (Batch, Ch, Freq, Time)
    
    # Recortar frequencia extra (Nyquist) se necessário para bater certo com 1024/2048
    # Normalmente librosa dá 1025. O nosso modelo aceita 1025.
    
    # Stack Real e Imag: Output final (Batch, Ch, Freq, Time, 2)
    spec = jnp.stack([Zxx.real, Zxx.imag], axis=-1)
    
    # Flatten dos ultimos 2 canais para ficar (Batch, Ch, Freq, Time*2) ou similar?
    # Não, o modelo DGAS espera (Batch, Freq, Time, Channels_in) tipicamente? 
    # VAMOS ADAPTAR AO MODELO ANTIGO: (Batch, Channels, Freq, Time) -> Complexo separado
    
    # O DGAS original recebia (Batch, In_Ch, Freq, Time). O canal de entrada eram 4 (L_real, L_imag, R_real, R_imag)
    # Vamos reformatar para isso:
    
    B, C, F, T, _ = spec.shape
    # (Batch, 2_canais_audio, 1025, Time, 2_complexo)
    
    # Queremos: (Batch, 4_canais_misturados, Freq, Time)
    spec = jnp.transpose(spec, (0, 1, 4, 2, 3)) # (B, C, 2, F, T)
    spec = spec.reshape(B, C * 2, F, T)         # (B, 4, F, T)
    
    # Transpor para (B, F, T, 4) se o modelo for channel-last, ou manter se for channel-first
    # O nosso modelo DGAS usava Channel First nas Convs? 
    # O Equinox Conv2d é (Channel, Height, Width).
    # Portanto (Batch, 4, 1025, 128) está CORRETO.
    
    # Recorte temporal fixo para garantir tamanho (128 frames)
    # Se vier com mais, cortamos.
    return spec[:, :, :, :128]

def loss_fn(model, mix_spec, target_spec):
    pred_spec = model(mix_spec)
    # L1 Loss (Magnitude + Complexo)
    loss = jnp.mean(jnp.abs(pred_spec - target_spec))
    return loss

@eqx.filter_jit
def train_step(models, opt_states, mix_wav, target_wav, optimizers, steps):
    # 1. CONVERSÃO DSP NA GPU (Ultra Rápido)
    mix_spec = gpu_stft(mix_wav)
    target_spec = gpu_stft(target_wav)
    
    gen, disc = models
    opt_gen, opt_disc = opt_states
    
    # (Simplificando para treino apenas do Gerador para teste de velocidade)
    # Se quiseres GAN completa, descomenta o discriminador depois.
    # Vamos focar na velocidade agora.
    
    def compute_gen_loss(g):
        pred = g(mix_spec)
        return jnp.mean(jnp.abs(pred - target_spec))

    loss, grads = eqx.filter_value_and_grad(compute_gen_loss)(gen)
    updates, new_opt_gen = optimizers[0].update(grads, opt_gen, gen)
    new_gen = eqx.apply_updates(gen, updates)

    return (new_gen, disc), (new_opt_gen, opt_disc), loss, 0.0, {}

def main():
    # Inicialização
    key = jax.random.PRNGKey(0)
    
    # Criar modelo dummy só para inicializar formas
    # O modelo espera 4 canais in (Stereo Real+Imag) e 4 canais out
    gen = Generator(key=key) 
    disc = Discriminator(key=key)
    
    # Optimizadores
    opt_gen = optax.adam(CONFIG["LEARNING_RATE"])
    opt_disc = optax.adam(CONFIG["LEARNING_RATE"])
    
    opt_state_gen = opt_gen.init(eqx.filter(gen, eqx.is_array))
    opt_state_disc = opt_disc.init(eqx.filter(disc, eqx.is_array))
    
    # Data Loader
    loader = AudioLoader(CONFIG["DATA_DIR"], CONFIG["BATCH_SIZE"])
    loader.start()
    
    print("Pipeline Ready. Starting GPU-Accelerated Training...")
    
    try:
        for step in range(CONFIG["STEPS"]):
            # Recebe AUDIO CRU
            mix_wav, tgt_wav = loader.get_batch()
            
            # Envia para GPU
            mix_wav = jnp.array(mix_wav)
            tgt_wav = jnp.array(tgt_wav)
            
            start_time = time.time()
            models = (gen, disc)
            opt_states = (opt_state_gen, opt_state_disc)
            
            models, opt_states, g_loss, d_loss, _ = train_step(
                models, opt_states, mix_wav, tgt_wav, (opt_gen, opt_disc), step
            )
            gen, disc = models
            opt_state_gen, opt_state_disc = opt_states
            
            # Forçar sincronização para medir tempo real
            jax.block_until_ready(g_loss)
            dt = time.time() - start_time
            
            if step % 10 == 0:
                print(f"Step {step:05d} | Loss: {g_loss:.4f} | Time: {dt*1000:.1f}ms | FPS: {CONFIG['BATCH_SIZE']/dt:.1f}")

    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        loader.stop()

if __name__ == "__main__":
    main()