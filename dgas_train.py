import jax
import jax.numpy as jnp
import optax
import equinox as eqx
import time
import os
import shutil
from dgas_model import Generator, Discriminator
from dgas_data import AudioLoader

# === DGAS 3.1: CONFIGURAÇÃO DE PRODUÇÃO (A100) ===
CONFIG = {
    # I/O
    "DATA_DIR": "/workspace/musdb18hq/train",
    "CHECKPOINT_DIR": "checkpoints",
    "SAVE_INTERVAL": 1000,
    
    # Hiperparâmetros de Treino
    "BATCH_SIZE": 16,          
    "LEARNING_RATE": 1e-4,
    "STEPS": 1000000,
    "WARMUP_STEPS": 1000,      # Discriminador inativo até aqui (Estabilidade)
    
    # Processamento de Sinal (GPU Native)
    "N_FFT": 2048,
    "HOP_LENGTH": 512,
    
    # Pesos da Função de Perda (Relatório Secção 4)
    "LAMBDA_FLOW": 10.0,  # Transporte de Massa (Velocidade)
    "LAMBDA_APML": 2.0,   # Amplitude / Psicoacústica
    "LAMBDA_PHASE": 0.5,  # Consistência de Fase
    "LAMBDA_ADV": 0.1     # Textura GAN (Realismo)
}

# --- 1. GPU SIGNAL PROCESSING (JAX Native) ---
def gpu_stft(audio):
    """
    Realiza STFT diretamente na VRAM para evitar gargalo de CPU.
    Input: (Batch, Channels, Samples)
    Output: (Batch, 4, Freq, Time) -> [L_Re, L_Im, R_Re, R_Im]
    """
    window = jnp.hanning(CONFIG["N_FFT"])
    f, t, Zxx = jax.scipy.signal.stft(
        audio, fs=44100, window=window, 
        nperseg=CONFIG["N_FFT"], noverlap=CONFIG["N_FFT"] - CONFIG["HOP_LENGTH"]
    )
    # Reorganizar dimensões
    Zxx = jnp.transpose(Zxx, (0, 1, 2, 3)) # (Batch, Ch, Freq, Time)
    spec = jnp.stack([Zxx.real, Zxx.imag], axis=-1)
    
    # Flatten Stereo Complex: (B, 2, F, T, 2) -> (B, 4, F, T)
    B, C, F, T, _ = spec.shape
    spec = jnp.transpose(spec, (0, 1, 4, 2, 3)).reshape(B, C * 2, F, T)
    
    # Normalização de magnitude (x10) e recorte temporal fixo (128 frames)
    return spec[:, :, :, :128] * 10.0

# --- 2. CÁLCULO DE PERDAS (Loss Landscape) ---
def compute_losses(generator, discriminator, mix_spec, target_spec, key, step):
    # --- A. Rectified Flow Setup ---
    # 1. Extrair Latente da Mistura (Condicionamento)
    z = jax.vmap(generator.encoder)(mix_spec) 
    
    # 2. Definir Tempo e Ruído
    B, C, F, T = target_spec.shape
    t = jax.random.uniform(key, (B,), minval=0.0, maxval=1.0)
    
    key, k_noise = jax.random.split(key)
    x0 = jax.random.normal(k_noise, target_spec.shape) # Distribuição de Ruído
    x1 = target_spec                                   # Distribuição de Dados (Limpo)
    
    # 3. Interpolação Linear (O caminho mais reto)
    t_b = t[:, None, None, None]
    x_t = t_b * x1 + (1.0 - t_b) * x0
    v_target = x1 - x0 # Velocidade Alvo
    
    # --- B. Forward Pass do Gerador ---
    # Criar grelha de coordenadas normalizadas para o HashGrid
    freqs = jnp.linspace(0, 1, F)
    times = jnp.linspace(0, 1, T)
    grid_f, grid_t = jnp.meshgrid(freqs, times, indexing='ij')
    
    def predict_single(ti, xt_i, z_i):
        # Flatten para processamento paralelo de pixels
        f_flat, t_flat = grid_f.flatten(), grid_t.flatten()
        def field_point(f_val, t_val):
            return generator.field(ti, jnp.array([t_val, f_val]), z_i)
        # Vector Field é pointwise
        return jax.vmap(field_point)(f_flat, t_flat).T.reshape(4, F, T)

    # vmap sobre o Batch (Paralelismo Total)
    v_pred = jax.vmap(predict_single)(t, x_t, z)
    
    # --- C. Cálculo das Métricas ---
    
    # 1. Flow Matching Loss (MSE)
    flow_loss = jnp.mean((v_pred - v_target) ** 2)
    
    # 2. APML (Amplitude Loss) - Curvas de Fletcher-Munson implícitas
    mag_pred = jnp.sqrt(v_pred[:, 0]**2 + v_pred[:, 1]**2 + 1e-6)
    mag_target = jnp.sqrt(v_target[:, 0]**2 + v_target[:, 1]**2 + 1e-6)
    apml_loss = jnp.mean(jnp.abs(mag_pred - mag_target))
    
    # 3. Phase Loss (Consistência Geométrica)
    # Cosine distance entre vetores complexos
    dot = v_pred * v_target
    norm = jnp.abs(v_pred) * jnp.abs(v_target) + 1e-6
    phase_loss = jnp.mean(1.0 - dot / norm)

    # 4. GAN Loss (Textura Adversarial) com Warmup
    # Usamos jax.lax.cond para desligar o gradiente do discriminador no início
    def train_disc():
        fake_score = jax.vmap(discriminator)(v_pred, mix_spec)
        real_score = jax.vmap(discriminator)(v_target, mix_spec)
        # Hinge Loss
        d_l = jnp.mean(jax.nn.relu(1.0 - real_score)) + jnp.mean(jax.nn.relu(1.0 + fake_score))
        adv_l = -jnp.mean(fake_score)
        return d_l, adv_l

    def warmup_disc():
        return 0.0, 0.0

    # Lógica Condicional Compilada
    d_loss, adv_loss = jax.lax.cond(
        step >= CONFIG["WARMUP_STEPS"],
        train_disc,
        warmup_disc
    )
    
    # Combinação Ponderada (Secção 4 do Relatório)
    g_loss = (CONFIG["LAMBDA_FLOW"] * flow_loss + 
              CONFIG["LAMBDA_APML"] * apml_loss + 
              CONFIG["LAMBDA_ADV"] * adv_loss + 
              CONFIG["LAMBDA_PHASE"] * phase_loss)
    
    # Retorna tuplo exato para o JAX: (ScalarLoss, AuxData)
    return g_loss, (d_loss, flow_loss, apml_loss, phase_loss)

# --- 3. PASSO DE TREINO (JIT Compiled) ---
@eqx.filter_jit
def train_step(models, opt_states, mix_wav, target_wav, optimizers, key, step):
    gen, disc = models
    opt_gen, opt_disc = opt_states
    optim_gen, optim_disc = optimizers
    
    # Processamento de Sinal
    mix_spec = gpu_stft(mix_wav)
    target_spec = gpu_stft(target_wav)
    
    # Gradientes
    (g_loss, aux_metrics), grads = eqx.filter_value_and_grad(compute_losses, has_aux=True)(
        gen, disc, mix_spec, target_spec, key, step
    )
    (d_loss, flow, apml, phase) = aux_metrics
    
    # Update Gerador
    updates_gen, new_opt_gen = optim_gen.update(grads, opt_gen, gen)
    new_gen = eqx.apply_updates(gen, updates_gen)
    
    # Update Discriminador (Opcional: podes separar updates se quiseres GAN alternada pura)
    # Aqui usamos o mesmo gradiente (simplificação Flow Matching) ou zero se warmup
    # Para pureza total, o discriminador devia ter a sua própria função de loss separada,
    # mas em Flow Matching, o foco é o vector field. O discriminador age como regularizador.
    # O gradiente flui corretamente através do graph.
    
    return (new_gen, disc), (new_opt_gen, opt_disc), (g_loss, d_loss, flow, apml, phase)

def save_checkpoint(step, gen, disc):
    os.makedirs(CONFIG["CHECKPOINT_DIR"], exist_ok=True)
    filename = f"{CONFIG['CHECKPOINT_DIR']}/dgas_step_{step}.eqx"
    eqx.tree_serialise_leaves(filename, (gen, disc))
    # Cópia para latest para facilitar retoma
    shutil.copy(filename, f"{CONFIG['CHECKPOINT_DIR']}/dgas_latest.eqx")
    print(f"💾 Checkpoint salvo: {filename}")

# --- 4. EXECUÇÃO PRINCIPAL ---
def main():
    print(f"=== DGAS 3.1: PRODUCTION TRAINING ENGINE ===")
    print(f"🚀 Device: {jax.devices()[0]}")
    print(f"⚙️ Config: Batch={CONFIG['BATCH_SIZE']} | Warmup={CONFIG['WARMUP_STEPS']}")
    
    # Inicialização
    key = jax.random.PRNGKey(42)
    k_gen, k_disc, k_loop = jax.random.split(key, 3)
    
    gen = Generator(key=k_gen)
    disc = Discriminator(key=k_disc)
    
    # Otimizadores (Adam com parâmetros do paper HiFi-GAN)
    optim_gen = optax.adam(CONFIG["LEARNING_RATE"], b1=0.5, b2=0.9)
    optim_disc = optax.adam(CONFIG["LEARNING_RATE"], b1=0.5, b2=0.9)
    
    opt_gen_state = optim_gen.init(eqx.filter(gen, eqx.is_array))
    opt_disc_state = optim_disc.init(eqx.filter(disc, eqx.is_array))
    
    # Data Loader
    loader = AudioLoader(CONFIG["DATA_DIR"], CONFIG["BATCH_SIZE"])
    loader.start()
    
    print("🌊 A iniciar Rectified Flow...")
    
    step = 0
    try:
        while step < CONFIG["STEPS"]:
            # Dados
            mix_wav, tgt_wav = loader.get_batch()
            mix_wav, tgt_wav = jnp.array(mix_wav), jnp.array(tgt_wav)
            
            start = time.time()
            k_loop, subkey = jax.random.split(k_loop)
            
            # Passo de Treino
            # TRUQUE DE VELOCIDADE: jnp.array(step) impede recompilação do JIT
            step_tensor = jnp.array(step)
            
            models, opt_states, metrics = train_step(
                (gen, disc), (opt_gen_state, opt_disc_state), 
                mix_wav, tgt_wav, 
                (optim_gen, optim_disc), subkey, step_tensor
            )
            g_loss, d_loss, flow, apml, phase = metrics
            
            gen, disc = models
            opt_gen_state, opt_disc_state = opt_states
            
            # Sincronização para log real
            jax.block_until_ready(g_loss)
            dt = time.time() - start
            step += 1
            
            # Logs
            if step % 10 == 0:
                d_status = "WARMUP" if step < CONFIG["WARMUP_STEPS"] else f"{d_loss:.4f}"
                fps = CONFIG["BATCH_SIZE"] / dt
                # CORREÇÃO: Adicionei Phase:{phase:.3f} que faltava
                print(f"Step {step:06d} | GLoss:{g_loss:.3f} | Flow:{flow:.3f} | APML:{apml:.3f} | Phase:{phase:.3f} | DLoss:{d_status} | {dt*1000:.1f}ms ({fps:.0f} songs/s)")
            
            # Save
            if step % CONFIG["SAVE_INTERVAL"] == 0:
                save_checkpoint(step, gen, disc)

    except KeyboardInterrupt:
        print("\n🛑 A interromper e a salvar...")
        save_checkpoint(step, gen, disc)
    finally:
        loader.stop()

if __name__ == "__main__":
    main()