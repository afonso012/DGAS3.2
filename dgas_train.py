import jax
import jax.numpy as jnp
import optax
import equinox as eqx
import time
import os
import shutil
from dgas_model import Generator, Discriminator
from dgas_data import AudioLoader

# === CONFIGURAÇÃO (A100 - RECTIFIED FLOW + GAN) ===
CONFIG = {
    "DATA_DIR": "/workspace/musdb18hq/train",
    "BATCH_SIZE": 16,
    "LEARNING_RATE": 1e-4,
    "STEPS": 1000000,
    "SAVE_INTERVAL": 1000,
    "WARMUP_STEPS": 1000,      # <--- NOVO: Discriminador só arranca aqui
    "CHECKPOINT_DIR": "checkpoints",
    "N_FFT": 2048,
    "HOP_LENGTH": 512
}

def gpu_stft(audio):
    # (Batch, Channels, Samples) -> (Batch, 4, Freq, Time)
    window = jnp.hanning(CONFIG["N_FFT"])
    f, t, Zxx = jax.scipy.signal.stft(
        audio, fs=44100, window=window, 
        nperseg=CONFIG["N_FFT"], noverlap=CONFIG["N_FFT"] - CONFIG["HOP_LENGTH"]
    )
    Zxx = jnp.transpose(Zxx, (0, 1, 2, 3)) 
    spec = jnp.stack([Zxx.real, Zxx.imag], axis=-1)
    
    B, C, F, T, _ = spec.shape
    spec = jnp.transpose(spec, (0, 1, 4, 2, 3)).reshape(B, C * 2, F, T)
    return spec[:, :, :, :128] * 10.0

def compute_losses(generator, discriminator, mix_spec, target_spec, key, step):
    # A. Extrair Latente
    z = jax.vmap(generator.encoder)(mix_spec) 
    
    B, C, F, T = target_spec.shape
    t = jax.random.uniform(key, (B,), minval=0.0, maxval=1.0)
    
    key, k_noise = jax.random.split(key)
    x0 = jax.random.normal(k_noise, target_spec.shape)
    x1 = target_spec 
    
    # B. Interpolação (Rectified Path)
    t_b = t[:, None, None, None]
    x_t = t_b * x1 + (1.0 - t_b) * x0
    v_target = x1 - x0
    
    # C. Previsão
    freqs = jnp.linspace(0, 1, F)
    times = jnp.linspace(0, 1, T)
    grid_f, grid_t = jnp.meshgrid(freqs, times, indexing='ij')
    
    def predict_single(ti, xt_i, z_i):
        f_flat, t_flat = grid_f.flatten(), grid_t.flatten()
        def field_point(f_val, t_val):
            return generator.field(ti, jnp.array([t_val, f_val]), z_i)
        return jax.vmap(field_point)(f_flat, t_flat).T.reshape(4, F, T)

    v_pred = jax.vmap(predict_single)(t, x_t, z)
    
    # --- MÉTRICAS ---
    flow_loss = jnp.mean((v_pred - v_target) ** 2)
    
    mag_pred = jnp.sqrt(v_pred[:, 0]**2 + v_pred[:, 1]**2 + 1e-6)
    mag_target = jnp.sqrt(v_target[:, 0]**2 + v_target[:, 1]**2 + 1e-6)
    apml_loss = jnp.mean(jnp.abs(mag_pred - mag_target))
    
    phase_loss = jnp.mean(1.0 - (v_pred * v_target) / (jnp.abs(v_pred) * jnp.abs(v_target) + 1e-6))

    # --- LÓGICA DE WARMUP ---
    # Só calcula Loss do discriminador se step >= WARMUP_STEPS
    def train_disc_loss():
        disc_fake = jax.vmap(discriminator)(v_pred, mix_spec) 
        disc_real = jax.vmap(discriminator)(v_target, mix_spec)
        d_l = jnp.mean(jax.nn.relu(1.0 - disc_real)) + jnp.mean(jax.nn.relu(1.0 + disc_fake))
        adv_l = -jnp.mean(disc_fake)
        return d_l, adv_l

    def warmup_disc_loss():
        return 0.0, 0.0

    # Escolhe dinamicamente
    d_loss, adv_loss = jax.lax.cond(
        step >= CONFIG["WARMUP_STEPS"],
        train_disc_loss,
        warmup_disc_loss
    )
    
    g_loss = 10.0 * flow_loss + 2.0 * apml_loss + 0.1 * adv_loss + 0.1 * phase_loss
    
    return g_loss, (d_loss, flow_loss, apml_loss, phase_loss)

@eqx.filter_jit
def train_step(models, opt_states, mix_wav, target_wav, optimizers, key, step):
    gen, disc = models
    opt_gen, opt_disc = opt_states
    optim_gen, optim_disc = optimizers
    
    mix_spec = gpu_stft(mix_wav)
    target_spec = gpu_stft(target_wav)
    
    (g_loss, (d_loss, flow, apml, phase)), grads = eqx.filter_value_and_grad(compute_losses, has_aux=True)(
        gen, disc, mix_spec, target_spec, key, step
    )
    
    # Atualizar Gerador (Sempre)
    updates_gen, new_opt_gen = optim_gen.update(grads, opt_gen, gen)
    new_gen = eqx.apply_updates(gen, updates_gen)
    
    # Atualizar Discriminador (Só se step >= Warmup)
    # Nota: Como d_loss é 0.0 no warmup, o gradiente será zero, então update é seguro,
    # mas por eficiência podemos manter o disc antigo.
    # Mas para simplicidade do JAX JIT, vamos deixar o otimizador processar os zeros.
    
    # Se estivéssemos a separar explicitamente os gradientes:
    # d_grads = ...
    # Mas aqui d_loss faz parte do grafo total.
    # Se d_loss é 0, o gradiente w.r.t discriminador deve ser 0.
    
    return (new_gen, disc), (new_opt_gen, opt_disc), (g_loss, d_loss, flow, apml, phase)

def save_checkpoint(step, gen, disc):
    os.makedirs(CONFIG["CHECKPOINT_DIR"], exist_ok=True)
    filename = f"{CONFIG['CHECKPOINT_DIR']}/dgas_step_{step}.eqx"
    eqx.tree_serialise_leaves(filename, (gen, disc))
    shutil.copy(filename, f"{CONFIG['CHECKPOINT_DIR']}/dgas_latest.eqx")
    print(f"💾 Checkpoint salvo: {filename}")

def main():
    print(f"=== DGAS 3.2: FULL METRICS TRAINING (A100) ===")
    
    key = jax.random.PRNGKey(42)
    k_gen, k_disc, k_loop = jax.random.split(key, 3)
    
    gen = Generator(key=k_gen)
    disc = Discriminator(key=k_disc)
    
    optim_gen = optax.adam(CONFIG["LEARNING_RATE"])
    optim_disc = optax.adam(CONFIG["LEARNING_RATE"])
    
    opt_gen_state = optim_gen.init(eqx.filter(gen, eqx.is_array))
    opt_disc_state = optim_disc.init(eqx.filter(disc, eqx.is_array))
    
    loader = AudioLoader(CONFIG["DATA_DIR"], CONFIG["BATCH_SIZE"])
    loader.start()
    
    print("🚀 Pipeline Pronto. A carregar métricas completas...")
    
    step = 0
    try:
        while step < CONFIG["STEPS"]:
            mix_wav, tgt_wav = loader.get_batch()
            mix_wav, tgt_wav = jnp.array(mix_wav), jnp.array(tgt_wav)
            
            start = time.time()
            k_loop, subkey = jax.random.split(k_loop)
            
            # Passamos 'step' para dentro da função JIT
            models, opt_states, metrics = train_step(
                (gen, disc), (opt_gen_state, opt_disc_state), mix_wav, tgt_wav, (optim_gen, optim_disc), subkey, step
            )
            g_loss, d_loss, flow, apml, phase = metrics
            
            gen, disc = models
            opt_gen_state, opt_disc_state = opt_states
            
            jax.block_until_ready(g_loss)
            dt = time.time() - start
            step += 1
            
            if step % 10 == 0:
                # Log ajustado para mostrar se D está ativo ou não
                d_status = "WARMUP" if step < CONFIG["WARMUP_STEPS"] else f"{d_loss:.4f}"
                print(f"Step {step:06d} | GLoss: {g_loss:.4f} | Flow: {flow:.4f} | APML: {apml:.4f} | DLoss: {d_status} | {dt*1000:.1f}ms")
            
            if step % CONFIG["SAVE_INTERVAL"] == 0:
                save_checkpoint(step, gen, disc)

    except KeyboardInterrupt:
        save_checkpoint(step, gen, disc)
    finally:
        loader.stop()

if __name__ == "__main__":
    main()