import jax
import jax.numpy as jnp
import optax
import equinox as eqx
import time
import os
import shutil
import sys
from dgas_model import Generator, Discriminator
from dgas_data import AudioLoader

# --- CONFIGURAÇÕES AJUSTADAS ---
CONFIG = {
    "DATA_DIR": "/workspace/musdb18hq/train",
    "CHECKPOINT_DIR": "checkpoints",
    "SAVE_INTERVAL": 1000,
    "BATCH_SIZE": 24,       
    "STEPS": 1000000,
    "WARMUP_STEPS": 5000, 
    "N_FFT": 2048, "HOP_LENGTH": 512,
    "LAMBDA_FLOW": 100.0,
    "LAMBDA_MRSTFT": 0.05, # Reduzido ligeiramente para equilibrar
    "LAMBDA_ADV": 0.05,   
    "LAMBDA_R1": 10.0,   
    "R1_INTERVAL": 16,    
}

def get_log_coords(F, T):
    times = jnp.linspace(0, 1, T)
    linear_freqs = jnp.linspace(0, 1, F)
    log_freqs = jnp.log1p(linear_freqs * 10.0) / jnp.log1p(10.0)
    return log_freqs, times

def complex_mag(spec):
    l_re, l_im = spec[:, 0], spec[:, 1]
    r_re, r_im = spec[:, 2], spec[:, 3]
    mag_l = jnp.sqrt(l_re**2 + l_im**2 + 1e-6)
    mag_r = jnp.sqrt(r_re**2 + r_im**2 + 1e-6)
    return jnp.stack([mag_l, mag_r], axis=1)

def stft_loss_fn(mag_pred, mag_target):
    sc_loss = jnp.mean(jnp.abs(mag_pred - mag_target)) / (jnp.mean(jnp.abs(mag_target)) + 1e-6)
    log_loss = jnp.mean(jnp.abs(jnp.log1p(mag_pred) - jnp.log1p(mag_target)))
    return sc_loss + log_loss

def safe_avg_pool(x, window_shape, strides):
    val = jax.lax.reduce_window(x, 0.0, jax.lax.add, window_shape, strides, 'SAME')
    ones_spatial = jnp.ones((1, 1, x.shape[2], x.shape[3]))
    count = jax.lax.reduce_window(ones_spatial, 0.0, jax.lax.add, window_shape, strides, 'SAME')
    return val / (count + 1e-6)

def compute_mrstft_loss(pred_complex, target_complex):
    m_pred = complex_mag(pred_complex)
    m_target = complex_mag(target_complex)
    loss_1 = stft_loss_fn(m_pred, m_target)
    p_t = safe_avg_pool(m_pred, (1, 1, 1, 4), (1, 1, 1, 2))
    t_t = safe_avg_pool(m_target, (1, 1, 1, 4), (1, 1, 1, 2))
    loss_2 = stft_loss_fn(p_t, t_t)
    p_f = safe_avg_pool(m_pred, (1, 1, 4, 1), (1, 1, 2, 1))
    t_f = safe_avg_pool(m_target, (1, 1, 4, 1), (1, 1, 2, 1))
    loss_3 = stft_loss_fn(p_f, t_f)
    return loss_1 + loss_2 + loss_3

def gpu_stft(audio):
    window = jnp.hanning(CONFIG["N_FFT"])
    f, t, Zxx = jax.scipy.signal.stft(audio, fs=44100, window=window, nperseg=CONFIG["N_FFT"], noverlap=CONFIG["N_FFT"] - CONFIG["HOP_LENGTH"])
    Zxx = jnp.transpose(Zxx, (0, 1, 2, 3))
    spec = jnp.stack([Zxx.real, Zxx.imag], axis=-1)
    B, C, F, T, _ = spec.shape
    spec = jnp.transpose(spec, (0, 1, 4, 2, 3)).reshape(B, C * 2, F, T)
    return spec * 5

def diff_spec_augment(x, key, strength):
    B, C, F, T = x.shape
    k1, k2, k3 = jax.random.split(key, 3)
    do_aug = jax.random.uniform(k1, (B, 1, 1, 1)) < strength
    f_width = int(F * 0.15) 
    f_pos = jax.random.randint(k2, (B, 1, 1, 1), 0, max(1, F - f_width))
    freq_grid = jnp.arange(F)[None, None, :, None]
    f_mask = jnp.where((freq_grid >= f_pos) & (freq_grid < (f_pos + f_width * strength)), 0.0, 1.0)
    return jnp.where(do_aug, x * f_mask, x)

def compute_disc_loss(discriminator, generator, mix_spec, target_spec, key, do_r1, aug_strength):
    z = jax.vmap(generator.encoder)(mix_spec)
    B, _, F, T = target_spec.shape
    t = jax.random.uniform(key, (B,), minval=0.0, maxval=1.0)
    key, k_noise, k_aug = jax.random.split(key, 3)
    x0 = jax.random.normal(k_noise, target_spec.shape)
    t_b = t[:, None, None, None]
    x_t = t_b * target_spec + (1.0 - t_b) * x0
    freqs, times = get_log_coords(F, T)
    grid_f, grid_t = jnp.meshgrid(freqs, times, indexing='ij')
    ff, tt = grid_f.flatten(), grid_t.flatten()
    
    def predict_batch_sample(ti, xti, zi):
        xti_flat = xti.reshape(4, -1).T
        # Correção no gerador: xti já está escalado, mas o modelo agora lida com isso
        v = jax.vmap(lambda f, t_val, x_val: generator.field(ti, jnp.array([t_val, f]), x_val, zi))(ff, tt, xti_flat)
        return v.T.reshape(4, F, T)
    
    v_pred = jax.vmap(predict_batch_sample)(t, x_t, z)
    
    # CORREÇÃO CRÍTICA: O Discriminador deve ver o X1 estimado, não o target puro ou v_pred
    # x1_pred = x_t + (1-t) * v_pred
    x1_pred = x_t + (1.0 - t_b) * v_pred
    
    target_aug = diff_spec_augment(target_spec, k_aug, aug_strength)
    fake_aug = diff_spec_augment(x1_pred, k_aug, aug_strength) # Discrimina o audio reconstruído
    
    real_scores = jax.vmap(discriminator)(target_aug, mix_spec)
    fake_scores = jax.vmap(discriminator)(fake_aug, mix_spec)
    
    d_loss = jnp.mean(jax.nn.relu(1.0 - real_scores)) + jnp.mean(jax.nn.relu(1.0 + fake_scores))
    
    def compute_r1():
        grads = jax.vmap(jax.grad(lambda x, c: jnp.squeeze(discriminator(x, c))), in_axes=(0, 0))(target_spec, mix_spec)
        return jnp.mean(jnp.sum(grads.reshape(B, -1)**2, axis=1)) * CONFIG["LAMBDA_R1"] * 0.5
    
    r1 = jax.lax.cond(do_r1, compute_r1, lambda: 0.0)
    return d_loss + r1, (d_loss, r1)

def compute_gen_loss(generator, discriminator, mix_spec, target_spec, key, step, aug_strength):
    z = jax.vmap(generator.encoder)(mix_spec)
    B, _, F, T = target_spec.shape
    t = jax.random.uniform(key, (B,), minval=0.0, maxval=1.0)
    key, k_noise, k_aug = jax.random.split(key, 3)
    x0 = jax.random.normal(key, target_spec.shape)
    t_b = t[:, None, None, None]
    
    x_t = t_b * target_spec + (1.0 - t_b) * x0
    v_target = target_spec - x0
    
    freqs, times = get_log_coords(F, T)
    grid_f, grid_t = jnp.meshgrid(freqs, times, indexing='ij')
    ff, tt = grid_f.flatten(), grid_t.flatten()
    
    def predict_batch_sample(ti, xti, zi):
        xti_flat = xti.reshape(4, -1).T
        v = jax.vmap(lambda f, t_val, x_val: generator.field(ti, jnp.array([t_val, f]), x_val, zi))(ff, tt, xti_flat)
        return v.T.reshape(4, F, T)
        
    v_pred = jax.vmap(predict_batch_sample)(t, x_t, z)
    
    # 1. Flow Loss (MSE no vetor) - Mantém a física correta
    flow_loss = jnp.mean((v_pred - v_target)**2)
    
    # 2. MRSTFT Loss - CORRIGIDA
    # Calculamos a perda no "Audio Limpo Estimado" (x1_pred) e não na velocidade
    # x1 = x_t + (1-t)v
    x1_pred = x_t + (1.0 - t_b) * v_pred
    mrstft_loss = compute_mrstft_loss(x1_pred, target_spec)
    
    # 3. Phase Loss (Cosseno)
    dot = jnp.sum(v_pred * v_target, axis=1)
    norm_p = jnp.linalg.norm(v_pred, axis=1) + 1e-6
    norm_t = jnp.linalg.norm(v_target, axis=1) + 1e-6
    raw_phase_loss = jnp.mean(1.0 - (dot / (norm_p * norm_t)))
    phase_weight = jnp.clip(step / 10000.0, 0.0, 1.0) * 1.0 
    
    # 4. Adversarial Loss (No audio reconstruido)
    fake_aug = diff_spec_augment(x1_pred, k_aug, aug_strength)
    fake_score = jax.vmap(discriminator)(fake_aug, mix_spec)
    adv_loss = -jnp.mean(fake_score)
    adv_weight = jax.lax.cond(step < CONFIG["WARMUP_STEPS"], lambda: 0.0, lambda: CONFIG["LAMBDA_ADV"])
    
    total = (CONFIG["LAMBDA_FLOW"] * flow_loss + CONFIG["LAMBDA_MRSTFT"] * mrstft_loss + phase_weight * raw_phase_loss + adv_weight * adv_loss)
    return total, (flow_loss, mrstft_loss, raw_phase_loss, adv_loss)

@eqx.filter_jit
def train_step(gen, disc, opt_gen, opt_disc, optim_gen, optim_disc, mix_wav, target_wav, key, step, pid_state):
    mix_spec = gpu_stft(mix_wav)
    target_spec = gpu_stft(target_wav)
    
    # Opcional: Debug print para garantir que o boost está lá
    # jax.debug.print("Target Max: {}", jnp.max(jnp.abs(target_spec)))

    k1, k2, k_aug = jax.random.split(key, 3)
    do_r1 = (step % CONFIG["R1_INTERVAL"] == 0)
    pid_int, aug_strength = pid_state
    
    def update_disc(d, g, o_state):
        (loss, (clean_d_loss, r1)), grads = eqx.filter_value_and_grad(compute_disc_loss, has_aux=True)(d, g, mix_spec, target_spec, k1, do_r1, aug_strength)
        updates, new_state = optim_disc.update(grads, o_state, d)
        new_d = eqx.apply_updates(d, updates)
        return new_d, new_state, clean_d_loss, r1
        
    (g_loss, gen_metrics), grads_g = eqx.filter_value_and_grad(compute_gen_loss, has_aux=True)(gen, disc, mix_spec, target_spec, k2, step, aug_strength)
    
    updates_g, new_opt_gen = optim_gen.update(grads_g, opt_gen, gen)
    new_gen = eqx.apply_updates(gen, updates_g)
    
    new_disc, new_opt_disc, d_loss, r1_val = jax.lax.cond(
        step >= CONFIG["WARMUP_STEPS"],
        lambda: update_disc(disc, gen, opt_disc),
        lambda: (disc, opt_disc, 1.0, 0.0)
    )
    
    target_d_loss = 0.4 
    error = target_d_loss - d_loss
    new_int = jnp.clip(pid_int + error * 0.05, -2.0, 5.0)
    new_aug = jnp.clip((new_int * 0.2) + (error * 0.2), 0.0, 0.8)
    new_aug = jax.lax.cond(step < CONFIG["WARMUP_STEPS"], lambda: 0.0, lambda: new_aug)
    
    return new_gen, new_disc, new_opt_gen, new_opt_disc, g_loss, d_loss, r1_val, gen_metrics, (new_int, new_aug)

def main():
    print(f"=== DGAS 3.6.8: FIXED LOSS & SCALED FIELD ===")
    
    if os.path.exists(CONFIG['CHECKPOINT_DIR']): 
        shutil.rmtree(CONFIG['CHECKPOINT_DIR'])
    os.makedirs(CONFIG['CHECKPOINT_DIR'], exist_ok=True)
    
    key = jax.random.PRNGKey(42)
    k_gen, k_disc, k_loop = jax.random.split(key, 3)
    gen = Generator(key=k_gen)
    disc = Discriminator(key=k_disc)
    
    # Cosine decay restart se for muito longo
    lr = optax.warmup_cosine_decay_schedule(1e-5, 3e-4, 2000, CONFIG["STEPS"], 1e-6)
    optim = optax.chain(optax.clip_by_global_norm(1.0), optax.adamw(lr, b1=0.5, b2=0.9, weight_decay=1e-4))
    
    opt_gen_state = optim.init(eqx.filter(gen, eqx.is_array))
    opt_disc_state = optim.init(eqx.filter(disc, eqx.is_array))
    pid_state = (0.0, 0.0)
    
    loader = AudioLoader(CONFIG["DATA_DIR"], CONFIG["BATCH_SIZE"])
    loader.start()
    step = 0
    try:
        while step < CONFIG["STEPS"]:
            mix, tgt = loader.get_batch()
            mix, tgt = jnp.array(mix), jnp.array(tgt)
            k_loop, subkey = jax.random.split(k_loop)
            
            start = time.time()
            gen, disc, opt_gen_state, opt_disc_state, g_loss, d_loss, r1, (flow, mrs, phase, adv), pid_state = train_step(
                gen, disc, opt_gen_state, opt_disc_state, optim, optim,
                mix, tgt, subkey, jnp.array(step), pid_state
            )
            jax.block_until_ready(g_loss)
            dt = time.time() - start
            step += 1
            
            if step % 10 == 0:
                print(f"S{step:05d} | G:{g_loss:.2f} D:{d_loss:.2f} | FL:{flow:.2f} MRS:{mrs:.2f} | {dt*1000:.0f}ms")
            if step % CONFIG["SAVE_INTERVAL"] == 0:
                eqx.tree_serialise_leaves(os.path.join(CONFIG["CHECKPOINT_DIR"], "dgas_latest.eqx"), (gen, disc))
                print(f"💾 Checkpoint: {step}")
    except KeyboardInterrupt: pass
    finally: loader.stop()

if __name__ == "__main__":
    main()