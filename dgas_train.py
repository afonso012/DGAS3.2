import jax
import jax.numpy as jnp
import optax
import equinox as eqx
import time
import os
import shutil
from dgas_model import Generator, Discriminator
from dgas_data import AudioLoader

# === DGAS 3.6: CYBERNETIC ADA ENGINE + MRSTFT + RE-COUPLING ===
# Innovation: PID-Controlled ADA, Re-coupled Flow Matching, MRSTFT Loss

CONFIG = {
    "DATA_DIR": "/workspace/musdb18hq/train",
    "CHECKPOINT_DIR": "checkpoints",
    "SAVE_INTERVAL": 1000,
    "BATCH_SIZE": 16,     
    "STEPS": 1000000,
    "WARMUP_STEPS": 5000, # Aumentado para garantir estabilidade física
    "N_FFT": 2048, "HOP_LENGTH": 512,
    
    # Pesos
    "LAMBDA_FLOW": 5.0,
    "LAMBDA_MRSTFT": 20.0, # SOTA: Substitui APML/Phase simples
    "LAMBDA_ADV": 1.0,     # Aumentado pois MRSTFT equilibra
    "LAMBDA_R1": 10.0,   
    "R1_INTERVAL": 16,    
    "TARGET_RATIO": 0.6   
}

# --- 1. SOTA: MULTI-RESOLUTION STFT LOSS ---
# Essencial para eliminar ruídos metálicos e artefatos de fase
def stft_loss_fn(pred_spec, target_spec, fft_size, hop_size):
    # Reconstrói áudio aproximado (ISTFT -> STFT) ou opera no espectro se compatível
    # Aqui aplicamos perda espectral direta nas magnitudes
    # Para simplicidade e velocidade em JAX, calculamos a perda na magnitude Log e Linear
    
    # Nota: Como o modelo opera no domínio STFT fixo (2048), a MRSTFT ideal requereria ISTFT->MultiSTFT.
    # Para evitar custo proibitivo de ISTFT no loop, usamos uma aproximação multi-escala no próprio espectrograma:
    # Average Pooling no espectrograma simula janelas maiores no tempo/frequência.
    
    diff = pred_spec - target_spec
    l1_loss = jnp.mean(jnp.abs(diff))
    
    # Log-Mag Loss
    mag_pred = jnp.abs(pred_spec) + 1e-7
    mag_target = jnp.abs(target_spec) + 1e-7
    log_loss = jnp.mean(jnp.abs(jnp.log(mag_pred) - jnp.log(mag_target)))
    
    return l1_loss + log_loss

def compute_mrstft_loss(pred, target):
    # Simulação eficiente de MRSTFT no domínio latente espectral
    # Escala 1: Full Res
    l1 = stft_loss_fn(pred, target, 2048, 512)
    
    # Escala 2: Tempo Baixo (Avg Pool Time)
    p_t = jax.nn.avg_pool(pred, (1, 2), strides=(1, 2), padding='SAME')
    t_t = jax.nn.avg_pool(target, (1, 2), strides=(1, 2), padding='SAME')
    l2 = stft_loss_fn(p_t, t_t, 0, 0)
    
    # Escala 3: Frequência Baixa (Avg Pool Freq)
    p_f = jax.nn.avg_pool(pred, (2, 1), strides=(2, 1), padding='SAME')
    t_f = jax.nn.avg_pool(target, (2, 1), strides=(2, 1), padding='SAME')
    l3 = stft_loss_fn(p_f, t_f, 0, 0)
    
    return l1 + l2 + l3

def gpu_stft(audio):
    window = jnp.hanning(CONFIG["N_FFT"])
    f, t, Zxx = jax.scipy.signal.stft(audio, fs=44100, window=window, nperseg=CONFIG["N_FFT"], noverlap=CONFIG["N_FFT"] - CONFIG["HOP_LENGTH"])
    Zxx = jnp.transpose(Zxx, (0, 1, 2, 3))
    spec = jnp.stack([Zxx.real, Zxx.imag], axis=-1)
    B, C, F, T, _ = spec.shape
    spec = jnp.transpose(spec, (0, 1, 4, 2, 3)).reshape(B, C * 2, F, T)
    return spec[:, :, :, :128] * 10.0

# --- 2. DIFFERENTIABLE AUGMENTATION (ADA) ---
def diff_spec_augment(x, key, strength):
    B, C, F, T = x.shape
    k1, k2, k3 = jax.random.split(key, 3)
    do_aug = jax.random.uniform(k1, (B, 1, 1, 1)) < strength
    f_width = int(F * 0.2) 
    f_pos = jax.random.randint(k2, (B, 1, 1, 1), 0, F - f_width)
    freq_grid = jnp.arange(F)[None, None, :, None]
    mask_vals = (freq_grid >= f_pos) & (freq_grid < (f_pos + f_width * strength)) 
    f_mask = jnp.where(mask_vals, 0.0, 1.0)
    
    t_width = int(T * 0.2)
    t_pos = jax.random.randint(k3, (B, 1, 1, 1), 0, T - t_width)
    time_grid = jnp.arange(T)[None, None, None, :]
    mask_vals_t = (time_grid >= t_pos) & (time_grid < (t_pos + t_width * strength))
    t_mask = jnp.where(mask_vals_t, 0.0, 1.0)
    x_aug = x * f_mask * t_mask
    return jnp.where(do_aug, x_aug, x)

# --- 3. LOSSES (COM ADA & RE-COUPLING) ---

def compute_disc_loss(discriminator, generator, mix_spec, target_spec, key, do_r1, aug_strength):
    z = jax.vmap(generator.encoder)(mix_spec)
    B, _, F, T = target_spec.shape
    
    t = jax.random.uniform(key, (B,), minval=0.0, maxval=1.0)
    key, k_noise, k_aug = jax.random.split(key, 3)
    x0 = jax.random.normal(k_noise, target_spec.shape)
    t_b = t[:, None, None, None]
    x_t = t_b * target_spec + (1.0 - t_b) * x0
    
    freqs, times = jnp.linspace(0, 1, F), jnp.linspace(0, 1, T)
    grid_f, grid_t = jnp.meshgrid(freqs, times, indexing='ij')
    ff, tt = grid_f.flatten(), grid_t.flatten()
    
    # SOTA: Re-acoplamento no Generator (Passamos x_t_flat)
    def predict_batch(ti, xti, zi):
        xti_flat = xti.reshape(4, -1).T # (F*T, 4)
        # O gerador agora recebe o valor do sinal xti_flat
        v = jax.vmap(lambda f, t, x_val: generator.field(ti, jnp.array([t, f]), x_val, zi))(ff, tt, xti_flat)
        return v.T.reshape(4, F, T)
    v_pred = jax.vmap(predict_batch)(t, x_t, z)
    
    target_aug = diff_spec_augment(target_spec, k_aug, aug_strength)
    fake_aug = diff_spec_augment(v_pred, k_aug, aug_strength)
    
    real_scores = jax.vmap(discriminator)(target_aug, mix_spec)
    fake_scores = jax.vmap(discriminator)(fake_aug, mix_spec)
    
    def compute_r1_penalty():
        def single_disc_score(x, c): return jnp.squeeze(discriminator(x, c)) 
        grads = jax.vmap(jax.grad(single_disc_score), in_axes=(0, 0))(target_spec, mix_spec)
        grads_flat = grads.reshape(B, -1)
        penalty = jnp.mean(jnp.sum(grads_flat ** 2, axis=1))
        return penalty * CONFIG["LAMBDA_R1"] * 0.5

    r1_penalty = jax.lax.cond(do_r1, compute_r1_penalty, lambda: 0.0)
    d_loss = jnp.mean(jax.nn.relu(1.0 - real_scores)) + jnp.mean(jax.nn.relu(1.0 + fake_scores))
    
    return d_loss + r1_penalty, d_loss

def compute_gen_loss(generator, discriminator, mix_spec, target_spec, key, step, aug_strength):
    z = jax.vmap(generator.encoder)(mix_spec)
    B, _, F, T = target_spec.shape
    t = jax.random.uniform(key, (B,), minval=0.0, maxval=1.0)
    key, k_noise, k_aug = jax.random.split(key, 3)
    x0 = jax.random.normal(key, target_spec.shape)
    t_b = t[:, None, None, None]
    x_t = t_b * target_spec + (1.0 - t_b) * x0
    v_target = target_spec - x0
    
    freqs, times = jnp.linspace(0, 1, F), jnp.linspace(0, 1, T)
    grid_f, grid_t = jnp.meshgrid(freqs, times, indexing='ij')
    ff, tt = grid_f.flatten(), grid_t.flatten()
    
    # SOTA: Re-acoplamento no Generator
    def predict_batch(ti, xti, zi):
        xti_flat = xti.reshape(4, -1).T
        v = jax.vmap(lambda f, t, x_val: generator.field(ti, jnp.array([t, f]), x_val, zi))(ff, tt, xti_flat)
        return v.T.reshape(4, F, T)
    v_pred = jax.vmap(predict_batch)(t, x_t, z)
    
    # Losses Físicas
    flow_loss = jnp.mean((v_pred - v_target)**2)
    
    # SOTA: MRSTFT Loss (Substitui a loss simples de magnitude)
    mrstft_loss = compute_mrstft_loss(v_pred, v_target)
    
    # Adversarial Loss
    fake_aug = diff_spec_augment(v_pred, k_aug, aug_strength)
    fake_score = jax.vmap(discriminator)(fake_aug, mix_spec)
    adv_loss = -jnp.mean(fake_score)
    
    # SOTA: Warmup Schedule Rigoroso
    # Durante warmup, adversarial loss é ZERO para permitir que o modelo aprenda física primeiro
    is_warmup = step < CONFIG["WARMUP_STEPS"]
    adv_weight = jax.lax.cond(is_warmup, lambda: 0.0, lambda: CONFIG["LAMBDA_ADV"])
    
    total = (CONFIG["LAMBDA_FLOW"] * flow_loss + 
             CONFIG["LAMBDA_MRSTFT"] * mrstft_loss + 
             adv_weight * adv_loss)
             
    return total, (flow_loss, mrstft_loss, 0.0, adv_loss)

# --- 4. STEP DE TREINO ---
@eqx.filter_jit
def train_step(gen, disc, opt_gen, opt_disc, optim_gen, optim_disc, mix_wav, target_wav, key, step, pid_state):
    mix_spec = gpu_stft(mix_wav)
    target_spec = gpu_stft(target_wav)
    k1, k2, k_aug = jax.random.split(key, 3)
    do_r1 = (step % CONFIG["R1_INTERVAL"] == 0)
    pid_int, aug_strength = pid_state
    
    # Update Discriminador (Condicional ao Warmup)
    # SOTA: Disc não treina durante warmup do Generator
    def update_disc(d, g, o_state):
        (loss, clean_d_loss), grads = eqx.filter_value_and_grad(compute_disc_loss, has_aux=True)(d, g, mix_spec, target_spec, k1, do_r1, aug_strength)
        updates, new_state = optim_disc.update(grads, o_state, d)
        new_d = eqx.apply_updates(d, updates)
        return new_d, new_state, clean_d_loss

    # Update Generator
    (g_loss, aux), grads_g = eqx.filter_value_and_grad(compute_gen_loss, has_aux=True)(gen, disc, mix_spec, target_spec, k2, step, aug_strength)
    updates_g, new_opt_gen = optim_gen.update(grads_g, opt_gen, gen)
    new_gen = eqx.apply_updates(gen, updates_g)
    
    flow, mrstft, _, g_adv_loss = aux
    
    # Executa update do disc apenas após warmup
    new_disc, new_opt_disc, final_d_loss = jax.lax.cond(
        step >= CONFIG["WARMUP_STEPS"],
        lambda: update_disc(disc, gen, opt_disc),
        lambda: (disc, opt_disc, 1.0) # Dummy loss 1.0
    )
    
    # PID ADA Logic
    target_d_loss = 0.4 
    error = target_d_loss - final_d_loss
    Kp, Ki = 0.2, 0.05
    p_term = error * Kp 
    new_int = jnp.clip(pid_int + error * Ki, -2.0, 5.0)
    raw_aug = (new_int * 0.2) + p_term
    new_aug_strength = jnp.clip(raw_aug, 0.0, 0.8)
    new_aug_strength = jax.lax.cond(step < CONFIG["WARMUP_STEPS"], lambda: 0.0, lambda: new_aug_strength)
    new_pid_state = (new_int, new_aug_strength)
    
    return new_gen, new_disc, new_opt_gen, new_opt_disc, g_loss, final_d_loss, aux, new_pid_state

# --- MAIN LOOP ---
def main():
    print(f"=== DGAS 3.6: SOTA ENGINE ACTIVATED ===")
    
    if os.path.exists(CONFIG['CHECKPOINT_DIR']):
        shutil.rmtree(CONFIG['CHECKPOINT_DIR'])
    os.makedirs(CONFIG['CHECKPOINT_DIR'], exist_ok=True)

    print(f"🚀 Device: {jax.devices()[0]}")
    
    key = jax.random.PRNGKey(42)
    k_gen, k_disc, k_loop = jax.random.split(key, 3)
    
    gen = Generator(key=k_gen)
    disc = Discriminator(key=k_disc)
    
    lr_schedule = optax.warmup_cosine_decay_schedule(1e-5, 1e-4, 1000, CONFIG["STEPS"], 1e-6)
    
    optim_gen = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(lr_schedule, b1=0.5, b2=0.9))
    optim_disc = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(lr_schedule, b1=0.5, b2=0.9))
    
    opt_gen_state = optim_gen.init(eqx.filter(gen, eqx.is_array))
    opt_disc_state = optim_disc.init(eqx.filter(disc, eqx.is_array))
    
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
            
            gen, disc, opt_gen_state, opt_disc_state, g_loss, d_loss, aux, pid_state = train_step(
                gen, disc, opt_gen_state, opt_disc_state, optim_gen, optim_disc,
                mix, tgt, subkey, jnp.array(step), pid_state
            )
            
            jax.block_until_ready(g_loss)
            dt = time.time() - start
            step += 1
            
            if step % 10 == 0:
                flow, mrstft, _, adv = aux
                # Status update: MRSTFT substitui APML
                print(f"S{step:05d} | GL:{g_loss:.3f} | DL:{d_loss:.3f} | F:{flow:.3f} | MRS:{mrstft:.3f} | ADA:{pid_state[1]:.3f} | {dt*1000:.0f}ms")
                
            if step % CONFIG["SAVE_INTERVAL"] == 0:
                ckpt_path = os.path.join(CONFIG["CHECKPOINT_DIR"], "dgas_latest.eqx")
                temp_path = ckpt_path + ".tmp"
                eqx.tree_serialise_leaves(temp_path, (gen, disc))
                os.replace(temp_path, ckpt_path)
                if step % 5000 == 0:
                    shutil.copy(ckpt_path, os.path.join(CONFIG["CHECKPOINT_DIR"], f"dgas_step_{step:06d}.eqx"))
                print(f"💾 CHECKPOINT GUARDADO: Step {step}")

    except KeyboardInterrupt:
        print("Interrompido.")
    finally:
        loader.stop()

if __name__ == "__main__":
    main()