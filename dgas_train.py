import jax
import jax.numpy as jnp
import optax
import equinox as eqx
import time
import os
import shutil
from dgas_model import Generator, Discriminator
from dgas_data import AudioLoader

# --- CONFIGURAÇÕES "PRODUCTION GRADE" ---
CONFIG = {
    "DATA_DIR": "/workspace/musdb18hq/train",
    "CHECKPOINT_DIR": "checkpoints",
    "SAVE_INTERVAL": 1000,
    "BATCH_SIZE": 16,       
    "STEPS": 1000000,
    "WARMUP_STEPS": 5000,
    
    # Geometria Sincronizada (128 frames)
    "CHUNK_SIZE": 65536, 
    "N_FFT": 2048, 
    "HOP_LENGTH": 512,
    
    # Hiperparâmetros de Loss Refinados
    "LAMBDA_FLOW": 100.0,
    "LAMBDA_MRSTFT": 1.0, 
    "LAMBDA_WAV": 10.0,    # NOVA: Waveform Loss (Phase Lock)
    "LAMBDA_ADV": 1.0,    
    "LAMBDA_R1": 10.0,
    "R1_INTERVAL": 16,
    
    # EMA (Estabilidade de Inferência)
    "EMA_DECAY": 0.999,
}

# --- 1. ENGENHARIA DE COORDENADAS LOGARÍTMICAS ---
def get_log_coords(B, F, T):
    times = jnp.linspace(0, 1, T)
    # Frequência Logarítmica Suave (Foca nos médios/graves)
    freqs_log = jnp.logspace(jnp.log10(1e-3), jnp.log10(1.0), F)
    freqs_log = (freqs_log - 1e-3) / (1.0 - 1e-3)
    grid_f, grid_t = jnp.meshgrid(freqs_log, times, indexing='ij')
    return grid_f.flatten(), grid_t.flatten()

# --- 2. DIFERENCIÁVEIS (ISTFT & Augment) ---
def diff_istft(spec):
    """Reconstrução diferenciável para cálculo de loss no tempo."""
    l_re, l_im = spec[:, 0], spec[:, 1]
    r_re, r_im = spec[:, 2], spec[:, 3]
    Z_l = l_re + 1j * l_im
    Z_r = r_re + 1j * r_im
    
    # Expansão Log-Inversa (Assumindo que a rede prevê espaço log-comprimido)
    def expand(z):
        mag = jnp.abs(z)
        phase = jnp.angle(z)
        mag_linear = jnp.expm1(mag * 10.0) / 1000.0 # Inverso aproximado da compressão
        return mag_linear * jnp.exp(1j * phase)

    Z_l = expand(Z_l)
    Z_r = expand(Z_r)

    window = jnp.hanning(CONFIG["N_FFT"])
    def single_istft(z):
        return jax.scipy.signal.istft(z, fs=44100, window=window, 
                                     nperseg=CONFIG["N_FFT"], 
                                     noverlap=CONFIG["N_FFT"] - CONFIG["HOP_LENGTH"])[1]
    
    wav_l = jax.vmap(single_istft)(Z_l)
    wav_r = jax.vmap(single_istft)(Z_r)
    wav_l = wav_l[:, :CONFIG["CHUNK_SIZE"]]
    wav_r = wav_r[:, :CONFIG["CHUNK_SIZE"]]
    return jnp.stack([wav_l, wav_r], axis=1)

def diff_spec_augment(x, key, strength):
    B, C, F, T = x.shape
    k1, k2 = jax.random.split(key)
    do_aug = jax.random.uniform(k1, (B, 1, 1, 1)) < strength
    f_width = int(F * 0.15) 
    f_pos = jax.random.randint(k2, (B, 1, 1, 1), 0, max(1, F - f_width))
    freq_grid = jnp.arange(F)[None, None, :, None]
    f_mask = jnp.where((freq_grid >= f_pos) & (freq_grid < (f_pos + f_width * strength)), 0.0, 1.0)
    return jnp.where(do_aug, x * f_mask, x)

# --- 3. LOSS FUNCTIONS ---
def stft_mag_loss(wav_pred, wav_target, n_fft, hop):
    window = jnp.hanning(n_fft)
    def get_mag(w):
        B, C, T = w.shape
        w_flat = w.reshape(B*C, T)
        _, _, Z = jax.scipy.signal.stft(w_flat, fs=44100, window=window, nperseg=n_fft, noverlap=n_fft-hop)
        return jnp.abs(Z)
    mag_p = get_mag(wav_pred)
    mag_t = get_mag(wav_target)
    sc_loss = jnp.mean(jnp.abs(mag_p - mag_t)) / (jnp.mean(jnp.abs(mag_t)) + 1e-6)
    log_loss = jnp.mean(jnp.abs(jnp.log1p(mag_p) - jnp.log1p(mag_t)))
    return sc_loss + log_loss

def compute_losses_sota(spec_pred, spec_target, wav_target):
    # 1. Reconstrução Temporal
    wav_pred = diff_istft(spec_pred)
    
    # 2. MR-STFT Loss (Spectral)
    l1 = stft_mag_loss(wav_pred, wav_target, 2048, 512)
    l2 = stft_mag_loss(wav_pred, wav_target, 1024, 256)
    l3 = stft_mag_loss(wav_pred, wav_target, 512, 128)
    mrstft = l1 + l2 + l3
    
    # 3. Waveform Loss (Phase Locking) - NOVA
    # Diferença direta átomo a átomo. Penaliza erros de fase minúsculos.
    wav_loss = jnp.mean(jnp.abs(wav_pred - wav_target))
    
    return mrstft, wav_loss

# --- 4. STEP FUNCTIONS (TRAIN & DISC) ---

def compute_disc_loss(discriminator, generator, mix_spec, target_spec, key, do_r1, aug_strength):
    # ... (Igual à versão anterior: LogCoords + Heun Step simulado) ...
    z = jax.vmap(generator.encoder)(mix_spec)
    B, _, F, T = target_spec.shape
    ff, tt = get_log_coords(B, F, T)
    
    key, k_noise, k_aug = jax.random.split(key, 3)
    x0 = jax.random.normal(k_noise, target_spec.shape)
    
    # Heun Step Simulado (2 passos para fake mais limpo)
    def predict_velocity(ti, xi):
        xi_flat = xi.reshape(4, -1).T
        v = jax.vmap(lambda f, t_val, x_val: generator.field(ti, jnp.array([t_val, f]), x_val, z))(ff, tt, xi_flat)
        return v.T.reshape(4, F, T)

    v0 = predict_velocity(jnp.zeros((B,)), x0)
    x_mid = x0 + 0.5 * v0
    v_mid = predict_velocity(jnp.ones((B,)) * 0.5, x_mid)
    x1_pred = x0 + v_mid 
    
    target_aug = diff_spec_augment(target_spec, k_aug, aug_strength)
    fake_aug = diff_spec_augment(x1_pred, k_aug, aug_strength)
    
    real_scores = jax.vmap(discriminator)(target_aug, mix_spec)
    fake_scores = jax.vmap(discriminator)(fake_aug, mix_spec)
    
    d_loss = jnp.mean(jax.nn.relu(1.0 - real_scores)) + jnp.mean(jax.nn.relu(1.0 + fake_scores))
    
    def compute_r1():
        grads = jax.vmap(jax.grad(lambda x, c: jnp.squeeze(discriminator(x, c))), in_axes=(0, 0))(target_spec, mix_spec)
        return jnp.mean(jnp.sum(grads.reshape(B, -1)**2, axis=1)) * CONFIG["LAMBDA_R1"] * 0.5
    
    r1 = jax.lax.cond(do_r1, compute_r1, lambda: 0.0)
    return d_loss + r1, (d_loss, r1)

def compute_gen_loss(generator, discriminator, mix_spec, target_spec, wav_target, key, step, aug_strength):
    z = jax.vmap(generator.encoder)(mix_spec)
    B, _, F, T = target_spec.shape
    ff, tt = get_log_coords(B, F, T)
    
    t = jax.random.uniform(key, (B,), minval=0.0, maxval=1.0)
    key, k_noise, k_aug = jax.random.split(key, 3)
    x0 = jax.random.normal(key, target_spec.shape)
    
    t_b = t[:, None, None, None]
    x_t = t_b * target_spec + (1.0 - t_b) * x0
    v_target = target_spec - x0
    
    xt_flat = x_t.reshape(B, 4, -1).transpose(0, 2, 1)
    
    def predict_batch_sample(ti, xti_flat, zi):
        v = jax.vmap(lambda f, t_val, x_val: generator.field(ti, jnp.array([t_val, f]), x_val, zi))(ff, tt, xti_flat)
        return v
        
    v_pred_flat = jax.vmap(predict_batch_sample)(t, xt_flat, z)
    v_pred = v_pred_flat.transpose(0, 2, 1).reshape(B, 4, F, T)
    
    flow_loss = jnp.mean((v_pred - v_target)**2)
    
    # Estimativa Final
    x1_est = x_t + (1.0 - t_b) * v_pred
    
    # SOTA LOSSES (MR-STFT + Waveform L1)
    mrstft_loss, wav_l1_loss = compute_losses_sota(x1_est, target_spec, wav_target)
    
    fake_aug = diff_spec_augment(x1_est, k_aug, aug_strength)
    fake_score = jax.vmap(discriminator)(fake_aug, mix_spec)
    adv_loss = -jnp.mean(fake_score)
    
    adv_weight = jax.lax.cond(step < CONFIG["WARMUP_STEPS"], lambda: 0.0, lambda: CONFIG["LAMBDA_ADV"])
    
    total = (CONFIG["LAMBDA_FLOW"] * flow_loss + 
             CONFIG["LAMBDA_MRSTFT"] * mrstft_loss + 
             CONFIG["LAMBDA_WAV"] * wav_l1_loss +  # Waveform weight
             adv_weight * adv_loss)
             
    return total, (flow_loss, mrstft_loss, wav_l1_loss, adv_loss)

# --- 5. LOOP PRINCIPAL COM EMA ---

@eqx.filter_jit
def train_step(gen, gen_ema, disc, opt_gen, opt_disc, optim_gen, optim_disc, mix_wav, target_wav, key, step, pid_state):
    # STFT Log-Aware (Input Prep)
    def gpu_stft_log(audio):
        window = jnp.hanning(CONFIG["N_FFT"])
        f, t, Zxx = jax.scipy.signal.stft(audio, fs=44100, window=window, nperseg=CONFIG["N_FFT"], noverlap=CONFIG["N_FFT"] - CONFIG["HOP_LENGTH"])
        mag = jnp.abs(Zxx)
        phase = jnp.angle(Zxx)
        mag = jnp.log1p(mag * 1000.0) * 0.1
        spec = jnp.stack([mag * jnp.cos(phase), mag * jnp.sin(phase)], axis=-1)
        B, C, F, T, _ = spec.shape
        return jnp.transpose(spec, (0, 1, 4, 2, 3)).reshape(B, C * 2, F, T)

    mix_spec = gpu_stft_log(mix_wav)
    target_spec = gpu_stft_log(target_wav)
    
    k1, k2, k_aug = jax.random.split(key, 3)
    do_r1 = (step % CONFIG["R1_INTERVAL"] == 0)
    pid_int, aug_strength = pid_state
    
    # Update Disc
    def update_disc(d, g, o_state):
        (loss, (clean_d, r1)), grads = eqx.filter_value_and_grad(compute_disc_loss, has_aux=True)(
            d, g, mix_spec, target_spec, k1, do_r1, aug_strength
        )
        updates, new_state = optim_disc.update(grads, o_state, d)
        new_d = eqx.apply_updates(d, updates)
        return new_d, new_state, clean_d, r1
        
    # Update Gen
    (g_loss, (flow, mrs, wav_l1, adv)), grads_g = eqx.filter_value_and_grad(compute_gen_loss, has_aux=True)(
        gen, disc, mix_spec, target_spec, target_wav, k2, step, aug_strength
    )
    
    updates_g, new_opt_gen = optim_gen.update(grads_g, opt_gen, gen)
    new_gen = eqx.apply_updates(gen, updates_g)
    
    # --- EMA UPDATE (O Segredo da Estabilidade) ---
    # gen_ema = decay * gen_ema + (1-decay) * new_gen
    # Utilizamos tree_map para aplicar a todos os pesos
    new_gen_ema = jax.tree_map(
        lambda e, n: CONFIG["EMA_DECAY"] * e + (1.0 - CONFIG["EMA_DECAY"]) * n,
        gen_ema, new_gen
    )
    
    new_disc, new_opt_disc, d_loss, r1_val = jax.lax.cond(
        step >= CONFIG["WARMUP_STEPS"],
        lambda: update_disc(disc, gen, opt_disc),
        lambda: (disc, opt_disc, 1.0, 0.0)
    )
    
    target_d_loss = 0.6
    error = target_d_loss - d_loss
    new_int = jnp.clip(pid_int + error * 0.01, -2.0, 5.0)
    new_aug = jnp.clip((new_int * 0.1) + (error * 0.1), 0.0, 0.8)
    new_aug = jax.lax.cond(step < CONFIG["WARMUP_STEPS"], lambda: 0.0, lambda: new_aug)
    
    return new_gen, new_gen_ema, new_disc, new_opt_gen, new_opt_disc, g_loss, d_loss, r1_val, (flow, mrs, wav_l1, adv), (new_int, new_aug)

def main():
    print(f"=== DGAS FINAL PRODUCTION: EMA & Phase-Lock Loss ===")
    
    if os.path.exists(CONFIG['CHECKPOINT_DIR']): shutil.rmtree(CONFIG['CHECKPOINT_DIR'])
    os.makedirs(CONFIG['CHECKPOINT_DIR'], exist_ok=True)
    
    key = jax.random.PRNGKey(42)
    k_gen, k_disc, k_loop = jax.random.split(key, 3)
    
    gen = Generator(key=k_gen)
    gen_ema = gen # Inicializa EMA com a mesma cópia
    disc = Discriminator(key=k_disc)
    
    lr = optax.warmup_cosine_decay_schedule(1e-6, 2e-4, 5000, CONFIG["STEPS"], 1e-6)
    optim = optax.chain(optax.clip_by_global_norm(1.0), optax.adamw(lr, b1=0.8, b2=0.99))
    
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
            
            gen, gen_ema, disc, opt_gen_state, opt_disc_state, g_loss, d_loss, r1, metrics, pid_state = train_step(
                gen, gen_ema, disc, opt_gen_state, opt_disc_state, optim, optim,
                mix, tgt, subkey, jnp.array(step), pid_state
            )
            jax.block_until_ready(g_loss)
            dt = time.time() - start
            step += 1
            
            if step % 10 == 0:
                flow, mrs, wav, adv = metrics
                print(f"S{step:05d} | G:{g_loss:.2f} D:{d_loss:.2f} | FL:{flow:.2f} WAV:{wav:.3f} | {dt*1000:.0f}ms")
            
            if step % CONFIG["SAVE_INTERVAL"] == 0:
                # SALVA O EMA PARA INFERÊNCIA (Qualidade Superior)
                eqx.tree_serialise_leaves(os.path.join(CONFIG["CHECKPOINT_DIR"], "dgas_sota_ema.eqx"), (gen_ema, disc))
                # Salva o training state também caso queiras retomar
                eqx.tree_serialise_leaves(os.path.join(CONFIG["CHECKPOINT_DIR"], "dgas_latest.eqx"), (gen, disc))
                print(f"💾 Checkpoint EMA Saved: {step}")
                
    except KeyboardInterrupt: pass
    finally: loader.stop()

if __name__ == "__main__":
    main()