import jax
import jax.numpy as jnp
import optax
import equinox as eqx
import time
import os
import shutil
from dgas_model import Generator, Discriminator
from dgas_data import AudioLoader

# === DGAS 3.5: CYBERNETIC ADA ENGINE ===
# Innovation: PID-Controlled Adaptive Discriminator Augmentation
# "Never Freeze, Just Adapt."

CONFIG = {
    "DATA_DIR": "/workspace/musdb18hq/train",
    "CHECKPOINT_DIR": "checkpoints",
    "SAVE_INTERVAL": 1000,
    "BATCH_SIZE": 16,     
    "STEPS": 1000000,
    "WARMUP_STEPS": 2000, 
    "N_FFT": 2048, "HOP_LENGTH": 512,
    
    # Pesos
    "LAMBDA_FLOW": 5.0,
    "LAMBDA_APML": 10.0,
    "LAMBDA_PHASE": 1.0,
    "LAMBDA_ADV": 0.5,
    "LAMBDA_R1": 10.0,   
    "R1_INTERVAL": 16,    
    "TARGET_RATIO": 0.6   
}

# --- 1. UTILS & SIGNAL PROCESSING ---
def get_fletcher_munson_weights(n_fft, sr=44100):
    freqs = jnp.linspace(0, sr/2, n_fft // 2 + 1)
    f_k = freqs / 1000.0 + 1e-6
    weights = (f_k**2 * 12200**2) / ((f_k**2 + 20.6**2) * (f_k**2 + 12200**2) * jnp.sqrt((f_k**2 + 107.7**2) * (f_k**2 + 737.9**2)))
    weights = weights + 2.0 * jnp.exp(-((f_k - 3.5)**2) / 2.0)
    return (weights / jnp.mean(weights))[None, None, :, None]

APML_WEIGHTS = get_fletcher_munson_weights(CONFIG["N_FFT"])

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
    """
    Aplica SpecAugment diferenciável.
    Strength (0.0 a 1.0) controla a probabilidade e intensidade.
    """
    B, C, F, T = x.shape
    
    # 1. Frequency Masking
    # Cria uma máscara aleatória suave
    k1, k2, k3 = jax.random.split(key, 3)
    
    # Probabilidade de aplicar aug baseada na strength
    do_aug = jax.random.uniform(k1, (B, 1, 1, 1)) < strength
    
    # Freq Mask
    f_width = int(F * 0.2) # Max 20% da banda
    f_pos = jax.random.randint(k2, (B, 1, 1, 1), 0, F - f_width)
    f_mask = jnp.ones((B, 1, F, 1))
    
    # Usamos uma lógica simples de meshgrid para criar a mascara vetorialmente
    # Nota: Implementação simplificada para JAX JIT
    freq_grid = jnp.arange(F)[None, None, :, None]
    mask_vals = (freq_grid >= f_pos) & (freq_grid < (f_pos + f_width * strength)) # Scaled width
    f_mask = jnp.where(mask_vals, 0.0, 1.0)
    
    # Time Mask
    t_width = int(T * 0.2)
    t_pos = jax.random.randint(k3, (B, 1, 1, 1), 0, T - t_width)
    time_grid = jnp.arange(T)[None, None, None, :]
    mask_vals_t = (time_grid >= t_pos) & (time_grid < (t_pos + t_width * strength))
    t_mask = jnp.where(mask_vals_t, 0.0, 1.0)
    
    # Aplica máscara apenas se do_aug for true, senão identidade
    # x é complex-like (4 channels), aplicamos a mesma máscara a tudo
    x_aug = x * f_mask * t_mask
    
    return jnp.where(do_aug, x_aug, x)

# --- 3. LOSSES (COM ADA) ---

def compute_disc_loss(discriminator, generator, mix_spec, target_spec, key, do_r1, aug_strength):
    z = jax.vmap(generator.encoder)(mix_spec)
    B, _, F, T = target_spec.shape
    
    # Gerar Fake
    t = jax.random.uniform(key, (B,), minval=0.0, maxval=1.0)
    key, k_noise, k_aug = jax.random.split(key, 3)
    x0 = jax.random.normal(k_noise, target_spec.shape)
    t_b = t[:, None, None, None]
    x_t = t_b * target_spec + (1.0 - t_b) * x0
    
    # Predict
    freqs, times = jnp.linspace(0, 1, F), jnp.linspace(0, 1, T)
    grid_f, grid_t = jnp.meshgrid(freqs, times, indexing='ij')
    ff, tt = grid_f.flatten(), grid_t.flatten()
    def predict_batch(ti, xti, zi):
        v = jax.vmap(lambda f, t: generator.field(ti, jnp.array([t, f]), zi))(ff, tt)
        return v.T.reshape(4, F, T)
    v_pred = jax.vmap(predict_batch)(t, x_t, z)
    
    # === APLICAÇÃO DA ADA (Adaptive Augmentation) ===
    # Distorcer TANTO o real COMO o fake antes de o Discriminador ver
    # Isto torna a tarefa do D mais difícil sem estragar o treino do G
    target_aug = diff_spec_augment(target_spec, k_aug, aug_strength)
    fake_aug = diff_spec_augment(v_pred, k_aug, aug_strength)
    
    # Scores (usando as versões distorcidas)
    real_scores = jax.vmap(discriminator)(target_aug, mix_spec)
    fake_scores = jax.vmap(discriminator)(fake_aug, mix_spec)
    
    # R1 Penalty (no audio limpo original, para estabilidade do gradiente real)
    def compute_r1_penalty():
        def single_disc_score(x, c): return jnp.squeeze(discriminator(x, c)) 
        grads = jax.vmap(jax.grad(single_disc_score), in_axes=(0, 0))(target_spec, mix_spec)
        grads_flat = grads.reshape(B, -1)
        penalty = jnp.mean(jnp.sum(grads_flat ** 2, axis=1))
        return penalty * CONFIG["LAMBDA_R1"] * 0.5

    r1_penalty = jax.lax.cond(do_r1, compute_r1_penalty, lambda: 0.0)

    # Hinge Loss
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
    def predict_batch(ti, xti, zi):
        v = jax.vmap(lambda f, t: generator.field(ti, jnp.array([t, f]), zi))(ff, tt)
        return v.T.reshape(4, F, T)
    v_pred = jax.vmap(predict_batch)(t, x_t, z)
    
    # Losses Físicas
    flow_loss = jnp.mean((v_pred - v_target)**2)
    mag_pred = jnp.sqrt(v_pred[:, 0]**2 + v_pred[:, 1]**2 + 1e-6)
    mag_target = jnp.sqrt(v_target[:, 0]**2 + v_target[:, 1]**2 + 1e-6)
    apml_loss = jnp.mean(jnp.abs(mag_pred - mag_target) * APML_WEIGHTS)
    dot = v_pred * v_target
    norm = jnp.abs(v_pred) * jnp.abs(v_target) + 1e-6
    phase_loss = jnp.mean(1.0 - dot / norm)
    
    # Adversarial Loss (Gerador tem de enganar o Discriminador AUGMENTADO)
    # Se D estiver "cego" pela ADA, G tem de produzir estruturas muito robustas para ser detetado como real
    fake_aug = diff_spec_augment(v_pred, k_aug, aug_strength)
    fake_score = jax.vmap(discriminator)(fake_aug, mix_spec)
    adv_loss = -jnp.mean(fake_score)
    
    adv_loss_weighted = jax.lax.cond(step >= CONFIG["WARMUP_STEPS"], lambda: adv_loss, lambda: 0.0)
    
    total = (CONFIG["LAMBDA_FLOW"] * flow_loss + 
             CONFIG["LAMBDA_APML"] * apml_loss + 
             CONFIG["LAMBDA_PHASE"] * phase_loss +
             CONFIG["LAMBDA_ADV"] * adv_loss_weighted)
             
    return total, (flow_loss, apml_loss, phase_loss, adv_loss)

# --- 4. STEP DE TREINO (PID CONTROLS AUGMENTATION) ---
@eqx.filter_jit
def train_step(gen, disc, opt_gen, opt_disc, optim_gen, optim_disc, mix_wav, target_wav, key, step, pid_state):
    mix_spec = gpu_stft(mix_wav)
    target_spec = gpu_stft(target_wav)
    k1, k2, k_aug = jax.random.split(key, 3)
    do_r1 = (step % CONFIG["R1_INTERVAL"] == 0)
    
    # 1. Recuperar Estado do PID
    pid_int, aug_strength = pid_state
    
    # 2. Update Discriminador
    def update_disc(d, g, o_state):
        (loss, clean_d_loss), grads = eqx.filter_value_and_grad(compute_disc_loss, has_aux=True)(d, g, mix_spec, target_spec, k1, do_r1, aug_strength)
        updates, new_state = optim_disc.update(grads, o_state, d)
        new_d = eqx.apply_updates(d, updates)
        return new_d, new_state, clean_d_loss

    # Update Generator
    (g_loss, aux), grads_g = eqx.filter_value_and_grad(compute_gen_loss, has_aux=True)(gen, disc, mix_spec, target_spec, k2, step, aug_strength)
    updates_g, new_opt_gen = optim_gen.update(grads_g, opt_gen, gen)
    new_gen = eqx.apply_updates(gen, updates_g)
    
    # 3. PID LOGIC: CONTROLAR AUG_STRENGTH
    # Objetivo: Manter d_loss em torno de TARGET_RATIO * g_adv_loss, mas simplificando:
    # ADA Target: Queremos que o Discriminador acerte ~0.6 (log(2) para GAN ideal, mas hinge é diferente).
    # Se d_loss < 0.2, está demasiado fácil -> Aumentar Augmentation
    # Se d_loss > 0.8, está muito difícil -> Diminuir Augmentation
    
    flow, apml, phase, g_adv_loss = aux
    
    # Loss de referência (calculada ou recuperada do update)
    # Executamos o update condicional primeiro
    new_disc, new_opt_disc, final_d_loss = jax.lax.cond(
        step >= CONFIG["WARMUP_STEPS"],
        lambda: update_disc(disc, gen, opt_disc),
        lambda: (disc, opt_disc, 1.0)
    )
    
    # Heurística ADA da NVIDIA:
    # Se Overfitting/TooStrong (Loss baixa), aumentar P.
    # Setpoint: Loss ~ 0.2 a 0.5 é saudável. Abaixo disso é perigo.
    # Erro = (Target_Loss - Current_Loss)
    # Se Loss=0.0, Erro positivo -> Aumentar P.
    
    target_d_loss = 0.4 
    error = target_d_loss - final_d_loss
    
    # === CORREÇÃO TURBO ===
    # Aumentar drasticamente a velocidade de reação (Gain x10)
    # Se D_Loss=0, error=0.4 -> new_int sobe 0.04 por step. Em 20 steps temos ADA forte.
    Kp = 0.2  # Proporcional (Reação Imediata)
    Ki = 0.05 # Integral (Acumulação)
    
    # Termo Proporcional ajuda a reagir AGORA ao colapso
    p_term = error * Kp 
    
    # Integrador
    new_int = jnp.clip(pid_int + error * Ki, -2.0, 5.0)
    
    # Augmentation Strength = Base Integral + Reação Imediata
    raw_aug = (new_int * 0.2) + p_term
    
    new_aug_strength = jnp.clip(raw_aug, 0.0, 0.8) # Max 80%
    
    # Se estiver em warmup, força aug = 0
    new_aug_strength = jax.lax.cond(step < CONFIG["WARMUP_STEPS"], lambda: 0.0, lambda: new_aug_strength)
    new_pid_state = (new_int, new_aug_strength)
    
    return new_gen, new_disc, new_opt_gen, new_opt_disc, g_loss, final_d_loss, aux, new_pid_state

# --- MAIN LOOP ---
def main():
    print(f"=== DGAS 3.5: CYBERNETIC ADA ENGINE ===")
    
    if os.path.exists(CONFIG['CHECKPOINT_DIR']):
        print("🧹 FRESH START...")
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
    
    # Estado PID: (Integral_Accumulator, Current_Aug_Strength)
    pid_state = (0.0, 0.0)
    
    loader = AudioLoader(CONFIG["DATA_DIR"], CONFIG["BATCH_SIZE"])
    loader.start()
    
    step = 0
    try:
        # --- O LOOP WHILE TEM DE ENVOLVER O TREINO ---
        while step < CONFIG["STEPS"]:
            mix, tgt = loader.get_batch()
            mix, tgt = jnp.array(mix), jnp.array(tgt)
            k_loop, subkey = jax.random.split(k_loop)
            
            start = time.time()
            
            # Passo de treino real
            gen, disc, opt_gen_state, opt_disc_state, g_loss, d_loss, aux, pid_state = train_step(
                gen, disc, opt_gen_state, opt_disc_state, optim_gen, optim_disc,
                mix, tgt, subkey, jnp.array(step), pid_state
            )
            
            jax.block_until_ready(g_loss)
            dt = time.time() - start
            step += 1
            
            # Logs a cada 10 steps
            if step % 10 == 0:
                flow, apml, phase, adv = aux
                print(f"S{step:05d} | GL:{g_loss:.3f} | DL:{d_loss:.3f} | F:{flow:.3f} | A:{apml:.3f} | ADA:{pid_state[1]:.3f} | {dt*1000:.0f}ms")
                
            # Salvamento Atómico a cada SAVE_INTERVAL
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