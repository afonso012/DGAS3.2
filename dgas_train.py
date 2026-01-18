import jax
import jax.numpy as jnp
import equinox as eqx
import optax
import numpy as np
import time
from typing import Tuple
from dgas_model import DGASField
from dgas_data import AudioLoader

CONFIG = {
    "DATA_DIR": "/Volumes/DSAG DRIVE/raw_data/musdb18hq/train", # Caminho corrigido
    "SAMPLE_RATE": 44100,
    "N_FFT": 2048,
    "HOP_LENGTH": 512,
    "LATENT_DIM": 128,
    "BATCH_SIZE": 4,
    "LEARNING_RATE": 3e-4,
    "WARMUP_STEPS": 1000,
    "TOTAL_STEPS": 100000,
    "GAN_WEIGHT_MAX": 0.1,
    "APML_WEIGHT": 10.0,
    "PHASE_WEIGHT": 0.5,
}

class LatentEncoder(eqx.Module):
    layers: list
    final_proj: eqx.nn.Linear

    # CORREÇÃO: input_channels=4 para aceitar Stereo Complexo (L_Re, L_Im, R_Re, R_Im)
    def __init__(self, key, input_channels=4, latent_dim=128):
        keys = jax.random.split(key, 5)
        self.layers = [
            eqx.nn.Conv1d(input_channels, 32, kernel_size=3, stride=2, key=keys[0]),
            eqx.nn.GroupNorm(8, 32),
            jax.nn.gelu,
            eqx.nn.Conv1d(32, 64, kernel_size=3, stride=2, key=keys[1]),
            eqx.nn.GroupNorm(8, 64),
            jax.nn.gelu,
            eqx.nn.Conv1d(64, 128, kernel_size=3, stride=2, key=keys[2]),
            jax.nn.gelu,
            eqx.nn.AdaptiveAvgPool1d(1)
        ]
        self.final_proj = eqx.nn.Linear(128, latent_dim, key=keys[3])

    def __call__(self, x):
        for layer in self.layers: x = layer(x)
        return self.final_proj(jnp.squeeze(x))

class MultiPeriodDiscriminator(eqx.Module):
    layers: list
    def __init__(self, key):
        keys = jax.random.split(key, 4)
        # Input channels=4 (Stereo)
        self.layers = [
            eqx.nn.Conv2d(4, 16, kernel_size=(3, 3), stride=(2, 2), key=keys[0]),
            jax.nn.leaky_relu,
            eqx.nn.Conv2d(16, 32, kernel_size=(3, 3), stride=(2, 2), key=keys[1]),
            jax.nn.leaky_relu,
            eqx.nn.Conv2d(32, 1, kernel_size=(3, 3), stride=(1, 1), key=keys[2])
        ]
    def __call__(self, x):
        for layer in self.layers: x = layer(x)
        return jnp.mean(x)

def get_apml_mask(n_freq_bins, sample_rate):
    freqs = jnp.linspace(0, sample_rate / 2, n_freq_bins)
    sensibilidade = 1.0 + 9.0 * jnp.exp(-0.5 * ((freqs - 3000) / 1000)**2)
    return sensibilidade[None, :, None]

def loss_fn_generator(generator_model, encoder_model, discriminator_model, batch_mix, batch_target, key, step_num):
    B, T, F, C = batch_mix.shape # C será 4 agora
    
    batch_mix_transposed = jnp.transpose(batch_mix, (0, 3, 1, 2))
    mix_flat = batch_mix_transposed.reshape(B, C, -1)
    z_batch = jax.vmap(encoder_model)(mix_flat)
    
    key_noise, key_flow = jax.random.split(key)
    z_noise = jax.random.normal(key_noise, z_batch.shape) * 0.1
    z_robust = z_batch + z_noise
    
    t_flow = jax.random.uniform(key_flow, (B, 1, 1, 1))
    x_t = (1 - t_flow) * batch_mix + t_flow * batch_target 
    target_velocity = batch_target - batch_mix
    
    t_space = jnp.linspace(0, 1, T)
    f_space = jnp.linspace(0, 1, F)
    
    def apply_model_point(t, f, z): return generator_model(t, f, z)

    predicted_velocity = jax.vmap(
        jax.vmap(jax.vmap(apply_model_point, in_axes=(None, 0, None)), in_axes=(0, None, None)),
        in_axes=(None, None, 0)
    )(t_space, f_space, z_robust)

    loss_flow = jnp.mean((predicted_velocity - target_velocity) ** 2)
    
    apml_mask = get_apml_mask(F, CONFIG["SAMPLE_RATE"])
    error = (predicted_velocity - target_velocity)
    loss_apml = jnp.mean((error ** 2) * apml_mask)

    # Phase Consistency Loss (Stereo Compatible)
    def get_phase_grad(spec):
        # spec: (Batch, Time, Freq, 4) -> [L_Re, L_Im, R_Re, R_Im]
        # Converter para complexo: (Batch, Time, Freq, 2) [Left_C, Right_C]
        left = spec[..., 0] + 1j * spec[..., 1]
        right = spec[..., 2] + 1j * spec[..., 3]
        angle_l = jnp.angle(left)
        angle_r = jnp.angle(right)
        # Stack para calcular gradiente
        angle = jnp.stack([angle_l, angle_r], axis=-1)
        return angle[:, 1:, :, :] - angle[:, :-1, :, :]

    pred_reconstructed = batch_mix + predicted_velocity
    grad_pred = get_phase_grad(pred_reconstructed)
    grad_target = get_phase_grad(batch_target)
    loss_phase = jnp.mean(jnp.abs(grad_pred - grad_target))

    progress = jnp.clip((step_num - CONFIG["WARMUP_STEPS"]) / CONFIG["TOTAL_STEPS"], 0.0, 1.0)
    lambda_adv = progress * CONFIG["GAN_WEIGHT_MAX"]
    
    def compute_adv_loss(_):
        pred_transposed = jnp.transpose(pred_reconstructed, (0, 3, 1, 2))
        fake_score = jax.vmap(discriminator_model)(pred_transposed)
        return jnp.mean((fake_score - 1.0) ** 2)
    def no_adv_loss(_): return 0.0

    loss_adv = jax.lax.cond(lambda_adv > 0, compute_adv_loss, no_adv_loss, operand=None)
    
    total_loss = loss_flow + (CONFIG["APML_WEIGHT"] * loss_apml) + (lambda_adv * loss_adv) + (CONFIG["PHASE_WEIGHT"] * loss_phase)
    return total_loss, (loss_flow, loss_apml, loss_adv, loss_phase)

def loss_fn_discriminator(disc_model, gen_model, enc_model, batch_mix, batch_target, key):
    B, T, F, C = batch_mix.shape
    mix_flat = jnp.transpose(batch_mix, (0, 3, 1, 2)).reshape(B, C, -1)
    z_batch = jax.vmap(enc_model)(mix_flat)
    
    predicted_velocity = batch_target - batch_mix 
    fake_audio = batch_mix + predicted_velocity
    
    real_score = jax.vmap(disc_model)(jnp.transpose(batch_target, (0, 3, 1, 2)))
    fake_score = jax.vmap(disc_model)(jnp.transpose(fake_audio, (0, 3, 1, 2)))
    return 0.5 * (jnp.mean((real_score - 1.0) ** 2) + jnp.mean((fake_score - 0.0) ** 2))

@eqx.filter_jit
def train_step(models, opt_states, batch_mix, batch_target, key, step_num, optimizers):
    gen, enc, disc = models
    opt_g, opt_d = optimizers
    key_gen, key_disc = jax.random.split(key)

    def combined_loss(params_g_combo, d_model, b_mix, b_tgt, k, s):
        g, e = params_g_combo
        return loss_fn_generator(g, e, d_model, b_mix, b_tgt, k, s)

    params_g = (gen, enc)
    (g_loss, aux), g_grads = eqx.filter_value_and_grad(combined_loss, has_aux=True)(
        params_g, disc, batch_mix, batch_target, key_gen, step_num
    )
    updates_g, new_opt_g = opt_g.update(g_grads, opt_states[0], params_g)
    gen, enc = eqx.apply_updates(params_g, updates_g)

    is_warmup = step_num < CONFIG["WARMUP_STEPS"]
    d_diff, d_static = eqx.partition(disc, eqx.is_array)
    
    def update_disc(d_diff, d_state):
        d_mod = eqx.combine(d_diff, d_static)
        d_loss, d_grads = eqx.filter_value_and_grad(loss_fn_discriminator)(
            d_mod, gen, enc, batch_mix, batch_target, key_disc
        )
        updates_d, new_d_state = opt_d.update(d_grads, d_state, d_mod)
        new_d_mod = eqx.apply_updates(d_mod, updates_d)
        new_d_diff, _ = eqx.partition(new_d_mod, eqx.is_array)
        return new_d_diff, new_d_state, d_loss

    def skip_disc(d_diff, d_state): return d_diff, d_state, 0.0

    new_d_diff, new_opt_d, d_loss = jax.lax.cond(is_warmup, skip_disc, update_disc, d_diff, opt_states[1])
    disc = eqx.combine(new_d_diff, d_static)
    
    return (gen, enc, disc), (new_opt_g, new_opt_d), g_loss, d_loss, aux

def main():
    print("=== DGAS 3.2: RESUMING TRAINING (SAVING MEMORY MODE) ===")
    
    # 1. Loader com Batch Size reduzido (definido na CONFIG)
    loader = AudioLoader(CONFIG["DATA_DIR"], CONFIG["BATCH_SIZE"])
    loader.start()
    
    try:
        key = jax.random.PRNGKey(42)
        k1, k2, k3 = jax.random.split(key, 3)
        
        # Inicializar a estrutura do modelo
        models = (DGASField(k1), LatentEncoder(k2, input_channels=4), MultiPeriodDiscriminator(k3))
        
        # --- NOVO: CARREGAR O CHECKPOINT 5000 ---
        try:
            print("Attempting to load checkpoint: dgas_stereo_step_20000.eqx ...")
            models = eqx.tree_deserialise_leaves("dgas_stereo_step_20000.eqx", models)
            print(">>> SUCESSO! Pesos carregados do Step 20000.")
            start_step = 20001
        except FileNotFoundError:
            print(">>> Checkpoint não encontrado. A começar do ZERO.")
            start_step = 0
        # ----------------------------------------

        sched = optax.warmup_cosine_decay_schedule(0.0, CONFIG["LEARNING_RATE"], CONFIG["WARMUP_STEPS"], CONFIG["TOTAL_STEPS"])
        opt_g = optax.chain(optax.clip_by_global_norm(1.0), optax.adamw(learning_rate=sched))
        opt_d = optax.adamw(learning_rate=2e-4)
        
        opt_states = (
            opt_g.init(eqx.filter((models[0], models[1]), eqx.is_array)),
            opt_d.init(eqx.filter(models[2], eqx.is_array))
        )
        
        print("Pipeline Ready. Waiting for data...")
        time.sleep(3)
        print(f"RESUMING TRAINING FROM STEP {start_step}...")
        
        for step in range(start_step, CONFIG["TOTAL_STEPS"]):
            key, subkey = jax.random.split(key)
            batch_mix, batch_target = loader.get_batch()
            
            models, opt_states, g_loss, d_loss, aux = train_step(
                models, opt_states, batch_mix, batch_target, subkey, step, (opt_g, opt_d)
            )
            
            if step % 10 == 0:
                 # Desempacotar as auxiliares para ver o log completo
                l_flow, l_apml, l_adv, l_phase = aux
                status = "WARMUP" if step < CONFIG["WARMUP_STEPS"] else "HYBRID"
                
                print(f"Step {step:05d} [{status}] | "
                      f"G_Loss: {g_loss:.4f} "
                      f"(Flow: {l_flow:.4f}, APML: {l_apml:.4f}, Phase: {l_phase:.4f}) | "
                      f"D_Loss: {d_loss:.4f}")
            
            if step > 0 and step % 1000 == 0:
                # Salvar novo checkpoint
                eqx.tree_serialise_leaves(f"dgas_stereo_step_{step}.eqx", models)
                print(f"Checkpoint saved: dgas_stereo_step_{step}.eqx")

    except KeyboardInterrupt: print("Interrupted.")
    finally: loader.stop()

if __name__ == "__main__": main()