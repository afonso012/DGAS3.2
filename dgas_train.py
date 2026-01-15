import jax
import jax.numpy as jnp
import equinox as eqx
import optax
import numpy as np
import time
from typing import Tuple

# Importar Componentes do Projeto
from dgas_model import DGASField
from dgas_data import AudioLoader  # <--- NOVA IMPORTAÇÃO

# --- CONFIGURAÇÃO E HIPERPARÂMETROS ---
CONFIG = {
    "DATA_DIR": "./dataset_audio", # <--- CRIA ESTA PASTA E PÕE LÁ MÚSICA
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
    "PHASE_WEIGHT": 1.0,
}

# --- 1. COMPONENTES AUXILIARES (ENCODER & DISCRIMINATOR) ---

class LatentEncoder(eqx.Module):
    """Hypernetwork Encoder (Roadmap 3.2)."""
    layers: list
    final_proj: eqx.nn.Linear

    def __init__(self, key, input_channels=2, latent_dim=128):
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
        for layer in self.layers:
            if hasattr(layer, '__call__'):
                x = layer(x)
            else:
                x = layer(x)
        x = jnp.squeeze(x)
        return self.final_proj(x)

class MultiPeriodDiscriminator(eqx.Module):
    """Discriminador para o Refinamento Adversarial."""
    layers: list

    def __init__(self, key):
        keys = jax.random.split(key, 4)
        self.layers = [
            eqx.nn.Conv2d(2, 16, kernel_size=(3, 3), stride=(2, 2), key=keys[0]),
            jax.nn.leaky_relu,
            eqx.nn.Conv2d(16, 32, kernel_size=(3, 3), stride=(2, 2), key=keys[1]),
            jax.nn.leaky_relu,
            eqx.nn.Conv2d(32, 1, kernel_size=(3, 3), stride=(1, 1), key=keys[2])
        ]

    def __call__(self, x):
        for layer in self.layers:
            if hasattr(layer, '__call__'):
                x = layer(x)
            else:
                x = layer(x)
        return jnp.mean(x)

# --- 2. SISTEMA DE PERDAS (LOSS FUNCTIONS) ---

def get_apml_mask(n_freq_bins, sample_rate):
    freqs = jnp.linspace(0, sample_rate / 2, n_freq_bins)
    sensibilidade = 1.0 + 9.0 * jnp.exp(-0.5 * ((freqs - 3000) / 1000)**2)
    mask = sensibilidade[None, :, None] 
    return mask

def loss_fn_generator(generator_model, encoder_model, discriminator_model, batch_mix, batch_target, key, step_num):
    B, T, F, C = batch_mix.shape
    
    # Encoder
    batch_mix_transposed = jnp.transpose(batch_mix, (0, 3, 1, 2))
    mix_flat = batch_mix_transposed.reshape(B, C, -1)
    z_batch = jax.vmap(encoder_model)(mix_flat)

    # Rectified Flow Matching
    t_flow = jax.random.uniform(key, (B, 1, 1, 1))
    x_t = (1 - t_flow) * batch_mix + t_flow * batch_target 
    target_velocity = batch_target - batch_mix
    
    # Inferência Real (Vetorizada)
    t_space = jnp.linspace(0, 1, T)
    f_space = jnp.linspace(0, 1, F)
    
    def apply_model_point(t, f, z):
        return generator_model(t, f, z)

    predicted_velocity = jax.vmap(
        jax.vmap(jax.vmap(apply_model_point, in_axes=(None, 0, None)), in_axes=(0, None, None)),
        in_axes=(None, None, 0)
    )(t_space, f_space, z_batch)

    loss_flow = jnp.mean((predicted_velocity - target_velocity) ** 2)

    # APML
    apml_mask = get_apml_mask(F, CONFIG["SAMPLE_RATE"])
    error = (predicted_velocity - target_velocity)
    loss_apml = jnp.mean((error ** 2) * apml_mask)

    # Adversarial (Curriculum)
    progress = jnp.clip((step_num - CONFIG["WARMUP_STEPS"]) / CONFIG["TOTAL_STEPS"], 0.0, 1.0)
    lambda_adv = progress * CONFIG["GAN_WEIGHT_MAX"]
    
    def compute_adv_loss(_):
        prediction_reconstructed = batch_mix + predicted_velocity
        pred_transposed = jnp.transpose(prediction_reconstructed, (0, 3, 1, 2))
        fake_score = jax.vmap(discriminator_model)(pred_transposed)
        return jnp.mean((fake_score - 1.0) ** 2)

    def no_adv_loss(_):
        return 0.0

    loss_adv = jax.lax.cond(lambda_adv > 0, compute_adv_loss, no_adv_loss, operand=None)
    total_loss = loss_flow + (CONFIG["APML_WEIGHT"] * loss_apml) + (lambda_adv * loss_adv)
    
    return total_loss, (loss_flow, loss_apml, loss_adv)

def loss_fn_discriminator(discriminator_model, generator_model, encoder_model, batch_mix, batch_target, key):
    B, T, F, C = batch_mix.shape
    
    # Encoder
    batch_mix_transposed = jnp.transpose(batch_mix, (0, 3, 1, 2))
    mix_flat = batch_mix_transposed.reshape(B, C, -1)
    z_batch = jax.vmap(encoder_model)(mix_flat)
    
    # Mock para poupar VRAM no passo do discriminador
    predicted_velocity = batch_target - batch_mix 
    fake_audio = batch_mix + predicted_velocity
    
    target_transposed = jnp.transpose(batch_target, (0, 3, 1, 2))
    real_score = jax.vmap(discriminator_model)(target_transposed)
    
    fake_transposed = jnp.transpose(fake_audio, (0, 3, 1, 2))
    fake_score = jax.vmap(discriminator_model)(fake_transposed)
    
    loss_d = 0.5 * (jnp.mean((real_score - 1.0) ** 2) + jnp.mean((fake_score - 0.0) ** 2))
    return loss_d

# --- 3. PASSO DE TREINO (UPDATE STEP) ---

@eqx.filter_jit
def train_step(models, opt_states, batch_mix, batch_target, key, step_num, optimizers):
    gen_model, enc_model, disc_model = models
    opt_g_state, opt_d_state = opt_states
    opt_g, opt_d = optimizers
    key_gen, key_disc = jax.random.split(key)

    # Update G + E
    def combined_loss(params_g_combo, d_model, b_mix, b_tgt, k, s):
        g_model, e_model = params_g_combo
        return loss_fn_generator(g_model, e_model, d_model, b_mix, b_tgt, k, s)

    params_g = (gen_model, enc_model)
    (g_loss, aux_losses), g_grads = eqx.filter_value_and_grad(combined_loss, has_aux=True)(
        params_g, disc_model, batch_mix, batch_target, key_gen, step_num
    )
    
    updates_g, new_opt_g_state = opt_g.update(g_grads, opt_g_state, params_g)
    gen_model, enc_model = eqx.apply_updates(params_g, updates_g)

    # Update D
    is_warmup = step_num < CONFIG["WARMUP_STEPS"]
    d_diff, d_static = eqx.partition(disc_model, eqx.is_array)
    
    def update_disc_wrapper(d_diff_inner, d_state_inner):
        d_model_inner = eqx.combine(d_diff_inner, d_static)
        d_loss, d_grads = eqx.filter_value_and_grad(loss_fn_discriminator)(
            d_model_inner, gen_model, enc_model, batch_mix, batch_target, key_disc
        )
        updates_d, new_d_state = opt_d.update(d_grads, d_state_inner, d_model_inner)
        new_d_model = eqx.apply_updates(d_model_inner, updates_d)
        new_d_diff, _ = eqx.partition(new_d_model, eqx.is_array)
        return new_d_diff, new_d_state, d_loss

    def skip_disc_wrapper(d_diff_inner, d_state_inner):
        return d_diff_inner, d_state_inner, 0.0

    new_d_diff, new_opt_d_state, d_loss = jax.lax.cond(
        is_warmup, skip_disc_wrapper, update_disc_wrapper, d_diff, opt_d_state
    )
    new_disc_model = eqx.combine(new_d_diff, d_static)

    return (gen_model, enc_model, new_disc_model), (new_opt_g_state, new_opt_d_state), g_loss, d_loss, aux_losses

# --- 4. LOOP PRINCIPAL ---

def main():
    print("=== DGAS 3.2 TRAINING PIPELINE (REAL DATA MODE) ===")
    print(f"Configuration: {CONFIG}")
    
    # 1. Inicializar Data Loader
    loader = AudioLoader(
        data_dir=CONFIG["DATA_DIR"], 
        batch_size=CONFIG["BATCH_SIZE"]
    )
    loader.start()
    
    try:
        # 2. Inicializar Modelos
        key = jax.random.PRNGKey(42)
        k_gen, k_enc, k_disc = jax.random.split(key, 3)
        
        generator = DGASField(k_gen)
        encoder = LatentEncoder(k_enc)
        discriminator = MultiPeriodDiscriminator(k_disc)
        models = (generator, encoder, discriminator)
        
        # 3. Otimizadores
        scheduler = optax.warmup_cosine_decay_schedule(
            init_value=0.0, 
            peak_value=CONFIG["LEARNING_RATE"], 
            warmup_steps=CONFIG["WARMUP_STEPS"], 
            decay_steps=CONFIG["TOTAL_STEPS"]
        )
        optim_g = optax.chain(optax.clip_by_global_norm(1.0), optax.adamw(learning_rate=scheduler))
        optim_d = optax.adamw(learning_rate=2e-4)
        
        params_g = (generator, encoder)
        opt_g_state = optim_g.init(eqx.filter(params_g, eqx.is_array))
        opt_d_state = optim_d.init(eqx.filter(discriminator, eqx.is_array))
        
        optimizers = (optim_g, optim_d)
        opt_states = (opt_g_state, opt_d_state)
        
        print("Waiting for data buffer...")
        # Pré-aquecer o buffer de dados
        time.sleep(2) 
        
        print(f"System Initialized. Starting Training on {CONFIG['DATA_DIR']}...")
        
        # 4. Training Loop
        for step in range(CONFIG["TOTAL_STEPS"]):
            key, subkey = jax.random.split(key)
            
            # --- CARREGAR DADOS REAIS ---
            batch_mix, batch_target = loader.get_batch()
            
            models, opt_states, g_loss, d_loss, aux = train_step(
                models, opt_states, batch_mix, batch_target, subkey, step, optimizers
            )
            
            if step % 10 == 0:
                l_flow, l_apml, l_adv = aux
                status = "WARMUP" if step < CONFIG["WARMUP_STEPS"] else "HYBRID"
                print(f"Step {step:05d} [{status}] | G_Loss: {g_loss:.4f} (Flow: {l_flow:.4f}, APML: {l_apml:.4f}) | D_Loss: {d_loss:.4f}")

            if step == CONFIG["WARMUP_STEPS"]:
                print(f"\n>>> PHASE 2 ACTIVATED: ADVERSARIAL REFINEMENT ENABLED (Step {step})\n")
                
            # Opcional: Salvar Checkpoint a cada 1000 passos
            if step > 0 and step % 1000 == 0:
                eqx.tree_serialise_leaves(f"dgas_checkpoint_{step}.eqx", models)
                print(f"Checkpoint saved: dgas_checkpoint_{step}.eqx")

    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")
    finally:
        print("Shutting down Data Loader...")
        loader.stop()
        print("Done.")

if __name__ == "__main__":
    main()