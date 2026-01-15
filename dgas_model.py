import jax
import jax.numpy as jnp
import equinox as eqx
from typing import Callable, List

# --- MÓDULO 1: MATEMÁTICA DE SUAVIZAÇÃO (C2 CONTINUITY) ---

def smoother_step(x):
    """
    Função de interpolação C2-Continuous (Ordem 5).
    """
    x = jnp.clip(x, 0.0, 1.0)
    return x * x * x * (x * (x * 6 - 15) + 10)

# --- MÓDULO 2: REPRESENTAÇÃO (HASH GRID) ---

class ContinuousHashGrid(eqx.Module):
    embeddings: jnp.ndarray
    resolution: int
    grid_size: int
    output_dim: int

    def __init__(self, key, resolution=128, grid_size=4096, output_dim=2):
        self.resolution = resolution
        self.grid_size = grid_size
        self.output_dim = output_dim
        self.embeddings = jax.random.uniform(key, (grid_size, output_dim), minval=-0.01, maxval=0.01)

    def __call__(self, x: jnp.ndarray):
        x_scaled = x * self.resolution
        x0 = jnp.floor(x_scaled).astype(jnp.int32)
        x1 = x0 + 1
        weights = smoother_step(x_scaled - x0)
        
        primes = jnp.array([1, 2654435761], dtype=jnp.uint32)
        
        def get_hash_index(coords):
            hashed = jnp.sum(coords * primes[:coords.shape[0]], axis=0) % self.grid_size
            return hashed

        idx00 = get_hash_index(x0)
        idx01 = get_hash_index(jnp.array([x0[0], x1[1]]))
        idx10 = get_hash_index(jnp.array([x1[0], x0[1]]))
        idx11 = get_hash_index(x1)

        val00 = self.embeddings[idx00]
        val01 = self.embeddings[idx01]
        val10 = self.embeddings[idx10]
        val11 = self.embeddings[idx11]

        h0 = val00 * (1 - weights[1]) + val01 * weights[1]
        h1 = val10 * (1 - weights[1]) + val11 * weights[1]
        final_val = h0 * (1 - weights[0]) + h1 * weights[0]
        
        return final_val

# --- MÓDULO 3: MODULAÇÃO (FiLM) ---

class FiLMLayer(eqx.Module):
    weight: jnp.ndarray
    bias: jnp.ndarray
    
    def __init__(self, key, in_features, out_features):
        w_key, b_key = jax.random.split(key)
        self.weight = jax.random.normal(w_key, (out_features, in_features)) * 0.02
        self.bias = jnp.zeros((out_features,))

    def __call__(self, x, gamma, beta):
        out = self.weight @ x + self.bias
        return (1.0 + gamma) * out + beta

# --- MÓDULO 4: O MOTOR NEURAL (SIREN BACKBONE) ---

class DGASField(eqx.Module):
    layers: List[FiLMLayer]
    grids: List[ContinuousHashGrid]
    final_layer: eqx.nn.Linear
    latent_to_film: eqx.nn.Linear
    num_layers: int
    hidden_dim: int

    def __init__(self, key, num_layers=4, hidden_dim=256, latent_dim=128):
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        
        keys = jax.random.split(key, num_layers + 3)
        
        self.grids = [
            ContinuousHashGrid(keys[0], resolution=16, output_dim=16),
            ContinuousHashGrid(keys[1], resolution=64, output_dim=16),
            ContinuousHashGrid(keys[2], resolution=256, output_dim=16)
        ]
        input_dim = 16 * 3 + 2 

        self.layers = []
        for i in range(num_layers):
            self.layers.append(FiLMLayer(keys[i+3], input_dim if i==0 else hidden_dim, hidden_dim))

        film_param_size = 2 * num_layers * hidden_dim
        self.latent_to_film = eqx.nn.Linear(latent_dim, film_param_size, key=keys[-1])
        self.final_layer = eqx.nn.Linear(hidden_dim, 2, key=keys[-2])

    def __call__(self, t, f, z):
        coords = jnp.array([t, f])
        
        grid_features = [g(coords) for g in self.grids]
        x = jnp.concatenate(grid_features + [coords], axis=0)
        
        film_params = self.latent_to_film(z)
        
        # AQUI OCORRIA O ERRO: Reshape precisa de inteiros estáticos.
        # Com @eqx.filter_jit, self.num_layers é tratado como static int.
        film_params = film_params.reshape(self.num_layers, 2, self.hidden_dim)
        
        for i, layer in enumerate(self.layers):
            gamma = film_params[i, 0]
            beta = film_params[i, 1]
            x = layer(x, gamma, beta)
            x = jnp.sin(30.0 * x)

        v = self.final_layer(x)
        return v

# --- MÓDULO 5: RECTIFIED FLOW SOLVER (EULER) ---

# CORREÇÃO CRÍTICA: Substituir @jax.jit por @eqx.filter_jit
# Isto permite ao JAX distinguir entre pesos (Tracer) e inteiros de configuração (Static)
@eqx.filter_jit
def solve_single_step(model, mixture_spec, z_latent):
    """
    Simulação do Processo de Inferência (1-Step Generation).
    """
    
    def predict_velocity(t_idx, f_idx, val_real, val_imag):
        T_max, F_max, _ = mixture_spec.shape
        norm_t = t_idx / T_max
        norm_f = f_idx / F_max
        v = model(norm_t, norm_f, z_latent)
        return v

    T, F, _ = mixture_spec.shape
    t_indices = jnp.arange(T)
    f_indices = jnp.arange(F)
    ts, fs = jnp.meshgrid(t_indices, f_indices, indexing='ij')
    
    vals_real = mixture_spec[:, :, 0]
    vals_imag = mixture_spec[:, :, 1]
    
    velocity_field = jax.vmap(
        jax.vmap(predict_velocity, in_axes=(0, 0, None, None)), 
        in_axes=(0, 0, 0, 0)
    )(ts, fs, vals_real, vals_imag)
    
    source_prediction = mixture_spec + velocity_field * 1.0
    
    return source_prediction

# --- TESTE DE SANIDADE ---

def main():
    key = jax.random.PRNGKey(0)
    
    model = DGASField(key)
    
    # Simular dados
    dummy_mix = jax.random.normal(key, (128, 128, 2))
    dummy_z = jax.random.normal(key, (128,)) 
    
    print("DGAS 3.2 Core Initialized.")
    print("Running JIT Compilation & 1-Step Inference (Wait for XLA)...")
    
    # Primeira execução (compilação)
    prediction = solve_single_step(model, dummy_mix, dummy_z)
    
    print(f"Output Shape: {prediction.shape}")
    print("Status: SUCCESS. Neural Field is operative.")

if __name__ == "__main__":
    main()