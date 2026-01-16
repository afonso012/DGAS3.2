import jax
import jax.numpy as jnp
import equinox as eqx
from typing import Callable, List

# --- MÓDULO 1: MATEMÁTICA ---

def smoother_step(x):
    x = jnp.clip(x, 0.0, 1.0)
    return x * x * x * (x * (x * 6 - 15) + 10)

# --- MÓDULO 2: HASH GRID ---

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
        # CORREÇÃO SIREN INIT: Distribuição Uniforme com limite sqrt(6/in)
        limit = jnp.sqrt(6 / in_features)
        self.weight = jax.random.uniform(w_key, (out_features, in_features), minval=-limit, maxval=limit)
        self.bias = jnp.zeros((out_features,))

    def __call__(self, x, gamma, beta):
        out = self.weight @ x + self.bias
        return (1.0 + gamma) * out + beta

# --- MÓDULO 4: O MOTOR NEURAL ---

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
        
        # CORREÇÃO ESTÉREO: Output Dim agora é 4 (L_Re, L_Im, R_Re, R_Im)
        self.final_layer = eqx.nn.Linear(hidden_dim, 4, key=keys[-2])

    def __call__(self, t, f, z):
        coords = jnp.array([t, f])
        grid_features = [g(coords) for g in self.grids]
        x = jnp.concatenate(grid_features + [coords], axis=0)
        
        film_params = self.latent_to_film(z)
        film_params = film_params.reshape(self.num_layers, 2, self.hidden_dim)
        
        for i, layer in enumerate(self.layers):
            gamma = film_params[i, 0]
            beta = film_params[i, 1]
            x = layer(x, gamma, beta)
            x = jnp.sin(30.0 * x) # SIREN Activation

        v = self.final_layer(x)
        return v