import jax
import jax.numpy as jnp
import equinox as eqx
from typing import List

# --- MÓDULO AUXILIAR ---
class SinActivation(eqx.Module):
    def __call__(self, x):
        # SIREN activation (Secção 3.2 do Relatório)
        return jnp.sin(30.0 * x)

# --- FUNÇÃO CRÍTICA: C2 CONTINUITY (Secção 3.1) ---
def smoother_step(x):
    """
    Ken Perlin's Smootherstep function.
    Garante continuidade da 1ª e 2ª derivada (C2).
    Evita artefactos digitais e 'cliques' na fase.
    Eq: 6x^5 - 15x^4 + 10x^3
    """
    x = jnp.clip(x, 0.0, 1.0)
    return x * x * x * (x * (x * 6 - 15) + 10)

# --- 1. HASH GRID C2-CONTINUOUS ---
class ContinuousHashGrid(eqx.Module):
    embeddings: jnp.ndarray
    resolution: int
    grid_size: int

    def __init__(self, key, resolution=128, grid_size=4096, output_dim=16):
        self.resolution = resolution
        self.grid_size = grid_size
        limit = jnp.sqrt(6 / output_dim)
        self.embeddings = jax.random.uniform(key, (grid_size, output_dim), minval=-limit, maxval=limit)

    def __call__(self, x):
        # x shape: (2,) -> (Time, Freq) normalizados [0,1]
        x_scaled = x * self.resolution
        x0 = jnp.floor(x_scaled).astype(jnp.int32)
        x1 = x0 + 1
        
        # --- CORREÇÃO DGAS 3.1: USAR SMOOTHERSTEP EM VEZ DE LINEAR ---
        # Isto implementa a interpolação polinomial exigida no relatório
        weights = smoother_step(x_scaled - x0) 
        
        primes = jnp.array([1, 2654435761], dtype=jnp.uint32)
        def get_hash(coords):
            return jnp.sum(coords * primes[:coords.shape[0]], axis=0) % self.grid_size

        idx00 = get_hash(x0)
        idx01 = get_hash(jnp.array([x0[0], x1[1]]))
        idx10 = get_hash(jnp.array([x1[0], x0[1]]))
        idx11 = get_hash(x1)

        v00 = self.embeddings[idx00]
        v01 = self.embeddings[idx01]
        v10 = self.embeddings[idx10]
        v11 = self.embeddings[idx11]

        # Interpolação Bilinear com pesos C2 (Smooth)
        h0 = v00 * (1 - weights[1]) + v01 * weights[1]
        h1 = v10 * (1 - weights[1]) + v11 * weights[1]
        return h0 * (1 - weights[0]) + h1 * weights[0]

# --- 2. MODULAÇÃO FiLM (Secção 3.2) ---
class FiLMLayer(eqx.Module):
    linear: eqx.nn.Linear
    
    def __init__(self, key, in_dim, out_dim):
        self.linear = eqx.nn.Linear(in_dim, out_dim, key=key)
        
    def __call__(self, x, gamma, beta):
        out = self.linear(x)
        # Modulação Afim: Feature-wise Linear Modulation
        return (1.0 + gamma) * out + beta

# --- 3. VECTOR FIELD (Rectified Flow Engine) ---
class DGASField(eqx.Module):
    grids: List[ContinuousHashGrid]
    layers: List[FiLMLayer]
    to_film: eqx.nn.Linear
    final: eqx.nn.Linear
    hidden_dim: int
    
    def __init__(self, key, hidden_dim=256, latent_dim=128):
        keys = jax.random.split(key, 10)
        self.hidden_dim = hidden_dim
        
        # Multi-Resolução para capturar detalhes finos e globais
        self.grids = [
            ContinuousHashGrid(keys[0], resolution=16),
            ContinuousHashGrid(keys[1], resolution=64),
            ContinuousHashGrid(keys[2], resolution=256)
        ]
        
        # 16*3 (Grids) + 2 (Espacial) + 1 (Tempo Difusão) = 51
        in_dim = 16 * 3 + 3  
        
        self.layers = [
            FiLMLayer(keys[3], in_dim, hidden_dim),
            FiLMLayer(keys[4], hidden_dim, hidden_dim),
            FiLMLayer(keys[5], hidden_dim, hidden_dim),
            FiLMLayer(keys[6], hidden_dim, hidden_dim)
        ]
        
        # Mapeia o Latente Z para os parâmetros FiLM (Gamma, Beta)
        self.to_film = eqx.nn.Linear(latent_dim, 2 * 4 * hidden_dim, key=keys[7])
        self.final = eqx.nn.Linear(hidden_dim, 4, key=keys[8]) 

    def __call__(self, t, x_pos, cond):
        coords = jnp.concatenate([jnp.array([t]), x_pos]) 
        grid_feats = [g(x_pos) for g in self.grids]
        h = jnp.concatenate(grid_feats + [coords], axis=0)
        
        film_params = self.to_film(cond).reshape(4, 2, self.hidden_dim)
        
        for i, layer in enumerate(self.layers):
            gamma, beta = film_params[i]
            h = layer(h, gamma, beta)
            h = jnp.sin(30.0 * h) # SIREN Activation (Secção 3.2)
            
        return self.final(h)

# --- 4. ENCODER LATENTE ---
class LatentEncoder(eqx.Module):
    layers: List[eqx.nn.Conv2d]
    final: eqx.nn.Linear
    
    def __init__(self, key, input_channels=4):
        keys = jax.random.split(key, 5)
        self.layers = [
            eqx.nn.Conv2d(input_channels, 32, kernel_size=3, stride=2, key=keys[0]),
            eqx.nn.Conv2d(32, 64, kernel_size=3, stride=2, key=keys[1]),
            eqx.nn.Conv2d(64, 128, kernel_size=3, stride=2, key=keys[2]),
            eqx.nn.Conv2d(128, 128, kernel_size=3, stride=2, key=keys[3]),
        ]
        self.final = eqx.nn.Linear(128, 128, key=keys[4])

    def __call__(self, x):
        for layer in self.layers:
            x = jax.nn.leaky_relu(layer(x))
        x = jnp.mean(x, axis=(1, 2))
        return self.final(x)

# --- 5. WRAPPERS ---
class Generator(eqx.Module):
    encoder: LatentEncoder
    field: DGASField

    def __init__(self, key):
        k1, k2 = jax.random.split(key)
        self.encoder = LatentEncoder(k1)
        self.field = DGASField(k2)

    def __call__(self, t, x_pos, mix_spec):
        z = self.encoder(mix_spec)
        return self.field(t, x_pos, z)

class Discriminator(eqx.Module):
    layers: List[eqx.nn.Conv2d]
    final: eqx.nn.Linear
    
    def __init__(self, key):
        keys = jax.random.split(key, 5)
        self.layers = [
            eqx.nn.Conv2d(8, 32, kernel_size=3, stride=2, key=keys[0]),
            eqx.nn.Conv2d(32, 64, kernel_size=3, stride=2, key=keys[1]),
            eqx.nn.Conv2d(64, 128, kernel_size=3, stride=2, key=keys[2]),
        ]
        self.final = eqx.nn.Linear(128, 1, key=keys[3])

    def __call__(self, x, cond):
        h = jnp.concatenate([x, cond], axis=0)
        for layer in self.layers:
            h = jax.nn.leaky_relu(layer(h))
        h = jnp.mean(h, axis=(1, 2))
        return self.final(h)