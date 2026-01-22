import jax
import jax.numpy as jnp
import equinox as eqx
from typing import List

# --- UTILS & INTERPOLAÇÃO ---
def smoother_step(x):
    x = jnp.clip(x, 0.0, 1.0)
    return x * x * x * (x * (x * 6 - 15) + 10)

def siren_init(weight: jnp.ndarray, key, w0=1.0):
    out_dim, in_dim = weight.shape
    limit = jnp.sqrt(6 / in_dim) / w0
    return jax.random.uniform(key, (out_dim, in_dim), minval=-limit, maxval=limit)

def sample_grid(grid, coords):
    """
    Realiza amostragem bilinear na grelha latente.
    grid: [Channels, Freq(H), Time(W)]
    coords: [2] (t, f) normalizados entre 0.0 e 1.0
    """
    C, H, W = grid.shape
    
    # coords[0] é Tempo (eixo W), coords[1] é Frequência (eixo H)
    x = coords[0] * (W - 1)
    y = coords[1] * (H - 1)
    
    x0 = jnp.floor(x).astype(int)
    x1 = x0 + 1
    y0 = jnp.floor(y).astype(int)
    y1 = y0 + 1
    
    x0, x1 = jnp.clip(x0, 0, W-1), jnp.clip(x1, 0, W-1)
    y0, y1 = jnp.clip(y0, 0, H-1), jnp.clip(y1, 0, H-1)
    
    wa = (x1 - x) * (y1 - y)
    wb = (x1 - x) * (y - y0)
    wc = (x - x0) * (y1 - y)
    wd = (x - x0) * (y - y0)
    
    # grid[:, y, x] (C, H, W)
    val = (grid[:, y0, x0] * wa + 
           grid[:, y1, x0] * wb + 
           grid[:, y0, x1] * wc + 
           grid[:, y1, x1] * wd)
    return val

# --- MÓDULOS DE FIELD ---
class SinusoidalEmbedding(eqx.Module):
    frequencies: jnp.ndarray
    def __init__(self, embedding_dim: int, max_freq: float = 1000.0):
        half_dim = embedding_dim // 2
        self.frequencies = jnp.exp(jnp.linspace(0, jnp.log(max_freq), half_dim))
    def __call__(self, x):
        args = x * self.frequencies
        return jnp.concatenate([jnp.sin(args), jnp.cos(args)])

class ContinuousHashGrid(eqx.Module):
    embeddings: jnp.ndarray
    resolution: int
    grid_size: int

    def __init__(self, key, resolution=128, grid_size=16384, output_dim=16):
        self.resolution = resolution
        self.grid_size = grid_size
        limit = jnp.sqrt(6 / output_dim)
        self.embeddings = jax.random.uniform(key, (grid_size, output_dim), minval=-limit, maxval=limit)
        
    @jax.checkpoint
    def __call__(self, x):
        x = jnp.clip(x, 0.0, 1.0)
        x_scaled = x * self.resolution
        x0 = jnp.floor(x_scaled).astype(jnp.int32)
        x1 = x0 + 1
        weights = smoother_step(x_scaled - x0)
        primes = jnp.array([1, 2654435761], dtype=jnp.uint32) 
        def get_hash(coords):
            return jnp.sum(coords * primes[:coords.shape[0]], axis=0) % self.grid_size
        idx00, idx01 = get_hash(x0), get_hash(jnp.array([x0[0], x1[1]]))
        idx10, idx11 = get_hash(jnp.array([x1[0], x0[1]])), get_hash(x1)
        v00, v01 = self.embeddings[idx00], self.embeddings[idx01]
        v10, v11 = self.embeddings[idx10], self.embeddings[idx11]
        h0 = v00 * (1 - weights[1]) + v01 * weights[1]
        h1 = v10 * (1 - weights[1]) + v11 * weights[1]
        return h0 * (1 - weights[0]) + h1 * weights[0]

class FiLMLayer(eqx.Module):
    linear: eqx.nn.Linear
    def __init__(self, key, in_dim, out_dim):
        self.linear = eqx.nn.Linear(in_dim, out_dim, use_bias=True, key=key)
        w_init = jax.nn.initializers.xavier_uniform()(key, (out_dim, in_dim))
        self.linear = eqx.tree_at(lambda l: l.weight, self.linear, w_init)

    def __call__(self, x, gamma, beta):
        out = self.linear(x)
        return (1.0 + gamma) * out + beta

class DGASField(eqx.Module):
    grids: List[ContinuousHashGrid]
    time_embed: SinusoidalEmbedding
    freq_embed: SinusoidalEmbedding 
    layers: List[FiLMLayer]
    to_film: eqx.nn.Linear
    val_proj: eqx.nn.Linear 
    final: eqx.nn.Linear
    hidden_dim: int
    
    def __init__(self, key, hidden_dim=256, latent_dim=128):
        keys = jax.random.split(key, 15)
        self.hidden_dim = hidden_dim
        
        self.grids = [
            ContinuousHashGrid(keys[0], resolution=16),
            ContinuousHashGrid(keys[1], resolution=64),
            ContinuousHashGrid(keys[2], resolution=256)
        ]
        
        self.time_embed = SinusoidalEmbedding(32)
        self.freq_embed = SinusoidalEmbedding(32)
        self.val_proj = eqx.nn.Linear(4, 32, key=keys[10])

        in_dim = 48 + 2 + 32 + 32 + 32
        
        self.layers = [
            FiLMLayer(keys[3], in_dim, hidden_dim),
            FiLMLayer(keys[4], hidden_dim, hidden_dim),
            FiLMLayer(keys[5], hidden_dim, hidden_dim),
            FiLMLayer(keys[6], hidden_dim, hidden_dim)
        ]
        
        # Projeta o cond local (128) para os parâmetros FiLM
        self.to_film = eqx.nn.Linear(latent_dim, 2 * 4 * hidden_dim, key=keys[7])
        self.final = eqx.nn.Linear(hidden_dim, 4, key=keys[8])

    def __call__(self, t, x_pos, x_val, cond_grid):
        x_pos = jnp.clip(x_pos, 0.0, 1.0)
        
        # --- AMUSTRAGEM LOCAL (A GRANDE CORREÇÃO) ---
        # Em vez de usar um vetor cond global, vamos buscar o cond local
        # x_pos[0] = t (tempo), x_pos[1] = f (frequência)
        local_cond = sample_grid(cond_grid, x_pos)
        
        t_emb = self.time_embed(t)
        f_coord = x_pos[1] 
        f_emb = self.freq_embed(f_coord)
        val_emb = self.val_proj(x_val) 
        
        grid_feats = [g(x_pos) for g in self.grids]
        
        h = jnp.concatenate(grid_feats + [x_pos, t_emb, f_emb, val_emb], axis=0)
        
        film_params = self.to_film(local_cond).reshape(4, 2, self.hidden_dim)
        for i, layer in enumerate(self.layers):
            gamma, beta = film_params[i]
            h = layer(h, gamma, beta)
            h = jax.nn.swish(h) 
            
        return self.final(h)

# --- NETWORKS ---

class LatentEncoder(eqx.Module):
    layers: List[eqx.nn.Conv2d]
    # Removemos a camada final Linear para preservar a estrutura espacial

    def __init__(self, key, input_channels=4):
        keys = jax.random.split(key, 5)
        self.layers = [
            # Compressão espacial gradual, mas mantendo o mapa (H, W)
            eqx.nn.Conv2d(input_channels, 32, 3, stride=2, padding=1, key=keys[0]), # /2
            eqx.nn.Conv2d(32, 64, 3, stride=2, padding=1, key=keys[1]),  # /4
            eqx.nn.Conv2d(64, 128, 3, stride=2, padding=1, key=keys[2]), # /8
            eqx.nn.Conv2d(128, 128, 3, stride=1, padding=1, key=keys[3]), # Mantém resolução
        ]
        
    def __call__(self, x):
        for layer in self.layers: 
            x = jax.nn.leaky_relu(layer(x))
        # REMOVIDO: jnp.mean(x, axis=(1, 2)) 
        # Agora retorna o tensor [Channels, F_down, T_down]
        return x

class Discriminator(eqx.Module):
    layers: List[eqx.nn.Conv2d]
    final: eqx.nn.Linear
    def __init__(self, key):
        keys = jax.random.split(key, 5)
        self.layers = [
            eqx.nn.Conv2d(8, 32, 3, stride=2, key=keys[0]),
            eqx.nn.Conv2d(32, 64, 3, stride=2, key=keys[1]),
            eqx.nn.Conv2d(64, 128, 3, stride=2, key=keys[2]),
            eqx.nn.Conv2d(128, 256, 3, stride=2, key=keys[3]),
        ]
        self.final = eqx.nn.Linear(256, 1, key=keys[4])
    def __call__(self, x, cond):
        h = jnp.concatenate([x, cond], axis=0)
        for layer in self.layers: h = jax.nn.leaky_relu(layer(h), negative_slope=0.2)
        h = jnp.mean(h, axis=(1, 2))
        return self.final(h)

class Generator(eqx.Module):
    encoder: LatentEncoder
    field: DGASField
    def __init__(self, key):
        k1, k2 = jax.random.split(key)
        self.encoder = LatentEncoder(k1)
        self.field = DGASField(k2)
    def __call__(self, t, x_pos, x_val, mix_spec):
        z_grid = self.encoder(mix_spec) # z agora é uma grelha
        return self.field(t, x_pos, x_val, z_grid)