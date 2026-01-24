import jax
import jax.numpy as jnp
import equinox as eqx
from typing import List, Tuple

# ==============================================================================
# UTILS & MATH
# ==============================================================================

def smoother_step(x):
    x = jnp.clip(x, 0.0, 1.0)
    return x * x * x * (x * (x * 6 - 15) + 10)

def sample_grid_multiscale(grid, coords):
    """Amostrador Bilinear Diferenciável."""
    C, H, W = grid.shape
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
    val = (grid[:, y0, x0] * wa + 
           grid[:, y1, x0] * wb + 
           grid[:, y0, x1] * wc + 
           grid[:, y1, x1] * wd)
    return val

# ==============================================================================
# BLOCOS DE ENGENHARIA AVANÇADA (RESNET & COORDCONV)
# ==============================================================================

class ResBlock(eqx.Module):
    """Bloco Residual com Dilatação."""
    conv1: eqx.nn.Conv2d
    conv2: eqx.nn.Conv2d
    norm1: eqx.nn.GroupNorm
    norm2: eqx.nn.GroupNorm
    
    def __init__(self, channels, dilation=1, key=None):
        k1, k2 = jax.random.split(key)
        pad = dilation
        self.conv1 = eqx.nn.Conv2d(channels, channels, 3, padding=pad, dilation=dilation, key=k1)
        self.conv2 = eqx.nn.Conv2d(channels, channels, 3, padding=1, key=k2)
        self.norm1 = eqx.nn.GroupNorm(min(32, channels // 4), channels)
        self.norm2 = eqx.nn.GroupNorm(min(32, channels // 4), channels)
        
    def __call__(self, x):
        h = self.norm1(x)
        h = jax.nn.swish(h)
        h = self.conv1(h)
        h = self.norm2(h)
        h = jax.nn.swish(h)
        h = self.conv2(h)
        return x + h 

def add_coord_channels(x):
    """Implementação CoordConv."""
    C, H, W = x.shape
    y_coords = jnp.linspace(0, 1, H)
    y_grid = jnp.tile(y_coords[:, None], (1, W))
    x_coords = jnp.linspace(0, 1, W)
    x_grid = jnp.tile(x_coords[None, :], (H, 1))
    coords = jnp.stack([y_grid, x_grid], axis=0)
    return jnp.concatenate([x, coords], axis=0)

# ==============================================================================
# NEURAL FIELDS (Memória & Decodificação)
# ==============================================================================

class SinusoidalEmbedding(eqx.Module):
    frequencies: jnp.ndarray
    def __init__(self, embedding_dim: int, min_freq=1.0, max_freq=1000.0):
        # Gera embedding_dim frequências no total (sin + cos)
        half_dim = embedding_dim // 2
        self.frequencies = jnp.exp(jnp.linspace(jnp.log(min_freq), jnp.log(max_freq), half_dim))
    def __call__(self, x):
        args = x * self.frequencies
        return jnp.concatenate([jnp.sin(args), jnp.cos(args)])

class ContinuousHashGrid(eqx.Module):
    embeddings: jnp.ndarray
    resolution: int
    grid_size: int
    def __init__(self, key, resolution=128, grid_size=16384, output_dim=4):
        self.resolution = resolution
        self.grid_size = grid_size
        self.embeddings = jax.random.uniform(key, (grid_size, output_dim), minval=-1e-4, maxval=1e-4)
    def __call__(self, x):
        x = jnp.clip(x, 0.0, 1.0)
        x_scaled = x * self.resolution
        x0 = jnp.floor(x_scaled).astype(jnp.int32)
        x1 = x0 + 1
        weights = smoother_step(x_scaled - x0)
        primes = jnp.array([1, 2654435761], dtype=jnp.uint32) 
        def get_hash(coords):
            p = coords * primes[:coords.shape[0]]
            h = jax.lax.bitwise_xor(p[0], p[1])
            return h % self.grid_size
        idx00 = get_hash(jnp.array([x0[0], x0[1]]))
        idx01 = get_hash(jnp.array([x0[0], x1[1]]))
        idx10 = get_hash(jnp.array([x1[0], x0[1]]))
        idx11 = get_hash(jnp.array([x1[0], x1[1]]))
        v00, v01 = self.embeddings[idx00], self.embeddings[idx01]
        v10, v11 = self.embeddings[idx10], self.embeddings[idx11]
        h0 = v00 * (1 - weights[1]) + v01 * weights[1]
        h1 = v10 * (1 - weights[1]) + v11 * weights[1]
        return h0 * (1 - weights[0]) + h1 * weights[0]

class FiLMLayer(eqx.Module):
    linear: eqx.nn.Linear
    def __init__(self, key, in_dim, out_dim):
        self.linear = eqx.nn.Linear(in_dim, out_dim, key=key)
        w_init = jax.nn.initializers.he_uniform()(key, (out_dim, in_dim))
        self.linear = eqx.tree_at(lambda l: l.weight, self.linear, w_init)
    def __call__(self, x, gamma, beta):
        out = self.linear(x)
        return (1.0 + gamma) * out + beta

# ==============================================================================
# ENCODER PROFISSIONAL (Dilated ResNet + CoordConv)
# ==============================================================================

class MultiScaleEncoder(eqx.Module):
    """Encoder SOTA corrigido para evitar FrozenInstanceError."""
    # Declaração explícita de campos (Obrigatório no Equinox)
    init_conv: eqx.nn.Conv2d
    blocks_l0: List[ResBlock]
    down0: eqx.nn.Conv2d
    blocks_l1: List[ResBlock]
    down1: eqx.nn.Conv2d
    blocks_l2: List[ResBlock]
    
    def __init__(self, key):
        keys = jax.random.split(key, 10)
        
        # Input: 4 (Audio) + 2 (Coords) = 6 canais
        self.init_conv = eqx.nn.Conv2d(6, 32, kernel_size=3, padding=1, key=keys[0])
        
        # Nível 0
        self.blocks_l0 = [
            ResBlock(32, dilation=1, key=keys[1]),
            ResBlock(32, dilation=2, key=keys[2])
        ]
        self.down0 = eqx.nn.Conv2d(32, 64, 3, stride=2, padding=1, key=keys[3])
        
        # Nível 1
        self.blocks_l1 = [
            ResBlock(64, dilation=1, key=keys[4]),
            ResBlock(64, dilation=2, key=keys[5])
        ]
        self.down1 = eqx.nn.Conv2d(64, 128, 3, stride=2, padding=1, key=keys[6])
        
        # Nível 2
        self.blocks_l2 = [
            ResBlock(128, dilation=1, key=keys[7]),
            ResBlock(128, dilation=2, key=keys[8])
        ]

    def __call__(self, x):
        x = add_coord_channels(x) 
        
        h = jax.nn.swish(self.init_conv(x))
        for block in self.blocks_l0: h = block(h)
        f0 = h 
        
        h = jax.nn.swish(self.down0(h))
        for block in self.blocks_l1: h = block(h)
        f1 = h 
        
        h = jax.nn.swish(self.down1(h))
        for block in self.blocks_l2: h = block(h)
        f2 = h 
        
        return f0, f1, f2

# ==============================================================================
# DECODER FIELD & DISCRIMINATOR
# ==============================================================================

class DGASField(eqx.Module):
    grids: List[ContinuousHashGrid]
    time_embed: SinusoidalEmbedding
    freq_embed: SinusoidalEmbedding 
    layers: List[FiLMLayer]
    to_film: eqx.nn.Linear
    val_proj: eqx.nn.Linear 
    final: eqx.nn.Linear
    
    def __init__(self, key, hidden_dim=256):
        keys = jax.random.split(key, 15)
        self.grids = [
            ContinuousHashGrid(keys[0], resolution=64, output_dim=4),
            ContinuousHashGrid(keys[1], resolution=128, output_dim=4)
        ]
        self.time_embed = SinusoidalEmbedding(32)
        self.freq_embed = SinusoidalEmbedding(32)
        self.val_proj = eqx.nn.Linear(4, 32, key=keys[2])

        # CORREÇÃO CRÍTICA AQUI
        in_dim = 8 + 2 + 32 + 32 + 32 # 106
        
        # Camada 1: 106 -> 256
        layer1 = FiLMLayer(keys[3], in_dim, hidden_dim)
        # Camadas seguintes: 256 -> 256
        layer2 = FiLMLayer(keys[4], hidden_dim, hidden_dim)
        layer3 = FiLMLayer(keys[5], hidden_dim, hidden_dim)
        layer4 = FiLMLayer(keys[6], hidden_dim, hidden_dim)
        
        self.layers = [layer1, layer2, layer3, layer4]
        
        # Contexto: 32 (L0) + 64 (L1) + 128 (L2) = 224
        self.to_film = eqx.nn.Linear(224, 2 * 4 * hidden_dim, key=keys[7])
        
        self.final = eqx.nn.Linear(hidden_dim, 4, key=keys[8])
        self.final = eqx.tree_at(lambda l: l.weight, self.final, jnp.zeros((4, hidden_dim)))
        self.final = eqx.tree_at(lambda l: l.bias, self.final, jnp.zeros((4,)))

    def __call__(self, t, x_pos, x_val, multi_scale_conds):
        c0, c1, c2 = multi_scale_conds
        
        # Amostragem Multi-Escala
        ctx_fine = sample_grid_multiscale(c0, x_pos)
        ctx_mid  = sample_grid_multiscale(c1, x_pos)
        ctx_coarse = sample_grid_multiscale(c2, x_pos)
        local_cond = jnp.concatenate([ctx_fine, ctx_mid, ctx_coarse], axis=0)
        
        # Embeddings
        t_emb = self.time_embed(t)
        f_emb = self.freq_embed(x_pos[1])
        val_emb = self.val_proj(x_val) 
        grid_feats = [g(x_pos) for g in self.grids]
        
        h = jnp.concatenate(grid_feats + [x_pos, t_emb, f_emb, val_emb], axis=0)
        
        film_params = self.to_film(local_cond).reshape(4, 2, -1)
        
        for i, layer in enumerate(self.layers):
            gamma, beta = film_params[i]
            h = layer(h, gamma, beta)
            h = jax.nn.swish(h) 
            
        return self.final(h)

class Discriminator(eqx.Module):
    """Discriminador com ResBlocks."""
    init_conv: eqx.nn.Conv2d
    blocks: List[eqx.Module]
    final: eqx.nn.Linear
    
    def __init__(self, key):
        keys = jax.random.split(key, 8)
        self.init_conv = eqx.nn.Conv2d(8, 32, 4, stride=2, padding=1, key=keys[0])
        
        self.blocks = [
            ResBlock(32, key=keys[1]),
            eqx.nn.Conv2d(32, 64, 4, stride=2, padding=1, key=keys[2]),
            ResBlock(64, key=keys[3]),
            eqx.nn.Conv2d(64, 128, 4, stride=2, padding=1, key=keys[4]),
            ResBlock(128, key=keys[5]),
            eqx.nn.Conv2d(128, 256, 4, stride=2, padding=1, key=keys[6]),
        ]
        self.final = eqx.nn.Linear(256, 1, key=keys[7])
        
    def __call__(self, x, cond):
        h = jnp.concatenate([x, cond], axis=0)
        h = jax.nn.leaky_relu(self.init_conv(h), negative_slope=0.2)
        
        for layer in self.blocks:
            if isinstance(layer, ResBlock):
                h = layer(h)
            else:
                h = layer(h)
                h = jax.nn.leaky_relu(h, negative_slope=0.2)
                
        h = jnp.mean(h, axis=(1, 2))
        return self.final(h)

# Wrappers Finais
class Generator(eqx.Module):
    encoder: MultiScaleEncoder
    field: DGASField
    def __init__(self, key):
        k1, k2 = jax.random.split(key)
        self.encoder = MultiScaleEncoder(k1)
        self.field = DGASField(k2)
    def __call__(self, t, x_pos, x_val, mix_spec): pass