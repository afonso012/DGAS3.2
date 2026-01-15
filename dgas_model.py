import jax
import jax.numpy as jnp
import equinox as eqx
from typing import Callable, List

# --- MÓDULO 1: MATEMÁTICA DE SUAVIZAÇÃO (C2 CONTINUITY) ---

def smoother_step(x):
    """
    Função de interpolação C2-Continuous (Ordem 5).
    Garante que a 1ª e 2ª derivadas são zero nas extremidades [0, 1].
    Evita 'cliques' de fase nas fronteiras da grid.
    Fórmula: 6x^5 - 15x^4 + 10x^3
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
        # Tabela de Hash aprendível
        self.embeddings = jax.random.uniform(key, (grid_size, output_dim), minval=-0.01, maxval=0.01)

    def __call__(self, x: jnp.ndarray):
        """
        Input x: Coordenada (tempo, frequência) normalizada [0, 1]
        """
        # Escalar coordenadas para a resolução da grid
        x_scaled = x * self.resolution
        
        # Índices inteiros (canto inferior esquerdo da célula)
        x0 = jnp.floor(x_scaled).astype(jnp.int32)
        x1 = x0 + 1
        
        # Pesos de interpolação (fracionários) suavizados pelo Smoother Step
        weights = smoother_step(x_scaled - x0)
        
        # Hashing espacial simples (XOR mixing para simular colisão espacial)
        # Nota: Em produção C++, usaremos Taint/Prime hashing otimizado.
        primes = jnp.array([1, 2654435761], dtype=jnp.uint32) # Primos para 2D
        
        def get_hash_index(coords):
            # Hashing simples das coordenadas inteiras
            hashed = jnp.sum(coords * primes[:coords.shape[0]], axis=0) % self.grid_size
            return hashed

        # Lookup nos 4 cantos (2D) da célula
        # (Para simplificar este snippet, assumimos 2D. 3D seria 8 cantos)
        idx00 = get_hash_index(x0)
        idx01 = get_hash_index(jnp.array([x0[0], x1[1]]))
        idx10 = get_hash_index(jnp.array([x1[0], x0[1]]))
        idx11 = get_hash_index(x1)

        val00 = self.embeddings[idx00]
        val01 = self.embeddings[idx01]
        val10 = self.embeddings[idx10]
        val11 = self.embeddings[idx11]

        # Interpolação Bilinear suavizada
        # Lerp no eixo Y
        h0 = val00 * (1 - weights[1]) + val01 * weights[1]
        h1 = val10 * (1 - weights[1]) + val11 * weights[1]
        # Lerp no eixo X
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
        # Transformação Linear Padrão
        out = self.weight @ x + self.bias
        # Modulação FiLM: Escala e Shift condicionados pelo vetor latente Z
        # Gamma multiplica (energia), Beta soma (bias harmónico)
        return (1.0 + gamma) * out + beta

# --- MÓDULO 4: O MOTOR NEURAL (SIREN BACKBONE) ---

class DGASField(eqx.Module):
    layers: List[FiLMLayer]
    grids: List[ContinuousHashGrid]
    final_layer: eqx.nn.Linear
    
    latent_to_film: eqx.nn.Linear  # Projeta Z para todos os gammas/betas
    num_layers: int
    hidden_dim: int

    def __init__(self, key, num_layers=4, hidden_dim=256, latent_dim=128):
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        
        keys = jax.random.split(key, num_layers + 3)
        
        # Multi-Resolution Hash Grids (Baixa freq -> Alta freq)
        self.grids = [
            ContinuousHashGrid(keys[0], resolution=16, output_dim=16),
            ContinuousHashGrid(keys[1], resolution=64, output_dim=16),
            ContinuousHashGrid(keys[2], resolution=256, output_dim=16)
        ]
        input_dim = 16 * 3 + 2 # Grid features + raw coords (t, f)

        # Camadas FiLM (Siren Backbone)
        self.layers = []
        for i in range(num_layers):
            self.layers.append(FiLMLayer(keys[i+3], input_dim if i==0 else hidden_dim, hidden_dim))

        # Projetor do Latente Z -> Parâmetros FiLM (Gamma, Beta)
        # Tamanho total = 2 (gamma+beta) * num_layers * hidden_dim
        film_param_size = 2 * num_layers * hidden_dim
        self.latent_to_film = eqx.nn.Linear(latent_dim, film_param_size, key=keys[-1])

        # Saída: Vetor de Velocidade (Real, Imag)
        self.final_layer = eqx.nn.Linear(hidden_dim, 2, key=keys[-2])

    def __call__(self, t, f, z):
        """
        Inferência de um único ponto (pixel de áudio).
        t: tempo [0,1]
        f: frequência [0,1]
        z: vetor latente da mistura (Audio DNA)
        """
        # 1. Preparar input
        coords = jnp.array([t, f])
        
        # 2. Consultar Hash Grids (Feature Extraction)
        grid_features = [g(coords) for g in self.grids]
        x = jnp.concatenate(grid_features + [coords], axis=0)
        
        # 3. Gerar parâmetros de modulação a partir de Z
        film_params = self.latent_to_film(z)
        
        # 4. Passar pela rede (SIREN loop)
        # Reshape film_params para acesso fácil por camada
        film_params = film_params.reshape(self.num_layers, 2, self.hidden_dim)
        
        for i, layer in enumerate(self.layers):
            gamma = film_params[i, 0]
            beta = film_params[i, 1]
            
            x = layer(x, gamma, beta)
            
            # Ativação Sine (Crucial para INRs e detalhe de sinal)
            # Frequência 30.0 é um hiperparâmetro padrão do paper SIREN
            x = jnp.sin(30.0 * x)

        # 5. Saída: Velocidade do Fluxo
        v = self.final_layer(x)
        return v

# --- MÓDULO 5: RECTIFIED FLOW SOLVER (EULER) ---

@jax.jit
def solve_single_step(model, mixture_spec, z_latent):
    """
    Simulação do Processo de Inferência (1-Step Generation).
    Assume que x_0 é a mistura (ou ruído) e queremos ir para x_1 (fonte).
    
    mixture_spec: Array (Time, Freq, 2) - O espectrograma complexo da mistura
    z_latent: O vetor de condicionamento
    """
    
    # Função wrapper para vmap (aplicar a todos os pixels da imagem de áudio de uma vez)
    def predict_velocity(t_idx, f_idx, val_real, val_imag):
        # Normalizar coords
        T_max, F_max, _ = mixture_spec.shape
        norm_t = t_idx / T_max
        norm_f = f_idx / F_max
        
        # Predizer o campo vetorial v
        v = model(norm_t, norm_f, z_latent)
        return v

    # Vetorização massiva sobre a Grid Tempo-Frequência
    # Cria índices
    T, F, _ = mixture_spec.shape
    t_indices = jnp.arange(T)
    f_indices = jnp.arange(F)
    ts, fs = jnp.meshgrid(t_indices, f_indices, indexing='ij')
    
    # Extrair valores atuais
    vals_real = mixture_spec[:, :, 0]
    vals_imag = mixture_spec[:, :, 1]
    
    # VMAP Mágico do JAX: Corre o modelo em paralelo para todos os pontos
    velocity_field = jax.vmap(jax.vmap(predict_velocity, in_axes=(None, 0, None, None)), in_axes=(0, None, 0, 0))(
        ts, fs, vals_real, vals_imag
    )
    
    # Passo de Euler (dt = 1.0 para 1-Step Rectified Flow)
    # x_1 = x_0 + v * dt
    source_prediction = mixture_spec + velocity_field * 1.0
    
    return source_prediction

# --- TESTE DE SANIDADE ---

def main():
    key = jax.random.PRNGKey(0)
    
    # Instanciar Modelo
    model = DGASField(key)
    
    # Dados Fake (Espectrograma 128x128, Complexo)
    dummy_mix = jax.random.normal(key, (128, 128, 2))
    dummy_z = jax.random.normal(key, (128,)) # Latente do Conformer
    
    print("DGAS 3.2 Core Initialized.")
    print("Running JIT Compilation & 1-Step Inference...")
    
    # Primeira execução (compilação)
    prediction = solve_single_step(model, dummy_mix, dummy_z)
    
    print(f"Output Shape: {prediction.shape}")
    print("Status: SUCCESS. Neural Field is operative.")

if __name__ == "__main__":
    main()