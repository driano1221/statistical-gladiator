# config.py (Versão 2.0 - Fase 2)

"""
Painel de Controle do Projeto "Statistical Gladiator" - Fase 2

Expandimos os parâmetros para incluir múltiplos gladiadores e
condições de batalha.
"""

# Parâmetros da Simulação Global
N_SIMULATIONS = 1000
RANDOM_STATE = 42

# Parâmetros para Geração de Dados
N_GROUPS = 3
N_PER_GROUP_DEFAULT = 30
EFFECT_SIZE = 0.8

# --- LISTA DE GLADIADORES ---
# Aqui definimos os competidores que entrarão na arena
GLADIATORS = {
    'ANOVA': {'type': 'statistical'},
    'Kruskal-Wallis': {'type': 'statistical'},
    'RandomForest': {'type': 'ml'},
    'XGBoost': {'type': 'ml'},
}

# --- CONDIÇÕES DE BATALHA ---
# Cada condição é um dicionário de parâmetros que sobrepõe os padrões
BATTLE_CONDITIONS = {
    'baseline_normal': {
        'contamination_level': 0.0,
        'skewness_level': 0.0,
        'n_per_group': 30
    },
    'outliers_10_percent': {
        'contamination_level': 0.10,
        'outlier_intensity': 5,
        'skewness_level': 0.0,
        'n_per_group': 30
    },
    'high_skewness': {
        'contamination_level': 0.0,
        'skewness_level': 4.0, # Nível de assimetria (0 é normal)
        'n_per_group': 30
    },
    'small_samples': {
        'contamination_level': 0.0,
        'skewness_level': 0.0,
        'n_per_group': 10 # Amostras bem pequenas
    }
}