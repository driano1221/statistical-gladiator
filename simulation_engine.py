# simulation_engine.py (Versão 2.0 - Fase 2)

"""
Motor de Simulação do Projeto "Statistical Gladiator"

Módulo aprimorado para gerar dados com múltiplas condições,
incluindo outliers e assimetria (skewness).
"""

import numpy as np
import pandas as pd
from scipy.stats import zscore

def generate_battle_data(n_groups, n_per_group, effect_size=0, contamination_level=0, outlier_intensity=5, skewness_level=0, random_state=None):
    """
    Gera um DataFrame para uma única batalha, agora com opção de assimetria.
    """
    if random_state:
        np.random.seed(random_state)

    data = []
    labels = []

    for i in range(n_groups):
        # Gera os dados base
        if skewness_level > 0:
            # Usa a distribuição Gama para gerar dados com assimetria positiva
            # O parâmetro 'shape' controla a assimetria
            shape = skewness_level
            group_data = np.random.gamma(shape, scale=1.0, size=n_per_group)
        else:
            # Gera dados normais se não houver assimetria
            group_data = np.random.normal(loc=0, scale=1, size=n_per_group)

        # Padroniza os dados (z-score) para que tenham média 0 e desvio padrão 1
        # Isso garante que o effect_size e a outlier_intensity sejam comparáveis entre os cenários
        group_data = zscore(group_data)

        # Adiciona o tamanho do efeito (diferença entre as médias)
        mean_offset = i * effect_size
        group_data += mean_offset

        # Injeta outliers (mesma lógica de antes)
        n_outliers = int(n_per_group * contamination_level)
        if n_outliers > 0:
            outlier_indices = np.random.choice(n_per_group, n_outliers, replace=False)
            outliers = np.random.normal(loc=mean_offset, scale=outlier_intensity, size=n_outliers)
            group_data[outlier_indices] = outliers

        data.extend(group_data)
        labels.extend([f'Group_{i+1}' for _ in range(n_per_group)])

    return pd.DataFrame({'value': data, 'group': labels})