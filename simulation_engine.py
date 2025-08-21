# simulation_engine.py

"""
Motor de Simulação do Projeto "Statistical Gladiator"

Este módulo contém as funções para gerar os conjuntos de dados
para cada batalha, incluindo as condições ideais e as contaminadas.
"""

import numpy as np
import pandas as pd

def generate_battle_data(n_groups, n_per_group, effect_size=0, contamination_level=0, outlier_intensity=5, random_state=None):
    """
    Gera um DataFrame para uma única batalha.

    Args:
        n_groups (int): Número de grupos a serem comparados.
        n_per_group (int): Número de amostras por grupo.
        effect_size (float): A diferença entre as médias dos grupos.
                             Se 0, a hipótese nula (H0) é verdadeira.
        contamination_level (float): A proporção (0 a 1) de outliers a serem injetados.
        outlier_intensity (float): Multiplicador do desvio padrão para definir o quão extremo é o outlier.
        random_state (int, opcional): Semente para o gerador de números aleatórios para reprodutibilidade.

    Returns:
        pd.DataFrame: Um DataFrame com as colunas 'value' e 'group'.
    """
    if random_state:
        np.random.seed(random_state)

    data = []
    labels = []

    for i in range(n_groups):
        # A média de cada grupo é deslocada pelo effect_size
        mean = i * effect_size
        
        # Gera dados normais para o grupo
        group_data = np.random.normal(loc=mean, scale=1, size=n_per_group)

        # Injeta outliers se a contaminação for maior que zero
        n_outliers = int(n_per_group * contamination_level)
        if n_outliers > 0:
            # Seleciona índices aleatórios para transformar em outliers
            outlier_indices = np.random.choice(n_per_group, n_outliers, replace=False)
            
            # Gera outliers a partir de uma distribuição muito mais ampla
            outliers = np.random.normal(loc=mean, scale=outlier_intensity, size=n_outliers)
            
            # Substitui os dados nos índices selecionados pelos outliers
            group_data[outlier_indices] = outliers
            
        data.extend(group_data)
        labels.extend([f'Group_{i+1}' for _ in range(n_per_group)])
        
    return pd.DataFrame({'value': data, 'group': labels})