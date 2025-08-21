# config.py

"""
Painel de Controle do Projeto "Statistical Gladiator"

Aqui centralizamos todas as variáveis e parâmetros para
facilitar a experimentação e a manutenção do código.
"""

# Parâmetros da Simulação Global
N_SIMULATIONS = 1000  # Número de vezes que cada cenário será simulado
RANDOM_STATE = 42      # Semente para garantir a reprodutibilidade dos resultados

# Parâmetros para Geração de Dados (MVP)
N_GROUPS = 3
N_PER_GROUP = 30
# Tamanho do efeito quando a H0 é falsa. 0 para H0 verdadeira.
EFFECT_SIZE = 0.8
git add .
git commit -m "Fase 1: Implementado config e motor de geração de dados"
# Parâmetros da "Condição de Batalha": Outliers
CONTAMINATION_LEVEL = 0.10  # 10% dos dados serão outliers
# Quão extremos os outliers são (x vezes o desvio padrão)
OUTLIER_INTENSITY = 5
