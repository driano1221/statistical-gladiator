# main_runner.py (Versão 3.0 - Fase 2)

import pandas as pd
import config as cfg
from simulation_engine import generate_battle_data
from scipy.stats import f_oneway, kruskal
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import balanced_accuracy_score
from xgboost import XGBClassifier
from tqdm import tqdm
import warnings

# Ignorar warnings para manter o output mais limpo
warnings.filterwarnings('ignore', category=UserWarning, module='xgboost')

# --- Funções dos Gladiadores ---
# Cada gladiador tem sua própria função para análise


def run_anova(df):
    """Executa a análise ANOVA e retorna o p-valor."""
    groups = [df['value'][df['group'] == g] for g in df['group'].unique()]
    _, p_value = f_oneway(*groups)
    return {'p_value': p_value}


def run_kruskal(df):
    """Executa a análise Kruskal-Wallis e retorna o p-valor."""
    groups = [df['value'][df['group'] == g] for g in df['group'].unique()]
    _, p_value = kruskal(*groups)
    return {'p_value': p_value}


def run_random_forest(df):
    """Executa a análise com RandomForest e retorna a acurácia."""
    X = df[['value']]
    y = df['group']
    model = RandomForestClassifier(random_state=cfg.RANDOM_STATE)
    model.fit(X, y)
    predictions = model.predict(X)
    accuracy = balanced_accuracy_score(y, predictions)
    return {'accuracy': accuracy}


def run_xgboost(df):
    """Executa a análise com XGBoost e retorna a acurácia."""
    X = df[['value']]
    y = df['group']
    # XGBoost precisa de labels numéricos (0, 1, 2...)
    y_encoded = pd.factorize(y)[0]
    model = XGBClassifier(random_state=cfg.RANDOM_STATE,
                          use_label_encoder=False, eval_metric='mlogloss')
    model.fit(X, y_encoded)
    predictions = model.predict(X)
    accuracy = balanced_accuracy_score(y_encoded, predictions)
    return {'accuracy': accuracy}


# Mapeia os nomes dos gladiadores às suas funções
GLADIATOR_FUNCTIONS = {
    'ANOVA': run_anova,
    'Kruskal-Wallis': run_kruskal,
    'RandomForest': run_random_forest,
    'XGBoost': run_xgboost,
}

# --- Ponto de Entrada do Script ---
if __name__ == "__main__":
    print("⚔️  Iniciando Simulação Completa da Fase 2 ⚔️")

    all_results = []

    # Loop principal: itera sobre cada condição de batalha
    for condition_name, params in cfg.BATTLE_CONDITIONS.items():
        print(f"\n--- Iniciando Batalha na Condição: {condition_name} ---")

        # Loop secundário: itera sobre cada gladiador
        for gladiator_name, gladiator_func in GLADIATOR_FUNCTIONS.items():

            print(f"  - Convocando Gladiador: {gladiator_name}...")

            # Loop de simulação: executa N vezes
            for i in tqdm(range(cfg.N_SIMULATIONS), leave=False):
                # Gera os dados para a condição atual
                battle_df = generate_battle_data(
                    n_groups=cfg.N_GROUPS,
                    n_per_group=params.get(
                        'n_per_group', cfg.N_PER_GROUP_DEFAULT),
                    effect_size=cfg.EFFECT_SIZE,
                    contamination_level=params.get('contamination_level', 0.0),
                    outlier_intensity=params.get('outlier_intensity', 5),
                    skewness_level=params.get('skewness_level', 0.0),
                    random_state=cfg.RANDOM_STATE + i
                )

                # Executa a função do gladiador atual
                result_metrics = gladiator_func(battle_df)

                # Monta o dicionário de resultados da rodada
                round_result = {
                    'simulation_id': i,
                    'condition': condition_name,
                    'gladiator': gladiator_name,
                    # Adiciona as métricas específicas (p_value ou accuracy)
                    **result_metrics
                }
                all_results.append(round_result)

    # Converte a lista completa de resultados em um DataFrame
    results_df = pd.DataFrame(all_results)

    # --- Análise e Salvamento ---
    print("\n--- Simulação da Fase 2 Concluída! ---")

    # Processa os resultados para criar uma tabela de resumo
    summary_list = []
    for condition in results_df['condition'].unique():
        for gladiator in results_df['gladiator'].unique():
            subset = results_df[(results_df['condition'] == condition) & (
                results_df['gladiator'] == gladiator)]

            # Se for um teste estatístico, calcula o poder
            if 'p_value' in subset.columns:
                power = (subset['p_value'] < 0.05).mean()
                summary_list.append(
                    {'condition': condition, 'gladiator': gladiator, 'metric': 'Power', 'value': power})
            # Se for um modelo de ML, calcula a acurácia média
            elif 'accuracy' in subset.columns:
                avg_accuracy = subset['accuracy'].mean()
                summary_list.append({'condition': condition, 'gladiator': gladiator,
                                    'metric': 'Avg. Accuracy', 'value': avg_accuracy})

    summary_df = pd.DataFrame(summary_list)

    print("\n📋 Tabela de Resumo dos Resultados Finais 📋")
    # Formata a tabela de resumo para melhor visualização (pivot)
    summary_pivot = summary_df.pivot_table(
        index='gladiator', columns='condition', values='value')
    print(summary_pivot.to_string(float_format="%.3f"))

    # Salva os resultados detalhados
    output_path = "phase2_results.csv"
    results_df.to_csv(output_path, index=False)
    print(f"\nResultados detalhados salvos em: '{output_path}'")
