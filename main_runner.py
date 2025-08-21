# main_runner.py (Versão 2.0 - Simulação Completa)

import pandas as pd
import config as cfg
from simulation_engine import generate_battle_data
from scipy.stats import f_oneway
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import balanced_accuracy_score
from tqdm import tqdm # Importamos uma barra de progresso!

def run_single_battle(simulation_id):
    """
    Executa uma única batalha e retorna os resultados.
    Esta função será chamada em um loop.
    """
    # Geramos dados para um cenário onde H0 é Falsa (existe efeito)
    battle_df = generate_battle_data(
        n_groups=cfg.N_GROUPS,
        n_per_group=cfg.N_PER_GROUP,
        effect_size=cfg.EFFECT_SIZE,
        contamination_level=cfg.CONTAMINATION_LEVEL,
        outlier_intensity=cfg.OUTLIER_INTENSITY,
        random_state=cfg.RANDOM_STATE + simulation_id # Semente diferente para cada simulação
    )
    
    # --- Análise ANOVA ---
    groups = [battle_df['value'][battle_df['group'] == g] for g in battle_df['group'].unique()]
    _, p_value = f_oneway(*groups)
    anova_detects_effect = p_value < 0.05
    
    # --- Análise Random Forest ---
    X = battle_df[['value']]
    y = battle_df['group']
    rf_model = RandomForestClassifier(random_state=cfg.RANDOM_STATE)
    rf_model.fit(X, y)
    predictions = rf_model.predict(X)
    rf_accuracy = balanced_accuracy_score(y, predictions)
    
    # Retorna um dicionário com os resultados da rodada
    return {
        'simulation_id': simulation_id,
        'anova_p_value': p_value,
        'anova_detects_effect': anova_detects_effect,
        'rf_accuracy': rf_accuracy
    }

# --- Ponto de Entrada do Script ---
if __name__ == "__main__":
    print(f"⚔️  Iniciando Simulação Completa do MVP ({cfg.N_SIMULATIONS} batalhas)... ⚔️")
    
    # Lista para armazenar os resultados de cada batalha
    results_list = []
    
    # Loop que executa a simulação N vezes com uma barra de progresso (tqdm)
    for i in tqdm(range(cfg.N_SIMULATIONS)):
        result = run_single_battle(simulation_id=i)
        results_list.append(result)
        
    # Converte a lista de resultados em um DataFrame do Pandas
    results_df = pd.DataFrame(results_list)
    
    print("\nSimulação concluída!")
    print("-" * 50)
    
    # --- Análise dos Resultados Agregados ---
    # Poder da ANOVA: % de vezes que detectou o efeito corretamente
    anova_power = results_df['anova_detects_effect'].mean()
    
    # Performance Média do Random Forest
    avg_rf_accuracy = results_df['rf_accuracy'].mean()
    
    print("Resultados Finais da Arena (MVP):")
    print(f"Condição: {cfg.CONTAMINATION_LEVEL:.0%} de contaminação por outliers.")
    print(f"Poder Estatístico da ANOVA: {anova_power:.2%}")
    print(f"Acurácia Média do Random Forest: {avg_rf_accuracy:.2%}")
    print("-" * 50)
    
    # Salva os resultados detalhados em um arquivo CSV para análise futura
    output_path = "mvp_results.csv"
    results_df.to_csv(output_path, index=False)
    print(f"Resultados detalhados salvos em: '{output_path}'")