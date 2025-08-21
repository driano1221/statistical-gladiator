# dashboard.py

import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# --- Configuração da Página ---
st.set_page_config(
    page_title="Statistical Gladiator Dashboard",
    page_icon="⚔️",
    layout="wide"
)

# --- Carregamento e Cache dos Dados ---
# Usamos cache para não recarregar os dados a cada interação


@st.cache_data
def load_data():
    try:
        df = pd.read_csv('phase2_results.csv')
        return df
    except FileNotFoundError:
        return None


results_df = load_data()

# --- Título e Introdução ---
st.title("⚔️ Statistical Gladiator: A Arena de Batalha")
st.markdown("""
Bem-vindo ao Dashboard de resultados do projeto 'Statistical Gladiator'.
Aqui, analisamos a performance de métodos estatísticos clássicos e algoritmos de Machine Learning
em diferentes cenários de dados imperfeitos.
""")

# --- Lógica de Exibição ---
if results_df is None:
    st.error("Arquivo 'phase2_results.csv' não encontrado. Por favor, rode a simulação 'main_runner.py' primeiro.")
else:
    st.header("🔥 Heatmap de Performance dos Gladiadores 🔥")

    # --- Lógica de sumarização (a mesma do notebook) ---
    summary_list = []
    for condition in results_df['condition'].unique():
        for gladiator in results_df['gladiator'].unique():
            subset = results_df[(results_df['condition'] == condition) & (
                results_df['gladiator'] == gladiator)]
            if subset['p_value'].notna().any():
                power = (subset['p_value'] < 0.05).mean()
                summary_list.append(
                    {'condition': condition, 'gladiator': gladiator, 'metric': 'Power', 'value': power})
            elif subset['accuracy'].notna().any():
                avg_accuracy = subset['accuracy'].mean()
                summary_list.append({'condition': condition, 'gladiator': gladiator,
                                    'metric': 'Avg. Accuracy', 'value': avg_accuracy})

    summary_df = pd.DataFrame(summary_list)
    summary_pivot = summary_df.pivot_table(
        index='gladiator', columns='condition', values='value')

    # --- Desenho do Gráfico ---
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(
        summary_pivot,
        annot=True,
        fmt=".3f",
        cmap='viridis',
        linewidths=.5,
        ax=ax,
        cbar_kws={'label': 'Performance (Poder ou Acurácia Média)'}
    )
    ax.set_title('Performance dos Gladiadores em Diferentes Arenas',
                 fontsize=16, pad=20)
    ax.set_xlabel('Condição de Batalha (Arena)', fontsize=12)
    ax.set_ylabel('Gladiador', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)

    # Exibe o gráfico no Streamlit
    st.pyplot(fig)

    st.markdown("---")
    st.header("📜 Tabela de Resumo dos Resultados")
    st.dataframe(summary_pivot.style.format(
        "{:.3f}").background_gradient(cmap='viridis'))

    st.header("📊 Dados Brutos da Simulação")
    st.dataframe(results_df)
