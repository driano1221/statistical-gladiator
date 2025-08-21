# ⚔️ Statistical Gladiator: ANOVA vs. Machine Learning

Este projeto é uma comparação sistemática e robusta entre métodos estatísticos clássicos e algoritmos de Machine Learning para a tarefa de detectar diferenças entre grupos, especialmente sob condições de dados não ideais.

## 🎯 Pergunta Central

*100 anos de estatística vs 20 anos de ML. Quem vence quando os dados não cooperam?*

## ⚖️ Métricas-Chave do MVP

Para garantir uma comparação justa entre ANOVA e Random Forest no MVP, definimos as seguintes métricas:

* **Cenário de Poder (H₀ Falsa):** Mediremos a capacidade de detectar um efeito real usando **Acurácia Balanceada**.
* **Cenário de Erro (H₀ Verdadeira):** Mediremos o controle de falsos alarmes usando a **Taxa de Erro Tipo I** (para ANOVA) e a **Taxa de Falsos Alarmes** (para Random Forest).