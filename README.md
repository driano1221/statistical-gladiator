# Statistical Gladiator: Uma Análise Comparativa da Robustez de Métodos Estatísticos e de Machine Learning

![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)
![Libraries](https://img.shields.io/badge/Bibliotecas-Pandas%20%7C%20Scikit--Learn%20%7C%20XGBoost-orange.svg)
![License](https://img.shields.io/badge/Licen%C3%A7a-MIT-green.svg)

## Introdução

A prática contemporânea da ciência de dados situa-se na intersecção entre a inferência estatística clássica e os algoritmos de aprendizado de máquina (Machine Learning). Uma questão fundamental, contudo, permanece subexplorada: qual a robustez relativa desses dois paradigmas diante de dados não ideais, frequentemente encontrados em cenários reais?

Este projeto conduz uma investigação sistemática para quantificar e comparar a performance de métodos estatísticos consagrados (e.g., ANOVA) e algoritmos de ML de alta performance (e.g., XGBoost) na tarefa de detecção de diferenças entre grupos. A análise se dá sob um conjunto controlado de condições adversas, como a presença de outliers, assimetria na distribuição dos dados e amostras de tamanho reduzido.

O objetivo central é gerar um mapa de desempenho empírico que sirva como um guia prático para pesquisadores e analistas na seleção da ferramenta mais apropriada para o seu contexto de dados específico.

## Metodologia

O estudo emprega uma simulação de Monte Carlo para gerar e analisar um vasto número de conjuntos de dados sintéticos, permitindo um controle rigoroso sobre as condições experimentais.

###  Delineamento Experimental

O problema fundamental consiste em testar a diferença entre as médias de $k=3$ grupos. A hipótese nula ($H_0$) e a hipótese alternativa ($H_1$) são definidas como:

$$H_0: \mu_1 = \mu_2 = \mu_3$$
$$H_1: \exists \ i,j \ | \ \mu_i \neq \mu_j$$

O modelo de geração de dados base para cada observação $Y_{ij}$ no grupo $i$ e amostra $j$ é:

$$Y_{ij} = \mu_i + \epsilon_{ij}, \quad \text{onde} \quad \epsilon_{ij} \sim N(0, \sigma^2)$$

Um efeito real ($\delta$) foi introduzido para simular a condição de $H_1$ verdadeira, tal que $\mu_i = (i-1)\delta$. O estudo foi conduzido com um total de 1.000 simulações para cada combinação de método e condição.

### 2.2. Fatores do Estudo

A simulação foi estruturada em torno de dois fatores principais: os métodos de análise ("Gladiadores") e as condições dos dados ("Arenas").

| Fator | Níveis | Descrição |
| :--- | :--- | :--- |
| **Gladiador** | ANOVA | Teste F de Análise de Variância. |
| (Método) | Kruskal-Wallis | Teste H, a alternativa não-paramétrica baseada em ranks. |
| | RandomForest | Algoritmo de ensemble baseado em árvores de decisão. |
| | XGBoost | Implementação de Gradient Boosting de alta performance. |
| **Arena** | Baseline Normal | Dados normais e homocedásticos, condição ideal. |
| (Condição) | Outliers (10%) | 10% das amostras de cada grupo substituídas por valores extremos. |
| | Alta Assimetria | Dados gerados a partir de uma distribuição Gama, com forte assimetria à direita. |
| | Amostras Pequenas | Tamanho amostral reduzido para $n=10$ por grupo. |

### 2.3. Modelagem das Condições Adversas

* **Outliers:** A contaminação foi modelada pela substituição de uma proporção `p=0.10` das amostras de cada grupo por novas amostras oriundas de uma distribuição com variância ampliada: $N(\mu_i, \sigma^2 \times k)$, onde $k=5$ é o fator de intensidade.
* **Assimetria:** Dados assimétricos foram gerados a partir da distribuição Gama, $\Gamma(\alpha, \beta)$, e subsequentemente padronizados (Z-score) para manter a média e variância controladas antes da adição do efeito.

## 3. Métricas de Avaliação

Para uma comparação justa entre os paradigmas, as métricas de sucesso foram definidas da seguinte forma:

* **Testes Estatísticos (ANOVA, Kruskal-Wallis):** A métrica de performance é o **Poder Estatístico**, definido como a proporção de simulações em que a hipótese nula foi corretamente rejeitada ($p < 0.05$) quando um efeito real existia.
  
    $$ \text{Poder} = P(\text{rejeitar } H_0 | H_1 \text{ é verdadeira}) $$
  
* **Modelos de Machine Learning (RF, XGBoost):** A métrica é a **Acurácia Balanceada Média**. Ela avalia a capacidade do modelo de classificar corretamente as amostras em seus respectivos grupos de origem e é robusta a desbalanceamentos de classe.

## 4. Resultados

A análise das 16.000 batalhas simuladas produziu um conjunto de resultados claros e consistentes.

### 4.1. Panorama Geral da Performance

O mapa de calor abaixo resume a performance média de cada gladiador em cada arena. Valores mais altos (amarelo) indicam melhor desempenho.

<img width="1313" height="865" alt="heat" src="https://github.com/user-attachments/assets/de5e47f8-612e-4904-b388-5d3e78fba05c" />


A visualização geral já indica a alta performance de Kruskal-Wallis e RandomForest em todos os cenários, em contraste com a vulnerabilidade específica da ANOVA.

### 4.2. Análise de Robustez a Outliers

A condição com outliers revelou a diferença mais dramática de robustez. A análise de sensibilidade abaixo demonstra como o poder estatístico da ANOVA e do Kruskal-Wallis decai à medida que a porcentagem de outliers nos dados aumenta.

<img width="1366" height="865" alt="linhas" src="https://github.com/user-attachments/assets/639e5361-b662-41ab-b3e9-8e624c58e35d" />


Observa-se que o poder da ANOVA sofre uma degradação catastrófica a partir de 10% de contaminação, enquanto o Kruskal-Wallis mantém uma performance quase perfeita mesmo em níveis de contaminação elevados.

### 4.3. Análise de Desempenho com Amostras Pequenas

A arena com amostras de tamanho reduzido ($n=10$) testou a consistência dos métodos. O gráfico de boxplot ilustra a distribuição da performance de cada gladiador ao longo das 1.000 simulações.

<img width="1366" height="865" alt="boxplot" src="https://github.com/user-attachments/assets/e8d20a28-60a1-4e7d-bcac-341770b6c728" />

Enquanto ANOVA, Kruskal-Wallis e RandomForest apresentaram performance máxima e com variância nula, o XGBoost demonstrou não apenas uma média de acurácia inferior, mas também uma variabilidade de resultados significativamente maior, indicando menor confiabilidade neste cenário.

## 5. Discussão e Conclusões

Os resultados agregados permitem a formulação de um ranking geral de robustez, calculado pela média de performance de cada método em todas as quatro arenas.

<img width="1165" height="666" alt="pontuação" src="https://github.com/user-attachments/assets/34bf36c8-cefd-4a13-b7a7-bfb29ab3d4b6" />

As conclusões deste estudo são:

1.  **Os Campeões da Robustez:** Kruskal-Wallis e RandomForest emergiram como os métodos mais versáteis e robustos, apresentando desempenho quase perfeito em todos os cenários testados. A escolha entre eles pode ser guiada pelo objetivo da análise (inferência estatística vs. predição).

2.  **A Vulnerabilidade da ANOVA:** O pressuposto de normalidade da ANOVA, embora robusto a violações de simetria, é extremamente sensível à presença de outliers, que podem inflar a variância e mascarar efeitos reais. Seu uso exige cautela e uma análise exploratória rigorosa dos dados.

3.  **O Contexto para Algoritmos Complexos:** O desempenho relativamente inferior do XGBoost, especialmente com amostras pequenas, sugere que algoritmos de alta complexidade não são universalmente superiores. Em problemas de baixa dimensionalidade e com dados limitados, a sua capacidade pode ser subaproveitada ou mesmo prejudicial se não houver ajuste fino de hiperparâmetros.

Em suma, a escolha da ferramenta analítica correta depende criticamente das características dos dados. Este estudo fornece um guia empírico para auxiliar nessa decisão, demonstrando que tanto os métodos estatísticos robustos quanto os algoritmos de ensemble de Machine Learning representam escolhas excelentes para a análise de dados imperfeitos.

## 6. Como Reproduzir o Estudo

Para replicar os resultados desta análise, siga os passos abaixo:

1.  Clone o repositório:
    ```bash
    git clone [https://github.com/SEU_USUARIO/statistical-gladiator.git](https://github.com/SEU_USUARIO/statistical-gladiator.git)
    cd statistical-gladiator
    ```
2.  Crie e ative um ambiente virtual:
    ```bash
    python -m venv venv
    # Windows: .\venv\Scripts\activate
    # Mac/Linux: source venv/bin/activate
    ```
3.  Instale as dependências:
    ```bash
    pip install -r requirements.txt
    ```
4.  Execute a simulação completa (pode levar vários minutos):
    ```bash
    python main_runner.py
    ```
5.  Execute a análise e visualize os resultados:
    ```bash
    jupyter lab analysis_phase2.ipynb
    ```
