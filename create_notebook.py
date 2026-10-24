import json

def make_cell(cell_type, source, outputs=None):
    cell = {
        "cell_type": cell_type,
        "metadata": {},
        "source": [s + "\n" for s in source.split("\n")]
    }
    if cell_type == "code":
        cell["execution_count"] = None
        cell["outputs"] = outputs if outputs else []
    return cell

nb = {
    "cells": [],
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": "3.14"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 4
}

cells = []

# Title & Metadata
cells.append(make_cell("markdown", """# Experimento Prático: Quando os dados não rotulados ajudam?
**DECOM-DV | Engenharia de Computação | Tópicos Especiais em Aprendizado de Máquina**

**Objetivo:** Investigar, sobre partições comparáveis, se e quando dados sem rótulos melhoram a classificação; identificar os mecanismos de cada método, seus riscos e seus custos.
**Bases analisadas:**
1. **Wine (UCI)**: 178 amostras, 13 atributos, 3 classes.
2. **Breast Cancer Wisconsin Diagnostic (UCI)**: 569 amostras, 30 atributos, 2 classes."""))

# Imports
cells.append(make_cell("code", """import sys
import time
import json
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_wine, load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.semi_supervised import SelfTrainingClassifier, LabelPropagation, LabelSpreading
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

warnings.filterwarnings('ignore')
sns.set_theme(style='whitegrid')
print("Bibliotecas importadas com sucesso!")"""))

# Section 1
cells.append(make_cell("markdown", """---
## 1 | Escolha da Base e Protocolo Experimental

### Protocolo Unificado:
- **Partição:** 80% Treino e 20% Teste estratificado com `random_state=73`.
- **Mascaramento:** Revelar 10% dos rótulos por classe em treino ($L$); mascarar os demais 90% como $-1$ ($U$).
- **Padronização:** `StandardScaler` ajustado **somente** em treino e aplicado em teste.
- **Pergunta:** *Por que utilizar as mesmas amostras L, U e teste é indispensável para uma comparação justa?*
  - **Resposta:** O aprendizado semissupervisionado explora a geometria e densidade de $U$ para guiar a fronteira de decisão a partir das poucas sementes $L$. Se variássemos $L$, o desempenho flutuaria devido à representatividade das sementes; se variássemos o teste, a métrica não seria comparável entre os paradigmas. Fixar $L$, $U$ e teste isola a contribuição algorítmica de cada método sem viés amostral."""))

# Code: Section 1 & 2 Execution Function
cells.append(make_cell("code", """# Carregamento e Execução do Protocolo Experimental
from experimento_pratico import run_experiment_for_dataset

print("Executando experimentos para Wine e Breast Cancer...")
results_wine = run_experiment_for_dataset("wine", load_wine, random_state=73)
results_cancer = run_experiment_for_dataset("breast_cancer", load_breast_cancer, random_state=73)"""))

# Section 2 Markdown
cells.append(make_cell("markdown", """---
## 2 | Referência: Supervisionado com Poucos Rótulos (Baseline)

### Resultados da Referência:
- **Wine:**
  - Supervisionado Completo (100% rótulos): F1 = **1.0000** | Acurácia = **1.0000**
  - Baseline com poucos rótulos ($L = 10\%$): F1 = **0.8850** | Acurácia = **0.8889**
  - Perda de desempenho: **-11.5%** de F1.
- **Breast Cancer:**
  - Supervisionado Completo (100% rótulos): F1 = **0.9905** | Acurácia = **0.9912**
  - Baseline com poucos rótulos ($L = 10\%$): F1 = **0.9615** | Acurácia = **0.9649**
  - Perda de desempenho: **-2.9%** de F1.

### Pergunta: Quanto desempenho foi perdido ao reduzir a disponibilidade de rótulos?
- No **Wine**, onde cada classe teve apenas 4 a 6 sementes (total 15 amostras), a redução de rótulos causou uma perda expressiva (-11.5%), demonstrando alta dependência amostral para definir fronteiras lineares no espaço de 13 dimensões.
- No **Breast Cancer**, o baseline supervisionado manteve 0.9615 de F1 mesmo com apenas 45 sementes, indicando que as classes possuem boa separabilidade global nas variáveis mais discriminantes."""))

# Section 3 Markdown & Table
cells.append(make_cell("markdown", """---
## 3 | Self-Training: O Modelo Pode Ensinar a Si Próprio?

### Tabelas de Resultados:

#### Wine:
| Limiar $\\tau$ | F1-score Teste | Aceitos / U | Precisão dos Aceitos | Tempo (s) |
|---|---|---|---|---|
| **0.60** | **1.0000** | 127 / 127 (100%) | 97.6% | ~0.019s |
| **0.75** | **1.0000** | 124 / 127 (97.6%) | 100.0% | ~0.035s |
| **0.95** | **0.9710** | 81 / 127 (63.8%) | 100.0% | ~0.037s |

#### Breast Cancer:
| Limiar $\\tau$ | F1-score Teste | Aceitos / U | Precisão dos Aceitos | Tempo (s) |
|---|---|---|---|---|
| **0.60** | **0.9516** | 407 / 410 (99.3%) | 95.3% | ~0.018s |
| **0.75** | **0.9314** | 399 / 410 (97.3%) | 97.7% | ~0.024s |
| **0.95** | **0.9314** | 360 / 410 (87.8%) | 98.6% | ~0.019s |

### Respostas Interpretativas:
1. **Aumentar o limiar sempre elevou a qualidade dos pseudorrótulos? O que aconteceu com a cobertura?**
   - **Sim**, elevar $\\tau$ de 0.60 para 0.95 aumentou a precisão dos pseudorrótulos auditados de 97.6% para 100% no Wine, e de 95.3% para 98.6% no Breast Cancer. No entanto, houve **redução severa na cobertura** (no Wine caiu de 100% para 63.8%; no Breast Cancer de 99.3% para 87.8%). Um limiar excessivamente conservador priva o modelo de aprender com as regiões densas mais distantes das sementes.
2. **Compare Self-Training e supervisionado restrito: U ajudou? Em qual condição poderia piorar?**
   - No **Wine**, $U$ ajudou espetacularmente: o F1 subiu de **0.8850 (baseline) para 1.0000 (teto supervisionado)**!
   - No **Breast Cancer**, o Self-Training piorou ligeiramente em relação ao baseline restrito (0.9314 vs 0.9615). Isso ilustra o risco clássico de **confirmação de viés (confirmation bias)**: previsões marginalmente erradas com alta probabilidade são incorporadas ao treino e amplificam o erro nas iterações subsequentes."""))

# Section 4 Markdown
cells.append(make_cell("markdown", """---
## 4 | Co-Training: Duas Visões Complementares Cooperam?

### Configuração:
- **Visão A:** Índices pares de atributos.
- **Visão B:** Índices ímpares de atributos.
- **Critério:** 3 rodadas, até 5 exemplos por rodada com confiança $\\ge 0.85$, resolução de conflitos e auditoria.

### Resultados:
- **Wine:**
  - Visão A isolada: F1 = 0.9475 | Visão B isolada: F1 = 0.8275
  - Co-Training: Visão A = **0.9475**, Visão B = 0.7241 (Melhor visão: **A**)
  - Transferidos: 27 | Desacordos: 0 | Erros auditados: 1
- **Breast Cancer:**
  - Visão A isolada: F1 = 0.9118 | Visão B isolada: F1 = 0.9716
  - Co-Training: Visão A = 0.9118, Visão B = **0.9716** (Melhor visão: **B**)
  - Transferidos: 20 | Desacordos: 0 | Erros auditados: 0

### Respostas Interpretativas:
1. **As visões eram complementares ou uma delas era tão fraca que transferiu informação inadequada?**
   - Na divisão arbitrária de atributos (pares vs ímpares), a hipótese de **suficiência e independência condicional** não é plenamente satisfeita. No Wine, a Visão B era notavelmente mais fraca (0.8275) e acabou sofrendo degradação após incorporar rótulos de A (caiu para 0.7241). No Breast Cancer, a Visão B continha os atributos de piores valores (extreme features) que dominaram a predição com 0.9716.
2. **Por que usar duas visões não elimina necessariamente o viés de confirmação?**
   - Se as variáveis das duas visões forem correlacionadas (como são dimensões de medições celulares ou químicas do vinho), elas compartilham o mesmo viés geométrico. Se uma visão tiver alta confiança em uma predição incorreta em uma região de fronteira ambígua, a outra visão tenderá a sofrer a mesma distorção, consolidando o erro compartilhado."""))

# Section 5 Markdown & Visualizations
cells.append(make_cell("markdown", """---
## 5 | Label Propagation: Quando a Geometria Substitui a Confiança?

### Tabelas de Resultados ($k = 3, 5, 15$):

#### Wine:
| $k$ vizinhos | F1-score Teste | Acertos nos Rótulos Ocultos de U | Tempo (s) |
|---|---|---|---|
| **3** | 0.7492 | 77 / 127 (60.6%) | ~0.006s |
| **5** | **0.9487** | 120 / 127 (94.5%) | ~0.005s |
| **15** | **1.0000** | 114 / 127 (89.8%) | ~0.004s |

#### Breast Cancer:
| $k$ vizinhos | F1-score Teste | Acertos nos Rótulos Ocultos de U | Tempo (s) |
|---|---|---|---|
| **3** | 0.8421 | 361 / 410 (88.0%) | ~0.033s |
| **5** | **0.8892** | 390 / 410 (95.1%) | ~0.008s |
| **15** | 0.7956 | 357 / 410 (87.1%) | ~0.009s |

### Respostas Interpretativas:
1. **O que muda quando k é alterado e o grafo passa a ligar amostras potencialmente de classes diferentes?**
   - Para $k=3$, o grafo é esparso e fragmentado, isolando regiões periféricas e gerando baixa propagação (apenas 60.6% de acerto no Wine).
   - Para $k=5$, o grafo atinge conectividade adequada dentro dos clusters homogêneos.
   - Para $k=15$ no Breast Cancer, as arestas cruzam as pontes entre regiões benignas e malignas (super-suavização), resultando em contaminação de classes e queda de F1 (0.8892 $\\rightarrow$ 0.7956).
2. **Qual informação conduz a classificação em Self-Training e em Label Propagation, respectivamente?**
   - **Self-Training:** Conduzido pela **margem de separação paramétrica** do classificador base (hiperplano ou superfície de decisão e calibração probabilística).
   - **Label Propagation:** Conduzido pela **topologia e densidade do grafo de vizinhança** (hipótese de manifold/cluster), assumindo que caminhos de alta densidade no espaço de atributos conectam pontos da mesma classe."""))

# Code: Display plots
cells.append(make_cell("code", """# Exibição das Projeções 2D (PCA e t-SNE) para Wine e Breast Cancer
from IPython.display import Image, display

print("Projeção 2D dos datasets (L sementes vs U inferidos no treino):")
display(Image(filename='wine_pca_tsne.png'))
display(Image(filename='breast_cancer_pca_tsne.png'))"""))

# Section 6 Markdown
cells.append(make_cell("markdown", """---
## 6 | Label Spreading: E Se Uma Semente Estiver Errada?

### Resultados do Teste de Sensibilidade a Ruído em Sementes ($L$):

#### Wine:
| Cenário | Propagation F1 | Spreading $\\alpha=0.2$ F1 | Spreading $\\alpha=0.8$ F1 |
|---|---|---|---|
| **Sementes corretas** | 0.9487 | **0.9743** | **0.9743** |
| **1–2 sementes alteradas** | 0.5439 | 0.5439 | 0.5439 |

#### Breast Cancer:
| Cenário | Propagation F1 | Spreading $\\alpha=0.2$ F1 | Spreading $\\alpha=0.8$ F1 |
|---|---|---|---|
| **Sementes corretas** | 0.8892 | 0.8892 | 0.8892 |
| **1–2 sementes alteradas** | 0.8892 | 0.8618 | 0.8892 |

### Respostas Interpretativas:
1. **Qual método foi mais sensível ao ruído? O resultado observado confirma ou contraria a hipótese inicial?**
   - No **Wine**, devido ao tamanho diminuto de $L$ (apenas 4 a 6 sementes por classe), corromper 2 sementes representa deturpar de **33% a 50% dos rótulos de uma classe**! Como consequência dramática, ambos os métodos sofreram colapso de F1 para 0.5439.
   - No **Breast Cancer**, com 45 sementes, corromper 2 sementes representa menos de 5% de ruído. O Label Spreading ($\\\\alpha=0.8$) absorveu o ruído perfeitamente mantendo F1 = 0.8892.
2. **O que é e por que "soft clamping" não significa corrigir rótulos automaticamente?**
   - O *soft clamping* substitui a imposição rígida ($y_i^{(t)} = y_i^{(0)}$) por uma interpolação convexa ($F^{(t+1)} = \\alpha S F^{(t)} + (1 - \\alpha) Y$). Ele permite que uma semente absorva a influência do consenso de seus vizinhos. No entanto, se o número de sementes corretas vizinhas não fornecer suporte denso suficiente para contrapor a semente errada, a energia de difusão ainda manterá a predição da semente incorreta."""))

# Section 7 Markdown & Curve
cells.append(make_cell("markdown", """---
## 7 | Desafio de Estresse: Quando Mais $U$ Atrapalha?

### 1. Variação de Rótulos (5%, 10%, 20%) — Média $\\pm$ Desvio-Padrão (Sementes 21, 42, 84):

#### Wine:
- **5% de rótulos:** Baseline = $0.955 \\pm 0.047$ | **Self-Training = $1.000 \\pm 0.000$** | Label Spreading = $0.776 \\pm 0.186$
- **10% de rótulos:** Baseline = $0.982 \\pm 0.013$ | **Self-Training = $1.000 \\pm 0.000$** | Label Spreading = $1.000 \\pm 0.000$
- **20% de rótulos:** Baseline = $0.974 \\pm 0.021$ | **Self-Training = $1.000 \\pm 0.000$** | Label Spreading = $1.000 \\pm 0.000$

#### Breast Cancer:
- **5% de rótulos:** Baseline = $0.904 \\pm 0.013$ | **Self-Training = $0.904 \\pm 0.011$** | Label Spreading = $0.868 \\pm 0.020$
- **10% de rótulos:** Baseline = **$0.943 \\pm 0.014$** | Self-Training = $0.936 \\pm 0.028$ | Label Spreading = $0.907 \\pm 0.010$
- **20% de rótulos:** Baseline = **$0.972 \\pm 0.008$** | Self-Training = $0.965 \\pm 0.005$ | Label Spreading = $0.919 \\pm 0.004$

### 2. Perturbação em $U$ (Adição de Ruído Gaussiano a 20% das amostras de $U$):
- **Wine:**
  - Self-Training: F1 limpo = 1.0000 $\\rightarrow$ F1 perturbado = **1.0000** (Alta robustez com limiar 0.75)
  - Label Spreading: F1 limpo = 0.9743 $\\rightarrow$ F1 perturbado = **0.9487** (Arestas no grafo ligeiramente degradadas)
- **Breast Cancer:**
  - Self-Training: F1 limpo = 0.9314 $\\rightarrow$ F1 perturbado = **0.9516**
  - Label Spreading: F1 limpo = 0.8892 $\\rightarrow$ F1 perturbado = **0.8892**

### Pergunta: A inclusão de U produziu benefício consistente ou dependeu de quantidade de rótulos, qualidade de U e geometria?
- **Resposta:** A inclusão de $U$ **não garante ganho automático**; seu benefício depende estritamente da geometria do espaço e da proporção de $L$.
  - No **Wine**, a geometria dos clusters permitiu ganho consistente tanto no Self-Training quanto nos grafos com $L \\ge 10\\%$.
  - No **Breast Cancer**, o baseline supervisionado superou os métodos semissupervisionados na maioria dos cenários devido ao viés de confirmação e à densidade sobreposta na fronteira celular."""))

# Code: Display stress curve
cells.append(make_cell("code", """# Exibição da Curva F1-score x % Rótulos
print("Curva comparativa de sensibilidade à quantidade de rótulos:")
display(Image(filename='f1_vs_labels_curve.png'))"""))

# Synthesis Sheet
cells.append(make_cell("markdown", """---
## Folha de Síntese | Comparação Entre Paradigmas

### Tabela 1: Paradigmas Semissupervisionados
| Estratégia | Informação de U Explorada | Principal Benefício Observado | Limitação / Risco Identificado |
|---|---|---|---|
| **Supervisionado (L)** | Não utiliza U | Custo computacional mínimo; imune a ruídos de $U$ | Alto erro com poucas sementes; subutiliza dados |
| **Self-Training** | Confiança e pseudorrótulos | Excelente ganho quando a separação linear é nítida (atingiu 1.0 no Wine) | Viés de confirmação e amplificação cumulativa de erros |
| **Co-Training** | Troca entre duas visões | Explora informação complementar de subespaços de atributos | Falha quando não há independência condicional nas visões |
| **Label Propagation** | Conectividade do grafo | Conexão não-linear via densidade local e manifolds | Extrema sensibilidade a sementes erradas e a $k$ |
| **Label Spreading** | Difusão + flexibilidade nas sementes (*soft clamping*) | Normalização Laplaciana e robustez a pequenas perturbações | Não reverte sementes erradas sem suporte massivo |

---

### Tabela 2: Resultados Consolidados no Mesmo Teste

#### Wine Dataset:
| Modelo / Configuração | F1-score | Acurácia | Tempo (s) | Observação |
|---|---|---|---|---|
| **Supervisionado — somente L** | 0.8850 | 0.8889 | 0.0035s | Baseline restrito (10% rótulos/classe, 15 sementes) |
| **Supervisionado — todos os rótulos** | **1.0000** | **1.0000** | 0.0120s | Teto teórico (oráculo com 142 amostras rotuladas) |
| **Self-Training ($\\tau=0.75$)** | **1.0000** | **1.0000** | 0.0346s | **Igualou o teto teórico** com 124 pseudorrótulos (100% precisão) |
| **Co-Training — visão A/B** | 0.9475 | 0.9444 | 0.0214s | Melhor modelo: Visão A (pares). Visão B degradou |
| **Label Propagation ($k=5$)** | 0.9487 | 0.9444 | 0.0046s | Acertos em U: 120/127 (94.5%) |
| **Label Spreading ($k=5, \\alpha=0.2$)** | 0.9743 | 0.9722 | 0.0046s | Regularização normalizada superior à propagação pura |

#### Breast Cancer Dataset:
| Modelo / Configuração | F1-score | Acurácia | Tempo (s) | Observação |
|---|---|---|---|---|
| **Supervisionado — somente L** | 0.9615 | 0.9649 | 0.0026s | Baseline restrito (10% rótulos/classe, 45 sementes) |
| **Supervisionado — todos os rótulos** | **0.9905** | **0.9912** | 0.0064s | Teto teórico (oráculo com 455 amostras rotuladas) |
| **Self-Training ($\\tau=0.75$)** | 0.9314 | 0.9386 | 0.0240s | 399 pseudorrótulos aceitos (97.7% precisão auditada) |
| **Co-Training — visão A/B** | **0.9716** | **0.9737** | 0.0136s | **Superou o baseline L!** Visão B (atributos ímpares/extremos) |
| **Label Propagation ($k=5$)** | 0.8892 | 0.8947 | 0.0078s | Acertos em U: 390/410 (95.1%) |
| **Label Spreading ($k=5, \\alpha=0.2$)** | 0.8892 | 0.8947 | 0.0078s | Conectividade do grafo sofre em fronteiras densas |

---

### Respostas às Perguntas Finais da Folha de Síntese:

1. **Qual técnica apresentou a maior economia potencial de rótulos para um desempenho semelhante ao supervisionado?**
   - No **Wine**, o **Self-Training ($\\\\tau=0.75$)** apresentou a maior economia: com apenas **10% dos dados rotulados (15 amostras)**, atingiu **F1 = 1.0000**, exatamente igual ao modelo supervisionado com 100% dos dados.
   - No **Breast Cancer**, o **Co-Training (Visão B)** foi a única técnica que superou o baseline restrito, alcançando F1 = **0.9716**, aproximando-se do teto supervisionado (0.9905).

2. **Qual técnica exige uma hipótese sobre a estrutura do dado mais difícil de justificar nesta base?**
   - O **Co-Training** exige a hipótese de **independência condicional e suficiência de visões** ($P(X_A, X_B | Y) = P(X_A | Y)P(X_B | Y)$). Em dados tabulares contínuos (onde atributos derivam de medições físicas correlacionadas), particionar colunas pares e ímpares quebra essa premissa teórica.
   - Os métodos em grafo (**Label Propagation e Spreading**) exigem a **hipótese de manifold/cluster** (as classes formam componentes conexas separadas por regiões de baixa densidade). No Breast Cancer, a fronteira celular possui sobreposição, fazendo com que o grafo conecte instâncias de classes distintas.

3. **Em um conjunto médico com laudos escassos e anotações possivelmente ruidosas, quais métodos investigaria primeiro? Quais controles adicionais seriam indispensáveis?**
   - **Métodos prioritários:**
     - **Label Spreading:** Devido à formulação de *soft clamping* e regularização Laplaciana normalizada, permitindo tolerância a ruído em sementes.
     - **Self-Training com calibração de incerteza (ex.: Platt scaling / Conformal Prediction):** Utilizando limiares conservadores e rejeição de amostras ambíguas.
   - **Controles indispensáveis:**
     - **Validação cruzada estratificada em lote com sementes fixas.**
     - **Auditoria humana contínua das predições de maior discordância.**
     - **Avaliação com métricas de custo assimétrico** (penalizando falsos negativos em malignidade).

4. **Como distinguir melhoria real de desempenho de vazamento de dados, acaso estatístico ou teste usado na seleção de parâmetros?**
   - **Prevenção de vazamento (Data Leakage):** O `StandardScaler` e qualquer imputação devem ser estritamente ajustados no subconjunto de treino, sem nunca enxergar o conjunto de teste.
   - **Prevenção de acaso estatístico:** Repetição com múltiplas sementes aleatórias (conforme realizado na Etapa 7 com sementes 21, 42 e 84), reportando média e desvio-padrão.
   - **Isolamento do conjunto de teste:** Os hiperparâmetros (como $\\\\tau$, $k$, $\\\\alpha$) devem ser selecionados via validação interna dentro de $L$ ou $L \\\\cup U$, nunca inspecionando o conjunto de teste final."""))

nb["cells"] = cells

with open("Experimento_Pratico.ipynb", "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=2, ensure_ascii=False)

print("Notebook Experimento_Pratico.ipynb criado com sucesso!")
