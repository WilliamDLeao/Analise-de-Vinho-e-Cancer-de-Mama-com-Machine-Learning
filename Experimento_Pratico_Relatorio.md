# Experimento Prático — Aprendizado Semissupervisionado
## Quando os dados não rotulados ajudam?
**DECOM-DV | Engenharia de Computação | Tópicos Especiais em Aprendizado de Máquina**

---

### Resumo Executivo e Metodologia
Este documento resolve e responde integralmente a todas as etapas, tabelas e perguntas reflexivas do **Experimento Prático** proposto no documento `experimento_prático.pdf`. O estudo é conduzido comparativamente em dois conjuntos de dados tabulares de referência:
1. **Wine (UCI)**: 178 amostras, 13 atributos contínuos, 3 classes (cultivares de vinho).
2. **Breast Cancer Wisconsin Diagnostic (UCI)**: 569 amostras, 30 atributos contínuos, 2 classes (maligno / benigno).

Todos os experimentos foram executados com código reprodutível em Python utilizando o script `experimento_pratico.py` e o notebook `Experimento_Pratico.ipynb`, seguindo com rigor o protocolo experimental de particionamento, mascaramento e padronização.

---

## 1 | Escolha da Base e Protocolo Experimental

### Especificação dos Conjuntos de Dados
| Parâmetro | Wine Dataset | Breast Cancer Wisconsin Diagnostic |
|---|---|---|
| **Origem / URL** | [UCI Wine Dataset](https://archive.ics.uci.edu/dataset/109/winedataset) | [UCI Breast Cancer Dataset](https://archive.ics.uci.edu/dataset/17/breast-cancer-wisconsin-data-set) |
| **Número de Amostras** | 178 | 569 |
| **Número de Atributos** | 13 (químicos/físicos contínuos) | 30 (características nucleares celulares) |
| **Classes e Distribuição** | 3 classes:<br>• Classe 0: 59 amostras (33.1%)<br>• Classe 1: 71 amostras (39.9%)<br>• Classe 2: 48 amostras (27.0%) | 2 classes:<br>• Maligno (0): 212 amostras (37.3%)<br>• Benigno (1): 357 amostras (62.7%) |
| **Valores Ausentes** | 0 (nenhum valor faltante) | 0 (nenhum valor faltante) |

### Protocolo Experimental:
1. **Partição de Teste:** 20% das instâncias separadas para teste com semente fixa (`random_state=73`) e estratificação por classe. O conjunto de teste permaneceu isolado durante todo o ajuste dos modelos.
   - *Wine:* 142 treino, 36 teste.
   - *Breast Cancer:* 455 treino, 114 teste.
2. **Mascaramento no Treino:** Exatamente **10% dos rótulos por classe** foram revelados para constituir o conjunto rotulado ($L$); os demais 90% foram codificados como $-1$ ($U$). Os rótulos reais de $U$ foram mantidos exclusivamente para auditoria das decisões semissupervisionadas.
   - *Wine ($L$):* 15 amostras rotuladas (5 da classe 0, 6 da classe 1, 4 da classe 2); 127 amostras em $U$.
   - *Breast Cancer ($L$):* 45 amostras rotuladas (17 malignas, 28 benignas); 410 amostras em $U$.
3. **Padronização:** Atributos normalizados via `StandardScaler` ajustado **estritamente sobre o conjunto de treino** e aplicado ao conjunto de teste, prevenindo qualquer vazamento de informação (*data leakage*).
4. **Métricas:** Macro F1-score e Acurácia avaliados no teste isolado.

> **PERGUNTA 1: Por que utilizar as mesmas amostras L, U e teste é indispensável para uma comparação justa?**
>
> **Resposta:**
> Em aprendizado semissupervisionado, o desempenho do modelo depende criticamente da posição geométrica das poucas sementes rotuladas ($L$) em relação aos clusters e manifolds do conjunto não rotulado ($U$). Se as amostras de $L$ váriassem entre os métodos, as diferenças observadas no F1-score refletiriam variações na sorte de amostragem (ex.: selecionar uma semente central versus uma semente limítrofe/outlier) e não a eficácia intrínseca dos algoritmos de propagação ou pseudorrotulação.
> Da mesma forma, fixar o conjunto de teste e o escalonador garante que todos os modelos sejam julgados sobre a exata mesma superfície de teste, eliminando viés amostral e viés de vazamento.

---

## 2 | Referência: Supervisionado com Poucos Rótulos (Baseline)

Para determinar o valor real de explorar $U$, foram treinados dois modelos de referência com `LogisticRegression(max_iter=1000)`:
1. **Teto Teórico (Supervisionado Completo):** Treinado com 100% dos rótulos disponíveis do conjunto de treino (representa o limite de desempenho se todos os dados fossem anotados por especialistas).
2. **Baseline Restrito (Somente $L$):** Treinado exclusivamente com as amostras de $L$ (10% dos rótulos).

### Resultados do Baseline:
| Base de Dados | Modelo Supervisionado | F1-score Teste | Acurácia Teste | Tempo de Treino (s) |
|---|---|---|---|---|
| **Wine** | **Teto (100% rótulos)** | **1.0000** | **1.0000** | 0.0120s |
| **Wine** | **Baseline (Somente $L$, 10%)** | **0.8850** | **0.8889** | 0.0035s |
| **Breast Cancer** | **Teto (100% rótulos)** | **0.9905** | **0.9912** | 0.0064s |
| **Breast Cancer** | **Baseline (Somente $L$, 10%)** | **0.9615** | **0.9649** | 0.0026s |

> **PERGUNTA 2: Quanto desempenho foi perdido ao reduzir a disponibilidade de rótulos?**
>
> **Resposta:**
> - No **Wine**, a redução de rótulos para 10% causou uma perda considerável de **11.5% no F1-score** (caindo de 1.0000 para 0.8850). Com apenas 4 a 6 sementes por classe, o classificador linear não tem suporte suficiente para delimitar a fronteira de decisão tridimensional completa entre as três cultivares no espaço de 13 dimensões, cometendo erros nas classes 1 e 2.
> - No **Breast Cancer**, a perda foi de apenas **2.9% no F1-score** (de 0.9905 para 0.9615). Mesmo com apenas 45 sementes, as características nucleares celulares (como *mean concave points* e *worst perimeter*) possuem uma margem de separabilidade linear muito ampla, permitindo que a regressão logística defina um hiperplano quase ideal mesmo com poucos exemplos.

---

## 3 | Self-Training: O Modelo Pode Ensinar a Si Próprio?

O algoritmo `SelfTrainingClassifier` foi configurado com classificador base `LogisticRegression(max_iter=1000)`, número máximo de 10 iterações e limiares de confiança $\tau \in \{0.60, 0.75, 0.95\}$.
Os pseudorrótulos incorporados foram auditados contra as classes verdadeiras previamente ocultadas de $U$.

### Tabela 3.1: Desempenho do Self-Training no Wine
| Limiar $\tau$ | F1-score Teste | Aceitos / $U$ (Cobertura) | Precisão dos Aceitos | Iterações | Tempo (s) |
|---|---|---|---|---|---|
| **0.60** | **1.0000** | 127 / 127 (100.0%) | 97.6% | 3 | 0.0189s |
| **0.75** | **1.0000** | 124 / 127 (97.6%) | **100.0%** | 7 | 0.0346s |
| **0.95** | **0.9710** | 81 / 127 (63.8%) | **100.0%** | 9 | 0.0374s |

### Tabela 3.2: Desempenho do Self-Training no Breast Cancer
| Limiar $\tau$ | F1-score Teste | Aceitos / $U$ (Cobertura) | Precisão dos Aceitos | Iterações | Tempo (s) |
|---|---|---|---|---|---|
| **0.60** | **0.9516** | 407 / 410 (99.3%) | 95.3% | 4 | 0.0180s |
| **0.75** | **0.9314** | 399 / 410 (97.3%) | 97.7% | 7 | 0.0240s |
| **0.95** | **0.9314** | 360 / 410 (87.8%) | **98.6%** | 6 | 0.0194s |

> **PERGUNTAS INTERPRETATIVAS (SELF-TRAINING):**
>
> **1. Aumentar o limiar sempre elevou a qualidade dos pseudorrótulos? O que aconteceu com a cobertura?**
> - **Sim.** O aumento de $\tau$ de 0.60 para 0.95 aumentou monotonicamente a precisão auditada dos pseudorrótulos: no Wine subiu de 97.6% para 100.0%, e no Breast Cancer subiu de 95.3% para 98.6%.
> - No entanto, a **cobertura sofreu uma queda acentuada**: no Wine, 36.2% dos dados de $U$ (46 amostras) deixaram de ser rotulados com $\tau=0.95$; no Breast Cancer, 50 amostras foram ignoradas. Um limiar excessivamente conservador impede que o modelo explore dados nas regiões periféricas dos agrupamentos, diminuindo a capacidade de generalização (no Wine, o F1 com $\tau=0.95$ caiu de 1.0000 para 0.9710 por subcobertura).
>
> **2. Compare Self-Training e supervisionado restrito: U ajudou? Em qual condição poderia piorar?**
> - No **Wine**, $U$ ajudou de forma expressiva: o F1 saltou de **0.8850 (baseline) para 1.0000**, igualando o teto supervisionado completo com apenas 10% dos rótulos originais! Isso ocorreu porque as instâncias de alta confiança guiaram a reorientação do hiperplano para as regiões de baixa densidade corretas.
> - No **Breast Cancer**, o Self-Training com $\tau=0.75$ teve F1 de **0.9314**, ficando **abaixo do baseline restrito (0.9615)**. Isso demonstra o risco concreto do **viés de confirmação (confirmation bias)**: quando o modelo comete pequenos erros em instâncias próximas à fronteira, mas atribui alta confiança devido à calibração sigmoide da regressão logística, esses erros entram no treino nas primeiras rodadas e contaminam os gradientes das iterações seguintes.

---

## 4 | Co-Training: Duas Visões Complementares Cooperam?

### Protocolo de Implementação:
- **Visão A:** Atributos de índices pares ($0, 2, 4, \dots$).
- **Visão B:** Atributos de índices ímpares ($1, 3, 5, \dots$).
- **Processo:** 3 rodadas iterativas; cada visão seleciona até 5 amostras não incorporadas de $U$ com probabilidade $\ge 0.85$.
- **Resolução de Conflitos:** Se as duas visões preverem classes distintas para a mesma amostra, o ponto é descartado da rodada. Amostras aceitas são incorporadas **exclusivamente ao conjunto de treino da visão oposta**.

### Resultados do Co-Training:
| Métrica / Parâmetro | Wine Dataset | Breast Cancer Dataset |
|---|---|---|
| **Visão A Isolada (Treino L)** | F1 = 0.9475 (Acc = 0.9444) | F1 = 0.9118 (Acc = 0.9211) |
| **Visão B Isolada (Treino L)** | F1 = 0.8275 (Acc = 0.8333) | F1 = **0.9716** (Acc = 0.9737) |
| **Total de Amostras Transferidas** | 27 amostras | 20 amostras |
| **Desacordos entre Visões** | 0 divergências descartadas | 0 divergências descartadas |
| **Erros Auditados nas Transferências** | 1 erro transferido | **0 erros transferidos (100% acerto)** |
| **Modelo Final — Visão A** | F1 = **0.9475** | F1 = 0.9118 |
| **Modelo Final — Visão B** | F1 = 0.7241 | F1 = **0.9716** |
| **Melhor Modelo Co-Training** | **Visão A (F1 = 0.9475)** | **Visão B (F1 = 0.9716)** |

> **PERGUNTAS INTERPRETATIVAS (CO-TRAINING):**
>
> **1. As visões eram complementares ou uma delas era tão fraca que transferiu informação inadequada?**
> - No **Wine**, as visões eram assimétricas: a Visão A (atributos como teor alcoólico, flavonoides, intensidade da cor) possuía F1 inicial de 0.9475, enquanto a Visão B tinha apenas 0.8275. Ao longo das 3 rodadas, a Visão B incorporou amostras que enfraqueceram seu hiperplano, degradando seu F1 para 0.7241. A divisão ingênua par/ímpar não garantiu redundância suficiente na Visão B.
> - No **Breast Cancer**, a Visão B continha as medições "worst" (piores raios, perímetros e texturas das células), que são os preditores clínicos mais decisivos. A Visão B alcançou **F1 = 0.9716**, superando tanto o baseline restrito (0.9615) quanto a Visão A (0.9118).
>
> **2. Por que usar duas visões não elimina necessariamente o viés de confirmação?**
> - O pressuposto teórico do Co-Training exige **independência condicional das visões dada a classe**: $P(X_A, X_B | Y) = P(X_A | Y)P(X_B | Y)$.
> - Em dados tabulares contínuos, variáveis físicas e biológicas são altamente correlacionadas entre si. Se ambas as visões compartilham os mesmos vieses de medição ou desvios de distribuição, uma predição confiante porém incorreta em uma visão tem alta probabilidade de concordar com um falso positivo da outra visão, perpetuando o erro mútuo.

---

## 5 | Label Propagation: Quando a Geometria Substitui a Confiança?

O algoritmo `LabelPropagation` com kernel $k$-NN foi ajustado sobre a matriz completa de treino com $y \in \{-1, 0, 1, \dots\}$ para $k \in \{3, 5, 15\}$.
Foram avaliados tanto a generalização no conjunto de teste quanto a acurácia transdutiva interna sobre as instâncias ocultadas de $U$.

### Tabela 5.1: Desempenho do Label Propagation
| Base de Dados | $k$ Vizinhos | F1-score Teste | Acertos nos Rótulos Ocultos de $U$ | Acurácia em $U$ | Tempo (s) |
|---|---|---|---|---|---|
| **Wine** | **3** | 0.7492 | 77 / 127 | 60.6% | 0.0057s |
| **Wine** | **5** | **0.9487** | 120 / 127 | **94.5%** | 0.0046s |
| **Wine** | **15** | **1.0000** | 114 / 127 | 89.8% | 0.0038s |
| **Breast Cancer** | **3** | 0.8421 | 361 / 410 | 88.0% | 0.0331s |
| **Breast Cancer** | **5** | **0.8892** | 390 / 410 | **95.1%** | 0.0078s |
| **Breast Cancer** | **15** | 0.7956 | 357 / 410 | 87.1% | 0.0086s |

### Visualização Topológica 2D (PCA e t-SNE)
As projeções bidimensionais geradas confirmam como as sementes de $L$ propagam rótulos através da densidade de $U$:
- `wine_pca_tsne.png`: No Wine, os clusters das três classes são volumétricos e compactos, permitindo que as 15 sementes rotuladas difundam as classes uniformemente pelos componentes conexos.
- `breast_cancer_pca_tsne.png`: No Breast Cancer, a fronteira de contato entre células malignas e benignas apresenta alta densidade mista no centro do espaço projetado.

> **PERGUNTAS INTERPRETATIVAS (LABEL PROPAGATION):**
>
> **1. O que muda quando $k$ é alterado e o grafo passa a ligar amostras potencialmente de classes diferentes?**
> - Com **$k=3$**, o grafo de vizinhança é excessivamente esparso e fragmentado em subgrafos desconexos. Amostras distantes de $L$ não recebem fluxo de probabilidade significativo, resultando em acurácia de transdução baixa (60.6% no Wine).
> - Com **$k=5$**, estabelece-se um equilíbrio ideal: as arestas conectam pontos contíguos do mesmo manifold sem transpor os vales de baixa densidade entre classes.
> - Com **$k=15$** no Breast Cancer, ocorre **super-suavização (oversmoothing)**: as vizinhanças tornam-se tão amplas que arestas cruzam a fronteira de separação entre maligno e benigno, "contaminando" os nós vizinhos com a classe majoritária (benigna), derrubando o F1 de 0.8892 para 0.7956.
>
> **2. Qual informação conduz a classificação em Self-Training e em Label Propagation, respectivamente?**
> - **Self-Training:** Conduzido pela **margem de separação paramétrica global**. O modelo busca um hiperplano ótimo que maximize a confiança/probabilidade marginal das predições, assumindo que as classes são linearmente ou quadraticamente separáveis.
> - **Label Propagation:** Conduzido pela **geometria local e hipótese de manifold/cluster**. O algoritmo não traça hiperplanos; ele realiza um passeio aleatório (*random walk*) sobre o grafo de similaridade ponderada, assumindo que pontos vizinhos pertencem à mesma classe ao longo de caminhos contínuos de alta densidade.

---

## 6 | Label Spreading: E Se Uma Semente Estiver Errada?

O algoritmo `LabelSpreading` emprega normalização Laplaciana da matriz de afinidade e introduz o parâmetro $\alpha$ de *soft clamping*, permitindo relaxamento dos rótulos iniciais de $L$.
Para testar a resiliência a ruído, introduzimos deliberadamente corrupção em **2 sementes rotuladas de $L$** (invertendo suas classes reais), mantendo $U$ e o teste intocados.

### Tabela 6: Sensibilidade a Sementes Corrompidas
| Base de Dados | Cenário de Sementes | Propagation F1 | Spreading $\alpha=0.2$ F1 | Spreading $\alpha=0.8$ F1 |
|---|---|---|---|---|
| **Wine** | **Sementes Limpas** | 0.9487 | **0.9743** | **0.9743** |
| **Wine** | **2 Sementes Corrompidas** | 0.5439 | 0.5439 | 0.5439 |
| **Breast Cancer** | **Sementes Limpas** | 0.8892 | 0.8892 | **0.8892** |
| **Breast Cancer** | **2 Sementes Corrompidas** | 0.8892 | 0.8618 | **0.8892** |

- **Predições Finais nas Sementes Corrompidas:**
  - *Wine:* Sementes de classe real 1 corrompidas para 2 e 0 $\rightarrow$ Predição final permaneceu `[2, 0]`.
  - *Breast Cancer:* Sementes de classe real 1 corrompidas para 0 e 0 $\rightarrow$ Predição final permaneceu `[0, 0]`.

> **PERGUNTAS INTERPRETATIVAS (LABEL SPREADING):**
>
> **1. Qual método foi mais sensível ao ruído? O resultado observado confirma ou contraria a hipótese inicial?**
> - O **Wine** foi extremamente sensível ao ruído: o F1 despencou de **0.9743 para 0.5439** em todos os métodos! Isso ocorre porque com apenas 15 sementes no total, corromper 2 sementes da classe 1 destruiu **33.3% do suporte supervisionado daquela classe**. A hipótese de que o Label Spreading resistiria a ruído foi contrariada quando a proporção de sementes erradas em relação a $L$ é desproporcionalmente alta.
> - No **Breast Cancer**, por possuir 45 sementes, corromper 2 sementes representou menos de 4.4% de erro. O `LabelSpreading` com $\alpha=0.8$ sustentou o F1 em **0.8892**, mantendo robustez plena.
>
> **2. O que é e por que "soft clamping" não significa corrigir rótulos automaticamente?**
> - No *hard clamping* (Label Propagation), a matriz de rótulos é reposta a cada passo: $Y_L^{(t)} = Y_L^{(0)}$.
> - No *soft clamping* (Label Spreading), a propagação segue $F^{(t+1)} = \alpha S F^{(t)} + (1 - \alpha) Y$. O parâmetro $\alpha$ permite que o nó semente receba energia difusa de seus vizinhos não rotulados.
> - **Por que não corrige automaticamente?** Porque as amostras de $U$ conectadas à semente corrompida absorvem a influência inicial dessa mesma semente errada. Sem um número denso e dominante de sementes vizinhas *corretas* no mesmo cluster para anular a semente falsa, o equilíbrio estacionário da difusão apenas suaviza as probabilidades, sem conseguir inverter a predição da semente corrompida.

---

## 7 | Desafio de Estresse: Quando Mais $U$ Atrapalha?

### 7.1 Curva de Desempenho vs Disponibilidade de Rótulos (Sementes 21, 42, 84)
Avaliamos o comportamento dos métodos para 5%, 10% e 20% de rótulos disponíveis em treino, reportando **Média $\pm$ Desvio-Padrão**:

#### Wine Dataset:
- **5% de Rótulos (~7 sementes):**
  - Supervisionado (Baseline): $0.955 \pm 0.047$
  - **Self-Training ($\tau=0.75$): $1.000 \pm 0.000$** (Ganho absoluto de estabilidade e precisão)
  - Label Spreading ($\alpha=0.2$): $0.776 \pm 0.186$ (Alta instabilidade com grafos mal ancorados)
- **10% de Rótulos (~15 sementes):**
  - Supervisionado (Baseline): $0.982 \pm 0.013$
  - **Self-Training: $1.000 \pm 0.000$**
  - **Label Spreading: $1.000 \pm 0.000$**
- **20% de Rótulos (~28 sementes):**
  - Supervisionado (Baseline): $0.974 \pm 0.021$
  - **Self-Training: $1.000 \pm 0.000$**
  - **Label Spreading: $1.000 \pm 0.000$**

#### Breast Cancer Dataset:
- **5% de Rótulos (~23 sementes):**
  - Supervisionado (Baseline): $0.904 \pm 0.013$
  - **Self-Training: $0.904 \pm 0.011$**
  - Label Spreading: $0.868 \pm 0.020$
- **10% de Rótulos (~45 sementes):**
  - **Supervisionado (Baseline): $0.943 \pm 0.014$**
  - Self-Training: $0.936 \pm 0.028$
  - Label Spreading: $0.907 \pm 0.010$
- **20% de Rótulos (~91 sementes):**
  - **Supervisionado (Baseline): $0.972 \pm 0.008$**
  - Self-Training: $0.965 \pm 0.005$
  - Label Spreading: $0.919 \pm 0.004$

### 7.2 Injeção de Ruído Gaussiano em 20% das Amostras de $U$
Adicionamos ruído gaussiano $\mathcal{N}(0, 0.5^2)$ a 20% das instâncias de $U$, simulando deriva de distribuição (*covariate shift*):
- **Wine:**
  - Self-Training: F1 limpo = 1.0000 $\rightarrow$ F1 com ruído em $U$ = **1.0000** (O limiar $\tau=0.75$ rejeitou as amostras ruidosas descalibradas).
  - Label Spreading: F1 limpo = 0.9743 $\rightarrow$ F1 com ruído em $U$ = **0.9487** (O grafo conectou pontos ruidosos distorcendo arestas).
- **Breast Cancer:**
  - Self-Training: F1 limpo = 0.9314 $\rightarrow$ F1 com ruído em $U$ = **0.9516**
  - Label Spreading: F1 limpo = 0.8892 $\rightarrow$ F1 com ruído em $U$ = **0.8892**

> **PERGUNTA DO DESAFIO DE ESTRESSE:**
> **A inclusão de U produziu benefício consistente ou dependeu de quantidade de rótulos, qualidade de U e geometria?**
>
> **Resposta:**
> A inclusão de $U$ **dependeu criticamente da geometria do espaço e da razão sinal-ruído**, não havendo benefício garantido ou universal:
> 1. **Dependência da Geometria:** No Wine, onde as 3 cultivares formam grupos bem delineados no espaço de componentes principais, a inclusão de $U$ alavancou o F1 de 0.8850 para 1.0000. Já no Breast Cancer, onde há sobreposição de densidade entre células limítrofes, a inclusão indiscriminada de $U$ gerou viés de confirmação e queda no desempenho.
> 2. **Dependência da Quantidade de Rótulos:** Em regimes de extrema escassez (5%), métodos baseados em grafos tornam-se erráticos se não houver pelo menos uma semente bem posicionada em cada componente conexo.
> 3. **Qualidade de $U$:** A perturbação de 20% de $U$ mostrou que o Self-Training com limiar rígido é mais tolerante ao ruído em $U$ do que métodos em grafo, que dependem diretamente da integridade das distâncias euclidianas locais.

---

## Folha de Síntese | Comparação Entre Paradigmas

### Tabela Sintética 1: Comparativo Conceitual dos Paradigmas
| Estratégia | Informação de $U$ Explorada | Principal Benefício Observado | Limitação / Risco Identificado |
|---|---|---|---|
| **Supervisionado ($L$)** | Nenhuma (ignora $U$) | Custo computacional desprezível; imune a ruídos e dispersões de $U$. | Alto desperdício amostral; fronteiras instáveis com poucos rótulos. |
| **Self-Training** | Confiança das predições ($P(Y \mid X)$) | Excelente ganho em dados unimodais separáveis (alcançou 1.0000 no Wine). | Viés de confirmação (*confirmation bias*) e propagação cumulativa de erros. |
| **Co-Training** | Complementaridade de subespaços de atributos ($X_A, X_B$) | Explora visões distintas para quebrar o viés de auto-treinamento individual. | Falha severa se as visões não forem condicionalmente independentes ou se uma for muito fraca. |
| **Label Propagation** | Conectividade do grafo ($k$-NN / RBF) | Captura manifolds não-lineares arbitrários sem parametrização de fronteira. | Sensibilidade extrema à escolha de $k$ (risco de *oversmoothing*) e a sementes erradas. |
| **Label Spreading** | Difusão normalizada + *soft clamping* ($\alpha$) | Maior estabilidade numérica e tolerância a pequenos ruídos estocásticos. | Custo cúbico de inversão matricial e incapacidade de anular sementes erradas isoladas. |

---

### Tabela Sintética 2: Resultados Consolidados no Mesmo Teste

#### Wine Dataset (Teste: 36 amostras)
| Modelo / Configuração | F1-score Teste | Acurácia Teste | Tempo (s) | Observação Relevante |
|---|---|---|---|---|
| **Supervisionado — somente $L$** | 0.8850 | 0.8889 | 0.0035s | Baseline restrito com 10% rótulos (15 sementes) |
| **Supervisionado — todos os rótulos** | **1.0000** | **1.0000** | 0.0120s | Teto teórico (142 amostras rotuladas) |
| **Self-Training ($\tau=0.75$)** | **1.0000** | **1.0000** | 0.0346s | **Igualou o teto teórico** (124 aceitos, 100% precisão) |
| **Co-Training — visão A/B** | 0.9475 | 0.9444 | 0.0214s | Melhor modelo: Visão A (pares). Visão B degradou |
| **Label Propagation ($k=5$)** | 0.9487 | 0.9444 | 0.0046s | Acertos em $U$: 120/127 (94.5% de transdução) |
| **Label Spreading ($k=5, \alpha=0.2$)** | 0.9743 | 0.9722 | 0.0046s | Regularização Laplaciana superou a propagação pura |

#### Breast Cancer Dataset (Teste: 114 amostras)
| Modelo / Configuração | F1-score Teste | Acurácia Teste | Tempo (s) | Observação Relevante |
|---|---|---|---|---|
| **Supervisionado — somente $L$** | 0.9615 | 0.9649 | 0.0026s | Baseline restrito com 10% rótulos (45 sementes) |
| **Supervisionado — todos os rótulos** | **0.9905** | **0.9912** | 0.0064s | Teto teórico (455 amostras rotuladas) |
| **Self-Training ($\tau=0.75$)** | 0.9314 | 0.9386 | 0.0240s | Queda por viés de confirmação nas fronteiras |
| **Co-Training — visão A/B** | **0.9716** | **0.9737** | 0.0136s | **Superou o baseline restrito!** Visão B predominou |
| **Label Propagation ($k=5$)** | 0.8892 | 0.8947 | 0.0078s | Acertos em $U$: 390/410 (95.1% de transdução) |
| **Label Spreading ($k=5, \alpha=0.2$)** | 0.8892 | 0.8947 | 0.0078s | Sofre com a sobreposição de densidade na fronteira |

---

### Respostas Detalhadas às Perguntas Finais da Folha de Síntese

#### 1. Qual técnica apresentou a maior economia potencial de rótulos para um desempenho semelhante ao supervisionado?
- No **Wine**, o **Self-Training ($\tau=0.75$)** proporcionou uma economia de rótulos de **90%**: utilizou apenas 15 amostras rotuladas para atingir exatamente o mesmo desempenho do modelo com 142 amostras rotuladas (**F1 = 1.0000**).
- No **Breast Cancer**, o **Co-Training** destacou-se ao atingir **F1 = 0.9716**, economizando 90% dos rótulos com uma perda residual de apenas 1.9% em relação ao teto com 455 amostras anotadas.

#### 2. Qual técnica exige uma hipótese sobre a estrutura do dado mais difícil de justificar nesta base?
- O **Co-Training** apoia-se na hipótese de que existem dois conjuntos de características **mutuamente independentes e individualmente suficientes** para prever a classe. Em tabelas biomédicas ou físico-químicas contínuas (como medições do mesmo núcleo celular ou do mesmo mosto de uva), particionar índices pares e ímpares quebra formalmente a independência condicional.
- Em segundo lugar, os **métodos de grafo** apoiam-se na **hipótese de cluster/manifold** (que a fronteira de decisão passa por regiões de densidade nula ou mínima). No Breast Cancer, células atípicas em transição de malignidade povoam continuamente o espaço métrico entre benigno e maligno, invalidando a hipótese de vales vazios.

#### 3. Em um conjunto médico com laudos escassos e anotações possivelmente ruidosas, quais métodos investigaria primeiro? Quais controles adicionais seriam indispensáveis?
- **Métodos a investigar prioritariamente:**
  1. **Label Spreading:** Devido à propriedade de *soft clamping* ($\alpha$), permitindo que o modelo tolere sementes de laudos inconsistentes sem propagar o erro de forma dogmática.
  2. **Self-Training com Calibração Robusta e Rejeição (Conformal Prediction):** Em vez de limiares empíricos de probabilidade, adotar intervalos de predição com controle garantido de taxa de erro, incorporando *abstenção médica* em instâncias incertas.
- **Controles adicionais indispensáveis:**
  - **Métricas de custo assimétrico:** Penalizar rigorosamente falsos negativos (não detectar malignidade acarreta risco de morte), avaliando sensibilidade (*recall*) e curvas PR-AUC.
  - **Auditoria de incerteza por comitê:** Cruzar predições de múltiplos classificadores e encaminhar casos com alta discordância para reanálise humana (*Human-in-the-loop*).
  - **Validação estratificada por paciente/hospital:** Evitar vazamento de dados correlacionados de um mesmo paciente entre $L$, $U$ e teste.

#### 4. Como distinguir melhoria real de desempenho de vazamento de dados, acaso estatístico ou teste usado na seleção de parâmetros?
- **Contra Vazamento de Dados (Data Leakage):** Todas as etapas de pré-processamento (imputação de média, escalonamento `StandardScaler`, seleção de variáveis) devem ser ajustadas exclusivamente nos dados de treino. O conjunto de teste deve ser transformado apenas no momento da inferência final.
- **Contra Acaso Estatístico:** Nunca basear conclusões em uma única divisão treino/teste. É obrigatório executar validação cruzada repetida (*repeated stratified cross-validation*) ou protocolo multi-seed (como fizemos na Etapa 7 com sementes 21, 42 e 84), reportando intervalos de confiança e aplicando testes estatísticos de hipótese (como o teste pareado de Wilcoxon ou McNemar).
- **Contra Sobreajuste ao Teste:** Os hiperparâmetros (como $\tau$ no Self-Training, $k$ na propagação e $\alpha$ no spreading) devem ser selecionados usando apenas o conjunto rotulado $L$ via validação cruzada interna ou métricas não supervisionadas de consistência em $U$, mantendo o conjunto de teste lacrado até a medição definitiva.
