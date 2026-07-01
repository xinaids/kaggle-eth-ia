# Detecção de Contas Fraudulentas na Rede Ethereum

Projeto de Mineração de Dados aplicando a metodologia CRISP-DM para
classificar contas da rede Ethereum como fraudulentas ou legítimas, a
partir de atributos agregados do histórico de transações.

Trabalho desenvolvido para a disciplina de Mineração de Dados do curso
de Bacharelado em Ciência da Computação do IFRS — Campus Ibirubá.

Autor: Mateus Medeiros Schneider

## Resumo

Foram comparados seis algoritmos de classificação (Regressão Logística,
Árvore de Decisão, Random Forest, Naive Bayes, KNN e SVM) por meio de
validação cruzada estratificada de 5 dobras. Após a otimização de
hiperparâmetros via GridSearchCV, o Random Forest obteve o melhor
desempenho, com **F1-score de 0,9861** e **AUC-ROC de 0,9993** no
conjunto de teste, com apenas 1 falso positivo em 1.916 contas
legítimas avaliadas.

## Dataset

Ethereum Fraud Detection Dataset (Kaggle):
https://www.kaggle.com/datasets/vagifa/ethereum-frauddetection-dataset

9.841 contas da rede Ethereum, descritas por 50 atributos de
comportamento transacional agregado (número de transações, valores
movimentados, tempo entre transações, atividade com tokens ERC20,
etc.), classificadas como legítimas (77,86%) ou fraudulentas (22,14%).

O dataset **não está incluído** neste repositório. Baixe o arquivo CSV
da plataforma e salve-o na raiz do projeto como `address_data_k.csv`.

## Como executar

Instale as dependências:

```bash
pip install -r requirements.txt
```

Execute os scripts na ordem (cada um depende dos resultados do anterior):

```bash
python preprocessing.py          # limpa os dados e gera os conjuntos de treino/teste
python modeling.py               # compara os seis algoritmos via validação cruzada
python hyperparameter_tuning.py  # otimiza os dois melhores modelos
python evaluation.py             # avalia no conjunto de teste
python visualizations.py         # gera as figuras do artigo
```

## Estrutura do projeto

| Arquivo | Descrição |
|---|---|
| `preprocessing.py` | Limpeza, tratamento de ausentes, encoding e divisão treino/teste |
| `modeling.py` | Comparação dos seis algoritmos por validação cruzada |
| `hyperparameter_tuning.py` | Otimização via GridSearchCV |
| `evaluation.py` | Avaliação final e análise de atributos |
| `visualizations.py` | Geração das figuras |
| `requirements.txt` | Dependências Python |

## Etapas do processo (CRISP-DM)

1. **Compreensão dos dados** — 9.841 contas, 50 atributos, classe
   desbalanceada (78% legítimas / 22% fraude). Identificados também
   valores ausentes concentrados em atributos ERC20, colunas de
   variância zero, uma coluna duplicada e ~9,9% das contas com saldo
   total de Ether negativo (inconsistência preservada por estar
   associada sistematicamente à classe).
2. **Preparação dos dados** — remoção de colunas irrelevantes,
   unificação das representações de "sem atividade ERC20", redução de
   cardinalidade de tokens (300+ categorias → 10 + "Other"),
   transformação logarítmica assinada nos atributos monetários,
   one-hot encoding, split estratificado 75/25 e padronização
   (StandardScaler ajustado apenas no treino, evitando data leakage).
3. **Modelagem** — comparação dos seis algoritmos via validação
   cruzada estratificada de 5 dobras, usando F1-score como métrica
   principal devido ao desbalanceamento de classes.
4. **Otimização** — GridSearchCV para os dois modelos mais relevantes:
   Random Forest (melhor desempenho bruto) e Regressão Logística
   (maior interpretabilidade via coeficientes).
5. **Avaliação** — métricas no conjunto de teste (nunca visto nas
   etapas anteriores) e interpretação dos atributos mais relevantes
   por meio de importância de atributos (Random Forest) e coeficientes
   (Regressão Logística).

## Resultados

### Comparação entre algoritmos (validação cruzada, 5-fold)

| Modelo | F1 | AUC-ROC | Precisão | Recall | Acurácia |
|---|---|---|---|---|---|
| Random Forest | 0,9860 | 0,9994 | 1,0000 | 0,9725 | 0,9939 |
| Árvore de Decisão | 0,9629 | 0,9752 | 0,9656 | 0,9602 | 0,9836 |
| KNN | 0,9572 | 0,9881 | 0,9798 | 0,9357 | 0,9814 |
| SVM (linear) | 0,9569 | 0,9911 | 0,9358 | 0,9792 | 0,9805 |
| Regressão Logística | 0,9452 | 0,9943 | 0,9202 | 0,9718 | 0,9751 |
| Naive Bayes | 0,5139 | 0,8970 | 0,3507 | 0,9657 | 0,5927 |

### Modelos otimizados (conjunto de teste)

| Modelo | F1-Score | AUC-ROC | AUC-PR |
|---|---|---|---|
| Random Forest (otimizado) | 0,9861 | 0,9993 | 0,9978 |
| Regressão Logística (otimizada) | 0,9597 | 0,9950 | 0,9885 |

O Random Forest classificou corretamente 1.915 das 1.916 contas
legítimas (apenas 1 falso positivo) e identificou 531 das 545 contas
fraudulentas (14 falsos negativos).

### Principais atributos identificados

Os dois modelos convergem para padrões consistentes: contas
fraudulentas tendem a ter atividade transacional mais restrita,
sem diversidade de tokens ERC20, e menor volume total de transações
— comportamento característico de "carteiras de passagem" usadas em
golpes e descartadas em seguida.

## Limitações

- O dataset descreve comportamento agregado, sem a sequência temporal
  das transações.
- Os rótulos refletem fraudes já conhecidas; golpes com padrões novos
  podem não ser detectados.
- O Naive Bayes se mostrou inadequado para este conjunto de dados,
  por violar suas premissas de independência e normalidade em
  atributos mistos e correlacionados.

## Trabalhos futuros

- Incorporação de atributos temporais e de análise de grafo entre
  endereços.
- Validação dos modelos com dados mais recentes, a fim de verificar
  robustez frente à evolução das estratégias de fraude.