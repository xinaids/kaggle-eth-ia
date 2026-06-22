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
validação cruzada estratificada. Após a otimização de hiperparâmetros,
o Random Forest obteve o melhor desempenho, com F1-score de 0,9832 e
AUC-ROC de 0,9993 no conjunto de teste.

## Dataset

Ethereum Fraud Detection Dataset (Kaggle):
https://www.kaggle.com/datasets/vagifa/ethereum-frauddetection-dataset

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
| `artigo.tex` | Artigo científico (formato SBC) |
| `requirements.txt` | Dependências Python |

## Etapas do processo (CRISP-DM)

1. **Compreensão dos dados** — 9.841 contas, 50 atributos, classe
   desbalanceada (78% legítimas / 22% fraude).
2. **Preparação dos dados** — remoção de colunas irrelevantes,
   padronização de valores ausentes, redução de cardinalidade,
   transformação logarítmica e padronização.
3. **Modelagem** — comparação de seis algoritmos e otimização dos dois
   melhores.
4. **Avaliação** — métricas no conjunto de teste e interpretação dos
   atributos mais relevantes.
