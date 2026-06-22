"""
preprocessing.py

Pipeline de pré-processamento para o dataset Ethereum Fraud Detection.
Implementa a etapa "Preparação dos Dados" do processo CRISP-DM/KDD.

Decisões de pré-processamento documentadas:
  1. Remoção de identificadores e colunas com variância zero
  2. Padronização de representações inconsistentes de "ausência de
     atividade ERC20" (NaN, '0' string e string vazia tratados como
     uma única categoria)
  3. Redução de cardinalidade das colunas categóricas de token ERC20
     (top 10 tokens + categoria 'Other' + categoria 'None')
  4. Transformação logarítmica assinada para atributos monetários
     fortemente assimétricos (preserva sinal de valores negativos)
  5. Codificação one-hot das categóricas reduzidas
  6. Divisão treino/teste estratificada (preserva proporção de fraude)
  7. Padronização (StandardScaler) ajustada apenas no treino
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# Colunas identificadas na análise exploratória como tendo variância zero
ZERO_VARIANCE_COLS = [
    ' ERC20 avg time between sent tnx',
    ' ERC20 avg time between rec tnx',
    ' ERC20 avg time between rec 2 tnx',
    ' ERC20 avg time between contract tnx',
    ' ERC20 min val sent contract',
    ' ERC20 max val sent contract',
    ' ERC20 avg val sent contract',
]

# Identificadores que não generalizam para um modelo preditivo
ID_COLS = ['Unnamed: 0', 'Index', 'Address']

# Coluna duplicada por erro no cabeçalho original do CSV
DUPLICATED_COL = ' ERC20 uniq sent addr.1'

# Colunas categóricas de alta cardinalidade (tipo de token ERC20)
TOKEN_COLS = [' ERC20 most sent token type', ' ERC20_most_rec_token_type']

# Colunas monetárias/contagens com forte assimetria (aplicar log)
# Identificado na análise: skew entre 1.8 e 62.4
SKEWED_COLS = [
    'Avg min between sent tnx', 'Avg min between received tnx',
    'Time Diff between first and last (Mins)', 'Sent tnx', 'Received Tnx',
    'Number of Created Contracts', 'Unique Received From Addresses',
    'Unique Sent To Addresses', 'min value received', 'max value received ',
    'avg val received', 'min val sent', 'max val sent', 'avg val sent',
    'min value sent to contract', 'max val sent to contract',
    'avg value sent to contract',
    'total transactions (including tnx to create contract',
    'total Ether sent', 'total ether received', 'total ether sent contracts',
    'total ether balance',
    ' Total ERC20 tnxs', ' ERC20 total Ether received',
    ' ERC20 total ether sent', ' ERC20 total Ether sent contract',
    ' ERC20 uniq sent addr', ' ERC20 uniq rec addr',
    ' ERC20 uniq rec contract addr', ' ERC20 min val rec', ' ERC20 max val rec',
    ' ERC20 avg val rec', ' ERC20 min val sent', ' ERC20 max val sent',
    ' ERC20 avg val sent',
]

TARGET_COL = 'FLAG'
TOP_N_TOKENS = 10


def load_data(path):
    """
    Carrega o dataset bruto a partir de um arquivo CSV.

    Parâmetros:
        path (str): caminho para o arquivo CSV original.

    Retorna:
        pd.DataFrame: dataset bruto, sem nenhuma modificação.
    """
    return pd.read_csv(path)


def drop_irrelevant_columns(df):
    """
    Remove colunas que não agregam valor preditivo ou que são
    redundantes/inválidas, identificadas na análise exploratória.

    Remove:
        - Identificadores (Unnamed: 0, Index, Address): não generalizam
          para contas fora do dataset.
        - Colunas de variância zero: não discriminam entre classes.
        - Coluna duplicada por erro de cabeçalho no CSV original.

    Parâmetros:
        df (pd.DataFrame): dataset bruto.

    Retorna:
        pd.DataFrame: dataset sem as colunas irrelevantes.
    """
    cols_to_drop = ID_COLS + ZERO_VARIANCE_COLS + [DUPLICATED_COL]
    cols_to_drop = [c for c in cols_to_drop if c in df.columns]
    return df.drop(columns=cols_to_drop)


def standardize_missing_erc20(df):
    """
    Unifica as três representações inconsistentes de "sem atividade
    ERC20" encontradas nas colunas de tipo de token (NaN, string '0'
    e string vazia) em um único valor categórico 'None'.

    Também preenche com 0 os atributos numéricos ERC20 ausentes,
    pois a ausência nesses casos representa "zero transações ERC20"
    e não um dado realmente desconhecido.

    Parâmetros:
        df (pd.DataFrame): dataset após remoção de colunas irrelevantes.

    Retorna:
        pd.DataFrame: dataset com valores ausentes padronizados.
    """
    df = df.copy()

    for col in TOKEN_COLS:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()
            df[col] = df[col].replace({'0': 'None', '': 'None', 'nan': 'None'})

    # Demais colunas numéricas ausentes (estatísticas ERC20) -> 0
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].fillna(0)

    return df


def reduce_token_cardinality(df, top_n=TOP_N_TOKENS):
    """
    Reduz a cardinalidade das colunas de tipo de token ERC20,
    mantendo apenas os 'top_n' tokens mais frequentes por coluna e
    agrupando os demais na categoria 'Other'. A categoria 'None'
    (sem atividade ERC20) é sempre preservada.

    Isso evita explosão dimensional ao aplicar one-hot encoding em
    colunas que originalmente tinham 304 e 466 categorias únicas.

    Parâmetros:
        df (pd.DataFrame): dataset com valores ausentes já padronizados.
        top_n (int): quantidade de categorias mais frequentes a manter.

    Retorna:
        pd.DataFrame: dataset com colunas de token de cardinalidade
            reduzida.
    """
    df = df.copy()
    for col in TOKEN_COLS:
        if col not in df.columns:
            continue
        freq = df[col].value_counts()
        # 'None' sempre é mantida; selecionamos os top_n entre as demais
        top_categories = freq.drop('None', errors='ignore').head(top_n).index
        allowed = set(top_categories) | {'None'}
        df[col] = df[col].where(df[col].isin(allowed), other='Other')
    return df


def apply_signed_log_transform(df):
    """
    Aplica transformação logarítmica assinada às colunas monetárias e
    de contagem fortemente assimétricas (skew entre 1.8 e 62.4 na
    análise exploratória), reduzindo o impacto de valores extremos
    sem descartar informação -- importante porque valores extremos de
    transação podem ser justamente o sinal de comportamento fraudulento.

    A transformação usa sign(x) * log(1 + |x|) para preservar o sinal
    em colunas que podem ser negativas (ex.: total ether balance).

    Parâmetros:
        df (pd.DataFrame): dataset antes da transformação.

    Retorna:
        pd.DataFrame: dataset com as colunas assimétricas transformadas.
    """
    df = df.copy()
    for col in SKEWED_COLS:
        if col in df.columns:
            df[col] = np.sign(df[col]) * np.log1p(np.abs(df[col]))
    return df


def encode_categoricals(df):
    """
    Aplica codificação one-hot às colunas categóricas de tipo de
    token ERC20 já com cardinalidade reduzida.

    Parâmetros:
        df (pd.DataFrame): dataset com cardinalidade já reduzida.

    Retorna:
        pd.DataFrame: dataset totalmente numérico, pronto para
            divisão treino/teste e escalonamento.
    """
    existing_token_cols = [c for c in TOKEN_COLS if c in df.columns]
    return pd.get_dummies(df, columns=existing_token_cols, drop_first=False)


def split_and_scale(df, target_col=TARGET_COL, test_size=0.25, random_state=42):
    """
    Separa atributos (X) e alvo (y), realiza divisão treino/teste
    estratificada (preservando a proporção ~22%/78% de fraude/legítimo)
    e padroniza os atributos numéricos com StandardScaler, ajustado
    apenas no conjunto de treino para evitar vazamento de dados (data
    leakage) entre treino e teste.

    Parâmetros:
        df (pd.DataFrame): dataset totalmente numérico e pronto.
        target_col (str): nome da coluna alvo.
        test_size (float): proporção reservada para teste.
        random_state (int): semente para reprodutibilidade.

    Retorna:
        tuple: (X_train, X_test, y_train, y_test, scaler)
            X_train, X_test (pd.DataFrame): atributos padronizados.
            y_train, y_test (pd.Series): rótulos de treino e teste.
            scaler (StandardScaler): scaler ajustado no treino, para
                ser reaproveitado em dados novos.
    """
    X = df.drop(columns=[target_col])
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test), columns=X_test.columns, index=X_test.index
    )

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler


def run_pipeline(raw_path, test_size=0.25, random_state=42):
    """
    Executa o pipeline completo de pré-processamento, do dataset bruto
    até os conjuntos de treino e teste prontos para modelagem.

    Parâmetros:
        raw_path (str): caminho para o CSV original.
        test_size (float): proporção reservada para teste.
        random_state (int): semente para reprodutibilidade.

    Retorna:
        tuple: (X_train, X_test, y_train, y_test, scaler, df_processed)
            df_processed (pd.DataFrame): dataset completo após limpeza,
                útil para auditoria e geração de estatísticas para o
                artigo.
    """
    df = load_data(raw_path)
    df = drop_irrelevant_columns(df)
    df = standardize_missing_erc20(df)
    df = reduce_token_cardinality(df)
    df = apply_signed_log_transform(df)
    df = encode_categoricals(df)

    X_train, X_test, y_train, y_test, scaler = split_and_scale(
        df, test_size=test_size, random_state=random_state
    )

    return X_train, X_test, y_train, y_test, scaler, df


if __name__ == '__main__':
    X_train, X_test, y_train, y_test, scaler, df_processed = run_pipeline(
        'address_data_k.csv'
    )

    print("=== RESULTADO DO PRÉ-PROCESSAMENTO ===")
    print(f"Shape dataset processado (antes do split): {df_processed.shape}")
    print(f"Shape X_train: {X_train.shape}")
    print(f"Shape X_test:  {X_test.shape}")
    print(f"Valores ausentes restantes: {df_processed.isnull().sum().sum()}")
    print()
    print("Proporção da classe alvo:")
    print("  Treino:", y_train.value_counts(normalize=True).round(4).to_dict())
    print("  Teste: ", y_test.value_counts(normalize=True).round(4).to_dict())
    print()
    print("Colunas finais (primeiras 15):")
    print(X_train.columns.tolist()[:15])

    # Salva os conjuntos processados para uso na etapa de modelagem
    X_train.assign(FLAG=y_train).to_csv('train_processed.csv', index=False)
    X_test.assign(FLAG=y_test).to_csv('test_processed.csv', index=False)
    print("\nArquivos salvos: train_processed.csv, test_processed.csv")
