"""
modeling.py

Etapa de Modelagem do processo CRISP-DM/KDD.

Treina e compara múltiplos algoritmos de classificação para detecção
de contas fraudulentas na rede Ethereum:
  - Regressão Logística
  - Árvore de Decisão
  - Random Forest
  - Naive Bayes (Gaussian)
  - K-Nearest Neighbors (KNN)
  - SVM (kernel linear)

A comparação inicial usa validação cruzada estratificada (5-fold) no
conjunto de treino. Os dois algoritmos com melhor desempenho são então
otimizados via GridSearchCV antes da avaliação final no conjunto de
teste (ver evaluation.py).
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

RANDOM_STATE = 42


def load_processed_data(train_path, test_path, target_col='FLAG'):
    """
    Carrega os conjuntos de treino e teste já pré-processados.

    Parâmetros:
        train_path (str): caminho para train_processed.csv.
        test_path (str): caminho para test_processed.csv.
        target_col (str): nome da coluna alvo.

    Retorna:
        tuple: (X_train, y_train, X_test, y_test)
    """
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    X_train = train_df.drop(columns=[target_col])
    y_train = train_df[target_col]
    X_test = test_df.drop(columns=[target_col])
    y_test = test_df[target_col]

    return X_train, y_train, X_test, y_test


def get_candidate_models():
    """
    Define os modelos candidatos a serem comparados, todos
    configurados com tratamento para classes desbalanceadas
    (class_weight='balanced') onde aplicável, já que o dataset tem
    proporção de 78%/22% entre as classes legítima e fraudulenta.

    Retorna:
        dict: nome do modelo -> instância do classificador (não
            treinado).
    """
    return {
        'Logistic Regression': LogisticRegression(
            class_weight='balanced', max_iter=2000, random_state=RANDOM_STATE
        ),
        'Decision Tree': DecisionTreeClassifier(
            class_weight='balanced', random_state=RANDOM_STATE
        ),
        'Random Forest': RandomForestClassifier(
            class_weight='balanced', n_estimators=200, random_state=RANDOM_STATE,
            n_jobs=-1
        ),
        'Naive Bayes': GaussianNB(),
        'KNN': KNeighborsClassifier(n_neighbors=7),
        'SVM (linear)': SVC(
            kernel='linear', class_weight='balanced', random_state=RANDOM_STATE
        ),
    }


def compare_models_cv(X_train, y_train, models, n_splits=5):
    """
    Compara os modelos candidatos via validação cruzada estratificada,
    coletando F1-score, Recall, Precision e AUC-ROC médios em cada
    fold. A estratificação garante que a proporção ~78%/22% de classes
    seja preservada em cada fold.

    Parâmetros:
        X_train (pd.DataFrame): atributos de treino.
        y_train (pd.Series): rótulos de treino.
        models (dict): modelos candidatos (nome -> instância).
        n_splits (int): número de folds da validação cruzada.

    Retorna:
        pd.DataFrame: tabela com a média e desvio padrão de cada
            métrica, por modelo, ordenada pelo F1-score médio
            (decrescente).
    """
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    scoring = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']

    results = []
    for name, model in models.items():
        scores = cross_validate(
            model, X_train, y_train, cv=cv, scoring=scoring, n_jobs=-1
        )
        row = {'Modelo': name}
        for metric in scoring:
            key = f'test_{metric}'
            row[f'{metric}_mean'] = scores[key].mean()
            row[f'{metric}_std'] = scores[key].std()
        results.append(row)

    results_df = pd.DataFrame(results).sort_values('f1_mean', ascending=False)
    return results_df.reset_index(drop=True)


if __name__ == '__main__':
    X_train, y_train, X_test, y_test = load_processed_data(
        'train_processed.csv', 'test_processed.csv'
    )

    print(f"Treino: {X_train.shape}, Teste: {X_test.shape}")
    print(f"Proporção de fraude no treino: {y_train.mean():.4f}")
    print()

    models = get_candidate_models()
    print("Executando validação cruzada (5-fold) para todos os modelos...")
    results = compare_models_cv(X_train, y_train, models)

    pd.set_option('display.width', 200)
    pd.set_option('display.max_columns', None)
    cols_to_show = ['Modelo', 'f1_mean', 'f1_std', 'roc_auc_mean',
                     'precision_mean', 'recall_mean', 'accuracy_mean']
    print()
    print("=== COMPARAÇÃO DE MODELOS (validação cruzada, 5-fold) ===")
    print(results[cols_to_show].round(4).to_string(index=False))

    results.to_csv('model_comparison_cv.csv', index=False)
    print("\nResultado salvo em model_comparison_cv.csv")
