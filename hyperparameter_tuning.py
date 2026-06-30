"""
hyperparameter_tuning.py

Otimização de hiperparâmetros via GridSearchCV para os dois modelos
selecionados a partir da comparação inicial (modeling.py):
  - Random Forest: melhor desempenho bruto (F1 = 0.9778)
  - Regressão Logística: modelo interpretável, permite análise de
    coeficientes (replicando a abordagem do artigo de referência sobre
    detecção de SPAM)

Naive Bayes, Decision Tree, KNN e SVM linear não foram levados à
otimização por desempenho bruto inferior ou por não agregarem
interpretabilidade ao trabalho.
"""

import warnings
import pandas as pd
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from modeling import load_processed_data, RANDOM_STATE

warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)


def tune_random_forest(X_train, y_train, cv):
    """
    Executa GridSearchCV para o Random Forest, buscando a melhor
    combinação de profundidade da árvore, número de estimadores e
    critério de divisão mínima, otimizando para F1-score (métrica
    mais informativa que acurácia dado o desbalanceamento de classes).

    Parâmetros:
        X_train (pd.DataFrame): atributos de treino.
        y_train (pd.Series): rótulos de treino.
        cv (StratifiedKFold): estratégia de validação cruzada.

    Retorna:
        GridSearchCV: objeto ajustado, com o melhor estimador
            acessível em .best_estimator_.
    """
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [None, 20, 30],
        'min_samples_split': [2, 5],
        'class_weight': ['balanced'],
    }
    # Apenas 1 núcleo de CPU disponível no ambiente: paralelismo não
    # ajuda e n_jobs=-1 em ambos os níveis causava lentidão por
    # contenção de threads. Execução sequencial é mais previsível aqui.
    grid = GridSearchCV(
        RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=1),
        param_grid, cv=cv, scoring='f1', n_jobs=1, verbose=1
    )
    grid.fit(X_train, y_train)
    return grid


def tune_logistic_regression(X_train, y_train, cv):
    """
    Executa GridSearchCV para a Regressão Logística, buscando a
    melhor força de regularização (C) e tipo de penalidade,
    otimizando para F1-score.

    Parâmetros:
        X_train (pd.DataFrame): atributos de treino.
        y_train (pd.Series): rótulos de treino.
        cv (StratifiedKFold): estratégia de validação cruzada.

    Retorna:
        GridSearchCV: objeto ajustado, com o melhor estimador
            acessível em .best_estimator_.
    """
    param_grid = {
        'C': [0.01, 0.1, 1.0, 5.0, 10.0],
        'penalty': ['l1', 'l2'],
        'solver': ['liblinear'],
        'class_weight': ['balanced', {0: 1, 1: 3}, {0: 1, 1: 5}],
    }
    grid = GridSearchCV(
        LogisticRegression(max_iter=2000, random_state=RANDOM_STATE),
        param_grid, cv=cv, scoring='f1', n_jobs=1, verbose=1
    )
    grid.fit(X_train, y_train)
    return grid


if __name__ == '__main__':
    X_train, y_train, X_test, y_test = load_processed_data(
        'train_processed.csv', 'test_processed.csv'
    )
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    print("Otimizando Random Forest...")
    rf_grid = tune_random_forest(X_train, y_train, cv)
    print("Melhores parâmetros (RF):", rf_grid.best_params_)
    print(f"Melhor F1 (validação cruzada): {rf_grid.best_score_:.4f}")
    print()

    print("Otimizando Regressão Logística...")
    lr_grid = tune_logistic_regression(X_train, y_train, cv)
    print("Melhores parâmetros (LR):", lr_grid.best_params_)
    print(f"Melhor F1 (validação cruzada): {lr_grid.best_score_:.4f}")

    import joblib
    joblib.dump(rf_grid.best_estimator_, 'best_random_forest.joblib')
    joblib.dump(lr_grid.best_estimator_, 'best_logistic_regression.joblib')
    print("\nModelos salvos: best_random_forest.joblib, best_logistic_regression.joblib")
