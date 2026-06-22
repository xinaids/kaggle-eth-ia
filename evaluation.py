"""
evaluation.py

Etapa de Avaliação do processo CRISP-DM/KDD.

Avalia os dois modelos otimizados (Random Forest e Regressão
Logística) no conjunto de teste, que não foi usado em nenhum momento
do treinamento ou da otimização de hiperparâmetros. Gera:
  - Matriz de confusão
  - Relatório de classificação (precision, recall, F1 por classe)
  - AUC-ROC e AUC-PR (Precision-Recall)
  - Atributos mais importantes de cada modelo (feature importance da
    Random Forest e coeficientes da Regressão Logística), permitindo
    interpretar quais características de uma conta mais indicam
    comportamento fraudulento.
"""

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    confusion_matrix, classification_report, roc_auc_score,
    average_precision_score, f1_score
)

from modeling import load_processed_data


def evaluate_model(model, X_test, y_test, model_name):
    """
    Avalia um modelo treinado no conjunto de teste, calculando matriz
    de confusão, relatório de classificação completo, AUC-ROC e
    AUC-PR (mais informativa que AUC-ROC em datasets desbalanceados).

    Parâmetros:
        model: classificador treinado (com predict e predict_proba ou
            decision_function).
        X_test (pd.DataFrame): atributos de teste.
        y_test (pd.Series): rótulos verdadeiros de teste.
        model_name (str): nome do modelo, usado apenas na impressão.

    Retorna:
        dict: métricas resumidas (f1, roc_auc, pr_auc) e a matriz de
            confusão, para uso posterior em comparações/gráficos.
    """
    y_pred = model.predict(X_test)

    if hasattr(model, 'predict_proba'):
        y_score = model.predict_proba(X_test)[:, 1]
    else:
        y_score = model.decision_function(X_test)

    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=['Legítima', 'Fraude'])
    roc_auc = roc_auc_score(y_test, y_score)
    pr_auc = average_precision_score(y_test, y_score)
    f1 = f1_score(y_test, y_pred)

    print(f"\n{'=' * 60}")
    print(f"  {model_name}")
    print(f"{'=' * 60}")
    print("\nMatriz de Confusão:")
    print(f"                 Predito Legítima   Predito Fraude")
    print(f"Real Legítima    {cm[0][0]:>16}   {cm[0][1]:>14}")
    print(f"Real Fraude      {cm[1][0]:>16}   {cm[1][1]:>14}")
    print("\nRelatório de Classificação:")
    print(report)
    print(f"AUC-ROC: {roc_auc:.4f}")
    print(f"AUC-PR:  {pr_auc:.4f}")

    return {'modelo': model_name, 'f1': f1, 'roc_auc': roc_auc,
            'pr_auc': pr_auc, 'confusion_matrix': cm}


def top_features_random_forest(model, feature_names, top_n=15):
    """
    Extrai os atributos mais importantes do Random Forest, com base
    na redução média de impureza (Gini importance).

    Parâmetros:
        model (RandomForestClassifier): modelo treinado.
        feature_names (list): nomes dos atributos, na mesma ordem
            usada no treinamento.
        top_n (int): quantidade de atributos a retornar.

    Retorna:
        pd.DataFrame: atributos ordenados por importância
            decrescente.
    """
    importances = pd.Series(model.feature_importances_, index=feature_names)
    return importances.sort_values(ascending=False).head(top_n)


def top_features_logistic_regression(model, feature_names, top_n=15):
    """
    Extrai os atributos com maior peso (em módulo) nos coeficientes
    da Regressão Logística. Coeficientes positivos aumentam a
    probabilidade de a conta ser classificada como fraude; negativos
    diminuem.

    Parâmetros:
        model (LogisticRegression): modelo treinado.
        feature_names (list): nomes dos atributos, na mesma ordem
            usada no treinamento.
        top_n (int): quantidade de atributos a retornar (maiores em
            módulo, mantendo o sinal original).

    Retorna:
        pd.DataFrame: atributos ordenados por |coeficiente|
            decrescente, com o valor do coeficiente original.
    """
    coefs = pd.Series(model.coef_[0], index=feature_names)
    top_idx = coefs.abs().sort_values(ascending=False).head(top_n).index
    return coefs[top_idx]


if __name__ == '__main__':
    X_train, y_train, X_test, y_test = load_processed_data(
        'train_processed.csv', 'test_processed.csv'
    )

    rf_model = joblib.load('best_random_forest.joblib')
    lr_model = joblib.load('best_logistic_regression.joblib')

    results = []
    results.append(evaluate_model(rf_model, X_test, y_test, 'Random Forest (otimizado)'))
    results.append(evaluate_model(lr_model, X_test, y_test, 'Regressão Logística (otimizada)'))

    print(f"\n{'=' * 60}")
    print("  TOP 15 FEATURES - RANDOM FOREST (importância)")
    print(f"{'=' * 60}")
    print(top_features_random_forest(rf_model, X_train.columns.tolist()).round(4))

    print(f"\n{'=' * 60}")
    print("  TOP 15 FEATURES - REGRESSÃO LOGÍSTICA (coeficientes)")
    print(f"{'=' * 60}")
    print(top_features_logistic_regression(lr_model, X_train.columns.tolist()).round(4))

    summary = pd.DataFrame([
        {'Modelo': r['modelo'], 'F1': r['f1'], 'AUC-ROC': r['roc_auc'], 'AUC-PR': r['pr_auc']}
        for r in results
    ])
    summary.to_csv('final_test_results.csv', index=False)
    print(f"\n{'=' * 60}")
    print("  RESUMO FINAL (conjunto de teste)")
    print(f"{'=' * 60}")
    print(summary.round(4).to_string(index=False))
