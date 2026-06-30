"""
visualizations.py

Geração das figuras para o artigo científico.

Produz:
  1. Distribuição da classe alvo (desbalanceamento)
  2. Comparação de F1-score entre todos os modelos (validação cruzada)
  3. Matrizes de confusão dos dois modelos finais
  4. Top 15 features do Random Forest (importância)
  5. Top 15 coeficientes da Regressão Logística
  6. Curvas ROC e Precision-Recall comparativas
"""

import warnings
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix, roc_curve, auc, precision_recall_curve,
    average_precision_score
)

from modeling import load_processed_data
from evaluation import top_features_random_forest, top_features_logistic_regression

warnings.filterwarnings('ignore')
plt.rcParams['figure.dpi'] = 150
plt.rcParams['font.size'] = 10

BLUE = '#2c5f8a'
RED = '#c0392b'
GREEN = '#27ae60'


def plot_class_distribution(y_full, path):
    """Gera gráfico de barras da distribuição da classe alvo."""
    counts = y_full.value_counts().sort_index()
    labels = ['Legítima', 'Fraude']
    pct = (counts / counts.sum() * 100).round(1)

    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(labels, counts.values, color=[BLUE, RED])
    for bar, c, p in zip(bars, counts.values, pct.values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 80,
                f'{c}\n({p}%)', ha='center', va='bottom', fontsize=10)
    ax.set_ylabel('Número de contas')
    ax.set_title('Distribuição das classes no dataset')
    ax.set_ylim(0, max(counts.values) * 1.15)
    plt.tight_layout()
    plt.savefig(path, bbox_inches='tight')
    plt.close()


def plot_model_comparison(cv_results_path, path):
    """Gera gráfico de barras horizontais comparando F1 dos modelos."""
    df = pd.read_csv(cv_results_path).sort_values('f1_mean')

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.barh(df['Modelo'], df['f1_mean'], xerr=df['f1_std'],
                   color=BLUE, capsize=3)
    for bar, val in zip(bars, df['f1_mean']):
        ax.text(val - 0.04, bar.get_y() + bar.get_height() / 2,
                f'{val:.3f}', va='center', ha='right', color='white', fontsize=9)
    ax.set_xlabel('F1-Score médio (validação cruzada 5-fold)')
    ax.set_title('Comparação de desempenho entre algoritmos')
    ax.set_xlim(0, 1.0)
    plt.tight_layout()
    plt.savefig(path, bbox_inches='tight')
    plt.close()


def plot_confusion_matrices(models_dict, X_test, y_test, path):
    """Gera matrizes de confusão lado a lado para os modelos finais."""
    fig, axes = plt.subplots(1, len(models_dict), figsize=(11, 4.2))
    if len(models_dict) == 1:
        axes = [axes]

    for ax, (name, model) in zip(axes, models_dict.items()):
        y_pred = model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        im = ax.imshow(cm, cmap='Blues')
        ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
        ax.set_xticklabels(['Legítima', 'Fraude'])
        ax.set_yticklabels(['Legítima', 'Fraude'])
        ax.set_xlabel('Predito'); ax.set_ylabel('Real')
        ax.set_title(name)
        thresh = cm.max() / 2
        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(cm[i, j]), ha='center', va='center',
                        color='white' if cm[i, j] > thresh else 'black',
                        fontsize=14)
    plt.tight_layout()
    plt.savefig(path, bbox_inches='tight')
    plt.close()


def plot_rf_importances(rf_model, feature_names, path, top_n=15):
    """Gera gráfico de barras das features mais importantes do RF."""
    imp = top_features_random_forest(rf_model, feature_names, top_n).sort_values()
    labels = [c.strip()[:38] for c in imp.index]

    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.barh(labels, imp.values, color=GREEN)
    ax.set_xlabel('Importância (redução de impureza Gini)')
    ax.set_title(f'Top {top_n} atributos - Random Forest')
    plt.tight_layout()
    plt.savefig(path, bbox_inches='tight')
    plt.close()


def plot_lr_coefficients(lr_model, feature_names, path, top_n=15):
    """Gera gráfico de coeficientes da Regressão Logística (sinal preservado)."""
    coefs = top_features_logistic_regression(lr_model, feature_names, top_n)
    coefs = coefs.sort_values()
    labels = [c.strip()[:38] for c in coefs.index]
    colors = [RED if v > 0 else BLUE for v in coefs.values]

    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.barh(labels, coefs.values, color=colors)
    ax.axvline(0, color='black', linewidth=0.8)
    ax.set_xlabel('Coeficiente (positivo = indica fraude)')
    ax.set_title(f'Top {top_n} atributos - Regressão Logística')
    plt.tight_layout()
    plt.savefig(path, bbox_inches='tight')
    plt.close()


def plot_roc_pr_curves(models_dict, X_test, y_test, roc_path, pr_path):
    """Gera curvas ROC e Precision-Recall comparativas."""
    # ROC
    fig, ax = plt.subplots(figsize=(6, 5))
    for (name, model), color in zip(models_dict.items(), [BLUE, RED]):
        if hasattr(model, 'predict_proba'):
            y_score = model.predict_proba(X_test)[:, 1]
        else:
            y_score = model.decision_function(X_test)
        fpr, tpr, _ = roc_curve(y_test, y_score)
        ax.plot(fpr, tpr, color=color, label=f'{name} (AUC={auc(fpr, tpr):.4f})')
    ax.plot([0, 1], [0, 1], 'k--', linewidth=0.8)
    ax.set_xlabel('Taxa de Falsos Positivos')
    ax.set_ylabel('Taxa de Verdadeiros Positivos')
    ax.set_title('Curva ROC')
    ax.legend(loc='lower right', fontsize=8)
    plt.tight_layout()
    plt.savefig(roc_path, bbox_inches='tight')
    plt.close()

    # Precision-Recall
    fig, ax = plt.subplots(figsize=(6, 5))
    for (name, model), color in zip(models_dict.items(), [BLUE, RED]):
        if hasattr(model, 'predict_proba'):
            y_score = model.predict_proba(X_test)[:, 1]
        else:
            y_score = model.decision_function(X_test)
        prec, rec, _ = precision_recall_curve(y_test, y_score)
        ap = average_precision_score(y_test, y_score)
        ax.plot(rec, prec, color=color, label=f'{name} (AP={ap:.4f})')
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title('Curva Precision-Recall')
    ax.legend(loc='lower left', fontsize=8)
    plt.tight_layout()
    plt.savefig(pr_path, bbox_inches='tight')
    plt.close()


if __name__ == '__main__':
    X_train, y_train, X_test, y_test = load_processed_data(
        'train_processed.csv', 'test_processed.csv'
    )
    y_full = pd.concat([y_train, y_test])

    rf_model = joblib.load('best_random_forest.joblib')
    lr_model = joblib.load('best_logistic_regression.joblib')
    models = {'Random Forest': rf_model, 'Regressão Logística': lr_model}
    features = X_train.columns.tolist()

    plot_class_distribution(y_full, 'fig1_class_distribution.png')
    plot_model_comparison('model_comparison_cv.csv', 'fig2_model_comparison.png')
    plot_confusion_matrices(models, X_test, y_test, 'fig3_confusion_matrices.png')
    plot_rf_importances(rf_model, features, 'fig4_rf_importances.png')
    plot_lr_coefficients(lr_model, features, 'fig5_lr_coefficients.png')
    plot_roc_pr_curves(models, X_test, y_test,
                       'fig6_roc_curve.png', 'fig7_pr_curve.png')

    print("Figuras geradas com sucesso:")
    import os
    for f in sorted(os.listdir('.')):
        if f.endswith('.png'):
            print(f"  - {f}")
