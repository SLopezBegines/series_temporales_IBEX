#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Funciones de visualización para análisis estadístico
@author: santi
"""

# ============================================================================
# IMPORTS
# ============================================================================

# Nota: Se asume que matplotlib, seaborn, numpy y pandas 
#       ya están importados en main.py
# Si ejecutas este módulo de forma independiente, descomenta:
# import matplotlib.pyplot as plt
# import seaborn as sns
# import numpy as np
# import pandas as pd

# Import config
from config import CSV, PLOTS
from sklearn.metrics import confusion_matrix
import os
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd

# Configuración general
fig, ax = plt.subplots()
plt.style.use('default')
sns.set_palette("husl")
# ============================================================================
# FUNCIONES DE VISUALIZACIÓN
# ============================================================================

def plot_statistical_results_classification(results_df):
    """
    Visualiza los resultados de las comparaciones estadísticas
    """
    if results_df.empty:
        print("No hay resultados para visualizar")
        return

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Análisis Estadístico: Financial vs Sentiment', fontsize=16, fontweight='bold')

    # 1. Diferencias de accuracy
    ax = axes[0, 0]
    results_sorted = results_df.sort_values('diff', ascending=False)
    colors = ['green' if x > 0 else 'red' for x in results_sorted['diff']]

    ax.barh(range(len(results_sorted)), results_sorted['diff'], color=colors, alpha=0.6)
    ax.set_yticks(range(len(results_sorted)))
    ax.set_yticklabels([f"{row['model']} ({row['target']})"
                        for _, row in results_sorted.iterrows()], fontsize=8)
    ax.axvline(0, color='black', linestyle='--', linewidth=1)
    ax.set_xlabel('Diferencia de Accuracy (Sentiment - Financial)', fontsize=10)
    ax.set_title('Diferencias de Accuracy', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)

    # 2. P-values (escala logarítmica)
    ax = axes[0, 1]
    results_sorted = results_df.sort_values('mcnemar_pvalue')
    colors = ['green' if x else 'red' for x in results_sorted['mcnemar_significant']]

    ax.barh(range(len(results_sorted)), results_sorted['mcnemar_pvalue'],
            color=colors, alpha=0.6)
    ax.axvline(0.05, color='orange', linestyle='--', linewidth=2, label='p=0.05')
    ax.axvline(0.01, color='red', linestyle='--', linewidth=2, label='p=0.01')
    ax.set_yticks(range(len(results_sorted)))
    ax.set_yticklabels([f"{row['model']} ({row['target']})"
                        for _, row in results_sorted.iterrows()], fontsize=8)
    ax.set_xlabel('McNemar p-value', fontsize=10)
    ax.set_title('Significancia Estadística (McNemar Test)', fontsize=12, fontweight='bold')
    ax.set_xscale('log')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 3. Intervalos de confianza bootstrap
    ax = axes[1, 0]
    results_sorted = results_df.sort_values('diff', ascending=False)

    for i, (_, row) in enumerate(results_sorted.iterrows()):
        ci_lower = row['bootstrap_ci_lower']
        ci_upper = row['bootstrap_ci_upper']
        diff = row['diff']

        # Color: verde si significativo, rojo si no
        color = 'green' if row['bootstrap_significant'] else 'red'

        # Plot CI
        ax.plot([ci_lower, ci_upper], [i, i], 'o-', color=color, linewidth=2, markersize=4)
        ax.plot(diff, i, 'D', color=color, markersize=6)

    ax.axvline(0, color='black', linestyle='--', linewidth=1)
    ax.set_yticks(range(len(results_sorted)))
    ax.set_yticklabels([f"{row['model']} ({row['target']})"
                        for _, row in results_sorted.iterrows()], fontsize=8)
    ax.set_xlabel('Diferencia de Accuracy', fontsize=10)
    ax.set_title('Intervalos de Confianza 95% (Bootstrap)', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)

    # 4. Resumen: % significativos
    ax = axes[1, 1]

    summary_data = [
        ('McNemar\n(p<0.05)', results_df['mcnemar_significant'].sum()),
        ('Bootstrap\n(CI no incluye 0)', results_df['bootstrap_significant'].sum()),
        ('Mejora positiva\n(diff > 0)', (results_df['diff'] > 0).sum()),
        ('Total\ncomparaciones', len(results_df))
    ]

    labels, values = zip(*summary_data)
    colors_summary = ['green', 'green', 'blue', 'gray']

    bars = ax.bar(labels, values, color=colors_summary, alpha=0.6)
    ax.set_ylabel('Número de comparaciones', fontsize=10)
    ax.set_title('Resumen de Significancia', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')

    # Añadir valores en las barras
    for bar, value in zip(bars, values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(value)}',
                ha='center', va='bottom', fontweight='bold')

    plt.tight_layout()
    plt.show()

    # Tabla resumen
    print("\n" + "="*70)
    print("RESUMEN ESTADÍSTICO")
    print("="*70)

    print(f"\nTotal comparaciones: {len(results_df)}")
    print(f"Mejoras (sentiment > financial): {(results_df['diff'] > 0).sum()}")
    print(f"Empeoramientos (sentiment < financial): {(results_df['diff'] < 0).sum()}")
    print(f"\nSignificativas (McNemar p<0.05): {results_df['mcnemar_significant'].sum()} ({results_df['mcnemar_significant'].mean()*100:.1f}%)")
    print(f"Significativas (Bootstrap IC): {results_df['bootstrap_significant'].sum()} ({results_df['bootstrap_significant'].mean()*100:.1f}%)")

    # Mostrar casos significativos
    significant = results_df[results_df['mcnemar_significant'] | results_df['bootstrap_significant']]

    if len(significant) > 0:
        print(f"\n{'='*70}")
        print("DIFERENCIAS ESTADÍSTICAMENTE SIGNIFICATIVAS")
        print(f"{'='*70}\n")

        for _, row in significant.iterrows():
            direction = '📈' if row['diff'] > 0 else '📉'
            print(f"{direction} {row['model']} ({row['target']})")
            print(f"   Diff: {row['diff']:.4f} | p-value: {row['mcnemar_pvalue']:.4f} | "
                  f"CI: [{row['bootstrap_ci_lower']:.4f}, {row['bootstrap_ci_upper']:.4f}]")
    else:
        print("\n⚠ NO se encontraron diferencias estadísticamente significativas")

print("✓ Función de visualización definida")

def plot_statistical_results_regression(regression_stats, save_path=None):
    """
    Visualiza y analiza resultados de comparaciones estadísticas de regresión
    
    Args:
        regression_stats: DataFrame con resultados de run_regression_statistical_comparisons()
        save_path: Ruta para guardar el gráfico (opcional)
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np
    
    # Preparar datos para visualización
    regression_stats = regression_stats.copy()
    regression_stats['model_target'] = (regression_stats['model'] + '\n(' +
                                        regression_stats['target'].str.replace('returns_', 'T+') +
                                        ', ' + regression_stats['scaling'] + ')')
    
    regression_stats['sentiment_mejor'] = regression_stats['rmse_diff'] > 0
    regression_stats['financial_mejor'] = regression_stats['rmse_diff'] < 0
    
    # Crear figura
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.3)
    
    # PANEL 1: Diferencias de RMSE
    ax1 = fig.add_subplot(gs[0, :2])
    df_sorted = regression_stats.sort_values('rmse_diff', ascending=True)
    colors = ['green' if x > 0 else 'red' for x in df_sorted['rmse_diff']]
    
    bars = ax1.barh(range(len(df_sorted)), df_sorted['rmse_diff'],
                    color=colors, alpha=0.6, edgecolor='black', linewidth=0.5)
    
    for i, (bar, sig) in enumerate(zip(bars, df_sorted['dm_significant'])):
        if sig:
            bar.set_edgecolor('black')
            bar.set_linewidth(2.5)
    
    ax1.axvline(0, color='black', linestyle='--', linewidth=2)
    ax1.set_yticks(range(len(df_sorted)))
    ax1.set_yticklabels(df_sorted['model_target'], fontsize=9)
    ax1.set_xlabel('RMSE(Financial) - RMSE(Sentiment)', fontsize=11, fontweight='bold')
    ax1.set_title('Diferencias de RMSE: Financial vs Sentiment\n(Verde=Sentiment mejor | Rojo=Financial mejor | Borde grueso=DM significativo)',
                  fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='x')
    ax1.set_xlim(df_sorted['rmse_diff'].min()*1.2, df_sorted['rmse_diff'].max()*1.2)
    
    for i, (idx, row) in enumerate(df_sorted.iterrows()):
        if row['dm_significant']:
            ax1.text(row['rmse_diff'], i, f" p={row['dm_pvalue']:.3f}",
                    va='center', fontsize=8, fontweight='bold')
    
    # PANEL 2: P-values
    ax2 = fig.add_subplot(gs[0, 2])
    df_sorted_p = regression_stats.sort_values('dm_pvalue')
    colors_p = ['darkgreen' if p < 0.05 else 'gray' for p in df_sorted_p['dm_pvalue']]
    
    bars_p = ax2.barh(range(len(df_sorted_p)), df_sorted_p['dm_pvalue'],
                      color=colors_p, alpha=0.7)
    ax2.axvline(0.05, color='red', linestyle='--', linewidth=2, label='α=0.05')
    ax2.axvline(0.10, color='orange', linestyle='--', linewidth=1.5, label='α=0.10')
    ax2.set_yticks(range(len(df_sorted_p)))
    ax2.set_yticklabels(df_sorted_p['model_target'], fontsize=8)
    ax2.set_xlabel('P-value', fontsize=10, fontweight='bold')
    ax2.set_title('Test Diebold-Mariano\n(p-values)', fontsize=11, fontweight='bold')
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3, axis='x')
    ax2.set_xlim(0, 1)
    
    # PANEL 3: Intervalos de confianza Bootstrap
    ax3 = fig.add_subplot(gs[1, :])
    df_sorted_boot = regression_stats.sort_values('rmse_diff', ascending=True)
    
    for i, (idx, row) in enumerate(df_sorted_boot.iterrows()):
        ci_lower = row['boot_ci_lower']
        ci_upper = row['boot_ci_upper']
        diff = row['rmse_diff']
        
        color = 'green' if diff > 0 else 'red'
        alpha = 0.9 if row['dm_significant'] else 0.4
        
        ax3.plot([ci_lower, ci_upper], [i, i], 'o-', color=color,
                 linewidth=2, markersize=5, alpha=alpha)
        ax3.plot(diff, i, 'D', color=color, markersize=8, alpha=alpha,
                 markeredgecolor='black', markeredgewidth=1.5 if row['dm_significant'] else 0.5)
    
    ax3.axvline(0, color='black', linestyle='--', linewidth=2)
    ax3.set_yticks(range(len(df_sorted_boot)))
    ax3.set_yticklabels(df_sorted_boot['model_target'], fontsize=9)
    ax3.set_xlabel('Diferencia de RMSE (IC 95% Bootstrap)', fontsize=11, fontweight='bold')
    ax3.set_title('Intervalos de Confianza Bootstrap (95%)\n(Diamante=diferencia observada | Línea=IC | Borde grueso=DM significativo)',
                  fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='x')
    ax3.set_xlim(df_sorted_boot['rmse_diff'].min()*1.2, df_sorted_boot['rmse_diff'].max()*1.2)
    
    # PANEL 4: RMSE absolutos
    ax4 = fig.add_subplot(gs[2, 0])
    x_pos = np.arange(len(regression_stats))
    width = 0.35
    
    bars1 = ax4.bar(x_pos - width/2, regression_stats['rmse_financial'], width,
                    label='Financial', color='steelblue', alpha=0.7)
    bars2 = ax4.bar(x_pos + width/2, regression_stats['rmse_sentiment'], width,
                    label='Sentiment', color='coral', alpha=0.7)
    
    ax4.set_ylabel('RMSE', fontsize=10, fontweight='bold')
    ax4.set_title('RMSE Absolutos: Financial vs Sentiment', fontsize=11, fontweight='bold')
    ax4.set_xticks(x_pos)
    ax4.set_xticklabels(regression_stats['model_target'], rotation=45, ha='right', fontsize=7)
    ax4.legend(fontsize=9)
    ax4.grid(True, alpha=0.3, axis='y')
    
    # PANEL 5: Resumen estadístico
    ax5 = fig.add_subplot(gs[2, 1])
    summary_stats = {
        'Total\ncomparaciones': len(regression_stats),
        'DM signif.\n(p<0.05)': regression_stats['dm_significant'].sum(),
        'DM cuasi-signif.\n(p<0.10)': (regression_stats['dm_pvalue'] < 0.10).sum(),
        'Sentiment\nmejor RMSE': regression_stats['sentiment_mejor'].sum(),
        'Financial\nmejor RMSE': regression_stats['financial_mejor'].sum()
    }
    
    labels = list(summary_stats.keys())
    values = list(summary_stats.values())
    colors_summary = ['gray', 'darkgreen', 'orange', 'green', 'red']
    
    bars_summary = ax5.bar(labels, values, color=colors_summary, alpha=0.7, edgecolor='black')
    ax5.set_ylabel('Número de casos', fontsize=10, fontweight='bold')
    ax5.set_title('Resumen Estadístico', fontsize=11, fontweight='bold')
    ax5.grid(True, alpha=0.3, axis='y')
    
    for bar, value in zip(bars_summary, values):
        height = bar.get_height()
        ax5.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(value)}', ha='center', va='bottom', fontweight='bold', fontsize=11)
    
    # PANEL 6: Distribución de diferencias
    ax6 = fig.add_subplot(gs[2, 2])
    ax6.hist(regression_stats['rmse_diff'], bins=15, color='steelblue', alpha=0.6, edgecolor='black')
    ax6.axvline(0, color='red', linestyle='--', linewidth=2, label='Sin diferencia')
    ax6.axvline(regression_stats['rmse_diff'].mean(), color='green', linestyle='-', linewidth=2,
                label=f'Media={regression_stats["rmse_diff"].mean():.4f}')
    ax6.set_xlabel('RMSE(Financial) - RMSE(Sentiment)', fontsize=10, fontweight='bold')
    ax6.set_ylabel('Frecuencia', fontsize=10, fontweight='bold')
    ax6.set_title('Distribución de Diferencias', fontsize=11, fontweight='bold')
    ax6.legend(fontsize=8)
    ax6.grid(True, alpha=0.3, axis='y')
    
    fig.suptitle('Análisis Estadístico: Impacto de Datos de Sentimiento en Modelos de Regresión',
                 fontsize=16, fontweight='bold', y=0.995)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Gráfico guardado en {save_path}")
    
    plt.show()
    
    # INTERPRETACIÓN TEXTUAL
    print("\n" + "="*80)
    print("INTERPRETACIÓN DE RESULTADOS - REGRESIÓN")
    print("="*80)
    
    print("\n📊 RESUMEN GENERAL:")
    print(f"   • Total de comparaciones: {len(regression_stats)}")
    print(f"   • Casos donde Sentiment es mejor (RMSE menor): {regression_stats['sentiment_mejor'].sum()} "
          f"({regression_stats['sentiment_mejor'].sum()/len(regression_stats)*100:.1f}%)")
    print(f"   • Casos donde Financial es mejor (RMSE menor): {regression_stats['financial_mejor'].sum()} "
          f"({regression_stats['financial_mejor'].sum()/len(regression_stats)*100:.1f}%)")
    print(f"   • Diferencia media de RMSE: {regression_stats['rmse_diff'].mean():.4f}")
    
    print("\n🔬 SIGNIFICANCIA ESTADÍSTICA:")
    print(f"   • Test Diebold-Mariano (p<0.05): {regression_stats['dm_significant'].sum()} casos significativos")
    print(f"   • Test Diebold-Mariano (p<0.10): {(regression_stats['dm_pvalue'] < 0.10).sum()} casos cuasi-significativos")
    
    if regression_stats['dm_significant'].sum() > 0:
        print("\n   Casos DM significativos (p<0.05):")
        sig_cases = regression_stats[regression_stats['dm_significant']][
            ['model', 'target', 'scaling', 'rmse_diff', 'dm_pvalue']
        ]
        for idx, row in sig_cases.iterrows():
            direction = "Financial MEJOR" if row['rmse_diff'] < 0 else "Sentiment MEJOR"
            print(f"      - {row['model']} ({row['target']}, {row['scaling']}): "
                  f"diff={row['rmse_diff']:.4f}, p={row['dm_pvalue']:.4f} → {direction}")
    
    print("\n📈 MAGNITUD DE DIFERENCIAS:")
    print(f"   • Diferencia mínima: {regression_stats['rmse_diff'].min():.4f}")
    print(f"   • Diferencia máxima: {regression_stats['rmse_diff'].max():.4f}")
    print(f"   • Desviación estándar: {regression_stats['rmse_diff'].std():.4f}")
    if regression_stats['dm_significant'].sum() > 0:
        max_sig_diff = regression_stats[regression_stats['dm_significant']]['rmse_diff'].abs().max()
        print(f"   • Mayor diferencia significativa: {max_sig_diff:.4f}")
    
    sentiment_mejor_sig = regression_stats[regression_stats['dm_significant'] & regression_stats['sentiment_mejor']]
    financial_mejor_sig = regression_stats[regression_stats['dm_significant'] & regression_stats['financial_mejor']]
    
    print("\n🎯 CONCLUSIONES:")
    print("\n1. SIGNIFICANCIA ESTADÍSTICA:")
    print(f"   - Solo {regression_stats['dm_significant'].sum()} de {len(regression_stats)} comparaciones "
          f"({regression_stats['dm_significant'].sum()/len(regression_stats)*100:.1f}%) muestran diferencias significativas.")
    
    if len(sentiment_mejor_sig) > 0:
        print(f"\n   - En {len(sentiment_mejor_sig)} caso(s) significativo(s), SENTIMENT es MEJOR:")
        for idx, row in sentiment_mejor_sig.iterrows():
            print(f"      • {row['model']} ({row['target']}, {row['scaling']})")
            print(f"        RMSE: Financial={row['rmse_financial']:.4f} vs Sentiment={row['rmse_sentiment']:.4f}")
    
    if len(financial_mejor_sig) > 0:
        print(f"\n   - En {len(financial_mejor_sig)} caso(s) significativo(s), FINANCIAL es MEJOR:")
        for idx, row in financial_mejor_sig.iterrows():
            print(f"      • {row['model']} ({row['target']}, {row['scaling']})")
            print(f"        RMSE: Financial={row['rmse_financial']:.4f} vs Sentiment={row['rmse_sentiment']:.4f}")
    
    print("\n2. MAGNITUD DE LAS DIFERENCIAS:")
    print(f"   - Diferencia media absoluta: {abs(regression_stats['rmse_diff'].mean()):.4f}")
    print(f"   - Rango: [{regression_stats['rmse_diff'].min():.4f}, {regression_stats['rmse_diff'].max():.4f}]")
    
    print("\n3. EVIDENCIA GLOBAL:")
    sentiment_mejor_total = regression_stats['sentiment_mejor'].sum()
    financial_mejor_total = regression_stats['financial_mejor'].sum()
    print(f"   - Sentiment mejor en {sentiment_mejor_total}/{len(regression_stats)} casos "
          f"({sentiment_mejor_total/len(regression_stats)*100:.1f}%).")
    print(f"   - Financial mejor en {financial_mejor_total}/{len(regression_stats)} casos "
          f"({financial_mejor_total/len(regression_stats)*100:.1f}%).")
    
    pct_no_sig = (1 - regression_stats['dm_significant'].sum()/len(regression_stats)) * 100
    print(f"\n   ⚠ El {pct_no_sig:.1f}% de comparaciones NO muestran diferencias significativas.")
    
    print("\n" + "="*80)

# Uso:
# plot_statistical_results_regression(regression_stats, save_path=f"{PLOT_PATH}/regression_statistical_comparison.png")

def plot_classification_horizon_analysis(results_df, horizon_results_df):
    """
    Visualiza análisis completo: Financial vs Sentiment Y horizontes temporales
    """

    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

    # 1. Accuracy por horizonte y modelo (scaled)
    ax1 = fig.add_subplot(gs[0, :])

    if not horizon_results_df.empty:
        # Filtrar solo scaled datasets
        scaled_data = horizon_results_df[horizon_results_df['dataset'].str.contains('scaled')]

        if not scaled_data.empty:
            # Crear matriz: modelos × horizontes
            pivot_data = []

            for model in scaled_data['model'].unique():
                for dataset in ['financial_scaled', 'sentiment_scaled']:
                    model_data = scaled_data[
                        (scaled_data['model'] == model) &
                        (scaled_data['dataset'] == dataset)
                    ]

                    if model_data.empty:
                        continue

                    # Extraer accuracies para cada horizonte
                    horizons_dict = {}
                    for _, row in model_data.iterrows():
                        horizons_dict[row['days_1']] = row['acc_horizon_1']
                        horizons_dict[row['days_2']] = row['acc_horizon_2']

                    for days, acc in sorted(horizons_dict.items()):
                        pivot_data.append({
                            'model': model,
                            'dataset_type': 'Financial' if 'financial' in dataset else 'Sentiment',
                            'days': days,
                            'accuracy': acc
                        })

            if pivot_data:
                pivot_df = pd.DataFrame(pivot_data)

                # Plot líneas
                for model in pivot_df['model'].unique():
                    for dtype in ['Financial', 'Sentiment']:
                        model_data = pivot_df[
                            (pivot_df['model'] == model) &
                            (pivot_df['dataset_type'] == dtype)
                        ]

                        if model_data.empty:
                            continue

                        linestyle = '-' if dtype == 'Financial' else '--'
                        marker = 'o' if dtype == 'Financial' else 's'

                        ax1.plot(model_data['days'], model_data['accuracy'],
                                marker=marker, linestyle=linestyle,
                                label=f"{model} ({dtype})", linewidth=2, markersize=8)

                ax1.axhline(0.5, color='red', linestyle=':', linewidth=2, label='Random (50%)')
                ax1.set_xlabel('Horizonte de Predicción (días)', fontsize=12, fontweight='bold')
                ax1.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
                ax1.set_title('Accuracy vs Horizonte Temporal (Scaled)', fontsize=14, fontweight='bold')
                ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
                ax1.grid(True, alpha=0.3)
                ax1.set_xticks([1, 5, 10, 20])

    # 2. ¿La diferencia Financial-Sentiment cambia con el horizonte?
    ax2 = fig.add_subplot(gs[1, 0])

    if not results_df.empty:
        # Agregar columna de días
        results_df['days'] = results_df['target'].apply(lambda x:
            1 if x.endswith('_next') else int(x.split('_')[-1])
        )

        # Calcular diferencia promedio por horizonte
        diff_by_horizon = results_df.groupby('days')['diff'].agg(['mean', 'std', 'count'])

        ax2.bar(diff_by_horizon.index, diff_by_horizon['mean'],
                yerr=diff_by_horizon['std'], capsize=5, alpha=0.7, color='steelblue')
        ax2.axhline(0, color='red', linestyle='--', linewidth=2)
        ax2.set_xlabel('Horizonte (días)', fontsize=11, fontweight='bold')
        ax2.set_ylabel('Diferencia Promedio\n(Sentiment - Financial)', fontsize=11, fontweight='bold')
        ax2.set_title('¿Sentiment Mejora con Horizonte Más Largo?', fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='y')
        ax2.set_xticks([1, 5, 10, 20])

    # 3. P-values de comparaciones entre horizontes
    ax3 = fig.add_subplot(gs[1, 1])

    if not horizon_results_df.empty:
        # Contar significancias por modelo
        sig_counts = horizon_results_df.groupby('model')['mcnemar_significant'].sum()
        total_counts = horizon_results_df.groupby('model').size()

        pct_significant = (sig_counts / total_counts * 100).sort_values(ascending=False)

        colors = ['green' if x > 50 else 'orange' if x > 0 else 'gray'
                  for x in pct_significant.values]

        ax3.barh(range(len(pct_significant)), pct_significant.values, color=colors, alpha=0.7)
        ax3.set_yticks(range(len(pct_significant)))
        ax3.set_yticklabels(pct_significant.index, fontsize=10)
        ax3.set_xlabel('% Comparaciones Significativas', fontsize=11, fontweight='bold')
        ax3.set_title('Modelos con Mayor Variación\nentre Horizontes', fontsize=12, fontweight='bold')
        ax3.grid(True, alpha=0.3, axis='x')

    # 4. Heatmap: Accuracy por modelo y horizonte
    ax4 = fig.add_subplot(gs[1, 2])

    if not horizon_results_df.empty:
        # Crear matriz para heatmap
        pivot_for_heatmap = []

        for model in horizon_results_df['model'].unique():
            model_data = horizon_results_df[
                (horizon_results_df['model'] == model) &
                (horizon_results_df['dataset'] == 'financial_scaled')
            ]

            horizons_dict = {}
            for _, row in model_data.iterrows():
                horizons_dict[row['days_1']] = row['acc_horizon_1']
                horizons_dict[row['days_2']] = row['acc_horizon_2']

            pivot_for_heatmap.append({
                'model': model,
                '1d': horizons_dict.get(1, np.nan),
                '5d': horizons_dict.get(5, np.nan),
                '10d': horizons_dict.get(10, np.nan),
                '20d': horizons_dict.get(20, np.nan)
            })

        if pivot_for_heatmap:
            heatmap_df = pd.DataFrame(pivot_for_heatmap).set_index('model')

            im = ax4.imshow(heatmap_df.values, cmap='RdYlGn', aspect='auto', vmin=0.45, vmax=0.65)

            ax4.set_xticks(range(len(heatmap_df.columns)))
            ax4.set_xticklabels(heatmap_df.columns, fontsize=10)
            ax4.set_yticks(range(len(heatmap_df.index)))
            ax4.set_yticklabels(heatmap_df.index, fontsize=10)

            # Añadir valores
            for i in range(len(heatmap_df.index)):
                for j in range(len(heatmap_df.columns)):
                    val = heatmap_df.iloc[i, j]
                    if not np.isnan(val):
                        ax4.text(j, i, f'{val:.3f}', ha='center', va='center',
                                color='black', fontsize=9, fontweight='bold')

            ax4.set_title('Accuracy por Horizonte\n(Financial Scaled)', fontsize=12, fontweight='bold')
            plt.colorbar(im, ax=ax4, label='Accuracy')

    # 5. Tendencia: ¿Accuracy mejora o empeora con horizonte?
    ax5 = fig.add_subplot(gs[2, :])

    if not horizon_results_df.empty:
        for dataset in ['financial_scaled', 'sentiment_scaled']:
            dataset_data = horizon_results_df[horizon_results_df['dataset'] == dataset]

            if dataset_data.empty:
                continue

            # Calcular tendencia promedio
            trend_data = []
            for days in [1, 5, 10, 20]:
                accs = []
                for _, row in dataset_data.iterrows():
                    if row['days_1'] == days:
                        accs.append(row['acc_horizon_1'])
                    if row['days_2'] == days:
                        accs.append(row['acc_horizon_2'])

                if accs:
                    trend_data.append({
                        'days': days,
                        'mean_acc': np.mean(accs),
                        'std_acc': np.std(accs)
                    })

            if trend_data:
                trend_df = pd.DataFrame(trend_data)
                label = 'Financial' if 'financial' in dataset else 'Sentiment'
                marker = 'o' if 'financial' in dataset else 's'

                ax5.plot(trend_df['days'], trend_df['mean_acc'],
                        marker=marker, linestyle='-', linewidth=3, markersize=10,
                        label=label)
                ax5.fill_between(trend_df['days'],
                                trend_df['mean_acc'] - trend_df['std_acc'],
                                trend_df['mean_acc'] + trend_df['std_acc'],
                                alpha=0.2)

        ax5.axhline(0.5, color='red', linestyle=':', linewidth=2, label='Random')
        ax5.set_xlabel('Horizonte de Predicción (días)', fontsize=12, fontweight='bold')
        ax5.set_ylabel('Accuracy Promedio', fontsize=12, fontweight='bold')
        ax5.set_title('Tendencia General: ¿Los Horizontes Más Largos Son Más Predecibles?',
                     fontsize=14, fontweight='bold')
        ax5.legend(fontsize=11)
        ax5.grid(True, alpha=0.3)
        ax5.set_xticks([1, 5, 10, 20])

    plt.tight_layout()
    plt.show()

print("✓ Función de visualización de horizontes definida")


def plot_regression_horizon_analysis(horizon_results_df):
    """
    Visualiza el análisis de horizontes para regresión
    """

    if horizon_results_df.empty:
        print("No hay datos para visualizar")
        return

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Análisis de Horizontes Temporales - REGRESIÓN',
                 fontsize=16, fontweight='bold', y=1.02)

    # 1. MAE por horizonte (Financial vs Sentiment)
    ax = axes[0, 0]

    for dataset in ['financial_scaled', 'sentiment_scaled']:
        data = horizon_results_df[horizon_results_df['dataset'] == dataset]
        if data.empty:
            continue

        for model in data['model'].unique():
            model_data = data[data['model'] == model]

            # Extraer MAEs por horizonte
            horizons_dict = {}
            for _, row in model_data.iterrows():
                horizons_dict[row['days_1']] = row['mae_horizon_1']
                horizons_dict[row['days_2']] = row['mae_horizon_2']

            days = sorted(horizons_dict.keys())
            maes = [horizons_dict[d] for d in days]

            label = f"{model} ({'Fin' if 'financial' in dataset else 'Sent'})"
            linestyle = '-' if 'financial' in dataset else '--'
            ax.plot(days, maes, marker='o', linestyle=linestyle, label=label, linewidth=2)

    ax.set_xlabel('Días', fontweight='bold')
    ax.set_ylabel('MAE', fontweight='bold')
    ax.set_title('MAE por Horizonte Temporal', fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # 2. RMSE por horizonte
    ax = axes[0, 1]

    for dataset in ['financial_scaled', 'sentiment_scaled']:
        data = horizon_results_df[horizon_results_df['dataset'] == dataset]
        if data.empty:
            continue

        for model in data['model'].unique():
            model_data = data[data['model'] == model]

            horizons_dict = {}
            for _, row in model_data.iterrows():
                horizons_dict[row['days_1']] = row['rmse_horizon_1']
                horizons_dict[row['days_2']] = row['rmse_horizon_2']

            days = sorted(horizons_dict.keys())
            rmses = [horizons_dict[d] for d in days]

            label = f"{model} ({'Fin' if 'financial' in dataset else 'Sent'})"
            linestyle = '-' if 'financial' in dataset else '--'
            ax.plot(days, rmses, marker='s', linestyle=linestyle, label=label, linewidth=2)

    ax.set_xlabel('Días', fontweight='bold')
    ax.set_ylabel('RMSE', fontweight='bold')
    ax.set_title('RMSE por Horizonte Temporal', fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # 3. Mapa de calor de diferencias significativas
    ax = axes[1, 0]

    # Matriz: modelos × comparaciones
    pivot_data = horizon_results_df.pivot_table(
        index='model',
        columns=['days_1', 'days_2'],
        values='dm_significant',
        aggfunc='mean'
    )

    if not pivot_data.empty:
        sns.heatmap(pivot_data, annot=True, fmt='.0f', cmap='RdYlGn',
                   center=0.5, ax=ax, cbar_kws={'label': 'Proporción sig.'})
        ax.set_title('Diferencias Significativas (DM test)', fontweight='bold')
        ax.set_xlabel('Comparación (días)', fontweight='bold')
        ax.set_ylabel('Modelo', fontweight='bold')

    # 4. Distribución de p-values
    ax = axes[1, 1]

    for dataset in ['financial_scaled', 'sentiment_scaled']:
        data = horizon_results_df[
            (horizon_results_df['dataset'] == dataset) &
            (horizon_results_df['dm_pvalue'].notna())
        ]

        if data.empty:
            continue

        label = 'Financial' if 'financial' in dataset else 'Sentiment'
        ax.hist(data['dm_pvalue'], bins=20, alpha=0.5, label=label, edgecolor='black')

    ax.axvline(0.05, color='red', linestyle='--', linewidth=2, label='α=0.05')
    ax.set_xlabel('P-value (DM test)', fontweight='bold')
    ax.set_ylabel('Frecuencia', fontweight='bold')
    ax.set_title('Distribución de P-values', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(f'{CSV}/regression_horizon_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

    print(f"\n✓ Gráfico guardado: {CSV}/regression_horizon_analysis.png")



def plot_training_history(history, model_name, dataset_name, target_name,
                          save_path=None, task_type='regression',
                          figsize=(12, 5), show_plot=True):
    """
    Genera gráficos de pérdida y métricas durante el entrenamiento.

    Parameters:
    -----------
    history : keras.callbacks.History or dict
        Objeto History o diccionario con history.history
    model_name : str
        Nombre del modelo
    dataset_name : str
        Nombre del dataset
    target_name : str
        Variable objetivo
    save_path : str, optional
        Directorio donde guardar el gráfico. Si None, no guarda
    task_type : str, default='regression'
        'regression' o 'classification'
    figsize : tuple, default=(12, 5)
        Tamaño de la figura
    show_plot : bool, default=False
        Si True, muestra el gráfico con plt.show()

    Returns:
    --------
    str or None : Ruta del archivo guardado, o None si save_path es None
    """

    # Extraer diccionario de history
    if hasattr(history, 'history'):
        hist_dict = history.history
    else:
        hist_dict = history

    # Determinar métrica adicional según tipo de tarea
    if task_type == 'regression':
        metric_key = 'mae' if 'mae' in hist_dict else 'mean_absolute_error'
        metric_label = 'MAE'
    else:  # classification
        metric_key = 'accuracy' if 'accuracy' in hist_dict else 'acc'
        metric_label = 'Accuracy'

    # Crear figura
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    epochs = range(1, len(hist_dict['loss']) + 1)

    # Plot 1: Loss
    axes[0].plot(epochs, hist_dict['loss'], 'b-', linewidth=2, label='Train Loss')
    axes[0].plot(epochs, hist_dict['val_loss'], 'r--', linewidth=2, label='Val Loss')
    axes[0].set_xlabel('Epoch', fontsize=11, fontweight='bold')
    axes[0].set_ylabel('Loss', fontsize=11, fontweight='bold')
    axes[0].set_title(f'Training & Validation Loss\n{model_name}',
                      fontsize=12, fontweight='bold')
    axes[0].legend(loc='best', fontsize=10)
    axes[0].grid(True, alpha=0.3)

    # Marcar mejor epoch
    best_epoch = hist_dict['val_loss'].index(min(hist_dict['val_loss'])) + 1
    best_val_loss = min(hist_dict['val_loss'])
    axes[0].axvline(best_epoch, color='green', linestyle=':', alpha=0.7, linewidth=1.5)
    axes[0].scatter([best_epoch], [best_val_loss], color='green', s=100,
                   zorder=5, marker='*', label=f'Best (epoch {best_epoch})')
    axes[0].legend(loc='best', fontsize=9)

    # Plot 2: Métrica adicional
    if metric_key in hist_dict and f'val_{metric_key}' in hist_dict:
        axes[1].plot(epochs, hist_dict[metric_key], 'b-', linewidth=2,
                    label=f'Train {metric_label}')
        axes[1].plot(epochs, hist_dict[f'val_{metric_key}'], 'r--', linewidth=2,
                    label=f'Val {metric_label}')
        axes[1].set_xlabel('Epoch', fontsize=11, fontweight='bold')
        axes[1].set_ylabel(metric_label, fontsize=11, fontweight='bold')
        axes[1].set_title(f'Training & Validation {metric_label}\n{model_name}',
                         fontsize=12, fontweight='bold')
        axes[1].legend(loc='best', fontsize=10)
        axes[1].grid(True, alpha=0.3)

        # Marcar mejor epoch en métrica
        if task_type == 'regression':
            best_metric_epoch = hist_dict[f'val_{metric_key}'].index(
                min(hist_dict[f'val_{metric_key}'])) + 1
        else:
            best_metric_epoch = hist_dict[f'val_{metric_key}'].index(
                max(hist_dict[f'val_{metric_key}'])) + 1

        axes[1].axvline(best_metric_epoch, color='green', linestyle=':',
                       alpha=0.7, linewidth=1.5)

    plt.suptitle(f'{dataset_name} - {target_name}',
                fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()

    # Guardar si se especifica ruta
    filepath = None
    if save_path:
        os.makedirs(save_path, exist_ok=True)
        filename = f"{dataset_name}_{target_name}_{model_name}_training_history.png"
        filepath = os.path.join(save_path, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"  ✓ Gráfico guardado: {filepath}")

    if show_plot:
        plt.show()
    else:
        plt.close()

    return filepath


def plot_confusion_matrix(y_true, y_pred, title="Confusion Matrix", model_name=None):
    """Plot confusion matrix"""
    mask = ~np.isnan(y_true)
    cm = confusion_matrix(y_true[mask], y_pred[mask])

    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Down (0)', 'Up (1)'],
                yticklabels=['Down (0)', 'Up (1)'])
    plt.title(title, fontsize=14, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    #Guardar figura
    if model_name:
        filename = model_name.replace(' ', '_').replace('/', '_').replace('\\', '_')
        filepath = os.path.join(PLOTS, f'{filename}_confusion_matrix.png')
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"   Confusion matrix guardada: {filepath}")
    #Mostrar figura
    plt.show()

# Mensaje de confirmación
print("✓ Funciones de análisis de horizontes para REGRESIÓN definidas")

# -----------------------------------------------------------------------------
# 1. EVOLUCIÓN DE MÉTRICAS POR HORIZONTE - CLASIFICACIÓN
# -----------------------------------------------------------------------------

def plot_classification_horizon_evolution(results_df):
    """
    Muestra cómo cambia el accuracy cuando aumenta el horizonte temporal
    """
    
    if results_df.empty:
        print("No hay datos para visualizar")
        return
    
    # Crear dataset largo con todos los puntos
    plot_data = []
    
    for _, row in results_df.iterrows():
        plot_data.append({
            'dataset': row['dataset'],
            'model': row['model'],
            'days': row['days_1'],
            'accuracy': row['acc_horizon_1']
        })
        plot_data.append({
            'dataset': row['dataset'],
            'model': row['model'],
            'days': row['days_2'],
            'accuracy': row['acc_horizon_2']
        })
    
    plot_df = pd.DataFrame(plot_data).drop_duplicates()
    
    # Calcular promedio por horizonte y dataset
    avg_by_horizon = plot_df.groupby(['dataset', 'days'])['accuracy'].mean().reset_index()
    
    # Determinar grid dinámicamente
    datasets = plot_df['dataset'].unique()
    n_datasets = len(datasets)
    n_cols = 2
    n_rows = (n_datasets + n_cols - 1) // n_cols  # Redondear hacia arriba
    
    # Plot
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 5 * n_rows))
    fig.suptitle('Evolución del Accuracy por Horizonte Temporal', fontsize=14, fontweight='bold')
    
    # Aplanar axes si es necesario
    if n_datasets == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    
    for idx, dataset in enumerate(datasets):
        ax = axes[idx]
        
        data_subset = plot_df[plot_df['dataset'] == dataset]
        avg_subset = avg_by_horizon[avg_by_horizon['dataset'] == dataset]
        
        # Líneas individuales por modelo (suaves)
        for model in data_subset['model'].unique():
            model_data = data_subset[data_subset['model'] == model].sort_values('days')
            ax.plot(model_data['days'], model_data['accuracy'], 
                   alpha=0.3, linewidth=1, marker='o', markersize=3)
        
        # Línea promedio (destacada)
        ax.plot(avg_subset['days'], avg_subset['accuracy'], 
               color='red', linewidth=3, marker='o', markersize=8,
               label='Promedio', zorder=10)
        
        # Línea de referencia 50%
        ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Azar (50%)')
        
        ax.set_xlabel('Días de predicción')
        ax.set_ylabel('Accuracy')
        ax.set_title(dataset)
        ax.set_xticks([1, 5, 10, 20])
        ax.grid(True, alpha=0.3)
        ax.legend()
    
    # Ocultar subplots vacíos
    for idx in range(n_datasets, len(axes)):
        axes[idx].set_visible(False)
    
    plt.tight_layout()
    filepath = os.path.join(PLOTS, 'horizon_evolution_classification.png')
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print("✓ Guardado: horizon_evolution_classification.png")
    plt.show()
print("Función horizon 1 cargada correctamente")
# -----------------------------------------------------------------------------
# 2. HEATMAP DE DIFERENCIAS SIGNIFICATIVAS - CLASIFICACIÓN
# -----------------------------------------------------------------------------
def plot_classification_significance_heatmap(results_df):
    """
    Heatmap mostrando qué comparaciones son significativamente diferentes
    """
    
    if results_df.empty:
        print("No hay datos para visualizar")
        return
    
    # Determinar grid dinámicamente
    datasets = results_df['dataset'].unique()
    n_datasets = len(datasets)
    n_cols = 2
    n_rows = (n_datasets + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 5 * n_rows))
    fig.suptitle('Diferencias Significativas entre Horizontes (Clasificación)', 
                 fontsize=14, fontweight='bold')
    
    # Aplanar axes
    if n_datasets == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    
    for idx, dataset in enumerate(datasets):
        ax = axes[idx]
        
        data_subset = results_df[results_df['dataset'] == dataset]
        
        # Crear matriz de diferencias
        models = sorted(data_subset['model'].unique())
        horizons = [(1, 5), (5, 10), (10, 20)]
        
        matrix = np.zeros((len(models), len(horizons)))
        
        for i, model in enumerate(models):
            model_data = data_subset[data_subset['model'] == model]
            
            for j, (h1, h2) in enumerate(horizons):
                row = model_data[(model_data['days_1'] == h1) & 
                                (model_data['days_2'] == h2)]
                
                if len(row) > 0:
                    # Positivo = mejora, Negativo = empeora
                    diff = row['diff'].values[0]
                    sig = row['mcnemar_significant'].values[0]
                    
                    # Solo marcar si es significativo
                    matrix[i, j] = diff if sig else 0
        
        # Plot heatmap
        im = ax.imshow(matrix, cmap='RdYlGn', aspect='auto', 
                      vmin=-0.1, vmax=0.1, interpolation='nearest')
        
        # Etiquetas
        ax.set_xticks(range(len(horizons)))
        ax.set_xticklabels([f'{h1}→{h2}d' for h1, h2 in horizons])
        ax.set_yticks(range(len(models)))
        ax.set_yticklabels(models, fontsize=8)
        
        # Valores en celdas
        for i in range(len(models)):
            for j in range(len(horizons)):
                if matrix[i, j] != 0:
                    text = ax.text(j, i, f'{matrix[i, j]:+.3f}',
                                 ha="center", va="center", 
                                 color="black", fontsize=8, fontweight='bold')
        
        ax.set_title(dataset)
        plt.colorbar(im, ax=ax, label='Δ Accuracy')
    
    # Ocultar subplots vacíos
    for idx in range(n_datasets, len(axes)):
        axes[idx].set_visible(False)
    
    plt.tight_layout()
    filepath = os.path.join(PLOTS, 'horizon_significance_heatmap_classification.png')
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print("✓ Guardado: horizon_significance_heatmap_classification.png")
    plt.show()

# -----------------------------------------------------------------------------
# 3. EVOLUCIÓN DE MAE/RMSE POR HORIZONTE - REGRESIÓN
# -----------------------------------------------------------------------------
def plot_regression_horizon_evolution(results_df):
    """
    Muestra cómo cambian MAE y RMSE cuando aumenta el horizonte
    """
    
    if results_df.empty:
        print("No hay datos para visualizar")
        return
    
    # Crear dataset largo
    plot_data_mae = []
    plot_data_rmse = []
    
    for _, row in results_df.iterrows():
        plot_data_mae.append({
            'dataset': row['dataset'],
            'model': row['model'],
            'days': row['days_1'],
            'mae': row['mae_horizon_1']
        })
        plot_data_mae.append({
            'dataset': row['dataset'],
            'model': row['model'],
            'days': row['days_2'],
            'mae': row['mae_horizon_2']
        })
        
        plot_data_rmse.append({
            'dataset': row['dataset'],
            'model': row['model'],
            'days': row['days_1'],
            'rmse': row['rmse_horizon_1']
        })
        plot_data_rmse.append({
            'dataset': row['dataset'],
            'model': row['model'],
            'days': row['days_2'],
            'rmse': row['rmse_horizon_2']
        })
    
    plot_df_mae = pd.DataFrame(plot_data_mae).drop_duplicates()
    plot_df_rmse = pd.DataFrame(plot_data_rmse).drop_duplicates()
    
    # Determinar grid dinámicamente
    datasets = plot_df_mae['dataset'].unique()
    n_datasets = len(datasets)
    n_cols = 2
    n_rows = (n_datasets + n_cols - 1) // n_cols
    
    # Plot MAE
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 5 * n_rows))
    fig.suptitle('Evolución del MAE por Horizonte Temporal', fontsize=14, fontweight='bold')
    
    # Aplanar axes
    if n_datasets == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    
    for idx, dataset in enumerate(datasets):
        ax = axes[idx]
        
        data_subset = plot_df_mae[plot_df_mae['dataset'] == dataset]
        avg_by_horizon = data_subset.groupby('days')['mae'].mean().reset_index()
        
        # Líneas individuales
        for model in data_subset['model'].unique():
            model_data = data_subset[data_subset['model'] == model].sort_values('days')
            ax.plot(model_data['days'], model_data['mae'], 
                   alpha=0.3, linewidth=1, marker='o', markersize=3)
        
        # Promedio
        ax.plot(avg_by_horizon['days'], avg_by_horizon['mae'], 
               color='red', linewidth=3, marker='o', markersize=8,
               label='Promedio', zorder=10)
        
        ax.set_xlabel('Días de predicción')
        ax.set_ylabel('MAE')
        ax.set_title(dataset)
        ax.set_xticks([1, 5, 10, 20])
        ax.grid(True, alpha=0.3)
        ax.legend()
    
    # Ocultar subplots vacíos
    for idx in range(n_datasets, len(axes)):
        axes[idx].set_visible(False)
    
    plt.tight_layout()
    filepath = os.path.join(PLOTS, 'horizon_evolution_mae_regression.png')
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print("✓ Guardado: horizon_evolution_mae_regression.png")
    plt.show()
# -----------------------------------------------------------------------------
# 4. BOXPLOT COMPARATIVO - REGRESIÓN
# -----------------------------------------------------------------------------
def plot_regression_horizon_boxplot(results_df):
    """
    Boxplot del MAE por horizonte temporal
    """
    
    if results_df.empty:
        print("No hay datos para visualizar")
        return
    
    # Crear dataset largo
    plot_data = []
    
    for _, row in results_df.iterrows():
        plot_data.append({
            'dataset': row['dataset'],
            'days': row['days_1'],
            'mae': row['mae_horizon_1']
        })
        plot_data.append({
            'dataset': row['dataset'],
            'days': row['days_2'],
            'mae': row['mae_horizon_2']
        })
    
    plot_df = pd.DataFrame(plot_data).drop_duplicates()
    
    # Determinar grid dinámicamente
    datasets = plot_df['dataset'].unique()
    n_datasets = len(datasets)
    n_cols = 2
    n_rows = (n_datasets + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 5 * n_rows))
    fig.suptitle('Distribución del MAE por Horizonte Temporal', 
                 fontsize=14, fontweight='bold')
    
    # Aplanar axes
    if n_datasets == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    
    for idx, dataset in enumerate(datasets):
        ax = axes[idx]
        
        data_subset = plot_df[plot_df['dataset'] == dataset]
        
        # Boxplot
        sns.boxplot(data=data_subset, x='days', y='mae', ax=ax, palette='Set2')
        
        # Puntos individuales
        sns.stripplot(data=data_subset, x='days', y='mae', ax=ax, 
                     color='black', alpha=0.3, size=3)
        
        ax.set_xlabel('Días de predicción')
        ax.set_ylabel('MAE')
        ax.set_title(dataset)
        ax.grid(True, alpha=0.3, axis='y')
    
    # Ocultar subplots vacíos
    for idx in range(n_datasets, len(axes)):
        axes[idx].set_visible(False)
    
    plt.tight_layout()
    filepath = os.path.join(PLOTS, 'horizon_boxplot_mae_regression.png')
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print("✓ Guardado: horizon_boxplot_mae_regression.png")
    plt.show()
# -----------------------------------------------------------------------------
# 5. RESUMEN EJECUTIVO - AMBOS
# -----------------------------------------------------------------------------

def plot_executive_summary(results_class, results_reg):
    """
    Panel resumen con las métricas clave
    """
    
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    fig.suptitle('Resumen Ejecutivo: Efecto del Horizonte Temporal', 
                 fontsize=16, fontweight='bold')
    
    # 1. Accuracy promedio por horizonte (clasificación)
    ax1 = fig.add_subplot(gs[0, :2])
    
    if not results_class.empty:
        plot_data = []
        for _, row in results_class.iterrows():
            plot_data.extend([
                {'days': row['days_1'], 'accuracy': row['acc_horizon_1']},
                {'days': row['days_2'], 'accuracy': row['acc_horizon_2']}
            ])
        
        plot_df = pd.DataFrame(plot_data).drop_duplicates()
        avg_acc = plot_df.groupby('days')['accuracy'].agg(['mean', 'std']).reset_index()
        
        ax1.errorbar(avg_acc['days'], avg_acc['mean'], yerr=avg_acc['std'],
                    marker='o', linewidth=2, markersize=8, capsize=5)
        ax1.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='Azar')
        ax1.set_xlabel('Días de predicción')
        ax1.set_ylabel('Accuracy promedio')
        ax1.set_title('Clasificación: Accuracy vs Horizonte')
        ax1.set_xticks([1, 5, 10, 20])
        ax1.grid(True, alpha=0.3)
        ax1.legend()
    
    # 2. % Comparaciones significativas (clasificación)
    ax2 = fig.add_subplot(gs[0, 2])
    
    if not results_class.empty:
        sig_pct = (results_class['mcnemar_significant'].sum() / 
                   len(results_class) * 100)
        
        ax2.bar(['Significativas', 'No significativas'], 
               [sig_pct, 100-sig_pct],
               color=['green', 'gray'])
        ax2.set_ylabel('% Comparaciones')
        ax2.set_title('Diferencias Significativas\n(Clasificación)')
        ax2.set_ylim([0, 100])
        
        for i, v in enumerate([sig_pct, 100-sig_pct]):
            ax2.text(i, v + 2, f'{v:.1f}%', ha='center', fontweight='bold')
    
    # 3. MAE promedio por horizonte (regresión)
    ax3 = fig.add_subplot(gs[1, :2])
    
    if not results_reg.empty:
        plot_data = []
        for _, row in results_reg.iterrows():
            plot_data.extend([
                {'days': row['days_1'], 'mae': row['mae_horizon_1']},
                {'days': row['days_2'], 'mae': row['mae_horizon_2']}
            ])
        
        plot_df = pd.DataFrame(plot_data).drop_duplicates()
        avg_mae = plot_df.groupby('days')['mae'].agg(['mean', 'std']).reset_index()
        
        ax3.errorbar(avg_mae['days'], avg_mae['mean'], yerr=avg_mae['std'],
                    marker='o', linewidth=2, markersize=8, capsize=5, color='orange')
        ax3.set_xlabel('Días de predicción')
        ax3.set_ylabel('MAE promedio')
        ax3.set_title('Regresión: MAE vs Horizonte')
        ax3.set_xticks([1, 5, 10, 20])
        ax3.grid(True, alpha=0.3)
    
    # 4. % Comparaciones significativas (regresión)
    ax4 = fig.add_subplot(gs[1, 2])
    
    if not results_reg.empty:
        sig_pct = (results_reg['dm_significant'].sum() / 
                   len(results_reg) * 100)
        
        ax4.bar(['Significativas', 'No significativas'], 
               [sig_pct, 100-sig_pct],
               color=['green', 'gray'])
        ax4.set_ylabel('% Comparaciones')
        ax4.set_title('Diferencias Significativas\n(Regresión)')
        ax4.set_ylim([0, 100])
        
        for i, v in enumerate([sig_pct, 100-sig_pct]):
            ax4.text(i, v + 2, f'{v:.1f}%', ha='center', fontweight='bold')
    
    # 5. Correlación días vs métrica
    ax5 = fig.add_subplot(gs[2, 0])
    
    if not results_class.empty:
        all_days = []
        all_accs = []
        for _, row in results_class.iterrows():
            all_days.extend([row['days_1'], row['days_2']])
            all_accs.extend([row['acc_horizon_1'], row['acc_horizon_2']])
        
        corr = np.corrcoef(all_days, all_accs)[0, 1]
        
        ax5.scatter(all_days, all_accs, alpha=0.3)
        z = np.polyfit(all_days, all_accs, 1)
        p = np.poly1d(z)
        ax5.plot([1, 20], [p(1), p(20)], "r--", linewidth=2)
        
        ax5.set_xlabel('Días')
        ax5.set_ylabel('Accuracy')
        ax5.set_title(f'Clasificación\nCorr = {corr:.3f}')
        ax5.grid(True, alpha=0.3)
    
    # 6. Correlación días vs MAE
    ax6 = fig.add_subplot(gs[2, 1])
    
    if not results_reg.empty:
        all_days = []
        all_maes = []
        for _, row in results_reg.iterrows():
            all_days.extend([row['days_1'], row['days_2']])
            all_maes.extend([row['mae_horizon_1'], row['mae_horizon_2']])
        
        corr = np.corrcoef(all_days, all_maes)[0, 1]
        
        ax6.scatter(all_days, all_maes, alpha=0.3, color='orange')
        z = np.polyfit(all_days, all_maes, 1)
        p = np.poly1d(z)
        ax6.plot([1, 20], [p(1), p(20)], "r--", linewidth=2)
        
        ax6.set_xlabel('Días')
        ax6.set_ylabel('MAE')
        ax6.set_title(f'Regresión\nCorr = {corr:.3f}')
        ax6.grid(True, alpha=0.3)
    
    # 7. Texto resumen
    ax7 = fig.add_subplot(gs[2, 2])
    ax7.axis('off')
    
    summary_text = "CONCLUSIONES:\n\n"
    
    if not results_class.empty:
        all_days = []
        all_accs = []
        for _, row in results_class.iterrows():
            all_days.extend([row['days_1'], row['days_2']])
            all_accs.extend([row['acc_horizon_1'], row['acc_horizon_2']])
        corr_class = np.corrcoef(all_days, all_accs)[0, 1]
        
        if corr_class > 0.3:
            summary_text += "• Clasificación: Mejora\n  con horizonte largo\n\n"
        elif corr_class < -0.3:
            summary_text += "• Clasificación: Empeora\n  con horizonte largo\n\n"
        else:
            summary_text += "• Clasificación: Sin\n  tendencia clara\n\n"
    
    if not results_reg.empty:
        all_days = []
        all_maes = []
        for _, row in results_reg.iterrows():
            all_days.extend([row['days_1'], row['days_2']])
            all_maes.extend([row['mae_horizon_1'], row['mae_horizon_2']])
        corr_reg = np.corrcoef(all_days, all_maes)[0, 1]
        
        if corr_reg > 0.3:
            summary_text += "• Regresión: Mayor error\n  con horizonte largo"
        elif corr_reg < -0.3:
            summary_text += "• Regresión: Menor error\n  con horizonte largo"
        else:
            summary_text += "• Regresión: Sin\n  tendencia clara"
    
    ax7.text(0.1, 0.5, summary_text, fontsize=10, 
            verticalalignment='center', fontweight='bold')
    
    filepath = os.path.join(PLOTS, 'executive_summary_horizons.png')
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print("✓ Guardado: executive_summary_horizons.png")
    plt.show()

print("✓ Funciones de visualización definidas")