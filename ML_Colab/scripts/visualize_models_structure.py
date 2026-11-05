#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Visualización de estructura interna de modelos ML
Genera plots explicativos para cada tipo de modelo entrenado

@author: santi
"""

import os
import joblib
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Imports específicos por tipo de modelo
import lightgbm as lgb
import xgboost as xgb
from sklearn.tree import plot_tree
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import plot_model

# Configuración
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10


def visualize_lightgbm(model_path, output_dir, tree_index=0):
    """
    Visualiza árbol de decisión de LightGBM
    
    Args:
        model_path: Ruta al modelo .joblib
        output_dir: Directorio de salida para imágenes
        tree_index: Índice del árbol a visualizar (default: 0)
    """
    model = joblib.load(model_path)
    model_name = Path(model_path).stem
    
    fig, ax = plt.subplots(figsize=(24, 16))
    lgb.plot_tree(
        model.booster_, 
        tree_index=tree_index, 
        ax=ax,
        show_info=['split_gain', 'internal_value', 'leaf_count']
    )
    ax.set_title(f'LightGBM - Árbol {tree_index}\n{model_name}', fontsize=14, fontweight='bold')
    
    output_path = output_dir / f'{model_name}_tree_{tree_index}.png'
    plt.savefig(output_path, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  ✓ Guardado: {output_path}")
    
    # Feature importance
    fig, ax = plt.subplots(figsize=(12, 8))
    lgb.plot_importance(model.booster_, ax=ax, max_num_features=20, importance_type='gain')
    ax.set_title(f'LightGBM - Feature Importance (Gain)\n{model_name}', fontsize=14, fontweight='bold')
    
    output_path = output_dir / f'{model_name}_importance.png'
    plt.savefig(output_path, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  ✓ Guardado: {output_path}")
    
    return model


def visualize_xgboost(model_path, output_dir, tree_index=0):
    """
    Visualiza árbol de decisión de XGBoost
    
    Args:
        model_path: Ruta al modelo .joblib
        output_dir: Directorio de salida para imágenes
        tree_index: Índice del árbol a visualizar (default: 0)
    """
    model = joblib.load(model_path)
    model_name = Path(model_path).stem
    
    fig, ax = plt.subplots(figsize=(24, 16))
    xgb.plot_tree(model, tree_idx=tree_index, ax=ax)
    ax.set_title(f'XGBoost - Árbol {tree_index}\n{model_name}', fontsize=14, fontweight='bold')
    
    output_path = output_dir / f'{model_name}_tree_{tree_index}.png'
    plt.savefig(output_path, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  ✓ Guardado: {output_path}")
    
    # Feature importance
    fig, ax = plt.subplots(figsize=(12, 8))
    xgb.plot_importance(model, ax=ax, max_num_features=20, importance_type='gain')
    ax.set_title(f'XGBoost - Feature Importance (Gain)\n{model_name}', fontsize=14, fontweight='bold')
    
    output_path = output_dir / f'{model_name}_importance.png'
    plt.savefig(output_path, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  ✓ Guardado: {output_path}")
    
    return model


def visualize_random_forest(model_path, output_dir, tree_index=0, max_depth=4):
    """
    Visualiza árbol de Random Forest (sklearn)
    
    Args:
        model_path: Ruta al modelo .joblib
        output_dir: Directorio de salida para imágenes
        tree_index: Índice del árbol a visualizar (default: 0)
        max_depth: Profundidad máxima a mostrar (para legibilidad)
    """
    model = joblib.load(model_path)
    model_name = Path(model_path).stem
    
    # Visualizar un árbol individual
    fig, ax = plt.subplots(figsize=(24, 16))
    plot_tree(
        model.estimators_[tree_index],
        max_depth=max_depth,
        filled=True,
        rounded=True,
        ax=ax,
        fontsize=8
    )
    ax.set_title(f'Random Forest - Árbol {tree_index} (max_depth={max_depth})\n{model_name}', 
                 fontsize=14, fontweight='bold')
    
    output_path = output_dir / f'{model_name}_tree_{tree_index}.png'
    plt.savefig(output_path, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  ✓ Guardado: {output_path}")
    
    # Feature importance
    fig, ax = plt.subplots(figsize=(12, 8))
    importances = model.feature_importances_
    indices = np.argsort(importances)[-20:]  # Top 20
    
    ax.barh(range(len(indices)), importances[indices], align='center')
    ax.set_yticks(range(len(indices)))
    ax.set_yticklabels([f'Feature {i}' for i in indices])
    ax.set_xlabel('Importance')
    ax.set_title(f'Random Forest - Feature Importance\n{model_name}', fontsize=14, fontweight='bold')
    
    output_path = output_dir / f'{model_name}_importance.png'
    plt.savefig(output_path, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  ✓ Guardado: {output_path}")
    
    return model


def visualize_mlp(model_path, output_dir):
    """
    Visualiza arquitectura de MLP (sklearn)
    Genera diagrama de capas y pesos
    
    Args:
        model_path: Ruta al modelo .joblib
        output_dir: Directorio de salida para imágenes
    """
    model = joblib.load(model_path)
    model_name = Path(model_path).stem
    
    # Extraer arquitectura
    n_layers = len(model.coefs_)
    layer_sizes = [model.coefs_[0].shape[0]]  # Input layer
    for i, coef in enumerate(model.coefs_):
        layer_sizes.append(coef.shape[1])
    
    # Crear visualización de arquitectura
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # Posiciones de las capas
    layer_positions = np.linspace(0.1, 0.9, len(layer_sizes))
    max_neurons = max(layer_sizes)
    
    for i, (pos, size) in enumerate(zip(layer_positions, layer_sizes)):
        # Limitar neuronas mostradas para legibilidad
        neurons_to_show = min(size, 20)
        neuron_positions = np.linspace(0.1, 0.9, neurons_to_show)
        
        for j, y in enumerate(neuron_positions):
            circle = plt.Circle((pos, y), 0.015, color='steelblue', ec='black', linewidth=1)
            ax.add_patch(circle)
        
        # Indicar si hay más neuronas
        if size > neurons_to_show:
            ax.text(pos, 0.02, f'...({size} total)', ha='center', fontsize=8)
        
        # Etiqueta de capa
        layer_type = 'Input' if i == 0 else ('Output' if i == len(layer_sizes)-1 else f'Hidden {i}')
        ax.text(pos, 0.98, f'{layer_type}\n({size})', ha='center', va='top', fontsize=10, fontweight='bold')
        
        # Conexiones (simplificadas)
        if i < len(layer_sizes) - 1:
            next_neurons = min(layer_sizes[i+1], 20)
            next_positions = np.linspace(0.1, 0.9, next_neurons)
            for y1 in neuron_positions[::max(1, len(neuron_positions)//5)]:
                for y2 in next_positions[::max(1, len(next_positions)//5)]:
                    ax.plot([pos, layer_positions[i+1]], [y1, y2], 
                           'gray', alpha=0.1, linewidth=0.5)
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title(f'MLP Architecture\n{model_name}\nLayers: {layer_sizes}', 
                fontsize=14, fontweight='bold')
    
    output_path = output_dir / f'{model_name}_architecture.png'
    plt.savefig(output_path, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  ✓ Guardado: {output_path}")
    
    # Loss curve (si está disponible)
    if hasattr(model, 'loss_curve_'):
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(model.loss_curve_, 'b-', linewidth=2)
        ax.set_xlabel('Iterations')
        ax.set_ylabel('Loss')
        ax.set_title(f'MLP Training Loss Curve\n{model_name}', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        output_path = output_dir / f'{model_name}_loss_curve.png'
        plt.savefig(output_path, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"  ✓ Guardado: {output_path}")
    
    return model


def visualize_keras_model(model_path, output_dir):
    """
    Visualiza arquitectura de modelo Keras (GRU, LSTM, etc.)
    
    Args:
        model_path: Ruta al modelo .keras
        output_dir: Directorio de salida para imágenes
    """
    model = load_model(model_path)
    model_name = Path(model_path).stem
    
    # Diagrama de arquitectura con plot_model
    output_path = output_dir / f'{model_name}_architecture.png'
    try:
        plot_model(
            model, 
            to_file=str(output_path),
            show_shapes=True,
            show_layer_names=True,
            show_layer_activations=True,
            dpi=150
        )
        print(f"  ✓ Guardado: {output_path}")
    except Exception as e:
        print(f"  ⚠ Error con plot_model (requiere graphviz/pydot): {e}")
        # Alternativa: resumen textual como imagen
        _create_text_summary(model, output_path, model_name)
    
    # Crear visualización manual de la arquitectura
    _plot_keras_architecture_manual(model, output_dir, model_name)
    
    return model


def visualize_lstm(model_path, output_dir):
    """
    Visualiza arquitectura LSTM con detalles específicos
    Soporta: small, medium, large, xlarge, bidirectional
    
    Args:
        model_path: Ruta al modelo .keras
        output_dir: Directorio de salida para imágenes
    """
    model = load_model(model_path)
    model_name = Path(model_path).stem
    
    # 1. Diagrama con plot_model (si graphviz disponible)
    output_path = output_dir / f'{model_name}_architecture.png'
    try:
        plot_model(
            model, 
            to_file=str(output_path),
            show_shapes=True,
            show_layer_names=True,
            show_layer_activations=True,
            dpi=150
        )
        print(f"  ✓ Guardado: {output_path}")
    except Exception as e:
        print(f"  ⚠ Error con plot_model: {e}")
        _create_text_summary(model, output_path, model_name)
    
    # 2. Visualización manual detallada para LSTM
    _plot_lstm_architecture(model, output_dir, model_name)
    
    # 3. Diagrama de flujo de información
    _plot_lstm_flow(model, output_dir, model_name)
    
    return model


def _plot_lstm_architecture(model, output_dir, model_name):
    """Crea visualización detallada de arquitectura LSTM"""
    fig, ax = plt.subplots(figsize=(12, 14))
    
    layers = model.layers
    n_layers = len(layers)
    y_positions = np.linspace(0.9, 0.1, n_layers)
    
    # Colores por tipo de capa
    colors = {
        'LSTM': '#1E88E5',           # Azul
        'Bidirectional': '#7B1FA2',  # Púrpura
        'GRU': '#43A047',            # Verde
        'Dense': '#FB8C00',          # Naranja
        'Dropout': '#9E9E9E',        # Gris
        'InputLayer': '#E0E0E0'      # Gris claro
    }
    
    for i, (layer, y) in enumerate(zip(layers, y_positions)):
        layer_type = layer.__class__.__name__
        
        # Detectar Bidirectional wrapper
        if layer_type == 'Bidirectional':
            inner_layer = layer.layer.__class__.__name__
            color = colors.get('Bidirectional', '#BBDEFB')
            display_name = f'Bidirectional({inner_layer})'
        else:
            color = colors.get(layer_type, '#BBDEFB')
            display_name = layer_type
        
        # Tamaño del rectángulo según tipo
        if layer_type in ['LSTM', 'Bidirectional', 'GRU']:
            width = 0.7
            height = 0.08
        else:
            width = 0.5
            height = 0.05
        
        # Dibujar capa
        rect = plt.Rectangle(
            (0.5 - width/2, y - height/2), 
            width, height,
            facecolor=color, 
            edgecolor='black', 
            linewidth=2,
            alpha=0.9
        )
        ax.add_patch(rect)
        
        # Información de la capa
        try:
            output_shape = layer.output_shape
            if isinstance(output_shape, list):
                output_shape = output_shape[0]
        except:
            output_shape = 'N/A'
        
        params = layer.count_params()
        
        # Extraer info específica de LSTM
        config_info = ""
        if hasattr(layer, 'units'):
            config_info = f"units={layer.units}"
        elif layer_type == 'Bidirectional' and hasattr(layer.layer, 'units'):
            config_info = f"units={layer.layer.units}×2"
        elif layer_type == 'Dropout' and hasattr(layer, 'rate'):
            config_info = f"rate={layer.rate:.1f}"
        elif layer_type == 'Dense' and hasattr(layer, 'units'):
            config_info = f"units={layer.units}"
        
        # Texto en la capa
        text_lines = [display_name]
        if config_info:
            text_lines.append(config_info)
        text_lines.append(f"out: {output_shape}")
        text_lines.append(f"params: {params:,}")
        
        ax.text(0.5, y, '\n'.join(text_lines),
               ha='center', va='center', fontsize=8, fontweight='bold')
        
        # Flecha de conexión
        if i < n_layers - 1:
            ax.annotate('', 
                       xy=(0.5, y_positions[i+1] + height/2 + 0.01), 
                       xytext=(0.5, y - height/2 - 0.01),
                       arrowprops=dict(arrowstyle='->', color='black', lw=1.5))
    
    # Leyenda
    legend_y = 0.02
    legend_items = [
        ('LSTM', colors['LSTM']),
        ('Bidirectional', colors['Bidirectional']),
        ('Dense', colors['Dense']),
        ('Dropout', colors['Dropout'])
    ]
    
    for i, (name, color) in enumerate(legend_items):
        x = 0.15 + i * 0.2
        rect = plt.Rectangle((x, legend_y), 0.05, 0.02, facecolor=color, edgecolor='black')
        ax.add_patch(rect)
        ax.text(x + 0.06, legend_y + 0.01, name, fontsize=8, va='center')
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    ax.set_title(f'LSTM Architecture\n{model_name}\nTotal params: {model.count_params():,}',
                fontsize=14, fontweight='bold')
    
    output_path = output_dir / f'{model_name}_architecture_detailed.png'
    plt.savefig(output_path, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  ✓ Guardado: {output_path}")


def _plot_lstm_flow(model, output_dir, model_name):
    """Crea diagrama de flujo de información en LSTM"""
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Detectar tipo de LSTM
    lstm_layers = []
    for layer in model.layers:
        layer_type = layer.__class__.__name__
        if layer_type == 'LSTM':
            lstm_layers.append(('LSTM', layer.units, layer.return_sequences))
        elif layer_type == 'Bidirectional':
            lstm_layers.append(('BiLSTM', layer.layer.units * 2, 
                              getattr(layer.layer, 'return_sequences', False)))
    
    if not lstm_layers:
        plt.close()
        return
    
    # Dibujar flujo temporal
    n_timesteps = 5  # Representación
    
    # Input sequence
    for t in range(n_timesteps):
        x = 0.1 + t * 0.15
        circle = plt.Circle((x, 0.8), 0.03, facecolor='#E0E0E0', edgecolor='black')
        ax.add_patch(circle)
        ax.text(x, 0.8, f't-{n_timesteps-1-t}' if t < n_timesteps-1 else 't', 
               ha='center', va='center', fontsize=8)
    
    ax.text(0.05, 0.8, 'Input\nSequence', ha='right', va='center', fontsize=10)
    
    # LSTM cells
    y_lstm = 0.5
    for t in range(n_timesteps):
        x = 0.1 + t * 0.15
        
        # Cell
        rect = plt.Rectangle((x-0.05, y_lstm-0.08), 0.1, 0.16,
                             facecolor='#1E88E5', edgecolor='black', linewidth=2)
        ax.add_patch(rect)
        
        # h y c states
        ax.annotate('', xy=(x+0.07, y_lstm), xytext=(x+0.15-0.07, y_lstm) if t < n_timesteps-1 else (x+0.07, y_lstm),
                   arrowprops=dict(arrowstyle='->', color='#43A047', lw=2) if t < n_timesteps-1 else dict(arrowstyle='-', color='white'))
        
        # Input arrow
        ax.annotate('', xy=(x, y_lstm+0.08), xytext=(x, 0.8-0.03),
                   arrowprops=dict(arrowstyle='->', color='gray', lw=1))
    
    ax.text(0.05, y_lstm, 'LSTM\nCells', ha='right', va='center', fontsize=10)
    ax.text(0.85, y_lstm+0.05, 'h (hidden state)', color='#43A047', fontsize=9)
    
    # Output
    last_x = 0.1 + (n_timesteps-1) * 0.15
    ax.annotate('', xy=(last_x, 0.25), xytext=(last_x, y_lstm-0.08),
               arrowprops=dict(arrowstyle='->', color='black', lw=2))
    
    rect = plt.Rectangle((last_x-0.06, 0.15), 0.12, 0.1,
                         facecolor='#FB8C00', edgecolor='black', linewidth=2)
    ax.add_patch(rect)
    ax.text(last_x, 0.2, 'Dense', ha='center', va='center', fontsize=9, fontweight='bold')
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    ax.set_title(f'LSTM Information Flow\n{model_name}', fontsize=14, fontweight='bold')
    
    output_path = output_dir / f'{model_name}_flow.png'
    plt.savefig(output_path, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  ✓ Guardado: {output_path}")


def _create_text_summary(model, output_path, model_name):
    """Crea imagen con resumen textual del modelo"""
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Obtener resumen
    summary_lines = []
    model.summary(print_fn=lambda x: summary_lines.append(x))
    summary_text = '\n'.join(summary_lines)
    
    ax.text(0.05, 0.95, summary_text, transform=ax.transAxes, 
            fontsize=8, verticalalignment='top', fontfamily='monospace')
    ax.axis('off')
    ax.set_title(f'Model Summary: {model_name}', fontsize=14, fontweight='bold')
    
    plt.savefig(output_path, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  ✓ Guardado (text summary): {output_path}")


def _plot_keras_architecture_manual(model, output_dir, model_name):
    """Crea visualización manual de arquitectura Keras"""
    fig, ax = plt.subplots(figsize=(10, 12))
    
    layers = model.layers
    n_layers = len(layers)
    y_positions = np.linspace(0.9, 0.1, n_layers)
    
    for i, (layer, y) in enumerate(zip(layers, y_positions)):
        # Determinar color por tipo de capa
        layer_type = layer.__class__.__name__
        colors = {
            'GRU': '#4CAF50',
            'LSTM': '#2196F3', 
            'Dense': '#FF9800',
            'Dropout': '#9E9E9E',
            'InputLayer': '#E0E0E0'
        }
        color = colors.get(layer_type, '#BBDEFB')
        
        # Dibujar capa
        rect = plt.Rectangle((0.2, y-0.03), 0.6, 0.06, 
                             facecolor=color, edgecolor='black', linewidth=2)
        ax.add_patch(rect)
        
        # Información de la capa
        try:
            output_shape = layer.output_shape
        except:
            output_shape = 'N/A'
        
        params = layer.count_params()
        
        ax.text(0.5, y, f'{layer_type}\n{output_shape}\nParams: {params:,}',
               ha='center', va='center', fontsize=9, fontweight='bold')
        
        # Flecha de conexión
        if i < n_layers - 1:
            ax.annotate('', xy=(0.5, y_positions[i+1]+0.03), xytext=(0.5, y-0.03),
                       arrowprops=dict(arrowstyle='->', color='black', lw=1.5))
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    ax.set_title(f'Keras Model Architecture\n{model_name}\nTotal params: {model.count_params():,}',
                fontsize=14, fontweight='bold')
    
    output_path = output_dir / f'{model_name}_architecture_manual.png'
    plt.savefig(output_path, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  ✓ Guardado: {output_path}")


def visualize_ensemble(model_path, output_dir):
    """
    Visualiza estructura de modelo Ensemble (Stacking)
    
    Args:
        model_path: Ruta al modelo .joblib
        output_dir: Directorio de salida para imágenes
    """
    model = joblib.load(model_path)
    model_name = Path(model_path).stem
    
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # Dibujar base estimators
    estimators = model.estimators_
    n_estimators = len(estimators)
    
    # Posiciones
    base_y = 0.7
    meta_y = 0.3
    
    for i, (name, est) in enumerate(model.named_estimators_.items()):
        x = (i + 1) / (n_estimators + 1)
        
        # Caja del estimador base
        rect = plt.Rectangle((x-0.08, base_y-0.08), 0.16, 0.16,
                             facecolor='#4CAF50', edgecolor='black', linewidth=2)
        ax.add_patch(rect)
        ax.text(x, base_y, f'{name}\n{est.__class__.__name__}',
               ha='center', va='center', fontsize=10, fontweight='bold')
        
        # Flecha al meta-estimador
        ax.annotate('', xy=(0.5, meta_y+0.1), xytext=(x, base_y-0.08),
                   arrowprops=dict(arrowstyle='->', color='gray', lw=1.5))
    
    # Meta-estimador
    rect = plt.Rectangle((0.35, meta_y-0.08), 0.30, 0.16,
                         facecolor='#FF9800', edgecolor='black', linewidth=2)
    ax.add_patch(rect)
    ax.text(0.5, meta_y, f'Meta-Learner\n{model.final_estimator_.__class__.__name__}',
           ha='center', va='center', fontsize=10, fontweight='bold')
    
    # Flecha a output
    ax.annotate('', xy=(0.5, 0.1), xytext=(0.5, meta_y-0.08),
               arrowprops=dict(arrowstyle='->', color='black', lw=2))
    
    # Output
    circle = plt.Circle((0.5, 0.08), 0.05, facecolor='#2196F3', edgecolor='black', linewidth=2)
    ax.add_patch(circle)
    ax.text(0.5, 0.08, 'Output', ha='center', va='center', fontsize=10, fontweight='bold', color='white')
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    ax.set_title(f'Stacking Ensemble Architecture\n{model_name}', fontsize=14, fontweight='bold')
    
    output_path = output_dir / f'{model_name}_architecture.png'
    plt.savefig(output_path, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  ✓ Guardado: {output_path}")
    
    return model


def visualize_all_models(models_dir, output_dir):
    """
    Busca y visualiza todos los modelos en un directorio
    
    Args:
        models_dir: Directorio raíz de modelos
        output_dir: Directorio de salida para imágenes
    """
    models_dir = Path(models_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*60}")
    print("VISUALIZACIÓN DE MODELOS ML")
    print(f"{'='*60}")
    print(f"Directorio de modelos: {models_dir}")
    print(f"Directorio de salida: {output_dir}")
    
    # Buscar modelos recursivamente
    joblib_files = list(models_dir.rglob('*.joblib'))
    keras_files = list(models_dir.rglob('*.keras'))
    
    print(f"\nEncontrados: {len(joblib_files)} modelos .joblib, {len(keras_files)} modelos .keras")
    
    for model_path in joblib_files:
        model_name = model_path.stem.lower()
        relative_path = model_path.relative_to(models_dir)
        
        # Crear subdirectorio de salida manteniendo estructura
        model_output_dir = output_dir / relative_path.parent
        model_output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\n→ Procesando: {relative_path}")
        
        try:
            if 'lightgbm' in model_name or 'lgbm' in model_name:
                visualize_lightgbm(model_path, model_output_dir)
            elif 'xgboost' in model_name or 'xgb' in model_name:
                visualize_xgboost(model_path, model_output_dir)
            elif 'random_forest' in model_name or 'rf' in model_name:
                visualize_random_forest(model_path, model_output_dir)
            elif 'mlp' in model_name:
                visualize_mlp(model_path, model_output_dir)
            elif 'ensemble' in model_name or 'stacking' in model_name:
                visualize_ensemble(model_path, model_output_dir)
            elif 'linear' in model_name or 'ridge' in model_name or 'logistic' in model_name:
                print(f"  ⚠ Modelos lineales no tienen estructura de árbol para visualizar")
            else:
                print(f"  ⚠ Tipo de modelo no reconocido: {model_name}")
        except Exception as e:
            print(f"  ✗ Error: {e}")
    
    for model_path in keras_files:
        relative_path = model_path.relative_to(models_dir)
        model_output_dir = output_dir / relative_path.parent
        model_output_dir.mkdir(parents=True, exist_ok=True)
        
        model_name = model_path.stem.lower()
        print(f"\n→ Procesando: {relative_path}")
        
        try:
            # Detectar tipo de modelo Keras
            if 'lstm' in model_name:
                visualize_lstm(model_path, model_output_dir)
            elif 'gru' in model_name:
                visualize_keras_model(model_path, model_output_dir)
            else:
                # Cargar y detectar por contenido
                temp_model = load_model(model_path)
                layer_types = [l.__class__.__name__ for l in temp_model.layers]
                
                if 'LSTM' in layer_types or 'Bidirectional' in layer_types:
                    visualize_lstm(model_path, model_output_dir)
                else:
                    visualize_keras_model(model_path, model_output_dir)
        except Exception as e:
            print(f"  ✗ Error: {e}")
    
    print(f"\n{'='*60}")
    print("✓ Visualización completada")
    print(f"{'='*60}")


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    from config import MODELS, PLOTS
    
    # Directorio de salida para visualizaciones
    output_dir = PLOTS / 'model_structures'
    
    # Visualizar todos los modelos
    visualize_all_models(MODELS, output_dir)
    
    # Ejemplo de uso individual:
    # visualize_lightgbm(MODELS / 'dataset/target/LightGBM.joblib', output_dir)
    # visualize_xgboost(MODELS / 'dataset/target/XGBoost.joblib', output_dir)
