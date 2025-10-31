#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Comprehensive comparison of model performance across different T values

Analyzes:
- How sequence length (T) affects each model architecture
- Which models benefit most from longer sequences
- Optimal T value for each model
- Trade-offs between accuracy and computational cost
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pandas as pd

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Model names
MODELS = ['baseline', 'transformer', 'deep_tcn', 'local_hybrid', 'conformer']
T_VALUES = [50, 100, 150]
MODEL_DISPLAY_NAMES = {
    'baseline': 'Baseline (TCN-BiGRU)',
    'transformer': 'Transformer',
    'deep_tcn': 'Deep TCN',
    'local_hybrid': 'Local Hybrid',
    'conformer': 'Conformer'
}


def load_results():
    """Load all test results for different T values"""
    results = {T: {} for T in T_VALUES}
    
    for T in T_VALUES:
        for model in MODELS:
            # Determine checkpoint directory
            if T == 100:
                checkpoint_dir = f'checkpoints_{model}' if model != 'baseline' else 'checkpoints'
            else:
                checkpoint_dir = f'checkpoints_{model}_T{T}'
            
            result_path = Path(checkpoint_dir) / 'test_results.json'
            
            if result_path.exists():
                with open(result_path, 'r') as f:
                    data = json.load(f)
                
                # Normalize keys
                results[T][model] = {
                    'test_acc': data.get('test_acc', data.get('test_accuracy')),
                    'test_f1': data.get('test_f1', data.get('macro_f1')),
                    'num_parameters': data.get('num_parameters', data.get('n_params')),
                    'per_class_f1': data.get('per_class_f1', {})
                }
                print(f"✅ Loaded {model} T={T}: {results[T][model]['test_acc']:.4f}")
            else:
                print(f"⚠️  Missing {model} T={T}: {result_path}")
                results[T][model] = None
    
    return results


def create_comparison_plots(results):
    """Create comprehensive comparison visualizations"""
    
    fig = plt.figure(figsize=(20, 14))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # Prepare data
    models_list = []
    T_list = []
    acc_list = []
    f1_list = []
    
    for T in T_VALUES:
        for model in MODELS:
            if results[T][model] is not None:
                models_list.append(MODEL_DISPLAY_NAMES[model])
                T_list.append(T)
                acc_list.append(results[T][model]['test_acc'] * 100)
                f1_list.append(results[T][model]['test_f1'])
    
    df = pd.DataFrame({
        'Model': models_list,
        'T': T_list,
        'Accuracy': acc_list,
        'F1': f1_list
    })
    
    # 1. Accuracy vs T (line plot)
    ax1 = fig.add_subplot(gs[0, :2])
    for model in MODELS:
        model_name = MODEL_DISPLAY_NAMES[model]
        model_df = df[df['Model'] == model_name]
        if len(model_df) > 0:
            ax1.plot(model_df['T'], model_df['Accuracy'], 
                    marker='o', linewidth=2.5, markersize=10, 
                    label=model_name)
    
    ax1.set_xlabel('Sequence Length (T)', fontsize=13, fontweight='bold')
    ax1.set_ylabel('Test Accuracy (%)', fontsize=13, fontweight='bold')
    ax1.set_title('Model Performance vs Sequence Length', fontsize=15, fontweight='bold')
    ax1.legend(fontsize=11, loc='best')
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(T_VALUES)
    
    # 2. Best model at each T
    ax2 = fig.add_subplot(gs[0, 2])
    best_models = []
    best_accs = []
    
    for T in T_VALUES:
        T_results = [(model, results[T][model]['test_acc'] * 100) 
                     for model in MODELS if results[T][model] is not None]
        if T_results:
            best_model, best_acc = max(T_results, key=lambda x: x[1])
            best_models.append(MODEL_DISPLAY_NAMES[best_model])
            best_accs.append(best_acc)
    
    colors_best = ['#f39c12', '#2ecc71', '#3498db']
    bars = ax2.bar([f'T={T}' for T in T_VALUES], best_accs, color=colors_best, alpha=0.8)
    ax2.set_ylabel('Best Accuracy (%)', fontsize=12, fontweight='bold')
    ax2.set_title('Best Model at Each T', fontsize=14, fontweight='bold')
    ax2.set_ylim(80, 100)
    
    for i, (bar, acc, model) in enumerate(zip(bars, best_accs, best_models)):
        ax2.text(i, acc + 0.5, f'{acc:.2f}%\n{model}', 
                ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # 3. Accuracy improvement vs T=100 baseline
    ax3 = fig.add_subplot(gs[1, 0])
    
    for model in MODELS:
        if results[100][model] is not None:
            baseline_acc = results[100][model]['test_acc'] * 100
            improvements = []
            T_vals = []
            
            for T in T_VALUES:
                if results[T][model] is not None:
                    improvements.append(results[T][model]['test_acc'] * 100 - baseline_acc)
                    T_vals.append(T)
            
            ax3.plot(T_vals, improvements, marker='o', linewidth=2, 
                    markersize=8, label=MODEL_DISPLAY_NAMES[model])
    
    ax3.axhline(y=0, color='red', linestyle='--', linewidth=2, alpha=0.7)
    ax3.set_xlabel('Sequence Length (T)', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Accuracy Change vs T=100 (%)', fontsize=12, fontweight='bold')
    ax3.set_title('Impact of T on Performance', fontsize=14, fontweight='bold')
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3)
    ax3.set_xticks(T_VALUES)
    
    # 4. F1 Score comparison
    ax4 = fig.add_subplot(gs[1, 1])
    
    x = np.arange(len(MODELS))
    width = 0.25
    
    for i, T in enumerate(T_VALUES):
        f1_scores = [results[T][model]['test_f1'] if results[T][model] else 0 
                     for model in MODELS]
        ax4.bar(x + i*width, f1_scores, width, label=f'T={T}', alpha=0.8)
    
    ax4.set_ylabel('F1 Score', fontsize=12, fontweight='bold')
    ax4.set_title('F1 Score Comparison', fontsize=14, fontweight='bold')
    ax4.set_xticks(x + width)
    ax4.set_xticklabels([MODEL_DISPLAY_NAMES[m] for m in MODELS], rotation=45, ha='right')
    ax4.legend(fontsize=10)
    ax4.grid(axis='y', alpha=0.3)
    
    # 5. Heatmap: Model vs T
    ax5 = fig.add_subplot(gs[1, 2])
    
    heatmap_data = []
    for model in MODELS:
        row = [results[T][model]['test_acc'] * 100 if results[T][model] else 0 
               for T in T_VALUES]
        heatmap_data.append(row)
    
    im = ax5.imshow(heatmap_data, cmap='RdYlGn', aspect='auto', vmin=80, vmax=100)
    ax5.set_xticks(np.arange(len(T_VALUES)))
    ax5.set_yticks(np.arange(len(MODELS)))
    ax5.set_xticklabels([f'T={T}' for T in T_VALUES])
    ax5.set_yticklabels([MODEL_DISPLAY_NAMES[m] for m in MODELS])
    ax5.set_title('Accuracy Heatmap', fontsize=14, fontweight='bold')
    
    for i in range(len(MODELS)):
        for j in range(len(T_VALUES)):
            text = ax5.text(j, i, f'{heatmap_data[i][j]:.1f}',
                           ha="center", va="center", color="black", 
                           fontsize=11, fontweight='bold')
    
    plt.colorbar(im, ax=ax5, label='Accuracy (%)')
    
    # 6. Parameters vs Performance (scatter for each T)
    ax6 = fig.add_subplot(gs[2, 0])
    
    colors_scatter = {'50': '#f39c12', '100': '#2ecc71', '150': '#3498db'}
    
    for T in T_VALUES:
        params_list = []
        acc_list_T = []
        model_names = []
        
        for model in MODELS:
            if results[T][model] is not None:
                params_list.append(results[T][model]['num_parameters'] / 1e6)
                acc_list_T.append(results[T][model]['test_acc'] * 100)
                model_names.append(MODEL_DISPLAY_NAMES[model])
        
        ax6.scatter(params_list, acc_list_T, s=150, alpha=0.6, 
                   label=f'T={T}', color=colors_scatter[str(T)])
    
    ax6.set_xlabel('Parameters (Millions)', fontsize=12, fontweight='bold')
    ax6.set_ylabel('Test Accuracy (%)', fontsize=12, fontweight='bold')
    ax6.set_title('Efficiency: Params vs Performance', fontsize=14, fontweight='bold')
    ax6.legend(fontsize=10)
    ax6.grid(True, alpha=0.3)
    
    # 7. Model ranking at each T
    ax7 = fig.add_subplot(gs[2, 1:])
    
    ranking_data = []
    for T in T_VALUES:
        T_results = [(MODEL_DISPLAY_NAMES[model], results[T][model]['test_acc'] * 100) 
                     for model in MODELS if results[T][model] is not None]
        T_results.sort(key=lambda x: x[1], reverse=True)
        ranking_data.append(T_results)
    
    y_pos = np.arange(len(MODELS))
    x_offset = 0
    bar_width = 0.25
    
    for i, (T, rankings) in enumerate(zip(T_VALUES, ranking_data)):
        models_sorted = [r[0] for r in rankings]
        accs_sorted = [r[1] for r in rankings]
        
        bars = ax7.barh([y + i*bar_width for y in y_pos[:len(rankings)]], 
                        accs_sorted, bar_width, label=f'T={T}', alpha=0.8)
        
        for j, (bar, acc) in enumerate(zip(bars, accs_sorted)):
            ax7.text(acc + 0.5, y_pos[j] + i*bar_width, f'{acc:.1f}%', 
                    va='center', fontsize=9)
    
    ax7.set_yticks([y + bar_width for y in y_pos])
    ax7.set_yticklabels([f'Rank {i+1}' for i in range(len(MODELS))])
    ax7.set_xlabel('Test Accuracy (%)', fontsize=12, fontweight='bold')
    ax7.set_title('Model Rankings at Each T', fontsize=14, fontweight='bold')
    ax7.legend(fontsize=10, loc='lower right')
    ax7.set_xlim(80, 100)
    ax7.grid(axis='x', alpha=0.3)
    
    # Overall title
    fig.suptitle('Comprehensive Analysis: Impact of Sequence Length (T) on Model Performance', 
                fontsize=18, fontweight='bold', y=0.995)
    
    plt.tight_layout()
    plt.savefig('docs/T_value_comparison.png', dpi=300, bbox_inches='tight')
    print("\n✅ Saved visualization to docs/T_value_comparison.png")


def generate_summary_report(results):
    """Generate text summary report"""
    
    print("\n" + "="*80)
    print("📊 COMPREHENSIVE ANALYSIS: SEQUENCE LENGTH (T) IMPACT")
    print("="*80)
    
    # Best model at each T
    print("\n🥇 Best Model at Each T:")
    print("-"*80)
    for T in T_VALUES:
        T_results = [(model, results[T][model]['test_acc'] * 100, results[T][model]['test_f1']) 
                     for model in MODELS if results[T][model] is not None]
        if T_results:
            best_model, best_acc, best_f1 = max(T_results, key=lambda x: x[1])
            print(f"T={T:3d}: {MODEL_DISPLAY_NAMES[best_model]:25s} | Acc: {best_acc:6.2f}% | F1: {best_f1:.4f}")
    
    # Performance change for each model
    print("\n📈 Performance Change vs T=100 Baseline:")
    print("-"*80)
    print(f"{'Model':<25s} | {'T=50 Change':>12s} | {'T=150 Change':>13s}")
    print("-"*80)
    
    for model in MODELS:
        if results[100][model] is not None:
            baseline_acc = results[100][model]['test_acc'] * 100
            
            change_50 = results[50][model]['test_acc'] * 100 - baseline_acc if results[50][model] else 0
            change_150 = results[150][model]['test_acc'] * 100 - baseline_acc if results[150][model] else 0
            
            sign_50 = "+" if change_50 >= 0 else ""
            sign_150 = "+" if change_150 >= 0 else ""
            
            print(f"{MODEL_DISPLAY_NAMES[model]:<25s} | {sign_50}{change_50:6.2f}%p     | {sign_150}{change_150:6.2f}%p")
    
    # Key insights
    print("\n🎯 Key Insights:")
    print("-"*80)
    
    # Find model most benefiting from longer sequences
    max_improvement = -999
    best_long_seq_model = None
    for model in MODELS:
        if results[150][model] and results[50][model]:
            improvement = results[150][model]['test_acc'] - results[50][model]['test_acc']
            if improvement > max_improvement:
                max_improvement = improvement
                best_long_seq_model = model
    
    if best_long_seq_model:
        print(f"✅ {MODEL_DISPLAY_NAMES[best_long_seq_model]} benefits most from long sequences")
        print(f"   (+{max_improvement*100:.2f}%p from T=50 to T=150)")
    
    # Find most stable model
    min_variance = 999
    most_stable_model = None
    for model in MODELS:
        accs = [results[T][model]['test_acc'] * 100 
                for T in T_VALUES if results[T][model]]
        if len(accs) == 3:
            variance = np.std(accs)
            if variance < min_variance:
                min_variance = variance
                most_stable_model = model
    
    if most_stable_model:
        print(f"✅ {MODEL_DISPLAY_NAMES[most_stable_model]} is most stable across T values")
        print(f"   (Std dev: {min_variance:.2f}%)")
    
    # Optimal T recommendation
    print("\n💡 Recommendations:")
    print("-"*80)
    for model in MODELS:
        T_accs = [(T, results[T][model]['test_acc'] * 100) 
                  for T in T_VALUES if results[T][model]]
        if T_accs:
            best_T, best_acc = max(T_accs, key=lambda x: x[1])
            print(f"{MODEL_DISPLAY_NAMES[model]:<25s} → Optimal T={best_T} ({best_acc:.2f}%)")
    
    print("="*80)


def main():
    """Main analysis function"""
    print("Loading results...")
    results = load_results()
    
    print("\nGenerating visualizations...")
    create_comparison_plots(results)
    
    print("\nGenerating summary report...")
    generate_summary_report(results)
    
    print("\n✅ Analysis complete!")
    print("📊 Visualization: docs/T_value_comparison.png")


if __name__ == '__main__':
    main()
