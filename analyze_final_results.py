#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Final Comprehensive Analysis of All Experiments

This script analyzes:
1. T-variant experiments (T=50, 100, 150) with embeddings
2. Embedding ablation study (with vs without embeddings at T=100)
3. Model architecture comparison across all conditions

Usage:
    python analyze_final_results.py
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict


def load_test_results(checkpoint_dir):
    """Load test results from checkpoint directory"""
    results_file = Path(checkpoint_dir) / 'test_results.json'
    if results_file.exists():
        with open(results_file, 'r') as f:
            return json.load(f)
    return None


def main():
    print("="*100)
    print("FINAL COMPREHENSIVE ANALYSIS - ALL EXPERIMENTS")
    print("="*100)
    
    models = ['baseline', 'deep_tcn', 'local_hybrid', 'conformer', 'transformer']
    model_names = {
        'baseline': 'Baseline (TCN-BiGRU)',
        'deep_tcn': 'Deep TCN',
        'local_hybrid': 'Local Attention Hybrid',
        'conformer': 'Conformer',
        'transformer': 'Transformer'
    }
    
    # Collect all results
    results = defaultdict(lambda: defaultdict(dict))
    
    # T=50 with embeddings
    for model in models:
        res = load_test_results(f'checkpoints_{model}_T50')
        if res:
            results[model]['T50_emb'] = {
                'acc': res.get('test_accuracy', res.get('test_acc', 0)) * 100,
                'f1': res.get('test_f1', 0)
            }
    
    # T=100 with embeddings
    for model in models:
        res = load_test_results(f'checkpoints_{model}')
        if res:
            results[model]['T100_emb'] = {
                'acc': res.get('test_accuracy', res.get('test_acc', 0)) * 100,
                'f1': res.get('test_f1', 0)
            }
    
    # T=150 with embeddings
    for model in models:
        res = load_test_results(f'checkpoints_{model}_T150')
        if res:
            results[model]['T150_emb'] = {
                'acc': res.get('test_accuracy', res.get('test_acc', 0)) * 100,
                'f1': res.get('test_f1', 0)
            }
    
    # T=100 without embeddings (raw binary)
    for model in models:
        res = load_test_results(f'checkpoints_{model}_raw')
        if res:
            results[model]['T100_raw'] = {
                'acc': res.get('test_accuracy', res.get('test_acc', 0)) * 100,
                'f1': res.get('test_f1', 0)
            }
    
    # ========== PART 1: T-Variant Analysis (with embeddings) ==========
    print("\n" + "="*100)
    print("PART 1: SEQUENCE LENGTH IMPACT (T=50, 100, 150) - WITH EMBEDDINGS")
    print("="*100)
    
    print(f"\n{'Model':<25} {'T=50':<20} {'T=100':<20} {'T=150':<20}")
    print("-"*100)
    
    for model in models:
        name = model_names[model]
        t50 = results[model].get('T50_emb', {})
        t100 = results[model].get('T100_emb', {})
        t150 = results[model].get('T150_emb', {})
        
        t50_str = f"{t50.get('acc', 0):.2f}% (F1:{t50.get('f1', 0):.4f})" if t50 else "---"
        t100_str = f"{t100.get('acc', 0):.2f}% (F1:{t100.get('f1', 0):.4f})" if t100 else "---"
        t150_str = f"{t150.get('acc', 0):.2f}% (F1:{t150.get('f1', 0):.4f})" if t150 else "---"
        
        print(f"{name:<25} {t50_str:<20} {t100_str:<20} {t150_str:<20}")
    
    # Find best per T value
    print("\n" + "="*100)
    print("🏆 BEST MODEL PER SEQUENCE LENGTH:")
    print("="*100)
    
    for t_label in ['T50_emb', 'T100_emb', 'T150_emb']:
        t_display = t_label.replace('_emb', '').replace('T', 'T=')
        best_model = None
        best_acc = 0
        for model in models:
            if t_label in results[model]:
                acc = results[model][t_label]['acc']
                if acc > best_acc:
                    best_acc = acc
                    best_model = model
        
        if best_model:
            f1 = results[best_model][t_label]['f1']
            print(f"{t_display:<10}: {model_names[best_model]:<30} {best_acc:.2f}% (F1: {f1:.4f})")
    
    # ========== PART 2: Embedding Ablation Analysis ==========
    print("\n\n" + "="*100)
    print("PART 2: EMBEDDING CONTRIBUTION ANALYSIS (T=100)")
    print("="*100)
    
    print(f"\n{'Model':<25} {'WITH Embeddings (F=114)':<30} {'WITHOUT Embeddings (F=28)':<30} {'Delta':<15}")
    print("-"*100)
    
    embedding_impact = {}
    
    for model in models:
        name = model_names[model]
        with_emb = results[model].get('T100_emb', {})
        without_emb = results[model].get('T100_raw', {})
        
        with_str = f"{with_emb.get('acc', 0):.2f}% (F1:{with_emb.get('f1', 0):.4f})" if with_emb else "---"
        without_str = f"{without_emb.get('acc', 0):.2f}% (F1:{without_emb.get('f1', 0):.4f})" if without_emb else "---"
        
        if with_emb and without_emb:
            delta_acc = with_emb['acc'] - without_emb['acc']
            delta_f1 = with_emb['f1'] - without_emb['f1']
            delta_str = f"+{delta_acc:.2f}% (F1:+{delta_f1:.4f})"
            embedding_impact[model] = delta_acc
        else:
            delta_str = "---"
            embedding_impact[model] = 0
        
        print(f"{name:<25} {with_str:<30} {without_str:<30} {delta_str:<15}")
    
    # ========== PART 3: Overall Rankings ==========
    print("\n\n" + "="*100)
    print("PART 3: OVERALL MODEL RANKINGS")
    print("="*100)
    
    # Best overall performance (T=100 with embeddings)
    print("\n🥇 BEST OVERALL PERFORMANCE (T=100 WITH Embeddings):")
    print("-"*100)
    
    ranked = []
    for model in models:
        if 'T100_emb' in results[model]:
            ranked.append((
                model,
                results[model]['T100_emb']['acc'],
                results[model]['T100_emb']['f1']
            ))
    
    ranked.sort(key=lambda x: x[1], reverse=True)
    
    for i, (model, acc, f1) in enumerate(ranked, 1):
        medal = ['🥇', '🥈', '🥉', '4️⃣', '5️⃣'][i-1] if i <= 5 else f'{i}.'
        print(f"{medal} {model_names[model]:<30} {acc:.2f}% (F1: {f1:.4f})")
    
    # Most improved by embeddings
    print("\n\n📈 MODELS MOST BENEFITED BY EMBEDDINGS:")
    print("-"*100)
    
    for model in sorted(embedding_impact, key=embedding_impact.get, reverse=True):
        if embedding_impact[model] > 0:
            print(f"{model_names[model]:<30} +{embedding_impact[model]:.2f}% improvement")
    
    # Most robust across T values
    print("\n\n💪 MOST ROBUST ACROSS SEQUENCE LENGTHS:")
    print("-"*100)
    
    robustness = {}
    for model in models:
        accs = []
        for t_label in ['T50_emb', 'T100_emb', 'T150_emb']:
            if t_label in results[model]:
                accs.append(results[model][t_label]['acc'])
        
        if len(accs) >= 2:
            std = np.std(accs)
            mean = np.mean(accs)
            robustness[model] = (std, mean)
    
    for model in sorted(robustness, key=lambda x: robustness[x][0]):
        std, mean = robustness[model]
        print(f"{model_names[model]:<30} σ={std:.2f}%  (mean={mean:.2f}%)")
    
    # ========== PART 4: Key Insights ==========
    print("\n\n" + "="*100)
    print("PART 4: KEY INSIGHTS & RECOMMENDATIONS")
    print("="*100)
    
    # Find best overall
    best_overall = ranked[0]
    print(f"\n✨ BEST MODEL: {model_names[best_overall[0]]}")
    print(f"   • Accuracy: {best_overall[1]:.2f}%")
    print(f"   • F1 Score: {best_overall[2]:.4f}")
    
    # Optimal T value
    t_performance = {}
    for t_label in ['T50_emb', 'T100_emb', 'T150_emb']:
        accs = [results[m][t_label]['acc'] for m in models if t_label in results[m]]
        if accs:
            t_performance[t_label] = np.mean(accs)
    
    best_t = max(t_performance, key=t_performance.get)
    best_t_display = best_t.replace('_emb', '').replace('T', 'T=')
    print(f"\n🎯 OPTIMAL SEQUENCE LENGTH: {best_t_display}")
    print(f"   • Average accuracy: {t_performance[best_t]:.2f}%")
    
    # Embedding contribution
    avg_embedding_gain = np.mean([v for v in embedding_impact.values() if v > 0])
    print(f"\n🚀 EMBEDDING CONTRIBUTION:")
    print(f"   • Average improvement: +{avg_embedding_gain:.2f}%")
    print(f"   • Feature expansion: 28 → 114 features (4.07x)")
    
    print("\n\n" + "="*100)
    print("ANALYSIS COMPLETE!")
    print("="*100)
    
    # Save summary to file
    summary_file = Path('results/final_analysis_summary.txt')
    summary_file.parent.mkdir(exist_ok=True)
    
    with open(summary_file, 'w') as f:
        f.write("="*100 + "\n")
        f.write("FINAL COMPREHENSIVE ANALYSIS - SUMMARY\n")
        f.write("="*100 + "\n\n")
        
        f.write("BEST RESULTS:\n")
        f.write("-"*100 + "\n")
        for i, (model, acc, f1) in enumerate(ranked[:3], 1):
            f.write(f"{i}. {model_names[model]}: {acc:.2f}% (F1: {f1:.4f})\n")
        
        f.write(f"\nOPTIMAL SEQUENCE LENGTH: {best_t_display}\n")
        f.write(f"EMBEDDING CONTRIBUTION: +{avg_embedding_gain:.2f}% average improvement\n")
    
    print(f"\n📄 Summary saved to: {summary_file}")


if __name__ == '__main__':
    main()
