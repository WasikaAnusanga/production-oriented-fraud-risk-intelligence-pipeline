"""
Script to execute model training, hyperparameter optimization, and imbalanced learning comparison.
"""

import time
import numpy as np
import pandas as pd
import joblib

from src.data_splitting import get_partition_data
from src.model_training import (
    create_fraud_pipeline,
    evaluate_pipeline_cv,
    run_hyperparameter_optimization
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from imblearn.over_sampling import RandomOverSampler, SMOTE
from config.settings import RANDOM_STATE

def main():
    X_dev, y_dev, ids_dev = get_partition_data('development')

    print('--- Step 1: Running Hyperparameter Optimization (RandomizedSearchCV) ---')
    tune_results = run_hyperparameter_optimization(X_dev, y_dev, n_iter=10)

    hyper_rows = []
    for model_name, info in tune_results.items():
        hyper_rows.append({
            'model': model_name,
            'best_score_pr_auc': round(info['best_score_pr_auc'], 4),
            'search_time_sec': round(info['search_time_sec'], 2),
            'best_params': str(info['best_params'])
        })
        print(f"{model_name}: Best PR-AUC = {info['best_score_pr_auc']:.4f}")
    pd.DataFrame(hyper_rows).to_csv('reports/tables/hyperparameter_results.csv', index=False)
    print('Saved reports/tables/hyperparameter_results.csv')

    print('\n--- Step 2: Evaluating Imbalanced Learning Strategies & Model Architectures ---')
    models_to_test = {
        'LR_Default': create_fraud_pipeline(LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)),
        'LR_ClassWeighted': create_fraud_pipeline(LogisticRegression(class_weight='balanced', max_iter=1000, random_state=RANDOM_STATE)),
        'LR_SMOTE': create_fraud_pipeline(LogisticRegression(max_iter=1000, random_state=RANDOM_STATE), resampler=SMOTE(random_state=RANDOM_STATE)),
        'LR_ROS': create_fraud_pipeline(LogisticRegression(max_iter=1000, random_state=RANDOM_STATE), resampler=RandomOverSampler(random_state=RANDOM_STATE)),
        'LR_Tuned': tune_results['logistic_regression']['best_estimator'],
        
        'RF_Default': create_fraud_pipeline(RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1), scale_numeric=False),
        'RF_ClassWeighted': create_fraud_pipeline(RandomForestClassifier(class_weight='balanced', random_state=RANDOM_STATE, n_jobs=-1), scale_numeric=False),
        'RF_SMOTE': create_fraud_pipeline(RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1), resampler=SMOTE(random_state=RANDOM_STATE), scale_numeric=False),
        'RF_Tuned': tune_results['random_forest']['best_estimator'],
        
        'HGB_Default': create_fraud_pipeline(HistGradientBoostingClassifier(random_state=RANDOM_STATE), scale_numeric=False),
        'HGB_ClassWeighted': create_fraud_pipeline(HistGradientBoostingClassifier(class_weight='balanced', random_state=RANDOM_STATE), scale_numeric=False),
        'HGB_Tuned': tune_results['hist_gradient_boosting']['best_estimator'],
    }

    comparison_rows = []
    fold_results_all = []
    fitted_eval_objects = {}

    for name, pipe in models_to_test.items():
        eval_res = evaluate_pipeline_cv(pipe, X_dev, y_dev, threshold=0.20)
        s = eval_res['summary']
        fitted_eval_objects[name] = eval_res
        
        comparison_rows.append({
            'Model': name,
            'PR_AUC_mean': round(s['pr_auc_mean'], 4),
            'PR_AUC_std': round(s['pr_auc_std'], 4),
            'ROC_AUC_mean': round(s['roc_auc_mean'], 4),
            'ROC_AUC_std': round(s['roc_auc_std'], 4),
            'Balanced_Acc_mean': round(s['balanced_accuracy_mean'], 4),
            'Accuracy_mean': round(s['accuracy_mean'], 4),
            'Precision_mean': round(s['precision_mean'], 4),
            'Recall_mean': round(s['recall_mean'], 4),
            'F1_mean': round(s['f1_mean'], 4),
            'Fit_Time_sec': round(s['fit_time_mean'], 3)
        })
        
        for fold_i, p_auc in enumerate(eval_res['fold_metrics']['pr_auc']):
            fold_results_all.append({
                'Model': name,
                'Fold': fold_i + 1,
                'PR_AUC': round(p_auc, 4),
                'ROC_AUC': round(eval_res['fold_metrics']['roc_auc'][fold_i], 4),
                'Recall': round(eval_res['fold_metrics']['recall'][fold_i], 4),
                'Precision': round(eval_res['fold_metrics']['precision'][fold_i], 4),
                'F1': round(eval_res['fold_metrics']['f1'][fold_i], 4),
            })
        print(f"{name:<18}: PR-AUC = {s['pr_auc_mean']:.4f} +/- {s['pr_auc_std']:.4f} | ROC-AUC = {s['roc_auc_mean']:.4f} | Recall = {s['recall_mean']:.4f}")

    comp_df = pd.DataFrame(comparison_rows).sort_values(by='PR_AUC_mean', ascending=False).reset_index(drop=True)
    comp_df.to_csv('reports/tables/model_comparison.csv', index=False)
    pd.DataFrame(fold_results_all).to_csv('reports/tables/cross_validation_results.csv', index=False)
    print('\nSaved reports/tables/model_comparison.csv and reports/tables/cross_validation_results.csv!')

    # Save candidate models fitted on full dev partition for calibration
    print('\n--- Step 3: Saving Top Candidate Estimators for Calibration ---')
    top_hgb = models_to_test['HGB_Tuned']
    top_hgb.fit(X_dev, y_dev)
    joblib.dump(top_hgb, 'reports/intermediate/candidate_hgb_pipeline.joblib')

    top_rf = models_to_test['RF_Tuned']
    top_rf.fit(X_dev, y_dev)
    joblib.dump(top_rf, 'reports/intermediate/candidate_rf_pipeline.joblib')

    top_lr = models_to_test['LR_Tuned']
    top_lr.fit(X_dev, y_dev)
    joblib.dump(top_lr, 'reports/intermediate/candidate_lr_pipeline.joblib')

    print('Saved candidate pipelines under reports/intermediate/')

if __name__ == '__main__':
    main()
