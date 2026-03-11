# metrics.py
import os
import glob
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


def calculate_metrics(labels, scores):
  
    try:
        auc = roc_auc_score(labels, scores)
        ap = average_precision_score(labels, scores)
    except ValueError:
        auc, ap = 0.0, 0.0
    return auc, ap


def load_run_data(run_dir):

    score_files = sorted(glob.glob(os.path.join(run_dir, "*_score.csv")))
    
    list_frames = []
    list_instances = []

    for score_path in score_files:
        label_path = score_path.replace("_score.csv", "_label.csv")

        try:
            df_s = pd.read_csv(score_path, header=None)
            df_l = pd.read_csv(label_path, header=None)
        except Exception: 
            continue

        min_len = min(len(df_s), len(df_l))
        scores = df_s.iloc[:min_len, 0].values
        labels = df_l.iloc[:min_len, 0].values
        
        # Frame Data 
        df_video = pd.DataFrame({'score': scores, 'label': labels})
        
        # Segment Group
        df_video['group'] = (df_video['label'] != df_video['label'].shift()).cumsum()
        group_lens = df_video.groupby('group')['label'].transform('count')
        
        # 0=Normal, 1=Short Error (<15), 2=Long Error (>=15)
        conditions = [
            (df_video['label'] == 0),
            (df_video['label'] == 1) & (group_lens < 15),
            (df_video['label'] == 1) & (group_lens >= 15)
        ]
        choices = ['Normal', 'Short', 'Long']
        df_video['type'] = np.select(conditions, choices, default='Normal')
        list_frames.append(df_video)

        # Instance Data
        grouped = df_video.groupby('group').agg(
            instance_score=('score', 'mean'),
            instance_label=('label', 'first')
        )
        list_instances.append(grouped)

    if not list_frames: 
        return None, None
    return pd.concat(list_frames, ignore_index=True), pd.concat(list_instances, ignore_index=True)


def calc_metrics_scaled(y_true, y_score):
    if len(np.unique(y_true)) < 2: 
        return np.nan, np.nan
    auc = roc_auc_score(y_true, y_score)
    ap = average_precision_score(y_true, y_score)
    return auc * 100, ap * 100


def process_single_run(run_dir):
    df_frame, df_inst = load_run_data(run_dir)
    if df_frame is None: 
        return {}
    
    results = {}

    # All Test Data
    results['All_AUC'], results['All_AP'] = calc_metrics_scaled(df_frame['label'], df_frame['score'])
    results['All_AUC_ins'], results['All_AP_ins'] = calc_metrics_scaled(df_inst['instance_label'], df_inst['instance_score'])

    # Short Errors (Normal + Short)
    df_short = df_frame[df_frame['type'].isin(['Normal', 'Short'])]
    results['Short_AUC'], results['Short_AP'] = calc_metrics_scaled(df_short['label'], df_short['score'])

    # Long Errors (Normal + Long)
    df_long = df_frame[df_frame['type'].isin(['Normal', 'Long'])]
    results['Long_AUC'], results['Long_AP'] = calc_metrics_scaled(df_long['label'], df_long['score'])
    
    return results


def evaluate_multiple_runs(run_dirs):
    final_stats = {k: [] for k in ['All_AUC', 'All_AP', 'All_AUC_ins', 'All_AP_ins', 
                                   'Short_AUC', 'Short_AP', 'Long_AUC', 'Long_AP']}

    print(f"\n{'Run':<4} | {'All AUC':<7} {'All AP':<7} | {'AUC_ins':<7} {'AP_ins':<7} | {'Short AUC':<9} | {'Long AUC':<9}")
    print("-" * 85)

    for idx, run_dir in enumerate(run_dirs):
        res = process_single_run(run_dir)
        if not res: 
            print(f"{idx:<4} | [WARNING: No valid data found in {run_dir}]")
            continue
        
        for k, v in res.items(): 
            final_stats[k].append(v)
        
        print(f"{idx:<4} | {res['All_AUC']:.2f}    {res['All_AP']:.2f}    | {res['All_AUC_ins']:.2f}    {res['All_AP_ins']:.2f}    | {res['Short_AUC']:.2f}        | {res['Long_AUC']:.2f}")

    print("\n" + "="*85)
    print("FINAL RESULTS (Mean ± Std)")
    print("="*85)
    print("All Test Data:")
    print(f"  AUC     : {np.mean(final_stats['All_AUC']):.2f} ± {np.std(final_stats['All_AUC']):.2f}")
    print(f"  AP      : {np.mean(final_stats['All_AP']):.2f} ± {np.std(final_stats['All_AP']):.2f}")
    print(f"  AUC_ins : {np.mean(final_stats['All_AUC_ins']):.2f} ± {np.std(final_stats['All_AUC_ins']):.2f}")
    print(f"  AP_ins  : {np.mean(final_stats['All_AP_ins']):.2f} ± {np.std(final_stats['All_AP_ins']):.2f}")
    print("-" * 45)
    print("Short Errors (< 3s/15f) [Frame-level]:")
    print(f"  AUC     : {np.mean(final_stats['Short_AUC']):.2f} ± {np.std(final_stats['Short_AUC']):.2f}")
    print(f"  AP      : {np.mean(final_stats['Short_AP']):.2f} ± {np.std(final_stats['Short_AP']):.2f}")
    print("-" * 45)
    print("Long Errors (>= 3s/15f) [Frame-level]:")
    print(f"  AUC     : {np.mean(final_stats['Long_AUC']):.2f} ± {np.std(final_stats['Long_AUC']):.2f}")
    print(f"  AP      : {np.mean(final_stats['Long_AP']):.2f} ± {np.std(final_stats['Long_AP']):.2f}")
    print("="*85)

if __name__ == "__main__":
    TARGET_DIRS = [
        "EXPERIMENT_DIR_1",
        "EXPERIMENT_DIR_2",
        "EXPERIMENT_DIR_3",
    ] # Replace with actual directories containing the results CSVs
    
    evaluate_multiple_runs(TARGET_DIRS)