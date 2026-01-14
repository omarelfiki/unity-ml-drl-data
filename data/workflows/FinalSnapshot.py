import os
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))
data_dir = os.path.join(project_root, 'data')
csv_path = os.path.join(data_dir, 'collected_results.csv')

df = pd.read_csv(csv_path)
thresholds = {
    '3DBall': 2.56576,
    'Basic': 0.6096,
    'BigWallJump': -0.864552,
    'Crawler': -0.79424,
    'Hallway': -0.556176,
    'PushBlock': 2.46952
}

def define_thresholds(row):
    if pd.isna(row['threshold_value']):
        return thresholds.get(row['environment'])
    return row['threshold_value']


def identify_success(row):
    if row['steps_to_threshold'] == 'Not reached':
        return 0
    if pd.notna(row['steps_to_threshold']):
        return 1
    if row['final_reward_mean'] >= row['threshold_value']:
        return 1
    return 0

df['threshold_value'] = df.apply(define_thresholds, axis=1)
df['run_reached_threshold'] = df.apply(identify_success, axis=1)
columns_to_keep = ['run_id', 'environment', 'algorithm', 'learning_rate', 
                   'batch_size', 'buffer_size', 'final_reward_mean', 
                   'threshold_value', 'run_reached_threshold']
snapshot_df = df[columns_to_keep]
snapshot_df.to_csv('final_data_snapshot.csv', index=False)
print('Snapshot successfully processed and saved.')
print(snapshot_df.head())