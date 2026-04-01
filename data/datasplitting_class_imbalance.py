import os
import pandas as pd
import numpy as np

# Generate oversampled/downsampled data
tasks_config = {"CYP2C9_Substrate": 4, "CYP2D6_Inhibition": 4, "Tox21_NR_ER": 7}

for task, max_factor in tasks_config.items():
    for split_method in ['random', 'scaffold']:
        for i in range(5):
            seed = 2024 + i * 10
            input_path = f'data_with_group_{split_method}/{task}_{split_method}_{seed}.csv'
            data = pd.read_csv(input_path)
            training_data = data[data['group'] == 'training']
            valid_data = data[data['group'] == 'valid']
            test_data = data[data['group'] == 'test']

            class_0 = training_data[training_data[task] == 0]
            class_1 = training_data[training_data[task] == 1]
            n_majority, n_minority = len(class_0), len(class_1)

            if n_minority > 0 and n_minority < n_majority:
                factor = min(int(np.ceil(n_majority / n_minority)), max_factor)
                class_1 = pd.concat([class_1] * factor, ignore_index=True)

            new_training = pd.concat([class_0, class_1], ignore_index=True).sample(frac=1, random_state=seed).reset_index(drop=True)
            new_dataset = pd.concat([new_training, valid_data, test_data], ignore_index=True)
            new_dataset.rename(columns={task: f'{task}_oversampled'}, inplace=True)
            new_dataset.to_csv(f'data_with_group_{split_method}/{task}_oversampled_{split_method}_{seed}.csv', index=False)

            class_0_orig = training_data[training_data[task] == 0].sample(frac=1, random_state=seed).reset_index(drop=True)
            splits = np.array_split(class_0_orig, max_factor)
            for split_idx, split in enumerate(splits):
                new_training = pd.concat([split, training_data[training_data[task] == 1]], ignore_index=True).sample(frac=1, random_state=seed).reset_index(drop=True)
                new_dataset = pd.concat([new_training, valid_data, test_data], ignore_index=True)
                new_dataset.rename(columns={task: f'{task}_downsampled'}, inplace=True)
                new_dataset.to_csv(f'data_with_group_{split_method}/{task}_downsampled_{split_idx+1}_{split_method}_{seed}.csv', index=False)

            print(f'Generated sampling data for {task} {split_method} seed {seed}')

def get_origin_idx(smiles_list):
    train_idx = []
    for smiles in smiles_list:
        try:
            idx = list(data_origin['smiles']).index(smiles)
            train_idx.append(idx)
        except ValueError:
            pass
    return train_idx

select_tasks = ["CYP2C9_Substrate", "CYP2D6_Inhibition", "Tox21_NR_ER"]
split_methods = ["random", "scaffold"]

# Downsampling: CYP2C9_Substrate, CYP2D6_Inhibition (4 subsets each)
# Oversampling: Tox21_NR_ER

for i in range(5):
    seed = 2024 + i * 10
    for split_policy in split_methods:
        for task in select_tasks:
            data_origin = pd.read_csv(f'data_after_processing/{task}.csv')

            # Original
            data_splitted = pd.read_csv(f'data_with_group_{split_policy}/{task}_{split_policy}_{seed}.csv')
            train_smiles = data_splitted[data_splitted['group'] == 'training']['smiles'].tolist()
            valid_smiles = data_splitted[data_splitted['group'] == 'valid']['smiles'].tolist()
            test_smiles = data_splitted[data_splitted['group'] == 'test']['smiles'].tolist()
            train_idx = get_origin_idx(train_smiles)
            valid_idx = get_origin_idx(valid_smiles)
            test_idx = get_origin_idx(test_smiles)
            os.makedirs(f'data_split_for_kpgt/{task}/', exist_ok=True)
            np.save(f'data_split_for_kpgt/{task}/original_{split_policy}_{seed}.npy',
                    np.array([train_idx, valid_idx, test_idx], dtype=object), allow_pickle=True)

            # Oversampled
            data_splitted = pd.read_csv(f'data_with_group_{split_policy}/{task}_oversampled_{split_policy}_{seed}.csv')
            train_smiles = data_splitted[data_splitted['group'] == 'training']['smiles'].tolist()
            valid_smiles = data_splitted[data_splitted['group'] == 'valid']['smiles'].tolist()
            test_smiles = data_splitted[data_splitted['group'] == 'test']['smiles'].tolist()
            train_idx = get_origin_idx(train_smiles)
            valid_idx = get_origin_idx(valid_smiles)
            test_idx = get_origin_idx(test_smiles)
            np.save(f'data_split_for_kpgt/{task}/oversampled_{split_policy}_{seed}.npy',
                    np.array([train_idx, valid_idx, test_idx], dtype=object), allow_pickle=True)

            # Downsampled (4 subsets for CYP2C9_Substrate, CYP2D6_Inhibition; 7 for Tox21_NR_ER)
            if task == "Tox21_NR_ER":
                downsample_k = 7
            else:
                downsample_k = 4
            for k in range(1, downsample_k + 1):
                    data_splitted = pd.read_csv(f'data_with_group_{split_policy}/{task}_downsampled_{k}_{split_policy}_{seed}.csv')
                    train_smiles = data_splitted[data_splitted['group'] == 'training']['smiles'].tolist()
                    valid_smiles = data_splitted[data_splitted['group'] == 'valid']['smiles'].tolist()
                    test_smiles = data_splitted[data_splitted['group'] == 'test']['smiles'].tolist()
                    train_idx = get_origin_idx(train_smiles)
                    valid_idx = get_origin_idx(valid_smiles)
                    test_idx = get_origin_idx(test_smiles)
                    np.save(f'data_split_for_kpgt/{task}/downsampled_{k}_{split_policy}_{seed}.npy',
                            np.array([train_idx, valid_idx, test_idx], dtype=object), allow_pickle=True)

            print(f'{task} {split_policy} seed {seed} done')

# ChemProp format
select_items = []
base_tasks = ["CYP2C9_Substrate", "CYP2D6_Inhibition", "Tox21_NR_ER"]
downsample_counts = {"CYP2C9_Substrate": 4, "CYP2D6_Inhibition": 4, "Tox21_NR_ER": 7}

for t in base_tasks:
    select_items.append({"file_prefix": t, "label": t})
    select_items.append({"file_prefix": f"{t}_oversampled", "label": t})
    cnt = downsample_counts.get(t, 0)
    for k in range(1, cnt + 1):
        select_items.append({"file_prefix": f"{t}_downsampled_{k}", "label": t})

for i in range(5):
    seed = 2024 + i * 10
    for split_policy in split_methods:
        for item in select_items:
            prefix = item['file_prefix']
            label = item['label']
            input_path = f'data_with_group_{split_policy}/{prefix}_{split_policy}_{seed}.csv'
            if not os.path.exists(input_path):
                continue
            data_origin = pd.read_csv(input_path)
            if label not in data_origin.columns:
                candidates = [c for c in data_origin.columns if c not in ('group', 'smiles') and label in c]
                if candidates:
                    used_label = candidates[0]
                else:
                    continue
            else:
                used_label = label

            train_data = data_origin[data_origin['group'] == 'training'][['smiles', used_label]]
            valid_data = data_origin[data_origin['group'] == 'valid'][['smiles', used_label]]
            test_data = data_origin[data_origin['group'] == 'test'][['smiles', used_label]]

            out_dir = f'data_split_for_chemprop/{prefix}_{split_policy}_{seed}/'
            os.makedirs(out_dir, exist_ok=True)
            train_data.to_csv(f'{out_dir}{prefix}_{split_policy}_{seed}_training.csv', index=False)
            valid_data.to_csv(f'{out_dir}{prefix}_{split_policy}_{seed}_valid.csv', index=False)
            test_data.to_csv(f'{out_dir}{prefix}_{split_policy}_{seed}_test.csv', index=False)

            print(f'{prefix} {split_policy} seed {seed}: {len(train_data)}, {len(valid_data)}, {len(test_data)}')

# Merged CSV format: smiles, task_name, group
for split_policy in split_methods:
    for i in range(5):
        seed = 2024 + i * 10
        merged_dfs = []
        for item in select_items:
            prefix = item['file_prefix']
            label = item['label']
            input_path = f'data_with_group_{split_policy}/{prefix}_{split_policy}_{seed}.csv'
            if not os.path.exists(input_path):
                continue
            df = pd.read_csv(input_path)
            if label not in df.columns:
                candidates = [c for c in df.columns if c not in ('group', 'smiles') and label in c]
                if not candidates:
                    continue
                used_label = candidates[0]
            else:
                used_label = label
            df_subset = df[['smiles', used_label, 'group']].copy()
            df_subset.columns = ['smiles', 'task_name', 'group']
            merged_dfs.append(df_subset)
        if merged_dfs:
            merged = pd.concat(merged_dfs, ignore_index=True)
            os.makedirs('data_split_for_merged', exist_ok=True)
            merged.to_csv(f'data_split_for_merged/{split_policy}_{seed}.csv', index=False)
            print(f'Merged {split_policy} seed {seed}: {len(merged)} rows')