from AttentiveFP import build_dataset
import os

data_splitting_methods = ['random', 'scaffold']
task_list = ["PAMPA1"]

for sample_ratio in [0.1, 0.2, 0.4, 0.6, 0.8]:
    for i in range(5):
        seed=2024+i*10
        for method in data_splitting_methods:
            for task in task_list:
                print(f'current task: {task}, current method: {method}, current seed: {seed}, current sample ratio: {sample_ratio}')
                build_dataset.built_mol_graph_data_and_save(
                    origin_data_path=f'./data/origin_data/{task}_{method}_{seed}_{sample_ratio}.csv',
                    labels_name=task,
                    save_g_path=f'./data/Attentivefp_graph_data/{task}_{method}_{seed}_{sample_ratio}.bin',
                    save_g_group_path=f'./data/Attentivefp_graph_data/{task}_{method}_{seed}_{sample_ratio}_group.csv',
                    )