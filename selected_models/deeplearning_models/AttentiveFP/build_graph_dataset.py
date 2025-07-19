from AttentiveFP import build_dataset
import os

data_splitting_methods = ['Perimeter']
task_list = ["PAMPA1"]

for i in range(5):
    seed=2024+i*10
    for method in data_splitting_methods:
        for task in task_list:
            print(f'current task: {task}, current method: {method}, current seed: {seed}')
            if os.path.exists(f"data/Attentivefp_graph_data/{task}_{method}_{seed}.bin"):
                print("Data already exists!")
            else:
                build_dataset.built_mol_graph_data_and_save(
                    origin_data_path=f'./data/origin_data/{task}_{method}_{seed}.csv',
                    labels_name=task,
                    save_g_path=f'./data/Attentivefp_graph_data/{task}_{method}_{seed}.bin',
                    save_g_group_path=f'./data/Attentivefp_graph_data/{task}_{method}_{seed}_group.csv',
                    )