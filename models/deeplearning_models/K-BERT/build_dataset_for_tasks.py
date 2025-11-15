from experiment import build_data
import os
data_splitting_methods = ['MoleculeACE']
task_list = ["BBBP","hERG","Mutagenicity","oral_bioavailability","HLM_metabolic_stability", "Caco2","HalfLife","VDss","PAMPA1"]
selected_tasks = ['CHEMBL1862_Ki', 'CHEMBL1871_Ki', 'CHEMBL2034_Ki', 'CHEMBL2047_EC50', 
                    'CHEMBL204_Ki', 'CHEMBL2147_Ki', 'CHEMBL214_Ki', 'CHEMBL218_EC50', 
                    'CHEMBL219_Ki', 'CHEMBL228_Ki', 'CHEMBL231_Ki', 'CHEMBL233_Ki', 
                    'CHEMBL234_Ki', 'CHEMBL235_EC50', 'CHEMBL236_Ki', 'CHEMBL237_EC50', 
                    'CHEMBL237_Ki', 'CHEMBL238_Ki', 'CHEMBL239_EC50', 'CHEMBL244_Ki', 
                    'CHEMBL262_Ki', 'CHEMBL264_Ki', 'CHEMBL2835_Ki', 'CHEMBL287_Ki', 
                    'CHEMBL2971_Ki', 'CHEMBL3979_EC50', 'CHEMBL4005_Ki', 'CHEMBL4203_Ki', 
                    'CHEMBL4616_EC50', 'CHEMBL4792_Ki']

os.makedirs('./data/origin_data/',exist_ok=True)
os.makedirs('./data/token_data/',exist_ok=True)

for i in range(5):
    seed=2024+i*10
    for method in data_splitting_methods:
        for task in selected_tasks:
            build_data.built_data_and_save_for_splited(
                origin_path=f'./data/origin_data/{task}_{method}_{seed}.csv',
                save_path=f'./data/token_data/{task}_{method}_{seed}.npy')