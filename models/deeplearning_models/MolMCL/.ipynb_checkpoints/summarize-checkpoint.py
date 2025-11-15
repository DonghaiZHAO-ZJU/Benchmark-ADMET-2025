import argparse
import os
import re
import numpy as np


def main(args):
    """ Example result file (BBBP_random_2024.txt):
        Run #1 (seed=2024): best=0.924971198156682 last=0.9127880184331797
        Run #2 (seed=2034): best=0.9196716589861752 last=0.9169642857142858
        Run #3 (seed=2044): best=0.9152649769585254 last=0.9208237327188941
        Run #4 (seed=2054): best=0.9120103686635944 last=0.926036866359447
        Run #5 (seed=2064): best=0.9201036866359448 last=0.9298099078341013
        Average last score: 0.9212845622119815
        Average best score: 0.9184043778801844
    """
    result_dict = {}
    result_path = args.result_path
    for file in os.listdir(result_path):
        seed_ordering = []
        if file.endswith('.txt'):
            file_name = file.split('.')[0]
            seed = file_name.split('_')[-1]
            split_type = file_name.split('_')[-2]
            data_name = '_'.join(file_name.split('_')[:-2])
            with open(os.path.join(result_path, file), 'r') as f:
                best_score_list = []
                for line in f.readlines():
                    line = line.rstrip()
                    if 'best' in line and 'seed' in line:
                        best_score = re.match('.*best=(.*) last=(.*)', line).group(1)
                        best_score_list.append(float(best_score))

                if data_name not in result_dict and 'CHEMBL' not in data_name:
                    result_dict[data_name] = {'random': [], 'scaffold': [], 'Perimeter': []}
                elif data_name not in result_dict:
                    result_dict[data_name] = {'ac': []}

                result_dict[data_name][split_type].append((seed, best_score_list))

    sorted_data_name = sorted(result_dict.keys())
    split_types = ['random', 'scaffold', 'Perimeter']
    if 'AC' in result_path:
        split_types = ['ac']
    for split_type in split_types:
        print(f'[{split_type}]')
        for data_name in sorted_data_name:
            raw_best_score_list = result_dict[data_name][split_type]
            raw_best_score_list = [t[1] for t in sorted(raw_best_score_list, key=lambda x: x[0])]
            if raw_best_score_list:
                best_score_list = [round(float(np.mean(res)), 8) for res in raw_best_score_list]
                if best_score_list:
                    print(f'  > {data_name}: {np.mean(best_score_list)}\n  \t{best_score_list}')
                    for i in range(len(raw_best_score_list)):
                        print(f'  \t\t{[round(float(score), 6) for score in raw_best_score_list[i]]}')
        print()

    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--result_path', type=str, required=True, help='Path to the result file')
    args = parser.parse_args()
    main(args)
