from build_dataset import built_data_and_save_for_splited
import argparse
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", help="the name of the target dataset", type=str, default="admet")
    args = parser.parse_args()

    input_csv = '../data/origin_data/' + args.dataset + '.csv'

    output_g_attentivefp_bin = '../data/Attentivefp_graph_data/' + args.dataset + '.bin'
    output_csv = '../data/Attentivefp_graph_data/' + args.dataset + '_group.csv'

    built_data_and_save_for_splited(
        origin_path=input_csv,
        save_g_attentivefp_path=output_g_attentivefp_bin,
        group_path=output_csv
    )

