SEED_NO = 0
DEVICE =  "cuda"
EPOCHS = 300
NUM_GRAPHS_PER_BATCH = 256
NUM_FEATURES=32
NUM_TARGET = 1
EDGE_DIM=11
PATIENCE = 10
N_SPLITS = 5

MAXPATH = 2
MINPATH = 1
FPSIZE_RDKFP = 1024

MORGAN_RAD = 2
MORGAN_BITS = 1024

# Parameters for random forest model
best_params_MD = {'n_estimators': 76, 'max_depth': 96}
best_params_maccskeys = {'n_estimators': 85, 'max_depth': 46}
best_params_morgan = {'n_estimators': 89, 'max_depth': 40}
best_params_rdkfp = {'n_estimators': 28, 'max_depth': 11}

# Parameters for GNN from scratch
params_gin = {'num_layers': 1, 'hidden_size': 66, 'learning_rate': 0.00889495369073538}
params_gt = {'num_layers': 2, 'hidden_size': 439, 'n_heads': 1, 'dropout': 0.269754753387312, 'learning_rate': 0.007890910361468965}
params_parallel_gnn = {'num_gin_layers': 3, 'num_graph_trans_layers': 2, 'hidden_size': 248, 'n_heads': 2, 'dropout': 0.343018493669972, 'learning_rate': 0.002091824465119609}
params_vertical_gnn = {'num_gin_layers': 2, 'num_graph_trans_layers': 2, 'hidden_size': 122, 'n_heads': 2, 'dropout': 0.36738054656589025, 'learning_rate': 0.00452976319043267}

#Parameters for trf learning models
best_params_parallel = {'num_gin_layers': 2, 'num_graph_trans_layers': 2, 'hidden_size': 396, 'n_heads': 1, 'dropout': 0.21481760818350845, 'learning_rate': 0.0012104847904387125}
best_params_vertical = {'num_gin_layers': 2, 'num_graph_trans_layers': 2, 'hidden_size': 245, 'n_heads': 1, 'dropout': 0.30146027310173296, 'learning_rate': 0.0012649520485726895}