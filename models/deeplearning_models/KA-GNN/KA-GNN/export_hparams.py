hyperparameters = [
    ("the number of Fourier GNN layers", "3"),
    ("the number of KAN Linear layers in Readout", "2"),
    ("the KAN hidden layer size", "64"),
    ("the KAN output layer size", "32"),
    ("the KAN grid size (Fourier basis)", "1"),
    ("node feat size", "113 (92 Atom + 21 Bond)"),
    ("dropout", "0.1"),
    ("the number of epochs", "501"),
    ("the learning rate", "0.0001"),
    ("the patience of early stop", "30"),
    ("batch size", "128"),
    ("readout pooling", "Average Pooling"),
    ("activation function", "LeakyReLU")
]

file_path = "ka_gnn_hyperparameters.txt"

with open(file_path, mode='w', encoding='utf-8') as file:
    for key, value in hyperparameters:
        file.write(f"{key}: {value}\n")

# Also print it for immediate copying
for key, value in hyperparameters:
    print(f"{key}: {value}")
