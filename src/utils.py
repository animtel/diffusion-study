import pickle
from prettytable import PrettyTable


def write_pickle(filename, obj):
    with open(filename, 'wb') as handle:
        pickle.dump(obj, handle)


def read_pickle(filename):
    with open(filename, 'rb') as handle:
        return pickle.load(handle)


def count_parameters(model):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: continue
        params = parameter.numel()
        table.add_row([name, params])
        total_params+=params
    print(table)
    print(f"Total Trainable Params: {total_params}")
    return total_params