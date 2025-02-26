"""DeepCompFedL: A Flower / PyTorch app.

This module provides utility functions for model parameter inspection.

Functions:
    pretty_parameters(model): Prints a table of the model's trainable parameters and returns the total count.

Example usage:
    model = YourModel()
    total_params = pretty_parameters(model)

"""

from prettytable import PrettyTable

def pretty_parameters(model):
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
