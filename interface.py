import tkinter as tk
from tkinter import ttk, messagebox
from tkinter.font import Font

import toml
import subprocess
import sys

def save_parameters(parameters):
    # Save parameters as TOML
    try:
        with open("deepcompfedl/__pycache__/interface-override.toml", "w") as file:
            toml.dump(parameters, file)
    except Exception as e:
        messagebox.showerror("Save Error", f"An error occurred while saving:\n{e}")


def start_model():
    parameters = {
        "server-rounds": int(server_rounds_entry.get()),
        "server-enable-pruning" : bool(enable_server_pruning_var.get()),
        "server-pruning-rate": float(server_pruning_entry.get()),
        "server-enable-quantization" : bool(enable_server_quantization_var.get()),
        "server-bits-quantization": int(server_quantization_entry.get()),
        "aggregation-strategy": agg_strategy_var.get(),
        "client-epochs": int(client_epochs_entry.get()),
        "client-enable-pruning" : bool(enable_client_pruning_var.get()),
        "client-pruning-rate": float(client_pruning_entry.get()),
        "client-enable-quantization" : bool(enable_client_quantization_var.get()),
        "client-bits-quantization": int(client_quantization_entry.get()),
        "model": model_var.get(),
        "dataset": dataset_var.get(),
        "fraction-fit": float(fraction_clients_entry.get()),
        "alpha": float(alpha_entry.get())
    }
    save_parameters(parameters=parameters)
    command = ["flwr", "run",
               "--run-config", "deepcompfedl/__pycache__/interface-override.toml"]
    subprocess.run(command)
    subprocess.run(["rm","deepcompfedl/__pycache__/interface-override.toml"])
    root.quit()
    sys.exit()

def toggle_server_compression():
    """Enable or disable the server compresion entries based on the Combobox."""
    if agg_strategy_var.get() == "DeepCompFedLStrategy":
        enable_server_pruning_checkbox.config(state="normal")
        enable_server_quantization_checkbox.config(state="normal")
    else:
        enable_server_pruning_checkbox.config(state="disabled")
        enable_server_quantization_checkbox.config(state="disabled")


def toggle_server_pruning():
    """Enable or disable the server pruning entry based on the checkbox."""
    if enable_server_pruning_var.get():
        server_pruning_entry.config(state="normal")
    else:
        server_pruning_entry.config(state="disabled")

def toggle_server_quantization():
    """Enable or disable the server quantization entry based on the checkbox."""
    if enable_server_quantization_var.get():
        server_quantization_entry.config(state="normal")
    else:
        server_quantization_entry.config(state="disabled")

def toggle_client_pruning():
    """Enable or disable the client pruning entry based on the checkbox."""
    if enable_client_pruning_var.get():
        client_pruning_entry.config(state="normal")
    else:
        client_pruning_entry.config(state="disabled")

def toggle_client_quantization():
    """Enable or disable the client quantization entry based on the checkbox."""
    if enable_client_quantization_var.get():
        client_quantization_entry.config(state="normal")
    else:
        client_quantization_entry.config(state="disabled")

# Main window
root = tk.Tk()
root.title("Federated Learning Parameters Configuration")

# Pararmeters
italic_font = Font(family="Helvetica", size=10, slant="italic")

# Main frame
main_frame = ttk.Frame(root, padding="10")
main_frame.grid(row=0, column=0, sticky="W")


## Section 1: Server Parameters
server_frame = ttk.LabelFrame(main_frame, text="Server Parameters", padding="10")
server_frame.grid(row=0, column=0, padx=10, pady=5, sticky="W")

# Server Rounds
ttk.Label(server_frame, text="Number of server rounds:").grid(row=0, column=0, sticky="W")
server_rounds_entry = ttk.Entry(server_frame, width=10)
server_rounds_entry.insert(0, "3")
server_rounds_entry.grid(row=0, column=2, sticky="W")


# Aggregation strategy
ttk.Label(server_frame, text="Aggregation strategy:").grid(row=1, column=0, sticky="W")
agg_strategy_var = tk.StringVar(value="FedAvg")
agg_strategy_menu = ttk.Combobox(server_frame, textvariable=agg_strategy_var, values=["FedAvg", "DeepCompFedLStrategy", "other (None)"], state="readonly", postcommand=toggle_server_compression)
agg_strategy_menu.grid(row=1, column=2, sticky="W")


# Server Pruning
enable_server_pruning_var = tk.BooleanVar(value=False)
enable_server_pruning_checkbox = ttk.Checkbutton(server_frame, text="Enable Server Pruning ", variable=enable_server_pruning_var, command=toggle_server_pruning)
enable_server_pruning_checkbox.config(state="disabled")
enable_server_pruning_checkbox.grid(row=2, column=0, sticky="W")

ttk.Label(server_frame, text="effective rate: ", font=italic_font).grid(row=2, column=1, sticky="E")
server_pruning_entry = ttk.Entry(server_frame, width=10)
server_pruning_entry.insert(0, "0.0")
server_pruning_entry.config(state="disabled")
server_pruning_entry.grid(row=2, column=2, sticky="W")


# Server Quantization
enable_server_quantization_var = tk.BooleanVar(value=False)
enable_server_quantization_checkbox = ttk.Checkbutton(server_frame, text="Enable Server Quantization ", variable=enable_server_quantization_var, command=toggle_server_quantization)
enable_server_quantization_checkbox.config(state="disabled")
enable_server_quantization_checkbox.grid(row=3, column=0, sticky="W")

ttk.Label(server_frame, text="nb of bits: ", font=italic_font).grid(row=3, column=1, sticky="E")
server_quantization_entry = ttk.Entry(server_frame, width=10)
server_quantization_entry.insert(0, "32")
server_quantization_entry.config(state="disabled")
server_quantization_entry.grid(row=3, column=2, sticky="W")




## Section 2: Local Client Parameters
client_frame = ttk.LabelFrame(main_frame, text="Local Client Parameters", padding="10")
client_frame.grid(row=1, column=0, padx=10, pady=5, sticky="W")

# Client Epochs
ttk.Label(client_frame, text="Number of client epochs:").grid(row=0, column=0, sticky="W")
client_epochs_entry = ttk.Entry(client_frame, width=10)
client_epochs_entry.insert(0, "1")
client_epochs_entry.grid(row=0, column=2, sticky="W")


# Client Pruning
enable_client_pruning_var = tk.BooleanVar(value=False)
enable_client_pruning_checkbox = ttk.Checkbutton(client_frame, text="Enable Client Pruning ", variable=enable_client_pruning_var, command=toggle_client_pruning)
enable_client_pruning_checkbox.grid(row=1, column=0, sticky="W")

ttk.Label(client_frame, text="effective rate: ", font=italic_font).grid(row=1, column=1, sticky="E")
client_pruning_entry = ttk.Entry(client_frame, width=10)
client_pruning_entry.insert(0, "0.0")
client_pruning_entry.config(state="disabled")
client_pruning_entry.grid(row=1, column=2, sticky="W")


# Client Quantization
enable_client_quantization_var = tk.BooleanVar(value=False)
enable_client_quantization_checkbox = ttk.Checkbutton(client_frame, text="Enable Client Quantization ", variable=enable_client_quantization_var, command=toggle_client_quantization)
enable_client_quantization_checkbox.grid(row=2, column=0, sticky="W")

ttk.Label(client_frame, text="nb of bits: ", font=italic_font).grid(row=2, column=1, sticky="E")
client_quantization_entry = ttk.Entry(client_frame, width=10)
client_quantization_entry.insert(0, "32")
client_quantization_entry.config(state="disabled")
client_quantization_entry.grid(row=2, column=2, sticky="W")




## Section 3: Advanced Parameters
advanced_frame = ttk.LabelFrame(main_frame, text="Advanced Parameters", padding="10")
advanced_frame.grid(row=2, column=0, padx=10, pady=5, sticky="W")


# Model
ttk.Label(advanced_frame, text="Model: ").grid(row=0, column=0, sticky="W")
model_var = tk.StringVar(value="ResNet12")
model_menu = ttk.Combobox(advanced_frame, textvariable=model_var, values=["ResNet12", "ResNet18", "Net", "other (None)"], state="readonly")
model_menu.grid(row=0, column=1, sticky="W")


# Dataset
ttk.Label(advanced_frame, text="Dataset: ").grid(row=1, column=0, sticky="W")
dataset_var = tk.StringVar(value="CIFAR-10")
dataset_menu = ttk.Combobox(advanced_frame, textvariable=dataset_var, values=["CIFAR-10", "other (None)"], state="readonly")
dataset_menu.grid(row=1, column=1, sticky="W")


# Fraction of clients
ttk.Label(advanced_frame, text="Fraction of clients selected: ").grid(row=2, column=0, sticky="W")
fraction_clients_entry = ttk.Entry(advanced_frame, width=10)
fraction_clients_entry.insert(0, "0.2")
fraction_clients_entry.grid(row=2, column=1, sticky="W")


# Distribution
ttk.Label(advanced_frame, text="Alpha for LDA distribution: ").grid(row=3, column=0, sticky="W")
alpha_entry = ttk.Entry(advanced_frame, width=10)
alpha_entry.insert(0, "100")
alpha_entry.grid(row=2, column=1, sticky="W")



### Buttons
button_frame = ttk.Frame(main_frame, padding="10")
button_frame.grid(row=3, column=0, sticky="S")

start_button = ttk.Button(button_frame, text="Start Model", command=start_model)
start_button.grid(row=0, column=1, padx=5)

# Run the Tkinter main loop
root.mainloop()
