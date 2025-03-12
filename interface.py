import tkinter as tk
from tkinter import ttk, messagebox
from tkinter.font import Font
import toml
import subprocess
import sys

def save_parameters(parameters):
    try:
        with open("deepcompfedl/__pycache__/interface-override.toml", "w") as file:
            toml.dump(parameters, file)
    except Exception as e:
        messagebox.showerror("Save Error", f"An error occurred while saving:\n{e}")

def start_model():
    parameters = {
        "server-rounds": int(server_rounds_entry.get()),
        "aggregation-strategy": agg_strategy_var.get(),
        "model": model_var.get(),
        "dataset": dataset_var.get(),
        "fraction-fit": float(fraction_clients_entry.get()),
        "client-epochs": int(client_epochs_entry.get()),
        "alpha": float(alpha_entry.get()) if enable_alpha_var.get() else None,
        "batch-size": int(batch_size_entry.get()),
        "client-enable-pruning": enable_pruning_var.get() and upstream_var.get(),
        "server-enable-pruning": enable_pruning_var.get() and downstream_var.get(),
        "client-enable-quantization": enable_quantization_var.get() and upstream_var.get(),
        "server-enable-quantization": enable_quantization_var.get() and downstream_var.get(),
        "pruning-rate": float(pruning_rate_entry.get()),
        "bits-quantization": int(bits_quantization_entry.get()),
        "quantization-method": quantization_method_var.get(),
        "init-space-quantization": init_space_var.get(),
        "layer-quantization": bool(layer_quantization_var.get())
    }
    save_parameters(parameters=parameters)
    command = ["flwr", "run", "--run-config", "deepcompfedl/__pycache__/interface-override.toml"]
    subprocess.run(command)
    subprocess.run(["rm", "deepcompfedl/__pycache__/interface-override.toml"])
    root.quit()
    sys.exit()

def toggle_quantization():
    state = "normal" if enable_quantization_var.get() else "disabled"
    bits_quantization_entry.config(state=state)
    quantization_method_menu.config(state=state)
    init_space_menu.config(state=state if quantization_method_var.get() == "K-Means (PTQ)" else "disabled")
    layer_quantization_checkbox.config(state=state)

def toggle_pruning():
    pruning_rate_entry.config(state="normal" if enable_pruning_var.get() else "disabled")

def toggle_init_space():
    init_space_menu.config(state="normal" if quantization_method_var.get() == "K-Means (PTQ)" else "disabled")

# Main window
root = tk.Tk()
root.title("Federated Learning Configuration")

main_frame = ttk.Frame(root, padding="10")
main_frame.grid(row=0, column=0, sticky="W")

# Federated Learning Parameters
fl_frame = ttk.LabelFrame(main_frame, text="Federated Learning Parameters", padding="10")
fl_frame.grid(row=0, column=0, padx=10, pady=5, sticky="W")

ttk.Label(fl_frame, text="Server Rounds:").grid(row=0, column=0, sticky="W")
server_rounds_entry = ttk.Entry(fl_frame, width=10)
server_rounds_entry.insert(0, "3")
server_rounds_entry.grid(row=0, column=1, sticky="W")

# Aggregation Strategy
ttk.Label(fl_frame, text="Aggregation Strategy:").grid(row=1, column=0, sticky="W")
agg_strategy_var = tk.StringVar(value="DeepCompFedLStrategy")
agg_strategy_menu = ttk.Combobox(fl_frame, textvariable=agg_strategy_var, values=["FedAvg", "DeepCompFedLStrategy"], state="readonly")
agg_strategy_menu.grid(row=1, column=1, sticky="W")

# Model & Dataset
ttk.Label(fl_frame, text="Model:").grid(row=2, column=0, sticky="W")
model_var = tk.StringVar(value="ResNet12")
model_menu = ttk.Combobox(fl_frame, textvariable=model_var, values=["ResNet12", "ResNet18", "QResNet12", "QResNet18", "Net"], state="readonly")
model_menu.grid(row=2, column=1, sticky="W")

ttk.Label(fl_frame, text="Dataset:").grid(row=3, column=0, sticky="W")
dataset_var = tk.StringVar(value="CIFAR-10")
dataset_menu = ttk.Combobox(fl_frame, textvariable=dataset_var, values=["CIFAR-10", "None other yet"], state="readonly")
dataset_menu.grid(row=3, column=1, sticky="W")

# Fraction of Clients & Number of Client epochs & Alpha
ttk.Label(fl_frame, text="Fraction of Clients:").grid(row=4, column=0, sticky="W")
fraction_clients_entry = ttk.Entry(fl_frame, width=10)
fraction_clients_entry.insert(0, "0.2")
fraction_clients_entry.grid(row=4, column=1, sticky="W")

ttk.Label(fl_frame, text="Number of Client epochs:").grid(row=5, column=0, sticky="W")
client_epochs_entry = ttk.Entry(fl_frame, width=10)
client_epochs_entry.insert(0, "1")
client_epochs_entry.grid(row=5, column=1, sticky="W")

enable_alpha_var = tk.BooleanVar(value=True)
enable_alpha_checkbox = ttk.Checkbutton(fl_frame, text="Enable Alpha for LDA: ", variable=enable_alpha_var)
enable_alpha_checkbox.grid(row=6, column=0, sticky="W")
alpha_entry = ttk.Entry(fl_frame, width=10)
alpha_entry.insert(0, "100")
alpha_entry.grid(row=6, column=1, sticky="W")

# Batch Size
ttk.Label(fl_frame, text="Batch Size:").grid(row=7, column=0, sticky="W")
batch_size_entry = ttk.Entry(fl_frame, width=10)
batch_size_entry.insert(0, "32")
batch_size_entry.grid(row=7, column=1, sticky="W")

# Deep Compression Parameters
dc_frame = ttk.LabelFrame(main_frame, text="Deep Compression Parameters", padding="10")
dc_frame.grid(row=1, column=0, padx=10, pady=5, sticky="W")
italic_font = Font(family="Helvetica", size=10, slant="italic")

downstream_var = tk.BooleanVar()
upstream_var = tk.BooleanVar()
enable_pruning_var = tk.BooleanVar(value=False)
enable_quantization_var = tk.BooleanVar(value=False)

pruning_rate_entry = ttk.Entry(dc_frame, width=10)
pruning_rate_entry.insert(0, "0.0")
pruning_rate_entry.config(state="disabled")
bits_quantization_entry = ttk.Entry(dc_frame, width=10)
bits_quantization_entry.insert(0, "32")
bits_quantization_entry.config(state="disabled")
quantization_method_var = tk.StringVar(value="K-Means (PTQ)")
quantization_method_menu = ttk.Combobox(dc_frame, textvariable=quantization_method_var, values=["K-Means (PTQ)", "Brevitas (QAT)"], state="disabled", postcommand=toggle_init_space)
init_space_var = tk.StringVar(value="uniform")
init_space_menu = ttk.Combobox(dc_frame, textvariable=init_space_var, values=["uniform", "density", "random"], state="disabled")
layer_quantization_var = tk.BooleanVar(value=True)
layer_quantization_checkbox = ttk.Checkbutton(dc_frame, text="Layer-wise Quantization", variable=layer_quantization_var, state="disabled")

# Place widgets
ttk.Checkbutton(dc_frame, text="Downstream", variable=downstream_var).grid(row=0, column=0, sticky="W")
ttk.Checkbutton(dc_frame, text="Upstream", variable=upstream_var).grid(row=0, column=1, sticky="W")
ttk.Checkbutton(dc_frame, text="Enable Pruning", variable=enable_pruning_var, command=toggle_pruning).grid(row=2, column=0, sticky="W")
ttk.Label(dc_frame, text="Pruning rate: ", font=italic_font).grid(row=2, column=1, sticky="E")
pruning_rate_entry.grid(row=2, column=2, sticky="W")
ttk.Checkbutton(dc_frame, text="Enable Quantization", variable=enable_quantization_var, command=toggle_quantization).grid(row=3, column=0, sticky="W")
ttk.Label(dc_frame, text="Bits for quantization: ", font=italic_font).grid(row=3, column=1, sticky="E")
bits_quantization_entry.grid(row=3, column=2, sticky="W")
ttk.Label(dc_frame, text="Quantization method: ", font=italic_font).grid(row=4, column=1, sticky="E")
quantization_method_menu.grid(row=4, column=2, sticky="W")
ttk.Label(dc_frame, text="Init space for K-Means: ", font=italic_font).grid(row=5, column=1, sticky="E")
init_space_menu.grid(row=5, column=2, sticky="W")
layer_quantization_checkbox.grid(row=6, column=0, sticky="W")

start_button = ttk.Button(main_frame, text="Start Model", command=start_model)
start_button.grid(row=2, column=0, padx=5, pady=10)

root.mainloop()
