"""
This file aims to calculate all the metrics
relative to the fourth experiment.
It includes evaluating average values, show final performances,
and interpolating convergence speed.
"""

##### VALUE TO CHANGE #####
n = 4  # Scenario number (1 to 4)
###########################

rounds_for_scenario = {1: 99, 2: 50, 3: 350, 4: 30}
nrounds = rounds_for_scenario[n]

import numpy as np
import plotly.graph_objects as go
import pandas as pd
import wandb
from scipy.optimize import curve_fit

X = [(i+1)*0.1 for i in range(9)] + [0.95]
Y = [2**(i+2) for i in range(7)]
X, Y = np.meshgrid(X, Y)
Z = np.zeros((7, 10)) 

api = wandb.Api(timeout=60)
project = f"elouan50-rwth-aachen-university/deepcompfedl-scenario{n}"


print("---------------------------")
print(f"------  Scenario {n}  -------")
print("---------------------------")
print("")
print("Model: ResNet-12")
print("Dataset: " + ("FEMNIST" if 1-n%2 else "CIFAR-10"))
print(("non-" if n>2 else "") + "IID data")
print("")

# Define the first-order model
def first_order_response(t, K, tau, bias):
   return K*(1-np.exp(-t/tau)) + bias

baseline = np.zeros((nrounds))
accuracies = np.zeros((7, 10, nrounds))
t_selects = np.zeros((7, 10, nrounds))
t_trains = np.zeros((7, 10, nrounds))
t_compresses = np.zeros((7, 10, nrounds))
t_decodes = np.zeros((7, 10, nrounds))
t_aggregates = np.zeros((7, 10, nrounds))

pr_dic = {0.1: 0,
          0.2: 1,
          0.3: 2,
          0.4: 3,
          0.5: 4,
          0.6: 5,
          0.7: 6,
          0.8: 7,
          0.9: 8,
          0.95: 9
          }

runs = api.runs(project)

for run in runs:
    pr = run.config["pruning-rate"]
    qb = run.config["bits-quantization"]
    df = run.history(keys=["accuracy", 't_select', 't_train', 't_compress', 't_decode', 't_aggregate'], pandas=True, samples=nrounds)

    if qb==32: #Baseline
        baseline += np.array(df['accuracy'])/3
    else:
        accuracies[qb-2, pr_dic[pr]] += np.array(df['accuracy'])/3
        t_selects[qb-2, pr_dic[pr]] += np.array(df['t_select'])/3
        t_trains[qb-2, pr_dic[pr]] += np.array(df['t_train'])/3
        t_compresses[qb-2, pr_dic[pr]] += np.array(df['t_compress'])/3
        t_decodes[qb-2, pr_dic[pr]] += np.array(df['t_decode'])/3
        t_aggregates[qb-2, pr_dic[pr]] += np.array(df['t_aggregate'])/3
    
steps = np.array([i for i in range(1,nrounds+1)])

min_, max_ = min(baseline), max(baseline)

# Own method to calculate the model
K = max_ - min_
tau = steps[np.argmax(baseline > 0.63*K+min_)]
bias = min_

latex_header = "\\begin{table}[!ht] \n"
latex_header += "   \centering \n"
latex_header += "   \\begin{tabular}{|c|cccccccccc|}\n"
latex_header += "       \hline \n"
latex_header += "       x & 0.1 & 0.2 & 0.3 & 0.4 & 0.5 & 0.6 & 0.7 & 0.8 & 0.9 & 0.95 \\\\ \n"
latex_header += "       \hline \n"

# Calculate the model and evaluate it
accuracy_calc = first_order_response(steps, K, tau, bias)
r2 = 1-np.sum((baseline - accuracy_calc)**2)/np.sum((baseline - np.mean(baseline))**2)

accuracies_end = np.zeros((7, 10))
accuracy_end_latex_table = latex_header
tau_latex_table = latex_header
r2_latex_table = latex_header
client_overhead_table = latex_header
server_overhead_table = latex_header
computation_overhead_table = latex_header

print("")
print("Baseline ACCURACY: ", round(baseline[nrounds-1], 5))
print("")
print("Baseline TAU: ", tau)
print("Baseline r²: ", r2)
print("")

for qb in range(6, -1, -1):
    accuracy_end_latex_table += "       " + str(2**(qb+2)).ljust(3, " ") + " & "
    tau_latex_table += "       " + str(2**(qb+2)).ljust(3, " ") + " & "
    r2_latex_table += "       " + str(2**(qb+2)).ljust(3, " ") + " & "
    client_overhead_table += "       " + str(2**(qb+2)).ljust(3, " ") + " & "
    server_overhead_table += "       " + str(2**(qb+2)).ljust(3, " ") + " & "
    computation_overhead_table += "       " + str(2**(qb+2)).ljust(3, " ") + " & "
    
    for pr in range(10):
        accuracy = accuracies[qb, pr]
        accuracies_end[qb, pr] = accuracy[-1]
        
        accuracy_end_latex_table += str(round(accuracy[-1], 5)).ljust(7, "0") + (" \\\\ \n" if pr==9 else " & ")
        
        min_, max_ = min(accuracy), max(accuracy)
        K = max_ - min_
        tau = steps[np.argmax(accuracy > 0.63*K+min_)]
        bias = min_
        accuracy_calc = first_order_response(steps, K, tau, bias)
        r2 = 1-np.sum((accuracy - accuracy_calc)**2)/np.sum((accuracy - np.mean(accuracy))**2)
        
        tau_latex_table += str(tau).rjust(2, " ") + (" \\\\ \n" if pr==9 else " & ")
        r2_latex_table += str(round(r2, 2)).ljust(4, "0") + (" \\\\ \n" if pr==9 else " & ")
        
        t_select = np.average(t_selects[qb, pr, 1:])
        t_train = np.average(t_trains[qb, pr, 1:])
        t_compress = np.average(t_compresses[qb, pr, 1:])
        t_decode = np.average(t_decodes[qb, pr, 2:])
        t_aggregate = np.average(t_aggregates[qb, pr, :-1])
        
        client_overhead_table += str(round((t_train + t_compress)/t_train, 3)).ljust(5, "0") + (" \\\\ \n" if pr==9 else " & ")
        server_overhead_table += str(round((t_select + t_decode + t_aggregate)/(t_select + t_aggregate), 3)).ljust(5, "0") + (" \\\\ \n" if pr==9 else " & ")
        computation_overhead_table += str(round((t_select + t_train + t_compress + t_decode + t_aggregate)/(t_select + t_train + t_aggregate), 3)).ljust(5, "0") + (" \\\\ \n" if pr==9 else " & ")
        

accuracy_end_latex_table += "       \hline \n"
accuracy_end_latex_table += "   \end{tabular} \n"
accuracy_end_latex_table += "   \caption{Final accuracy values for scenario " + str(n) + "} \n"
accuracy_end_latex_table += "   \label{tab:acc-scenario" + str(n) + "} \n"
accuracy_end_latex_table += "\end{table}"

tau_latex_table += "       \hline \n"
tau_latex_table += "   \end{tabular} \n"
tau_latex_table += "   \caption{Tau values for scenario " + str(n) + "} \n"
tau_latex_table += "   \label{tab:tau-scenario" + str(n) + "} \n"
tau_latex_table += "\end{table}"

r2_latex_table += "       \hline \n"
r2_latex_table += "   \end{tabular} \n"
r2_latex_table += "   \caption{r² values for scenario " + str(n) + "} \n"
r2_latex_table += "   \label{tab:r2-scenario" + str(n) + "} \n"
r2_latex_table += "\end{table}"

client_overhead_table += "       \hline \n"
client_overhead_table += "   \end{tabular} \n"
client_overhead_table += "   \caption{Client overhead values for scenario " + str(n) + "} \n"
client_overhead_table += "   \label{tab:clientoverhead-scenario" + str(n) + "} \n"
client_overhead_table += "\end{table}"

server_overhead_table += "       \hline \n"
server_overhead_table += "   \end{tabular} \n"
server_overhead_table += "   \caption{Server overhead values for scenario " + str(n) + "} \n"
server_overhead_table += "   \label{tab:serveroverhead-scenario" + str(n) + "} \n"
server_overhead_table += "\end{table}"

computation_overhead_table += "       \hline \n"
computation_overhead_table += "   \end{tabular} \n"
computation_overhead_table += "   \caption{Computation overhead values for scenario " + str(n) + "} \n"
computation_overhead_table += "   \label{tab:computationoverhead-scenario" + str(n) + "} \n"
computation_overhead_table += "\end{table}"

print("Latex table for final accuracies:")
print("")
print(accuracy_end_latex_table)
print("")

print("Latex table for tau:")
print("")
print(tau_latex_table)
print("")

print("Latex table for r²:")
print("")
print(r2_latex_table)
print("")

print("Latex table for client overhead:")
print("")
print(client_overhead_table)
print("")

print("Latex table for server overhead:")
print("")
print(server_overhead_table)
print("")

print("Latex table for computation overhead:")
print("")
print(computation_overhead_table)
print("")


print("")
print("Best fit evaluation (with scipy.optimize.curve_fit):")
print("")

min_, max_ = min(baseline), max(baseline)
K = max_ - min_
tau = steps[np.argmax(baseline > 0.63*K+min_)]
bias = min_
params, _ = curve_fit(first_order_response, steps, baseline, p0=[K, tau, bias])
K_opt, tau_opt, bias_opt = params
accuracy_fit = first_order_response(steps, K_opt, int(tau_opt), bias_opt)
r2_fit = 1-np.sum((baseline - accuracy_fit)**2)/np.sum((baseline - np.mean(baseline))**2)

accuracy_opt_latex_table = latex_header
tau_opt_latex_table = latex_header

print("Baseline ACCURACY (opt): ", round((K_opt+bias_opt), 5))
print("Baseline TAU (opt): ", int(tau_opt))
print("Baseline r² (opt): ", r2_fit)
print("")
print("           |                                                                     Pruning rate")
print("Nb clusters|  0.1             0.2             0.3             0.4             0.5             0.6             0.7             0.8             0.9             0.95")
print("-----------+----------------------------------------------------------------------------------------------------------------------------------------------------------------")

accuracies_calc = np.zeros((7, 10))

for qb in range(6, -1, -1):
    print("   ", 2**(qb+2), "" if qb>=5 else (" " if qb >= 2 else "  "), "  | ", end="")
    accuracy_opt_latex_table += "       " + str(2**(qb+2)).ljust(3, " ") + " & "
    tau_opt_latex_table += "       " + str(2**(qb+2)).ljust(3, " ") + " & "
    for pr in range(10):
        accuracy = accuracies[qb, pr]
        min_, max_ = min(accuracy), max(accuracy)
        K = max_ - min_
        try:
            params, _ = curve_fit(first_order_response, steps, accuracy, p0=[max_-min_, steps[np.argmax(accuracy > 0.63*(max_-min_))], min_])
            K_opt, tau_opt, bias_opt = params
            accuracy_fit = first_order_response(steps, K_opt, 1 if int(tau_opt)==0 else int (tau_opt), bias_opt)
            r2_fit = 1-np.sum((accuracy - accuracy_fit)**2)/np.sum((accuracy - np.mean(accuracy))**2)
        except:
            K_opt = 0
            tau_opt = 0
            r2_fit = 0
        
        accuracies_calc[qb, pr] = K_opt + bias_opt
        
        print(str(int(tau_opt)).rjust(3, " ") + " (" + (".00" if r2_fit<=0. else str(round(r2_fit, 2))[1:].ljust(3, "0")) + ", " + ("none" if r2_fit<0.8 else str(round(K_opt+bias_opt,3))[1:].ljust(4, "0")) + ")", end=" ")
        accuracy_opt_latex_table += str(round(K_opt+bias_opt,3)).ljust(5, "0") + (" \\\\ \n" if pr==9 else " & ")
        tau_opt_latex_table += str(int(tau_opt)).rjust(3, " ") + (" \\\\ \n" if pr==9 else " & ")
    print("")
    
accuracy_opt_latex_table += "       \hline \n"
accuracy_opt_latex_table += "   \end{tabular} \n"
accuracy_opt_latex_table += "   \caption{Optimal accuracy values for scenario " + str(n) + "} \n"
accuracy_opt_latex_table += "   \label{tab:accopt-scenario" + str(n) + "} \n"
accuracy_opt_latex_table += "\end{table}"

tau_opt_latex_table += "       \hline \n"
tau_opt_latex_table += "   \end{tabular} \n"
tau_opt_latex_table += "   \caption{Optimal tau values for scenario " + str(n) + "} \n"
tau_opt_latex_table += "   \label{tab:tauopt-scenario" + str(n) + "} \n"
tau_opt_latex_table += "\end{table}"

print("")
print("Latex table for optimal accuracies:")
print("")
print(accuracy_opt_latex_table)
print("")
print("Latex table for optimal taus:")
print("")
print(tau_opt_latex_table)
print("")


# Plot 3d surface graph

# Plottable: accuracies_calc, accuracies_end
# For other graph, pay attention to the dtick of the z-axis
data = accuracies_end



dtick_for_scenario = {1: 0.05, 2: 0.005, 3: 0.1, 4: 0.01}

surface = pd.DataFrame(data,
                       index=[2**i for i in range(2,9)],
                       columns=[.1,.2,.3,.4,.5,.6,.7,.8,.9,.95]
                       )

fig = go.Figure(data=[go.Surface(x=surface.columns,
                                 y=np.log2(surface.index),
                                 z=surface.values,
                                 colorscale='Viridis',
                                 lighting=dict(
                                     ambient=0.8,
                                     diffuse=0.5,
                                     specular=0.2,
                                     roughness=0.9,
                                     fresnel=0.1
                                 ),
                                 lightposition=dict(x=100, y=200, z=0),  # Light from a better angle
                                 contours={
                                     "x": {"show": True, "start": 0, "end": 1, "size": 0.1},
                                     "y": {"show": True, "start": 0, "end": 8, "size": 1},
                                     "z": {"show": True, "start": 0, "end": 1, "size": 0.01}
                                 }
                                 )])

fig.update_layout(
    scene=dict(
        xaxis_title=dict(text='Pruning rate', font=dict(size=23)),
        yaxis_title=dict(text='Quantization clusters', font=dict(size=23)),
        zaxis_title=dict(text='Accuracy', font=dict(size=25)),
        xaxis=dict(
            tickmode='array',
            tickvals=surface.columns,
            ticktext=[str(i) for i in surface.columns],
            tickfont=dict(size=16),
            minallowed=0.1, maxallowed=0.95
        ),
        yaxis=dict(
            tickmode='array',
            tickvals=np.log2(surface.index),
            ticktext=[str(i) for i in surface.index],
            tickfont=dict(size=16),
            minallowed=2, maxallowed=8
        ),
        zaxis=dict(
            dtick=dtick_for_scenario[n],
            tickfont=dict(size=18),
        )
    ),
    margin=dict(l=0, r=0, t=0, b=100)
)

fig.update_traces(contours_z=dict(show=True,
                                  usecolormap=True,
                                  project_z=False))

fig.show()
