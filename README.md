# DeepCompFedL
> Apply Deep Compression Techniques to Federated Learning, in order to save energy and computation power.

This work was realized during and after my Master's Thesis. It aims to provide an experimental setup which allows to apply various compression techniques to a Federated Learning framework.

Feel free to contact me for any questions, and to fork this repository to improve it yourself.

---
## Install dependencies and project

All the code was written and executed in a Linux environment (`Ubuntu Mate 22.04`), with a `Python 3.10` version.

Requirements for the <i>Flower Framework</i> can be found in the `pyproject.toml` file. To install related `pip` dependencies, execute this command:

```bash
pip install -e .
```

If you also want to use the interface[^1], you'll need to install a `tkinter` package (not accessible via `pip`). For a Linux distribution, you can download it with:
```bash
sudo apt install python3-tk
```
[^1]: This feature wasn't updated for a long time and isn't garanteed to work.

[^2]: Use of such in-built interface is not recommanded, but I did it here for didactical reasons. For a similar experience, it would be better to build an API and a front-end application.</i>


In the adapted strategy file `./deepcompfedl/strategy/DeepCompFedLStrategy.py`, in the `__init__()`, `aggregate_fit()` and `aggregate_evaluate()` methods you'll find `wandb` calls. It is very useful for reporting online the results of the experiments you in a very user-friendly interface. If you want to use it, set the option save-online to true in the `pyproject.toml`. If you do so, you'll eventually need to connect your `wandb` account. I recommand to follow the documentation on the application own website.

---
## Run the project

To run the experiments described in my paper, execute this command (with `file_name.sh` being the file correspondign to the desired experiment):

```bash
sh file_name.sh
```

Elsewise, if you prefer to run your own experiment in a personalized setup, this project offers you two possibilities: either prepare your environment via the native framework _(recommanded)_, or with an interface I implemented myself.

### Run the native framework with the Simulation Engine

In the `DeepCompFedL` directory, use `flwr run` to run a local simulation:

```bash
flwr run .
```

Arguments can be modified in the `pyproject.toml` file, or passed manually within the command (see `sh` files for examples).

Additionally, run `flwr run --help` for more informations.

### Run the framework starting from the interface[^2]

The interface allows to modify parameters in a more intuitive manner.

In the `DeepCompFedL` directory, execute the `interface.py` file:
```bash
python3 interface.py
```

---
## Troubleshooting

It might be that at the first execution, you get such an Exception: `Exception ClientAppException occured. Message: module 'PIL.Image' has no attribute 'ExifTags'`. If so, simply update the package `pillow` by executing this command:
```bash
pip install --upgrade pillow
```
which should solve this issue.

Please also pay attention to the number of available CPUs and GPUs, defined at the end of the `pyproject.toml` file.
- If your machine is a Linux/Mac, to know the number of CPUs you can use, simply type `nproc` in a terminal.
- To know if you can use the GPU, run a python environment and try these lines:
```python
import torch
torch.cuda.is_available()
```
or for more informations:
```bash
nvidia-smi
```

If `cuda` is available, then it should be ok. Else, to avoid errors you have to set the number of available GPUs to 0 in the `pyproject.toml` configuration file.

Beware also with `wandb` use, you can't run two experiments with the same id within the same project, or else the execution will loop on the connection to the server. Check for `wandb.init()` in the `./deepcompfedl/strategy/DeepCompFedLStrategy.py` file.

## Acknowledgments

This project was developed on the basis of the [Flower Framework](https://flower.ai/docs/framework/index.html).

Compression functions are inspired from the work of Lucas Grativol ([here](https://github.com/lgrativol/fl_exps/), implementation of [his paper](https://ieeexplore.ieee.org/abstract/document/10382717)), and an implementation of [this paper](https://arxiv.org/abs/1510.00149) available [here](https://github.com/mightydeveloper/Deep-Compression-PyTorch).
