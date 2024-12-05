# DeepCompFedL
> Apply Deep Compression Techniques to Federated Learning, in order to save energy and computation power.

This work was realized during my Master's Thesis. It aims to provide an experimental setup which allows to apply various compression techniques to a Federated Learning framework.

Feel free to contact me for any questions, and to fork this repository to improve it yourself.

## Install dependencies and project

All the code was written and executed in a Linux environment (`Ubuntu Mate 22.04`), with a `Python 3.10` version.

Requirements for the <i>Flower Framework</i> can be found in the `pyproject.toml` file. To install related `pip` dependencies, execute this command:

```bash
pip install -e .
```

If you also want to use the interface, you'll need to install a `tkinter` package (not accessible via `pip`). For a Linux distribution, you can download it with:
```bash
sudo apt install python3-tk
```

<i>Note: use of such in-built interface is not recommanded, but I did it here for simplicity reasons. It would be better to build an API and a front-end application.</i>

## Run the project

You have two possibilities to run the project: either via the native framework, or with an interface I implemented myself.

### Run the native framework with the Simulation Engine

In the `DeepCompFedL` directory, use `flwr run` to run a local simulation:

```bash
flwr run .
```

Arguments can be modified in the `pyproject.toml` file.

Additionally, run `flwr run --help` for more informations.

### Run the framework starting from the interface

The interface allows to modify parameters in a more intuitive manner.

In the `DeepCompFedL` directory, execute the `interface.py` file:
```bash
python3 interface.py
```
Beware, for the moment all parameters are not truly used during the execution.
TODO: implement use of the parameters, and also eventually add some others.

## Troubleshooting

It might be that at the first execution, you get such an Exception: `Exception ClientAppException occured. Message: module 'PIL.Image' has no attribute 'ExifTags'`. If so, simply update the package `pillow` by executing this command:
```bash
pip install --upgrade pillow
```
which should solve this issue.

Please also pay attention to the number of available CPUs and GPUs, defined at the end of the `pyproject.toml` file.
- If you works on Linux/Mac, to know the number of CPUs you can use, simply type `nproc` in a terminal.
- To know if you can use the GPU, run a python environment and try these lines:
```python
import torch
torch.cuda.is_available()
```
or for more informations:
```bash
nvidia-smi
```

If `cuda` is available, then it should be ok. Else, either set the number of available GPUs to 0 or try to google your problem :\)

## Acknowledgments

This project was developed on the basis of the [Flower Framework](https://flower.ai/docs/framework/index.html).

Many functions are inspired from the work of Lucas Grativol ([here](https://github.com/lgrativol/fl_exps/), implementation of [his paper](https://ieeexplore.ieee.org/abstract/document/10382717)), and an implementation of [this paper](https://arxiv.org/abs/1510.00149) available [here](https://github.com/mightydeveloper/Deep-Compression-PyTorch).
