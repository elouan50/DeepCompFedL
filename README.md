# DeepCompFedL
> Apply Deep Compression Techniques to Federated Learning, in order to save energy and computation power.

This work was realized during my Master's Thesis. It aims to provide an experimental setup which allows to apply various compression techniques to a Federated Learning framework.

Feel free to contact me for any questions, and to fork this repository to improve it yourself.

## Install dependencies and project

All the code was written and executed in a Linux environment (`Ubuntu Mate 22.04`), with a `Python 3.10` version.

Requirements for the <i>Flower Framework</i> can be found in the `pyproject.toml` file. To install related dependencies, execute this command:

```bash
pip install -e .
```


## Run the native framework with the Simulation Engine

In the `DeepCompFedL` directory, use `flwr run` to run a local simulation:

```bash
flwr run .
```

## Run the framework starting from the interface

The interface allows to modify parameters in a more intuitive manner.

In the `DeepCompFedL` directory, execute the `interface.py` file:
```bash
python3 interface.py
```
Beware, for the moment all parameters are not truly used during the execution.
TODO: implement use of the parameters, and also eventually add some others.

# Troubleshooting

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

