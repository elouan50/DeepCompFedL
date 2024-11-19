# DeepCompFedL
> Apply Deep Compression Techniques to Federated Learning, in order to save energy and computation power.

This work was realized during my Master's Thesis. It aims to provide an experimental setup which allows to apply various compression techniques to a Federated Learning framework.

Feel free to contact me for any questions, and to fork this repository to improve it yourself.

## Install dependencies and project

All the code was written and executed in a Linux environment (´Ubuntu Mate 22.04´), with a `Python 3.10` version.

Requirements for the <i>Flower Framework</i> can be found in the `pyproject.toml` file. To install related dependencies, execute this command:

```bash
pip install -e .
```


## Run the native framework with the Simulation Engine

In the `DeepCompFedL` directory, use `flwr run` to run a local simulation:

```bash
flwr run .
```

## Debugging

It might be that at the first execution, you get such an Exception: `Exception ClientAppException occured. Message: module 'PIL.Image' has no attribute 'ExifTags'`. If so, simply update the package `pillow` by executing this command:
```bash
pip install --upgrade pillow
```
which should solve this issue.
