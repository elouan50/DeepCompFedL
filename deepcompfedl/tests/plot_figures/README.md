# Plot Figures

This folder allows to plot some of the figures displayed in the final thesis, and calculate data used for Excel figures.

It can be read as follows:

## Figures

The files that display figures are `1st-order.py` and `distribution.py`.
As their names already explain, the first one shows how an approximation of the learning curve can be looks like, for later fitting it to a FL learning curve; the second details the repartition of the data classes among the clients.

## Plot & Fit

All files named `plotnfit_...` allow to fetch data from `wandb.ai` and calculate the tau coefficient, the average times and the final accuracy.
Sadly, all data stored online are linked with my account with a student free plan, so they are probably not accessible by other people. 
If you ever really need these data, feel free to contact me via my GitHub account.

Moreover, the `plotnfit.py` file provides a template for such needs.

## Save & ZIP

The `saveandzip.py` file is used to read the save non-trained models with different parametrizations.
The size is supposed to be independant from the quality of training.

## Script

Lastly, the `script.py` file just provides a template or accessing `wandb` resources.
