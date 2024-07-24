# Modeling Neuron Encoding of Visual Stimuli in the Anteromedial Visual Area

**Authors**: Diego Cerretti, Beatrice Citterio, Giovanni De Muri, Mattia Martino, Sandro Mikautadze

# Research Question

*Can we develop mathematical models to predict individual neurons'
spike counts in response to static grating inputs in the anteromedial visual area (VISam) of a mouse brain?*

More mathematically, given orientation ($x_1$), spatial frequency ($x_2$), and phase ($x_3$) of a visual input, can we find a function $f(x_1,x_2,x_3)$ to predict the spike count for each neuron?

# TL;DR

We select VISam neurons responsive to static gratings, based on variance, range, and modularity criteria. We employ multi-layer perceptrons (MLPs) and linear regression models with various input features (linear, quadratic, sinusoidal, combined) to predict each neuron's spike count from the static gratings' inputs.

Our results show that MLPs perform poorly, while linear models with quadratic features best capture the relationship between stimulus features and neural responses. Phase has low statistical relevance, but orientation and spatial frequency are good predictors. Overall, responsive VISam neurons tend to exhibit quadratic responses to orientation and low spatial frequencies, aligning with previous findings in the field.

Our study highlights the potential of mathematical modeling to unravel the encoding principles of sensory neurons.

# Repo Structure

- `data` folder contains the cleaned datasets used for the analysis.
- `utils` folder contains various auxiliary functions used in the regressions.
- `data_analysis.ipynb` contains the exploratory part of the work.
- `neurons_range_selection.ipynb` and `neurons_variance_selection.ipynb` contain the regression models for modulated neurons selected based on range and variance, respectively. 
- `report.pdf` contains the report of the project.