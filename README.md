# Surrogate Modeling for Inverse Uncertainty Quantification

This repository contains a computational statistics project focused on **surrogate modeling for inverse parameter estimation** in a time-dependent elasticity problem.

The objective is to efficiently estimate unknown physical parameters by replacing an expensive full-order model with a reduced surrogate, and to compare **frequentist** and **Bayesian** approaches for uncertainty quantification.

---

## Overview

Inverse problems in computational mechanics often require repeated evaluations of complex forward models, leading to high computational costs.

In this project:
- a **surrogate model** is constructed using an autoencoder-based architecture  
- the surrogate is used to approximate the parametric solution map  
- the inverse problem is solved using both **frequentist** and **Bayesian** methods  

This approach enables efficient parameter estimation while maintaining good accuracy.

---

## Problem Description

We consider a model of sidewalk deformation under a pedestrian load.

The system is governed by a time-dependent elasticity problem and depends on two parameters:
- $\delta$: controls the thickness of the upper layer (indicator of structural condition)  
- $m$ : pedestrian mass  

The goal is to estimate:

$\mu = (\delta, m)$

from indirect observations of the system response.

Instead of observing the full solution, only the **average vertical deformation on the top boundary** is available at a small number of time instants.

---

## Methodology

### 1. Surrogate Modeling
- The dataset consists of simulated trajectories $u_μ(t)$
- Inputs are constructed as parameter-time pairs $(\mu, t)$ 
- A neural network with an **autoencoder structure** is used to build a reduced-order model
- **Fourier features** are used to encode the time variable and improve temporal representation

### 2. Efficiency Analysis
- The full-order model (FOM) requires ~8 seconds per trajectory
- The surrogate model (ROM) enables significantly faster evaluations
- The computational speed-up is quantified by comparing execution times

### 3. Inverse Problem

#### Frequentist approach
- Parameters are estimated by minimizing the discrepancy between observed and predicted quantities
- Confidence intervals are obtained via Jacobian-based approximations

#### Bayesian approach
- The posterior distribution of the parameters is estimated using **MCMC (Metropolis algorithm)**
- This provides a full characterization of uncertainty

---

## Results

- The surrogate model accurately approximates the full-order dynamics while drastically reducing computational cost  
- The frequentist approach provides point estimates but can be unstable with limited observations  
- The Bayesian approach offers a more robust framework for uncertainty quantification, especially in low-data settings  

---
## Execution and Dependencies

This project is designed to run in **Google Colab**, where the required environment and dependencies are handled automatically.

A key dependency is the `dlroms` library, a Python package for **deep learning-based reduced order models (DL-ROMs)**.  
DL-ROMs are surrogate models that approximate expensive numerical solvers using neural networks trained on simulation data, enabling fast evaluation and efficient parameter estimation.

In this project, `dlroms` is used to:
- construct the surrogate model for the parametric solution map  
- train the neural network  
- replace the full-order model during inference 
The library is installed automatically in the first cell of the notebook:

```python
import numpy as np
import matplotlib.pyplot as plt

try:
    from dlroms import *
except:
    !pip install --no-deps git+https://github.com/NicolaRFranco/dlroms.git
    from dlroms import *

```
## Repository Structure

```text
.
├── surrogate_modeling_inverse_uq.ipynb
├── requirements.txt
└── README.md
