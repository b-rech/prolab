# 🧪ProLab

Processing of ocean color laboratory measurements.

## 🚀 Instalation

In the Anaconda prompt, clone the repository:

```bash
git clone git@github.com:b-rech/prolab.git
```

Then, go to the folder and create a new Conda environment with the required dependencies:

```bash
cd prolab
conda env create -f environment.yml
conda activate prolab
```

Finally, you can install the module:

```bash
pip install .
```

Alternatively, install directly from GitHub, but make sure to have all dependencies installed:

```bash
pip install git+https://github.com/your-username/prolab.git
```

## 🕸️ Using Jupyter and Spyder

To run notebooks inside the `prolab` conda environment, make sure that you have installed Jupyter:

```bash
conda activate prolab
conda install -c conda-forge jupyterlab
```

To open it, activate the environment and run the command `jupyter lab`.

If you want to run the `.py` scripts, we recommend using Spyder, that can be installed with:

```bash
conda activate prolab
conda install -c conda-forge spyder
```

Activate the environment and use the command `spyder` to open it.

## ⚙️ Key Features

The module allows the processing of:

1. Absorption by colored dissolved organic matter (CDOM)
2. Particulate absorption (in implementation)
3. ...
   
   

## 🔎Processing Details

#### CDOM absorption

The processing of CDOM absorption measurements follows the specific [IOCCG protocol](https://ioccg.org/wp-content/uploads/2019/10/cdom_abs_protocol_public_draft-19oct-2019-sm.pdf).



## 👤Credits

Developed and mantained by [Bruno Rech](https://github.com/b-rech).


