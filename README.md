<p align="center">
  <img src="https://github.com/user-attachments/assets/af2b654a-b3d7-48d7-82d9-cc3334a43a84"" width=300 />

</p>
<h3 align="center">Growth Balance Analysis for Python</h3>

<p align="center">
<br />
<a href="https://github.com/charlesrocabert/gbapy/releases/latest"><img src="https://img.shields.io/github/release/charlesrocabert/gbapy/all.svg" /></a>&nbsp;
<a href="https://badge.fury.io/py/gba"><img src="https://badge.fury.io/py/gba.svg" alt="PyPI version"></a>&nbsp;
<a href="https://github.com/charlesrocabert/gbapy/actions"><img src="https://github.com/charlesrocabert/gbapy/workflows/Upload Python Package/badge.svg" /></a>&nbsp;
<a href="https://github.com/charlesrocabert/gbapy/LICENSE.html"><img src="https://img.shields.io/badge/License-GPLv3-blue.svg" /></a>
</p>

<p align="center">
  <a href="https://www.cs.hhu.de/en/research-groups/computational-cell-biology"><img src="https://github.com/user-attachments/assets/4e4b3b79-0d6a-4328-9c3f-3497401887e4" width=200 /></a>
  &nbsp;&nbsp;&nbsp;&nbsp;
  <a href="https://www.hhu.de/en/"><img src="https://github.com/user-attachments/assets/7db5c8f7-e37a-415f-88c3-1b06a49e1f28" width=200 /></a>
</p>

-----------------

```
pip install gba
```

<p align="justify">
<strong>gbapy</strong> is a Python package that provides tools for building and analyzing <strong>self-replicating cell (SRC)</strong> models based on the <strong>growth balance analysis (GBA)</strong> mathematical formalism (<a href="https://doi.org/10.1371/journal.pcbi.1011156">Dourado et al. 2023</a>).
This approach, built exclusively on the first principles of fitness maximization, mass conservation, nonlinear reaction kinetics, and constant cell density, allows to study resource allocation in models of whole self-replicating cells (<a href="https://doi.org/10.1101/2025.06.24.661369">Dourado et al. 2025</a>).
</p>

<p align="justify">
The module offers two core components:
  
- :wrench: A <strong>builder class</strong>, to construct SRC models of any size from first principles,
- :chart_with_upwards_trend: A <strong>model class</strong>, to manipulate and optimize models once they are built.
</p>

> [!TIP]
> Start by reading the <a href="https://doi.org/10.1371/journal.pcbi.1011156">complete description of GBA formalism</a>, then follow the tutorials below to learn the required format:
> - SRC models must comply to a standardized format. Guidelines are available in the <a href="https://github.com/charlesrocabert/gbapy/blob/main/tutorials/src_model_format_tutorial.md">🔗 Toy model tutorial</a>.
> - When building a SRC model, stoichiometric coefficients, and kinetic parameters must be converted following GBA formalism. See the <a href="https://github.com/charlesrocabert/gbapy/blob/main/tutorials/units_conversion_tutorial.ipynb">🔗 Units conversion tutorial</a>.

# Table of contents

- [Get started with a typical workflow](#typical_workflow)
- [Installation](#installation)
  - [Supported platforms](#supported_platforms)
  - [Dependencies](#dependencies)
  - [Manual installation](#manual_installation)
- [Documentation](#documentation)
- [Contributing](#contributing)
- [Copyright](#copyright)
- [License](#license)

# Get started with a typical workflow <a name="typical_workflow"></a>

<p align="center">
<img width="400" alt="image" src="https://github.com/user-attachments/assets/710fecd4-0381-41a2-ae6e-d952eb8c40ad" />
</p>

<p align="justify">
In this first step, you build the structure of a simple SRC model from scratch. The <code>Builder</code> object is used to declare the main biological entities of the system, define how they interact through reactions, and specify the external conditions that will later be tested during optimization.
</p>

```python
from gba import Builder, Model, Protein, Metabolite, Reaction
from gba import SpeciesLocation, ReactionType, ReactionDirection

builder = Builder(name="toy")

### Add general information to the model (stored in the ODS sheet named 'Info')
builder.add_info(category="General", key="Name", content="toy")
builder.add_info(category="General", key="Description", content="Toy model")

### Create and add the proteins used by the enzymes in the model (one protein per enzyme):
### - Protein masses are given in Da.
p1 = Protein(id="p1", mass=1000000.0)
p2 = Protein(id="p2", mass=1000000.0)
builder.add_proteins([p1, p2])

### Create and add the metabolites used in the model:
### - x_G is external glucose
### - G is internal glucose
### - Protein is a generic protein product
### - Metabolite masses are given in Da
x_G     = Metabolite(id="x_G", species_location=SpeciesLocation.EXTERNAL, mass=180.0)
G       = Metabolite(id="G", species_location=SpeciesLocation.INTERNAL, mass=180.0)
Protein = Metabolite(id="Protein", species_location=SpeciesLocation.INTERNAL,mass=180.0)
builder.add_metabolites([x_G, G, Protein])

### Create a transport reaction that imports glucose into the cell:
### - The enzyme is composed of one protein p1
### - The reaction is irreversible
### - kcat values are given in 1/h
### - KM values are given in g/L
rxn1 = Reaction(id="rxn1", lb=0.0, ub=1000.0,
                reaction_type=ReactionType.TRANSPORT,
                metabolites={"x_G":-1.0, "G": 1.0},
                proteins={"p1": 1.0})
rxn1.add_kcat_value(direction=ReactionDirection.FORWARD, kcat_value=45000.0)
rxn1.add_km_value(metabolite_id="x_G", km_value=0.00013)
rxn1.complete(kcat_value=0.0, km_value=0.0)
builder.add_reaction(rxn1)

### Create a ribosome-like reaction that uses internal glucose to produce protein:
### - The enzyme is composed of one protein p2
### - The reaction is irreversible
ribosome = Reaction(id="Ribosome", lb=0.0, ub=1000.0,
                    reaction_type=ReactionType.METABOLIC,
                    metabolites={"G":-1.0, "Protein": 1.0},
                    proteins={"p2": 1.0})
ribosome.add_kcat_value(direction=ReactionDirection.FORWARD, kcat_value=45000.0)
ribosome.add_km_value(metabolite_id="G", km_value=0.00013)
ribosome.complete(kcat_value=0.0, km_value=0.0)
builder.add_reaction(ribosome)

### Convert the model quantities to the GBA formalism (see Dourado et al. 2023)
builder.convert(ribosome_mass_kcat=4.55, ribosome_mass_km=8.3)
builder.build_GBA_model()

### Set the total cell density in g/L (here, the dry weight density)
builder.set_rho(340.0)

### Create a series of external conditions with decreasing external glucose concentration (g/L)
x_G_conc = 1.0
for i in range(25):
    builder.add_condition(condition_id=str(i+1), metabolites={"x_G": x_G_conc})
    x_G_conc *= 2/3

### Export the model to an ODS file
builder.export_to_ods()
```

<p align="center">
<img width="500" alt="image" src="https://github.com/user-attachments/assets/7d0ec598-2fb2-497f-8d89-f15e27e24d43" />
</p>

<p align="justify">
In the second step, the exported ODS file is loaded back as a <code>Model</code> object so that numerical computations can be performed. The workflow first finds a feasible initial state, then solves the optimization problem for each glucose concentration.
</p>

```python
from gba import read_ods_model

### Load the ODS model file created in the previous step
model = read_ods_model(name="toy")

### Find an initial feasible solution before running the optimization
model.find_initial_solution()

### Compute the optimal solution for each external condition
model.find_optimum_by_condition()

### Plot the growth rate mu as a function of external glucose concentration x_G
model.plot(x="x_G", y="mu", title="Growth rate", logx=True)

### Export the optimization results to a CSV file
model.export_optimization_data()
```

<p align="center">
<img width="550" alt="image" src="https://github.com/user-attachments/assets/16b7eb56-81d8-429c-b7f8-9aa4cac48de5" />
</p>

<p align="justify">
The final figure shows the predicted growth rate as a function of external glucose concentration.
</p>

### Reference files

- 🔗 <a href="https://github.com/charlesrocabert/gbapy/blob/main/tutorials/my_toy_model.ipynb">Toy model tutorial</a>,
- 🔗 <a href="https://github.com/charlesrocabert/gbapy/blob/main/tutorials/toy.ods">ODS file</a>,
- 🔗 <a href="https://github.com/charlesrocabert/gbapy/blob/main/tutorials/toy_optimization_data.csv">CSV optimization data</a>.

# Installation <a name="installation"></a>

The easiest way to install <strong>gbapy</strong> is from <a href="https://pypi.org/project/gba/">PyPI</a>:
 
```
pip install gba
```

> [!IMPORTANT]
<a href="https://github.com/charlesrocabert/gbacpp">gbacpp</a> software is required to run optimization tasks.

## Supported platforms <a name="supported_platforms"></a>
<strong>gbapy</strong> has been primilary developed for Unix/Linux and macOS systems.

## Dependencies <a name="dependencies"></a>

### • Software
* <a href="https://github.com/charlesrocabert/gbacpp">gbacpp</a> is required to run optimization tasks.

### • Licensed Python modules
* The Python API of <a href="https://www.gurobi.com/">GUROBI optimizer</a> must be installed and requires a user license (<a href="https://www.gurobi.com/academia/academic-program-and-licenses/" >free for academics</a>).

### • Other Python modules
* <a href="https://numpy.org/">NumPy</a>
* <a href="https://pandas.pydata.org/">pandas</a>
* <a href="https://ipython.org/">IPython</a>
* <a href="https://plotly.com/">plotly</a>
* <a href="https://opencobra.github.io/cobrapy/">cobrapy</a>
* <a href="https://github.com/cgohlke/molmass">molmass</a>
* <a href="https://biopython.org/">Biopython</a>

## Manual installation <a name="manual_installation"></a>

If you want to install <strong>gbapy</strong> manually, download the <a href="https://github.com/charlesrocabert/gbapy/releases/latest">latest release</a>, and save it to a directory of your choice. Open a terminal, navigate to the <code>gbapy/</code> directory and run:

```
sh install.sh
```

> [!TIP]
> You can later uninstall the module using <code>sh uninstall.sh</code>.

# Documentation <a name="documentation"></a>

<p align="center" style="font-size: 2.5em;">
Documentation coming soon ...
</p>

# Contributing <a name="contributing"></a>

If you wish to contribute, do not hesitate to reach <a href="mailto:charles DOT rocabert AT hhu DOT de">the developer</a>.

# Copyright <a name="copyright"></a>

Copyright © 2024-2026 Charles Rocabert, Furkan Mert, Jérémie Muller-Prokob.

# License <a name="license"></a>

<p align="justify">
This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
</p>

<p align="justify">
This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
</p>

<p align="justify">
You should have received a copy of the GNU General Public License along with this program. If not, see http://www.gnu.org/licenses/.
</p>
