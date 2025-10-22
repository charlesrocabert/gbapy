<p align="center">
  <img src="https://github.com/user-attachments/assets/e801eb01-4108-4fe0-a5ef-763002dd583f" width=250 />

</p>
<h3 align="center">Growth Balance Analysis for Python</h3>

<p align="center">
<br />
<a href="https://badge.fury.io/py/gba"><img src="https://badge.fury.io/py/gba.svg" alt="PyPI version"></a>&nbsp;
<a href="https://github.com/charlesrocabert/gbapy/actions"><img src="https://github.com/charlesrocabert/gbapy/workflows/Upload Python Package/badge.svg" /></a>&nbsp;
<a href="https://github.com/charlesrocabert/gbapy/LICENSE.html"><img src="https://img.shields.io/badge/License-GPLv3-blue.svg" /></a>
</p>

<p align="center">
  <a href="https://www.cs.hhu.de/en/research-groups/computational-cell-biology" target="_blank"><img src="https://github.com/user-attachments/assets/4e4b3b79-0d6a-4328-9c3f-3497401887e4" width=150 /></a>
  &nbsp;&nbsp;&nbsp;&nbsp;
  <a href="https://www.hhu.de/en/" target="_blank"><img src="https://github.com/user-attachments/assets/7db5c8f7-e37a-415f-88c3-1b06a49e1f28" width=150 /></a>
</p>

-----------------

```
pip install gba
```

<p align="justify">
<strong>gbapy</strong> is a Python package that provides tools for building and analyzing <strong>self-replicating cell (SRC)</strong> models based on the <strong>growth balance analysis (GBA)</strong> mathematical formalism (<a href="https://doi.org/10.1371/journal.pcbi.1011156" target="_blank">Dourado et al. 2023</a>).
</p>

### Features

<p align="justify">
The module offers two core components:
  
- :wrench: A <strong>builder class</strong>, to construct SRC models of any size from first principles,
- :chart_with_upwards_trend: A <strong>model class</strong>, to manipulate and optimize models once they are built.

</p>

# Typical workflow

<p align="center">
<img width="400" alt="image" src="https://github.com/user-attachments/assets/710fecd4-0381-41a2-ae6e-d952eb8c40ad" />
</p>

```python
import gba

builder = gba.Builder(name="toy")

### Add model information (ODS sheet 'Info')
builder.add_info(category="General", key="Name", content="toy")
builder.add_info(category="General", key="Description", content="Toy model")

### Create and add proteins (one protein per enzyme per reaction)
### - Masses in Da.
p1 = gba.Protein(id="p1", mass=1000000.0)
p2 = gba.Protein(id="p2", mass=1000000.0)
builder.add_proteins([p1, p2])

### Create and add metabolites:
### - External and internal glucose
### - One generic Protein product
### - Masses in Da.
x_G     = gba.Metabolite(id="x_G", species_location=gba.SpeciesLocation.EXTERNAL, mass=180.0)
G       = gba.Metabolite(id="G", species_location=gba.SpeciesLocation.INTERNAL, mass=180.0)
Protein = gba.Metabolite(id="Protein", species_location=gba.SpeciesLocation.INTERNAL,mass=180.0)
builder.add_metabolites([x_G, G, Protein])

### Create and add transporter to import glucose:
### - Enzyme is composed of one protein p1
### - Reaction is irreversible
### - kcat values in 1/h
### - KM values in g/L
rxn1 = gba.Reaction(id="rxn1", lb=0.0, ub=1000.0,
                    reaction_type=gba.ReactionType.TRANSPORT,
                    metabolites={"x_G":-1.0, "G": 1.0},
                    proteins={"p1": 1.0})
rxn1.add_kcat_value(direction=gba.ReactionDirection.FORWARD, kcat_value=45000.0)
rxn1.add_km_value(metabolite_id="x_G", km_value=0.001)
rxn1.complete(kcat_value=0.0, km_value=0.0)
builder.add_reaction(rxn1)

### Create and add ribosome reaction to produce proteins:
### - Enzyme is composed of one protein p2
### - Reaction is irreversible
ribosome = gba.Reaction(id="Ribosome", lb=0.0, ub=1000.0,
                        reaction_type=gba.ReactionType.METABOLIC,
                        metabolites={"G":-1.0, "Protein": 1.0},
                        proteins={"p2": 1.0})
ribosome.add_kcat_value(direction=gba.ReactionDirection.FORWARD, kcat_value=45000.0)
ribosome.add_km_value(metabolite_id="G", km_value=0.001)
ribosome.complete(kcat_value=0.0, km_value=0.0)
builder.add_reaction(ribosome)

### Convert the model to GBA formalism (cf. Dourado et al. 2023)
builder.build_GBA_model()

### Set cell's total density (g/L)
builder.set_rho(340.0)

### Create external conditions (in g/L)
x_G_conc = 100.0
for i in range(25):
    builder.add_condition(condition_id=str(i+1), metabolites={"x_G": x_G_conc})
    x_G_conc *= 2/3

### Save the model to an ODS file
builder.write_to_ods()
```

<p align="center">
<img width="500" alt="image" src="https://github.com/user-attachments/assets/7d0ec598-2fb2-497f-8d89-f15e27e24d43" />
</p>

```python
### Load the ODS model
model = gba.read_ods_model(name="toy")

### Find a valid initial solution
model.find_initial_solution()

### Optimize the model for all conditions
model.find_optimum_by_condition()

### Make a plot
model.plot(x="x_G", y="mu", title="Growth rate", logx=True)

### Export optimization data in CSV
model.export_optimization_data()
```

<p align="center">
<img width="550" alt="image" src="https://github.com/user-attachments/assets/16b7eb56-81d8-429c-b7f8-9aa4cac48de5" />
</p>

### Reference files

- 🔗 <a href="https://github.com/charlesrocabert/gbapy/blob/main/tutorials/my_toy_model.ipynb" target="_blank">Toy model tutorial</a>,
- 🔗 <a href="https://github.com/charlesrocabert/gbapy/blob/main/tutorials/toy.ods" target="_blank">ODS file</a>,
- 🔗 <a href="https://github.com/charlesrocabert/gbapy/blob/main/tutorials/toy_optimization_data.csv" target="_blank">CSV optimization data</a>.

# Table of contents

- [1) Installation](#installation)
  - [1.1) Supported platforms](#supported_platforms)
  - [1.2) Dependencies](#dependencies)
  - [1.3) Manual installation](#manual_installation)
- [2) Tutorials](#tutorials)
- [3) Documentation](#documentation)
- [4) Contributing](#contributing)
- [5) Copyright](#copyright)
- [6) License](#license)

# 1) Installation <a name="installation"></a>

> [!CAUTION]
> Module not deployed on PyPI yet

The easiest way to install <strong>gbapy</strong> is from <a href="https://pypi.org/project/gba/" target="_blank">PyPI</a>:
 
```
pip install gba
```

> [!IMPORTANT]
<a href="https://github.com/charlesrocabert/gbacpp" target="_blank">gbacpp</a> software is required to run optimization tasks.

### 1.1) Supported platforms <a name="supported_platforms"></a>
<strong>gbapy</strong> has been primilary developed for Unix/Linux and macOS systems.

### 1.2) Dependencies <a name="dependencies"></a>

#### • Software
* <a href="https://github.com/charlesrocabert/gbacpp" target="_blank">gbacpp</a> is required to run optimization tasks.

#### • Licensed Python modules
* The Python API of <a href="https://www.gurobi.com/" target="_blank">GUROBI optimizer</a> must be installed and requires a user license (<a href="https://www.gurobi.com/academia/academic-program-and-licenses/" target="_blank">free for academics</a>).

#### • Other Python modules
* <a href="https://numpy.org/" target="_blank">NumPy</a>
* <a href="https://pandas.pydata.org/" target="_blank">pandas</a>
* <a href="https://ipython.org/" target="_blank">IPython</a>
* <a href="https://plotly.com/" target="_blank">plotly</a>
* <a href="https://opencobra.github.io/cobrapy/" target="_blank">cobrapy</a>
* <a href="https://github.com/cgohlke/molmass" target="_blank">molmass</a>
* <a href="https://biopython.org/" target="_blank">Biopython</a>

### 1.3) Manual installation <a name="manual_installation"></a>

If you want to install <strong>gbapy</strong> manually, download the <a href="https://github.com/charlesrocabert/gbapy/releases/latest">latest release</a>, and save it to a directory of your choice. Open a terminal and use the <code>cd</code> command to navigate to this directory. Then follow the steps below to compile and build the executables.

```
sh install.sh
```

> [!TIP]
> You can later uninstall the module using <code>sh uninstall.sh</code>.

# 2) Tutorials <a name="tutorials"></a>

<p align="center" style="font-size: 2.5em;">
Tutorials coming soon ...
</p>

# 3) Documentation <a name="documentation"></a>

<p align="center" style="font-size: 2.5em;">
Documentation coming soon ...
</p>

# 4) Contributing <a name="contributing"></a>

If you wish to contribute, do not hesitate to reach <a href="mailto:charles DOT rocabert AT hhu DOT de">the developer</a>.

# 5) Copyright <a name="copyright"></a>

Copyright © 2024-2025 Charles Rocabert, Furkan Mert.

# 6) License <a name="license"></a>

<p align="justify">
This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
</p>

<p align="justify">
This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
</p>

<p align="justify">
You should have received a copy of the GNU General Public License along with this program. If not, see http://www.gnu.org/licenses/.
</p>
