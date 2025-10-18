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

<p align="center">
<img width="500" height="566" alt="image" src="https://github.com/user-attachments/assets/f1f8c4be-c9f1-46cf-b0f5-83a6bb596bd2" />
</p>

```python
import gba
model = gba.read_ods_model(name="./models/A")
model.find_initial_solution()
model.find_optimum_by_condition()
model.plot(x="x_C", y="mu", title="Growth rate", logx=True)
model.export_optimization_data()
```

<p align="center">
<img width="550" height="1198" alt="image" src="https://github.com/user-attachments/assets/88b91aa3-b7d4-49fc-8bb1-c46762c27014" />
</p>

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

> [!WARNING]
> Module not deployed on PyPI yet

The easiest way to install <strong>gbapy</strong> is from <a href="https://pypi.org/project/gba/" target="_blank">PyPI</a>:
 
```
pip install gba
```

> [!IMPORTANT]
<a href="https://github.com/charlesrocabert/gbacpp" target="_blank">gbacpp</a> software is required to run optimization tasks.

### 1.1) Supported platforms <a name="supported_platforms"></a>
<strong>gbapy</strong> software has been primilary developed for Unix/Linux and macOS systems.

### 1.2) Dependencies <a name="dependencies"></a>

#### • Software
* <a href="https://github.com/charlesrocabert/gbacpp" target="_blank">gbacpp</a> is required to run optimization tasks.

#### • Licensed Python modules
* The Python API of <a href="https://www.gurobi.com/" target="_blank">GUROBI optimizer</a> must be installed and requires a user license (<a href="https://www.gurobi.com/academia/academic-program-and-licenses/" target="_blank">free for academics</a>)

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

Copyright © 2024-2025 @charlesrocabert, Furkan Mert.

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
