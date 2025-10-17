<p align="center">
  <img src="https://github.com/user-attachments/assets/e801eb01-4108-4fe0-a5ef-763002dd583f" width=250 />

</p>
<h3 align="center">Growth Balance Analysis for Python</h3>

<p align="center">
<br />
<a href="https://github.com/charlesrocabert/gbapy/LICENSE.html"><img src="https://img.shields.io/badge/License-GPLv3-blue.svg" /></a>&nbsp;
<img src="https://img.shields.io/badge/Python-3776AB?logo=python&logoColor=fff" />
</p>

<p align="center">
  <a href="https://www.cs.hhu.de/en/research-groups/computational-cell-biology" target="_blank"><img src="https://github.com/user-attachments/assets/4e4b3b79-0d6a-4328-9c3f-3497401887e4" width=150 /></a>
  &nbsp;&nbsp;&nbsp;&nbsp;
  <a href="https://www.hhu.de/en/" target="_blank"><img src="https://github.com/user-attachments/assets/7db5c8f7-e37a-415f-88c3-1b06a49e1f28" width=150 /></a>
</p>

-----------------

<p align="center" style="font-size: 2.5em;">
Documentation coming soon ...
</p>

# Build a model

# Optimize a model

```python
import gba
model = gba.read_ods_model(name="A")
model.find_initial_solution()
model.find_optimum_by_condition()
model.plot(x="x_G", y="mu", title="Growth rate", xlabel="External glucose", ylabel="μ", logx=True)
```

## Copyright <a name="copyright"></a>
Copyright © 2024-2025 Charles Rocabert, Furkan Mert.

## License <a name="license"></a>

<p align="justify">
This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
</p>

<p align="justify">
This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
</p>

<p align="justify">
You should have received a copy of the GNU General Public License along with this program. If not, see http://www.gnu.org/licenses/.
</p>
