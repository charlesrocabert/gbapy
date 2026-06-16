<p align="center">
  <h1 align="center">Self-replicating cell (SRC) model format tutorial</h1>
</p>

<p align="center">
  <img src="https://github.com/user-attachments/assets/cb2ea68d-7cb8-4094-8e69-d0c84af8bbbf" width=200 />
</p>

-----------------

Self-replicating cell (SRC) models must comply to a standard format. Please refer to <a href="https://doi.org/10.1371/journal.pcbi.1011156" target="_blank">Dourado et al. (2023)</a> and the tutorial available on <a href="https://cellgrowthsim.com/" target="_blank">https://cellgrowthsim.com/</a> for other detailed sources.

Two formats are used to distribute SRC models: the OpenDocument Spreadsheet format ODS and the text format CSV. Data organization is strictly identical in both formats. ODS models are prefered for diffusion.

# Table of contents
- [1) Model file organization](#organization)
- [2) Sheet content](#content)
  - [2.1) Model information (<code>Info</code>)](#info)
  - [2.2) Mass fraction matrix (<code>M</code>)](#M)
  - [2.3) Forward and backward turnover rates vectors (<code>kcat</code>)](#kcat)
  - [2.4) Michaelis constants matrix (<code>K</code>)](#K)
  - [2.5) Cell's total density (<code>rho</code>)](#rho)
  - [2.6) External conditions matrix (<code>conditions</code>)](#conditions)
  - [2.7) Initial and optimal solutions (<code>q</code>)](#q)
  - [2.8) Activation constants matrix (<code>KA</code>)](#KA)
  - [2.9) Inhibition constants matrix (<code>KI</code>)](#KI)
  - [2.10) Enzyme to protein mass concentration mapping (<code>protein_contributions</code>)](#protein_contributions)

# 1) Model file organization <a name="organization"></a>

The SRC model is organized as a set of sheets inside a single ODS file. They usually contain GBA variables (such as matrices, or vectors), but also additional variables and information.

      └── ODS file
           ├── Info
           ├── M
           ├── kcat
           ├── K
           ├── rho
           ├── conditions
           ├── KA
           ├── KI
           └── protein_contributions

Some sheets are mandatory to set a minimal SRC model (<img src="https://img.shields.io/badge/mandatory-red" />).
These are:
- `M`
- `kcat`
- `K`
- `rho`
- `conditions`
- `q`

Other sheets are optional (<img src="https://img.shields.io/badge/optional-grey" />).

> [!NOTE]  
> CSV files replicate sheets' organization, with `;` separator, and `\n` line breaks.

# 2) Sheet content <a name="content"></a>

### 2.1) Model information (<code>Info</code>) <img src="https://img.shields.io/badge/optional-grey" /> <a name="info"></a>

The optional sheet `Infos` contains various information about the model (units, file description, ...).

The content is free but must follow a hierarchical structure:

      └── Category
           ├── Key 1: Descriptor 1
           ├── Key 2: Descriptor 2
           └── ...

Usually, categories are "General" (model name, short description, ...), "Units", "Sheets", etc...

For example:

      └── General
           ├── Name: A
           └── Description: Simplest model with two reactions
      └── Units
           ├── KM: g/L
           ├── kcat: 1/h ([mass of products]/[mass of protein]/h)
           └── rho: g/L
      └── Sheets
           ├── M: Mass fraction matrix
           ├── K: Forward Michaelis constant matrix
           ├── kcat: Turnover numbers
           ├── rho: Total density
           └── conditions: Value of external concentrations at different growth conditions

### 2.2) Mass fraction matrix $\mathbf{M}$ (<code>M</code>) <img src="https://img.shields.io/badge/mandatory-red" /> <a name="M"></a>

The sheet `M` contains the mass fraction matrix $\mathbf{M}$, which is the pendant of the stoichiometric matrix in normalized mass units (see <a href="https://doi.org/10.1371/journal.pcbi.1011156" target="_blank">Dourado et al., 2023</a>). This sheet is mandatory to have minimal kinetics:
- Metabolites are in row, reactions in columns.
- The last row corresponds to total protein amount (`Protein`).
- The last column corresponds to the ribosome reaction (`Ribosome`), producing the total protein amount.
- All metabolites starting with `x_` are external metabolites with constant concentration.

> [!WARNING]  
> Stoichiometric coefficients must be converted following GBA formalism (see the <a href="https://github.com/charlesrocabert/gbacpp/blob/main/tutorials/units_conversion_tutorial.ipynb" target="_blank">units conversion tutorial</a>).

For example, the model below has three reactions and four metabolites (three internal, one external):

|             | **rxn1** | **rnx2** | **Ribosome** |
|:-----------:|:--------:|:--------:|:------------:|
|   **x_G**   |    -1    |     0    |       0      |
|    **G**    |     1    |    -1    |       0      |
|    **AA**   |     0    |     1    |      -1      |
| **Protein** |     0    |     0    |       1      |

### 2.3) Forward and backward $k_\text{cat}$ vectors (<code>kcat</code>) <img src="https://img.shields.io/badge/mandatory-red" /> <a name="kcat"></a>

The sheet `kcat` contains the vectors of forward (`kcat_f`) and backward (`kcat_b`) turnover rates $k_\text{cat}$ (usually, in h<sup>-1</sup>). This sheet is mandatory to have minimal kinetics:
- Reactions are in column.
- `kcat_f` and `kcat_b` vectors are in row.
- For forward irreversible reactions, backward values will be zero.

> [!WARNING]  
> $k_\text{cat}$ values must be converted following GBA formalism (see the <a href="https://github.com/charlesrocabert/gbacpp/blob/main/tutorials/units_conversion_tutorial.ipynb" target="_blank">units conversion tutorial</a>).

For example, the model below has three irreversible reactions (`kcat_b = 0`):

|            | **rxn1** | **rnx2** | **Ribosome** |
|:----------:|:--------:|:--------:|:------------:|
| **kcat_f** |    150   |    50    |     4.55     |
| **kcat_b** |     0    |     0    |       0      |

### 2.4) Michaelis constants matrix $\mathbf{K}$ (<code>K</code>) <img src="https://img.shields.io/badge/mandatory-red" /> <a name="K"></a>

The sheet `K` contains the matrix of Michaelis constants $\mathbf{K}$ (usually, in g.L<sup>-1</sup>). This sheet is mandatory to have minimal kinetics:
- Metabolites are in row, reactions in columns (as in the matrix $\mathbf{M}$).
- The matrix maps Michaelis constants from reactions to substrates and products, therefore including forward and backward $K_\text{M}$ values.

> [!WARNING]
> $K_\text{M}$ values must be converted following GBA formalism (see the <a href="https://github.com/charlesrocabert/gbacpp/blob/main/tutorials/units_conversion_tutorial.ipynb" target="_blank">units conversion tutorial</a>).

For example, the model below has three irreversible reactions and four metabolites (three internal, one external):

|             | **rxn1** | **rnx2** | **Ribosome** |
|:-----------:|:--------:|:--------:|:------------:|
|   **x_G**   |    10    |     0    |       0      |
|    **G**    |     0    |    10    |       0      |
|    **AA**   |     0    |     0    |      8.3     |
| **Protein** |     0    |     0    |       0      |

### 2.5) Cell's total density (<code>rho</code>) <img src="https://img.shields.io/badge/mandatory-red" /> <a name="rho"></a>

The sheet `rho` contains the cell's total density $\rho$ in g/L. This sheet is mandatory to have minimal kinetics.
For example, the following model has a total density $\rho = 340 g/L$ (typical of <em>E. coli</em> dry mass):

|         | **(g/L)** |
|:-------:|:---------:|
| **rho** |  340      |

### 2.6) External conditions matrix (<code>conditions</code>) <img src="https://img.shields.io/badge/mandatory-red" /> <a name="conditions"></a>

The sheet `conditions` contains the list of external conditions. This sheet is mandatory to have minimal kinetics. Each condition contains:
- Condition identifiers in column (usually numbered from 1 to N),
- External metabolite concentrations in row (usually, in g.L<sup>-1</sup>).

For example, the following model has 25 conditions with a glucose gradient. Here are the first 5 conditions:

|            | **1** | **2** | **3** | **4** | **5** |
|:----------:|:-----:|:-----:|:-----:|:-----:|:-----:|
|   **x_G**  |  100  | 66.67 | 44.44 | 29.63 | 19.75 |

### 2.7) Initial and optimal solutions (<code>q</code>) <img src="https://img.shields.io/badge/optional-grey" /> <a name="q"></a>

This sheet `q.csv` contains two sorts of flux fraction vectors (vector $q$ in GBA formalism):
- An initial solution vector $q_0$, calculated with linear sub-routines. This solution is generally used as a starting point to optimize the model.
- If available, optimal solutions for each external condition.

For example, the model below has the following initial solution $f_0$:

| **Reaction** | **f0** |
|:------------:|:------:|
|   **rxn1**   |   1.0  |
|   **rxn2**   |  0.97  |
| **Ribosome** | 0.93   |

### 2.8) Activation constants matrix $\mathbf{K_\text{A}}$ (<code>KA</code>) <img src="https://img.shields.io/badge/optional-grey" /> <a name="KA"></a>

The optional sheet `KA` contains activation constants $K_\text{A}$ (usually in g.L<sup>-1</sup>), where some metabolites acts as activators of one or more reactions. The structure is the same than the Michaelis constants matrix.

### 2.9) Inhibition constants matrix $\mathbf{K_\text{I}}$ (<code>KI</code>) <img src="https://img.shields.io/badge/optional-grey" /> <a name="KI"></a>

The optional sheet `KI` contains inhibition constants $K_\text{I}$ (usually in g.L<sup>-1</sup>), where some metabolites acts as inhibitors of one or more reactions. The structure is the same than the Michaelis constants matrix.

### 2.10) Enzyme to protein mass concentration mapping (<code>protein_contributions</code>) <img src="https://img.shields.io/badge/optional-grey" /> <a name="protein_contributions"></a>

The optional sheet `protein_contributions` contains a mapping linking enzyme mass concentrations (the vector $p$ in GBA formalism) and protein mass concentrations. This sheet is useful to calculate predicted proteomics from estimated enzymatic concentrations (vector $p$ in GBA formalism).

> [!WARNING]  
> This sheet can only be obtained with biological knowledge of the constructed SRC model.

Here is an example:

| **Reaction** | **Protein**  | **Contribution**   |
|:------------:|:------------:|:------------------:|
| DADK         | protein_0651 | 1.0                |
| DADNK        | protein_0330 | 0.979131688089874  |
| DADNK        | protein_0382 | 1.0208683119101258 |
| DADNabc      | protein_0008 | 0.1358904650088565 |
| DADNabc      | protein_0009 | 0.3862941601280572 |

