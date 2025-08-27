#!/usr/bin/env python3
# coding: utf-8

#*******************************************************************************
# gbapy (Growth Balance Analysis for Python)
# Web: https://github.com/charlesrocabert/gbapy
# 
# MIT License
# 
# Copyright © 2024-2025 Charles Rocabert. All rights reserved.
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#*******************************************************************************


"""gbapy (Growth Balance Analysis for Python).

See:
https://github.com/charlesrocabert/gbapy
"""

from setuptools import setup, find_packages
from os import path

here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(here, 'README.md'), encoding='utf-8') as f:
	long_description = f.read()

setup(
	name                          = "gba",
	version                       = "0.2.0",
	license                       = "MIT License",
	description                   = "gbapy (Growth Balance Analysis for Python)",
	long_description              = long_description,
	long_description_content_type = "text/markdown",
	url                           = "https://github.com/charlesrocabert/gbapy",
	author                        = "Charles Rocabert",
	author_email                  = "charles.rocabert@hhu.de",
	maintainer                    = "Furkan Mert",
	classifiers = [
		"Development Status :: 4 - Beta",
		"Programming Language :: Python :: 3",
		"License :: OSI Approved :: MIT License",
		"Operating System :: OS Independent",
		"Intended Audience :: Science/Research",
		"Topic :: Scientific/Engineering :: Mathematics",
	],
	keywords     = "constraint-based-modeling growth-balance-analysis self-replicating-model systems-biology metabolic-network resource-allocation cellular-economics kinetic-modeling first-prnciples simulation evolutionary-algorithms predictive-evolution",
	packages     = find_packages(exclude=["contrib", "docs", "tests"]),
	package_data = {
        'gba.data': ['**/*.csv', '**/*.ods']
    },
	python_requires  = ">=3",
	install_requires = ["argparse", "numpy", "scipy", "pandas", "gurobipy", "molmass", "Bio", "cobra", "plotly", "openpyxl", "pyexcel_ods3", "pyexcel_xlsx"],
	project_urls     = {
	"Source": "https://github.com/charlesrocabert/gbapy"
	},
)

