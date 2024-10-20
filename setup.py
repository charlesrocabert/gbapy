#!/usr/bin/env python3
# coding: utf-8

#***********************************************************************
# gbapy
# -----
# Growth balance analysis in Python.
# 
# Copyright © 2023-2024 Charles Rocabert, Furkan Mert
# Web: https://github.com/charlesrocabert/gbapy
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
#***********************************************************************


"""gbapy (growth balance analysis in Python) Python Package.

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
	name                          = "gbapy",
	version                       = "0.01",
	license                       = "GNU General Public License v3 (GPLv3)",
	description                   = "gbapy (growth balance analysis in Python) Python Package",
	long_description              = long_description,
	long_description_content_type = "text/markdown",
	url                           = "https://github.com/charlesrocabert/gbapy",
	author                        = "Charles Rocabert, Furkan Mert",
	author_email                  = "",
	classifiers = [
		"Development Status :: 4 - Beta",
		"Programming Language :: Python :: 3",
		"License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
		"Operating System :: OS Independent",
		"Intended Audience :: Science/Research",
		"Topic :: Scientific/Engineering :: Mathematics",
	],
	keywords         = "constraint-based-modeling growth-balance-analysis systems-biology metabolic-network resource-allocation cellular-economics kinetic-modeling first-prnciples simulation evolutionary-algorithms predictive-evolution",
	packages         = find_packages(exclude=["contrib", "docs", "tests"]),
	package_data= {
        'gbapy.data': [ '**/*.csv', ]
    },
	python_requires  = ">=3",
	install_requires = ["argparse", "dill", "numpy", "pandas", "gurobipy", "matplotlib"],
	project_urls     = {
	"Source": "https://github.com/charlesrocabert/gbapy"
	},
)

