<h1 align="center">
<img src="src/ert/gui/resources/gui/img/ert_icon.svg" width="200">
</h1>

[![Build Status](https://github.com/equinor/ert/actions/workflows/build.yml/badge.svg)](https://github.com/equinor/ert/actions/workflows/build.yml)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/ert)](https://img.shields.io/pypi/pyversions/ert)
[![Code Style](https://github.com/equinor/ert/actions/workflows/style.yml/badge.svg)](https://github.com/equinor/ert/actions/workflows/style.yml)
[![Type checking](https://github.com/equinor/ert/actions/workflows/typing.yml/badge.svg)](https://github.com/equinor/ert/actions/workflows/typing.yml)
[![codecov](https://codecov.io/gh/equinor/ert/graph/badge.svg?token=keVAcWavZ1)](https://codecov.io/gh/equinor/ert)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

ert - Ensemble based Reservoir Tool - is designed for running
ensembles of dynamical models such as reservoir models,
in order to do sensitivity analysis and data assimilation.
ert supports data assimilation using the Ensemble Smoother (ES),
Ensemble Smoother with Multiple Data Assimilation (ES-MDA) and
Iterative Ensemble Smoother (IES).

## Installation

``` sh
$ pip install ert
$ ert --help
```

or, for the latest development version:

``` sh
$ pip install git+https://github.com/equinor/ert.git@main
$ ert --help
```

For examples and help with configuration, see the [ert Documentation](https://ert.readthedocs.io/en/latest/getting_started/configuration/poly_new/guide.html#configuration-guide).

## Developing

ert was originally written in C/C++ but is now only Python.

You might first want to make sure that some system level packages are installed
before attempting setup:

```
- pip
- python include headers
- (python) venv
- (python) setuptools
- (python) wheel
```

It is left as an exercise to the reader to figure out how to install these on
their respective system.

To start developing the Python code, we suggest installing ert in editable mode
into a [virtual environment](https://docs.python.org/3/library/venv.html) to
isolate the install (substitute the appropriate way of sourcing venv for your shell):

```sh
# Create and enable a virtualenv
python3 -m venv my_virtualenv
source my_virtualenv/bin/activate

# Update build dependencies
pip install --upgrade pip wheel setuptools

# Download and install ert
git clone https://github.com/equinor/ert
cd ert
pip install --editable .
```

### Test setup

Additional development packages must be installed to run the test suite:

```sh
pip install ".[dev]"
pytest tests/
```

[Git LFS](https://git-lfs.com/) must be installed to get all the files. This is packaged as `git-lfs` on Ubuntu, Fedora or macOS Homebrew. For Equinor RGS node users, it is possible to use `git` from Red Hat Software Collections:
```sh
source /opt/rh/rh-git227/enable
```
test-data/block_storage is a submodule and must be checked out.
```sh
git submodule update --init --recursive
```

If you checked out submodules without having git lfs installed, you can force git lfs to run in all submodules with:
```sh
git submodule foreach "git lfs pull"
```


### Style requirements

There are a set of style requirements, which are gathered in the `pre-commit`
configuration, to have it automatically run on each commit do:

``` sh
$ pip install pre-commit
$ pre-commit install
```

### Trouble with setup

As a simple test of your `ert` installation, you may try to run one of the
examples, for instance:

```
cd test-data/poly_example
# for non-gui trial run
ert test_run poly.ert
# for gui trial run
ert gui poly.ert
```

### Notes

The default maximum number of open files is normally relatively low on MacOS
and some Linux distributions. This is likely to make tests crash with mysterious
error-messages. You can inspect the current limits in your shell by issuing the
command `ulimit -a`. In order to increase maximum number of open files, run
`ulimit -n 16384` (or some other large number) and put the command in your
`.profile` to make it persist.

## Example usage

### Basic ert test
To test if ert itself is working, go to `test-data/poly_example` and start ert by running `poly.ert` with `ert gui`
```
cd test-data/poly_example
ert gui poly.ert
````
This opens up the ert graphical user interface.
Finally, test ert by starting and successfully running the experiment.

### ert with a reservoir simulator
To actually get ert to work at your site you need to configure details about
your system; at the very least this means you must configure where your
reservoir simulator is installed. In addition you might want to configure e.g.
queue system in the `site-config` file, but that is not strictly necessary for
a basic test.
