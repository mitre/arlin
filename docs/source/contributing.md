# Contributing
Right now, submission of [issues](https://github.com/mitre/arlin/issues) and
[pull requests](https://github.com/mitre/arlin/pulls) through GitHub is the best way to
suggest changes, and those will have to be applied and merged on the internal MITRE
GitLab and mirrored to GitHub. This process may change in the future.

**Contributing Guidelines**:
  1. Nobody can commit directly to `main`.
  2. You must create a merge request in order to publish changes to `main`.
  3. Before merging, all stages of the GitLab CI pipeline must pass. This includes
  linting with [`flake8`](https://flake8.pycqa.org/en/latest/), code-formatting with
  [`black`](https://github.com/psf/black), passing the Python unit tests, and creating
  the documentation.
  4. Once all pipeline stages have passed, then the branch can be merged into main.
  5. These pipeline stages can be tested locally to ensure that they are passed on the
  remote side (explained in [Using Pre-commit](#using-pre-commit))



## Docstrings

Do your best to make sure that all docstrings adhere to the following Google format found
in `xrl_dataset.py`
[here](https://github.com/mitre/arlin/-/blob/main/arlin/dataset/xrl_dataset.py). The
reasoning can be found
[here](https://www.sphinx-doc.org/en/main/usage/extensions/napoleon.html). We're using
double quotes for docstrings and strings per the formatting requirements of
[`black`](https://github.com/psf/black).

If you are editing a file and see that a docstring doesn't adhere to the Python3 Google
Style Guide exemplified in `xrl_dataset.py`
[here](https://github.com/mitre/arlin/-/blob/main/arlin/dataset/xrl_dataset.py), please
be a good steward and fix it so that it does.

## Issue & Merge Request Creation
Create or assign an issue to yourself at the
[Issue Board Page](https://github.com/mitre/arlin/-/boards), add the appropriate labels,
and move the issue to “in progress”.

Create a merge request based on that issue using the "Create Merge Request" button on the
issue page.

## Installation

*Note: ARLIN has only been tested on Ubuntu 18.04 and Python 3.9.*

1. **Clone the repository**

    ```bash
    git clone https://gitlab.mitre.org/advxai/arlin.git
    ```

2. **Install poetry**

    ```bash
    curl -sSL https://install.python-poetry.org | python3 -
    ```

    *Note: Don't forget to add `poetry` to your path.*
    ```bash
    export PATH="$HOME/.local/bin:$PATH"
    ```

3. **Install required packages**

    ```bash
    cd arlin
    poetry shell
    poetry install
    ```

    *Note: To re-enter the environment after this step, run `poetry shell`.*

## Using Pre-commit (**Highly Recommended**)

If you'd like, you can install [pre-commit](https://pre-commit.com/) to run linting and
code-formatting before you are able to commit. This will ensure that you pass this
portion of the remote pipelines when you push to your merge request.

```shell
pre-commit install
```

Now, every time you try to commit, your code that you have staged will be linted by
`flake8` and auto-formatted by `black`. If the linting doesn’t pass pre-commit, it will
tell you, and you’ll have to make those changes before committing those files.
If `black` autoformats your code during pre-commit, you can view those changes and then
you’ll have to stage them. Then you can commit and push.

Pre-commit can also be run manually on all files without having to commit.

```shell
pre-commit run --all-files
```

## Running Unit Tests

There are also unit tests that need to be passed, and to make sure you are passing those
locally (before pushing to your remote branch and running the pipeline) you can run the
following command in the root directory:

```shell
poetry run pytest
```

This will search for all `test_*.py` files and run the tests held in those files.

## Adding Packages & Modules

If you add a package or module, make sure to create a unit test for it and preferably all
classes and functions within it. When creating a package (directory with Python modules),
add a `_tests` directory that mirrors the package directory. See the example below:

```
arlin
├── dataset
│   ├── analysis
│   │── __init__.py
│   │── _tests
│   │   │── __init__.py
│   │   │── test_cluster_analysis.py
│   │   │── test_latent_analysis.py
│   │── cluster_analysis.py
│   │── latent_analysis.py
```

In the `analysis` package, there are two modules, `cluster_analysis` and
`latent_analysis`. These are mirrored in the `_tests` directory as
`test_cluster_analysis.py` and `test_latent_analysis.py`. And inside each of those, there
is a mirrored `Test` class for every class in each module:

### cluster_analysis.py
```python
class ClusterAnalyzer():
```

### test_cluster_analysis.py
```python
class TestClusterAnalyzer(unittest.TestCase):
```

And *usually*, there is a test for every testable function in that class. Try to adhere
to this standard when creating new packages, modules, classes, and functions.
