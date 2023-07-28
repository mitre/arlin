# ARLIN - Adversarial Reinforcement Learning Model Interrogation

ARLIn aims to extract key information from trained reinforcement learning models that can
be used to create more effective adversarial attacks. This information can be related to
the training environment, the architecture, the algorithms used, or the policy itself.


`ARLIN` is a research-oriented library written in Python aimed at enhancing adversarial
decision making via explainable RL techniques. ARLIN utilizes *matplotlib* and *networkx* 
to visualize a trained RL model's decision making process and provide meaningful 
vulnerability analysis to researchers. The modular library is structured to easily 
support different RL model algorithms and architectures, as well as analytics and 
provides a well documented and tested API for adversarial research development. You can
read more about `ARLIN` in our [white paper](TODO). Documentation is available 
[here](TODO).


```bash
>>> import arlin
      _       _______     _____     _____  ____  _____  
     / \     |_   __ \   |_   _|   |_   _||_   \|_   _| 
    / _ \      | |__) |    | |       | |    |   \ | |   
   / ___ \     |  __ /     | |   _   | |    | |\ \| |   
 _/ /   \ \_  _| |  \ \_  _| |__/ | _| |_  _| |_\   |_  
|____| |____||____| |___||________||_____||_____|\____| 

---  Explainable RL for Adversarial Decision Making ---

version (1.0.0)
MITRE 2023
```

⎈ **TOC** ⎈

- [Developer Installation](#developer-installation)
  - [Ubuntu Setup](#ubuntu-setup)
  - [Clone Repository](#clone-repository)
  - [Pyenv Install](#pyenv-install)
  - [Poetry Install](#poetry-install)
  - [Project Setup](#project-setup)
---


# Developer Installation

We (currently) use
- `pyenv` for python version management
- `poetry` to manage dependencies and packaging
- `black` for stylistic formatting of our code via `precommit` hooks
- `isort` for formatting imports via `pre-commit` hooks
- `flake8` for style enforcement via `pre-commit` hooks

## Ubuntu Setup
If using `Ubuntu`

```bash
sudo apt update
sudo apt upgrade
```

> If using MITRE's `gpu-servers` Ubuntu 18.04, you will need to select the local copy when prompted which version of `sshd_config` after update/upgrade. Then run:
>
> ```bash
> sudo apt install policykit-1
> sudo systemctl restart ssh
> ```
> then exit the container and restart by
> ```bash
> ssh container-name@gpu.mitre.org
> restart <container-name>
> ```

## Clone Repository
> Note: if you don't have your machine's ssh keys in your gitlab profile, you need to `ssh-keygen -t ed25519` (this command will tell you where it stores the key) and then copy/paste the `.pub` key to your gitlab profile/preferences/ssh keys/. Advice: don't add a passphrase.

Clone the repository:

```bash
git clone https://gitlab.mitre.org/advxai/arlin.git
```
> Note: you may need to generate ssh keys and add them to your gitlab profile

Navigate into `arlin`:

```bash
cd arlin
```

## Pyenv Install

We first need to install pre-requisites for `pyenv`:


```bash
# FOR Ubuntu
sudo apt-get install -y make build-essential libssl-dev zlib1g-dev \
libbz2-dev libreadline-dev libsqlite3-dev wget curl llvm libncurses5-dev \
libncursesw5-dev xz-utils tk-dev libffi-dev liblzma-dev python-openssl

# FOR MacOS
brew install openssl readline sqlite3 xz zlib
```

Now we can install `pyenv` with the official installer:

```bash
curl https://pyenv.run | bash
```

Load pyenv automatically by appending the following to `~/.bash_profile` if it exists, otherwise `~/.profile` (for login shells) and `~/.bashrc` (for interactive shells):

```bash
export PYENV_ROOT="$HOME/.pyenv"
command -v pyenv >/dev/null || export PATH="$PYENV_ROOT/bin:$PATH"
eval "$(pyenv init -)"
```

Now we can build a python environment with `pyenv`:

```bash
pyenv install 3.10.10
```

Now select the pyenv python with one of the following commands:
1. `pyenv global 3.10.10`: sets python as global system system_default.
2. `pyenv local 3.10.10`: sets python version as local default (do this inside `/arlin/`).
3. `pyenv shell 3.10.10`: sets python version as default for current shell session.

> Note: Python base interpreter requires some additional modules. Those are not installed with e.g. Ubuntu 18.04 as default. Hence the need to select a pyenv python version that does come prepackaged with those additional modules before poetry installation in the next step.

## Poetry Install
We will install `poetry` and then create a virtual environment useing the `poetry.lock` file in the repository.

First, install `poetry` for `Mac OS / Linux` with the official installer:

```bash
curl -sSL https://install.python-poetry.org | python3 -
```
Don't forget to add `poetry` to your path with
```bash
export PATH="$HOME/.local/bin:$PATH"
```

## Project Setup

Set up the virtual environment with poetry:

```bash
poetry config virtualenvs.in-project true # stores virtualenv in project directory
poetry env use 3.10.10 # if you want to use a different python version you can choose here; but you must have that python version installed
poetry shell
poetry install
```

To re-enter the environment after this step, run `poetry shell`