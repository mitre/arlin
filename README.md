# ARLIn - Adversarial Reinforcement Learning Model Interrogation

ARLIn aims to extract key information from trained reinforcement learning models that can
be used to create more effective adversarial attacks. This information can be related to
the training environment, the architecture, the algorithms used, or the policy itself.

MSR: ECRP
PI: Alex Tapley

# Perfect Timing - Adversarial Attack Timing Identification for Maximum Effectiveness

Knowing _when_ to perform an adversarial attack can be crucial to the outcome of
said attack. If an attack is performed at the wrong time, it may be easily detectable by
the victim or an overseeing party, or the overall attack may be less effective and result
in an outcome different than desired.

This work analyzes the MDP of a victim model to identify the best time to perform an
attack given a target end goal or state. Using the models SAMDP, the decision boundaries
between states can be identified. States closest to the boundaries require minimal effort
to transition from one action to another. By performing adversarial attacks in states
on the boundaries, less manipulation is necessary in order for an attack to work which
results in a less detectable attack. We can also analyze the SAMDP to find common
ancestors between the expected end state and a target end state. If we perform an attack
when the victim is in the common ancestor state, the victim is more likely to end in the
target state without the need for constant attacks, resulting in a more effective and
less detectable attack.


## Installation

#### Step 1: Clone the repository.
```shell
git clone https://gitlab.mitre.org/arlin/perfect_timing.git
cd perfect_timing/
```

#### Step 2: Create a Conda environment
```
conda create env -f conda-env.yaml
conda activate timing
```

#### Step 3: Install dependencies via Poetry
```
poetry install
```