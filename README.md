# ARLIn - Adversarial Reinforcement Learning Model Interrogation

ARLIn aims to extract key information from trained reinforcement learning models that can
be used to create more effective adversarial attacks. This information can be related to
the training environment, the architecture, the algorithms used, or the policy itself.

MSR: ECRP
PI: Alex Tapley

# ARLIn Toolkit - Adversarial Attack Timing Identification for Maximum Effectiveness

Knowing _when_ to perform an adversarial attack can be crucial to the outcome of
said attack. If an attack is performed at the wrong time, it may be easily detectable by
the victim or an overseeing party, or the overall attack may be less effective and result
in an outcome different than desired.

This work analyzes the SAMDP of a victim model to identify the best time to perform an
attack such that the attack will be less detectable and more effective. Utilizing a 
dimensionality reduction technique, we map out a policy's action decision boundaries.
The embeddings can be clustered together to create separate groups which make up the 
policy's SAMDP. Each group is then assigned an entropy score determined by the number of
decision boundaries within the group. Given a state, we can detect the nearby decision
boundaries to identify actions that the policy believes are incorrect but are still
reasonable given the current state. We can then use the policy's SAMDP to find paths that
route the agent from its current state to a state with a higher entropy or a state with a
high likelihood for early termination.


## Installation

#### Step 1: Clone the repository.
```shell
git clone https://gitlab.mitre.org/ecrp/arlin.git
cd arlin/
```

#### Step 2: Create a Conda environment
```
conda create env -f conda-env.yaml
conda activate arlin
```

#### Step 3: Install dependencies via Poetry
```
poetry install
```