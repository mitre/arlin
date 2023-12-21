# Overview
The ARLIN (Assured Reinforcement Learning Model Interrogation) Toolkit is a research
library written in Python that provides explainability outputs and vulnerability
detection for Deep Reinforcement Learning (DRL) models, specifically designed to increase
model assurance and identify potential points of failure within a trained model.
ARLIN utilizes [matplotlib](https://matplotlib.org/stable/) and
[networkx](https://networkx.org/documentation/stable/) to visualize
a trained RL model's decision making process and provide meaningful vulnerability
identification and analysis to researchers. The modular library is structured to easily
support custom architecture, algorithm, framework, and analytics modifications and
provides a well-documented and tested API for XRL research development and model
assurance.

Key functionalities of the ARLIN library include:
- Creation of an XRL dataset with user-defined datapoints from a trained policy with the
ability to support custom data, algorithms, and model architectures.
- Dimensionality reduction and embedding generation of a trained model's latent space.
- Unsupervised clustering of policy latent space outputs based on policy transition data
and available XRL attributes.
- Analysis and visualization of policy latent space embeddings and clusters.
- Semi-aggregated Markov decision process (SAMDP) generation and policy-specific path
analysis.
