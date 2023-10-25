.. ARLIN documentation master file, created by
   sphinx-quickstart on Tue Jul 25 13:17:13 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to ARLIN's Documentation!
=================================

The ARLIN (Assured Reinforcement Learning Model Interrogation) Toolkit is a research
library written in Python that provides explainability outputs and vulnerability
detection for Deep Reinforcement Learning (DRL) models, specifically designed to increase
model assurance and identify potential points of failure within a trained model.
ARLIN utilizes :doc:`matplotlib <matplotlib>` and :doc:`networkx <networkx>` to visualize
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


.. note::
   This work is funded by the 2023 MITRE Independent Research and Development Program's
   Early Career Research Program.

.. toctree::
   :maxdepth: 2

   tutorials/index
   contributing


.. Indices and tables
.. ==================

.. * :ref:`genindex`
.. * :ref:`modindex`
.. * :ref:`search`
