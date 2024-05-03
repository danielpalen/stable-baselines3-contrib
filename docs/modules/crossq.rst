.. _crossq:

.. automodule:: sb3_contrib.crossq


CrossQ
======

Implementation of CrossQ proposed in:
`Bhatt A.*, Palenicek D.* et al. 2022. Batch Normalization in Deep Reinforcement Learning for Greater Sample Efficiency and Simplicity. ICLR 2024.`

CrossQ is a simple and efficient algorithm that uses batch normalization to improve the sample efficiency of off-policy deep reinforcement learning algorithms.
It is based on the idea of carefully introducing batch normalization layers in the critic network and dropping target networks.
This yield a simpler and more sample-efficient algorithm without requiring high update-to-data ratios.

.. rubric:: Available Policies

.. autosummary::
    :nosignatures:

    MlpPolicy


Notes
-----

- Original paper: https://openreview.net/pdf?id=PczQtTsTIX
- Original Implementation: https://github.com/adityab/CrossQ


Can I use?
----------

-  Recurrent policies: ❌
-  Multi processing: ✔️
-  Gym spaces:


============= ====== ===========
Space         Action Observation
============= ====== ===========
Discrete      ❌      ✔️
Box           ✔️      ✔️
MultiDiscrete ❌      ✔️
MultiBinary   ❌      ✔️
Dict          ❌      ✔️
============= ====== ===========


Example
-------

.. code-block:: python

  import gymnasium as gym
  import numpy as np

  from sb3_contrib import CrossQ

  env = gym.make("Pendulum-v1", render_mode="human")

  model = CrossQ("MlpPolicy", env)
  model.learn(total_timesteps=10_000, log_interval=4)
  model.save("crossq_pendulum")


Results
-------



Comments
--------

This implementation is based on SB3 SAC implementation.


Parameters
----------

.. autoclass:: CrossQ
  :members:
  :inherited-members:

.. _crossq_policies:

CrossQ Policies
---------------

.. autoclass:: MlpPolicy
  :members:
  :inherited-members:

.. autoclass:: sb3_contrib.crossq.policies.CrossQPolicy
  :members:
  :noindex:

