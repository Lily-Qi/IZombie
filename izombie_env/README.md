# README for PvZ Emulator Environment

## Overview
This Python module provides an emulation environment for a simplified version of the popular game "Plants vs. Zombies". It is designed to interface with the `pvzemu` package to create a game world, manage game state, and simulate interactions between plants and zombies.

## Features
- Customizable plant and zombie types and counts.
- A grid-based world representing the lawn with distinct plant and zombie areas.
- Night scene emulation with no spawning of additional zombies.
- A reinforcement learning environment with state and reward system.

## Installation

Before you can run this script, you need to have Python installed on your system. Python 3.6 or higher is recommended. You also need to install the `pvzemu` package, which this script depends on, as well as `numpy`.

```bash
pip install numpy
pip install pvzemu # This might be a hypothetical package for the purpose of this README.
```

## Usage

To use this emulator, you need to import the `IZenv` class from this module and create an instance of it. Here's a simple example:

```python
# Initialize the environment
env = IZenv()

# Reset the environment to the starting state
state = env.reset()

# Run a simulation step with a given action
action = 0  # Replace with a valid action
reward, next_state, is_done = env.step(action) # either losing or winning will set is_done to True; if lost, reward is reset to zero
```

### Actions
The environment accepts an action which corresponds to the creation of a zombie in the game world. The action space is calculated based on the number of zombie types, lanes, and the length of the zombie area.

### States
The state is a representation of the game world, including the health and types of plants and zombies, the amount of available sun, and the status of the brains (end goal for zombies).

### Rewards
The reward is calculated based on the amount of sun collected, which is a proxy for the player's performance in the game.

### End Conditions
The game ends if there is not enough sun left to spawn zombies and no zombies are left on the screen, or if the zombies reach the house (brains status is zero).

## Development

This environment is designed to be used with reinforcement learning algorithms. You can extend the `IZenv` class to add more features or to customize the environment for your specific use case.

## Contributing

Contributions to this project are welcome. Please fork the repository and submit a pull request with your changes.

## License

Specify the license under which this code is made available.

## Example code

```python
env = IZenv()
env.step(1)
```
```
[ 1.          1.          1.          1.          1.          1.
  1.          1.          1.          1.          1.          1.
  1.          1.          1.          1.          1.          1.
  1.          1.          1.          0.          2.          3.
  1.          0.          0.          1.          0.          2.
  0.          0.          0.          2.          3.          0.
  1.          0.          1.          1.          0.          0.
  0.          0.          0.54        0.          0.          0.
  0.          0.          0.          0.          0.          0.
  0.          0.          0.          0.          0.          0.
  0.          0.          0.          0.          0.          0.
  0.          0.          0.          0.          0.          0.
  0.          0.          0.          0.          0.          0.
  0.          0.          0.          0.          0.          0.
  0.         -1.         -1.         -1.         -1.          0.
 -1.         -1.         -1.         -1.         -1.         -1.
 -1.         -1.         -1.         -1.         -1.         -1.
 -1.         -1.         -1.         -1.         -1.         -1.
 -1.         -1.         -1.         -1.         -1.         -1.
 -1.         -1.         -1.         -1.         -1.         -1.
 -1.         -1.         -1.         -1.         -1.         -1.
 -1.         -1.         -1.         -1.          0.05128205  1.
  1.          1.          1.          1.        ]
  ```