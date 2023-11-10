# README for IZenv

## Overview

This is designed to simulate an environment for the mini game "I, Zombie" from (PvZ). It makes use of the `pvzemu` emulator to create a world, manage the spawning of plants and zombies, and simulate the game mechanics. The module allows the creation of a random IZombie level 1 environment for AI training or simulation purposes.

## Features

- Initialization of a PvZ world in night scene.
- Customizable plant and zombie types and counts.
- Ability to reset the world for a new simulation.
- Methods to perform actions within the world and update the game state.
- Functionality to calculate rewards based on the game state.

## Requirements

- numpy
- json
- pvzemu (a custom emulator for PvZ)

## Usage

To use this module, you need to import it and instantiate the `IZenv` class. Once instantiated, you can use its methods to interact with the PvZ environment.

### Initialization
```python
from your_module import IZenv
env = IZenv()
```

### Reset the World with New Arrangement
```python
env.reset()
```

### Perform an Action
```python
reward = env.step(action)
```

- `action` is an integer representing the specific action to be taken in the environment.
- `0` means no action
- `1-25` implements normal zombies from row0col4 to row4col8, row by row
- `26-50` implements buckethead
- `51-75` implements football

### Observe state
- `array initialization`: Arrays for types are initialized to -1 to indicate that they are empty.

### Getters

- `_get_sun`: Returns the current amount of sun available in the game.
- `_get_reward`: Calculates the reward based on the current sun amount.

## Customization

You can customize the plant counts and zombie deck by modifying the `plant_counts` and `zombie_deck` variables, respectively.

```python
plant_counts = {
    PlantType.sunflower: 9,
    PlantType.pea_shooter: 6,
    PlantType.squash: 3,
    PlantType.snow_pea: 2
}

zombie_deck = [[ZombieType.zombie, 50], [ZombieType.buckethead, 125], [ZombieType.football, 175]]
```

## Notes

- The PvZ world is initialized with the night scene by default.
- Plant placement are randomized at the beginning of each reset.
- The environment supports a fixed number of lanes and plants per lane.
- The sun value is a critical part of the game's state, influencing the spawning of zombies and calculation of rewards.

Ensure that you have the `pvzemu` emulator properly set up and configured to work with this module.

## Example code

```python
env = IZenv()
env.step(1)
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