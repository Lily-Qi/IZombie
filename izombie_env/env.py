import numpy as np
from pvzemu import World, SceneType, ZombieType, PlantType

plant_counts = {
    PlantType.sunflower: 9,
    PlantType.pea_shooter: 6,
    PlantType.squash: 3,
    PlantType.snow_pea: 2
}

N_LANES = 5 # Height
LANE_LENGTH = 9 # Width
PLANT_LANE_LENGTH = 4 # Plant area width

class IZenv:
    def __init__(self):
        self.world = World(SceneType.night)
        self.world.scene.stop_spawn = True
        self.world.scene.is_iz = True
        self.world.scene.set_sun(150)
        plant_list = [plant for plant, count in plant_counts.items() for _ in range(count)]
        np.random.shuffle(plant_list)
        for index, plant in enumerate(plant_list):
            x = index // PLANT_LANE_LENGTH  # Row index
            y = index % PLANT_LANE_LENGTH   # Column index
            self.world.plant_factory.create(plant, x, y)
