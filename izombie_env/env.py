import numpy as np
import json
import math
from pvzemu import World, SceneType, ZombieType, PlantType

plant_counts = {
    PlantType.sunflower: 9,
    PlantType.pea_shooter: 6,
    PlantType.squash: 3,
    PlantType.snow_pea: 2
}

zombie_deck = [[ZombieType.zombie, 50], [ZombieType.buckethead, 125], [ZombieType.football, 175]]

N_LANES = 5 # Height
LANE_LENGTH = 9 # Width
P_LANE_LENGTH = 4 # Plant area width
N_ZOMBIE_TYPE = 3
Z_LANE_LENGTH = LANE_LENGTH - P_LANE_LENGTH

class IZenv:
    def __init__(self):
        self.action_no = N_ZOMBIE_TYPE * N_LANES * Z_LANE_LENGTH + 1
        self.reset()
    
    def reset(self):
        self.world = World(SceneType.night)
        self.world.scene.stop_spawn = True
        self.world.scene.is_iz = True
        self.world.scene.set_sun(150)
        plant_list = [plant for plant, count in plant_counts.items() for _ in range(count)]
        np.random.shuffle(plant_list)
        for index, plant in enumerate(plant_list):
            x = index // P_LANE_LENGTH  # Row index
            y = index % P_LANE_LENGTH   # Column index
            self.world.plant_factory.create(plant, x, y)
    
    def step(self, action):
        self._take_action(action)
        for _ in range(50):
            self.world.update()
        reward = self._get_reward()
        return reward
    
    def _take_action(self, action):
        if action > 0:
            action -= 1
            z_idx = action//(N_LANES * Z_LANE_LENGTH)
            action_area = action%(N_LANES * Z_LANE_LENGTH)
            row = action_area//N_LANES
            col = action_area % N_LANES + 4
            sun = self._get_sun()
            # print(sun)
            self.world.zombie_factory.create(zombie_deck[z_idx][0], row, col)
            sun -= zombie_deck[z_idx][1]
            # print(type(sun))
            # print(zombie_deck[z_idx][1])
            self.world.scene.set_sun(sun) 

    def _zombie_x_to_col(self, x):
        col = (x-10)/80-1
        return math.ceil(col)


    def _get_sun(self):
        data = json.loads(self.world.get_json())
        return data['sun']['sun']
    
    def _get_reward(self):
        sun = self._get_sun()
        return sun//25

    
