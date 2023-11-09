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
        ob = self._get_obs()
        return reward
    
    def _get_obs(self):
        data = json.loads(self.world.get_json())
        plantHP = np.zeros(len(data['plants']), dtype=int) #Each plant HP
        plantType = np.zeros(len(data['plants']), dtype=int) #Each plant type
        zombieHP = np.zeros(len(data['zombies']), dtype=int) #Each zombie HP
        zombieType = np.zeros(len(data['zombies']), dtype=int) #Each zombie type
        sunNum = np.array([self._get_sun()]) #Total sun number
        brainStatus = np.array([1, 1, 1, 1, 1]) #Each brain status

        for i, plant in enumerate(data['plants']):
            plantHP[i] = plant['hp']
            plantType[i] = plant['type']
        
        for i, zombie in enumerate(data['zombies']):
            zombieHP[i] = zombie['hp']
            zombieType[i] = zombie['type']
    
    def _take_action(self, action):
        if action > 0:
            action -= 1
            z_idx = action//(N_LANES * Z_LANE_LENGTH)
            action_area = action%(N_LANES * Z_LANE_LENGTH)
            row = action_area//N_LANES
            col = action_area % N_LANES + 4
            # print(sun)
            sun -= zombie_deck[z_idx][1]
            if sun < 0:
                print("error, sun cannot be negative")
                return
            self.world.zombie_factory.create(zombie_deck[z_idx][0], row, col)
            # print(type(sun))
            # print(zombie_deck[z_idx][1])
            self.world.scene.set_sun(sun)
    
    def get_valid_actions(self):
        # Example return:
        # [ 1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24
        # 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48
        # 49 50] or
        # [ 1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24
        # 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48
        # 49 50 51 52 53 54 55 56 57 58 59 60 61 62 63 64 65 66 67 68 69 70 71 72
        # 73 74 75] or
        # [ 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23
        # 24 25]
        actions = np.arange(self.action_no)
        masks = self.get_action_mask()
        return actions[masks]
    
    def get_action_mask(self):
        sun = self._get_sun()
        data = json.loads(self.world.get_json())
        z_no = len(data['zombies'])
        mask = np.zeros(self.action_no, dtype=bool)
        if z_no>0:
            mask[0] = True
        if sun >= 50:
            mask[1:26] = True
        if sun >= 125:
            mask[26:51] = True
        if sun >= 175:
            mask[51:] = True
        return mask


    def _zombie_x_to_col(self, x):
        col = (x-10)/80-1
        return math.ceil(col)


    def _get_sun(self):
        data = json.loads(self.world.get_json())
        return data['sun']['sun']
    
    def _get_reward(self):
        sun = self._get_sun()
        return sun//25
