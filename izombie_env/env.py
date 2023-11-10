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
P_MAX_HP = 300
Z_MAX_HP = 500
SUN_MAX = 1950
ZOMBIE_TYPE_START = 85
ZOMBIE_TYPE_END = 128

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
        self._data = json.loads(self.world.get_json())
    
    def step(self, action):
        self._data = json.loads(self.world.get_json())

        self._take_action(action)
        for _ in range(50):
            self.world.update()

        self._data = json.loads(self.world.get_json())

        reward = self._get_reward()
        state = self.get_state()
        print(state)
        isEnded = self._get_game_status(state)
        isWin = self._get_game_result(state)
        return reward, state, isEnded, isWin
    
    def _take_action(self, action):
        if action > 0:
            action -= 1
            z_idx = action//(N_LANES * Z_LANE_LENGTH)
            action_area = action%(N_LANES * Z_LANE_LENGTH)
            row = action_area//N_LANES
            col = action_area % N_LANES + 4
            # print(sun)
            sun = self._get_sun()
            self.world.zombie_factory.create(zombie_deck[z_idx][0], row, col)
            sun -= zombie_deck[z_idx][1]
            # print(type(sun))
            # print(zombie_deck[z_idx][1])
            self.world.scene.set_sun(sun) 
    
    def get_state(self):
        #Each plant HP, intialize to 0 to indicate if there is no plant
        plantHPArray = np.zeros(N_LANES * P_LANE_LENGTH, dtype=float)
        #Each plant type, intialize to -1 to indicate if there is no plant
        plantTypeArray = np.full(N_LANES * P_LANE_LENGTH, -1, dtype=int)
        #Each zombie HP, intialize to 0 to indicate if there is no zombie
        zombieHPArray = np.zeros(N_LANES * LANE_LENGTH, dtype=float)
        #Each zombie type, intialize to -1 to indicate if there is no zombie
        zombieTypeArray = np.full(N_LANES * LANE_LENGTH, -1, dtype=int)
        #Total sun
        sunNum = np.array([self._get_sun() / SUN_MAX])
        #Each brain status
        brainStatusArray = np.array([1, 1, 1, 1, 1])

        for plant in self._data['plants']:
            plantHPArray[plant['row'] * P_LANE_LENGTH + plant['col']] = plant['hp'] / P_MAX_HP

            #Assign plant type depending on the plant type
            plantType = 0
            if plant['type'] == "sunflower":
                plantType = 0
            elif plant['type'] == "pea_shooter":
                plantType = 1
            elif plant['type'] == "squash":
                plantType = 2
            elif plant['type'] == "snow_pea":
                plantType = 3
                
            plantTypeArray[plant['row'] * P_LANE_LENGTH + plant['col']] = plantType
        
        for zombie in self._data['zombies']:
            #Update brain status for each row
            if zombie['x'] < 25:
                brainStatusArray[zombie['row']] = 0
            
            #Calcullate zombie total HP and assign type depending on the zombie type
            zombieHP = 0
            zombieType = 0
            if zombie['type'] == "zombie":
                zombieType = 0
                zombieHP = zombie['hp']
            elif zombie['type'] == "buckethead":
                zombieType = 1
                zombieHP = zombie['hp'] + zombie['accessory_1']['hp']
            elif zombie['type'] == "football":
                zombieType = 2
                zombieHP = zombie['hp'] + zombie['accessory_1']['hp']

            zombieHPArray[zombie['row'] * LANE_LENGTH + self._zombie_x_to_col(zombie['x'])] += zombieHP / Z_MAX_HP
            zombieTypeArray[zombie['row'] * LANE_LENGTH + self._zombie_x_to_col(zombie['x'])] = zombieType
        
        return np.concatenate([plantHPArray, plantTypeArray, zombieHPArray, zombieTypeArray, sunNum, brainStatusArray])

    def _zombie_x_to_col(self, x):
        col = (x-10)/80-1
        return math.ceil(col)

    def _get_sun(self):
        return self._data['sun']['sun']
    
    def _get_reward(self):
        sun = self._get_sun()
        return sun//25
    
    def _get_zombie_num(self, state):
        zombieNum = 45 + state[ZOMBIE_TYPE_START:ZOMBIE_TYPE_END].sum()

        return zombieNum
    
    def _get_game_status(self, state):
        sun = self._get_sun()
        zombieNum = self._get_zombie_num(state)

        if sun < 50 and zombieNum == 0:

            return True
        
        return False

    def _get_game_result(self, state):
        sum = state[-N_LANES:].sum()
        if sum == 0:
            
            return True
        
        return False

    
