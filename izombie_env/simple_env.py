import numpy as np
from pvzemu import (
    World,
    SceneType,
    ZombieType,
    PlantType,
    Scene,
    SunData,
    PlantList,
    PlantStatus,
    ZombieList,
    ZombieStatus,
    ZombieAccessory1,
)
from . import config


plant_counts = {
    PlantType.sunflower: 9,
    PlantType.pea_shooter: 6,
    PlantType.squash: 3,
    PlantType.snow_pea: 2,
}

zombie_deck = [
    [ZombieType.zombie, 50],
    [ZombieType.buckethead, 125],
    [ZombieType.football, 175],
]

EAT_BRAIN_X_THRES = 25


class IZenv:
    def __init__(self, step_length=50, max_step=None, fix_rand=False):
        self.step_length = step_length
        self.max_step = max_step
        self.fix_rand = fix_rand

        self.state = None
        self._zombie_count = 0
        self._plant_count = 0
        self.step_count = 0
        self._brain_status = np.array([1, 1, 1, 1, 1])
        self.world = World(SceneType.night)
        self._reset_world()

    def reset(self) -> None:
        self._zombie_count = 0
        self._plant_count = 0
        self.step_count = 0
        self._brain_status = np.array([1, 1, 1, 1, 1])
        self._reset_world()

    def _reset_world(self) -> None:
        self.world = World(SceneType.night)
        self.world.scene.stop_spawn = True
        self.world.scene.is_iz = True
        self.world.scene.set_sun(150)
        plant_list = [
            plant for plant, count in plant_counts.items() for _ in range(count)
        ]
        if self.fix_rand:
            np.random.seed(0)
        np.random.shuffle(plant_list)
        for index, plant in enumerate(plant_list):
            self.world.plant_factory.create(
                plant,
                index // config.P_LANE_LENGTH,
                index % config.P_LANE_LENGTH,
            )
        self._sync_state_with_world()

    def get_state_and_mask(self):
        return self.state, self.get_action_mask()

    def _get_game_status(self):
        if self.max_step is not None and self.step_count >= self.max_step:
            return config.GameStatus.TIMEUP
        if self._get_brain_num() == 0:
            return config.GameStatus.WIN
        if self.get_sun() < 50 and self._get_zombie_num() == 0:
            return config.GameStatus.LOSE
        return config.GameStatus.CONTINUE

    def _get_reward(self, prev, action, game_status):
        reward = self.get_sun() - prev["sun"]
        # reward += 100 * (prev["brain_num"] - self._get_brain_num())
        if game_status == config.GameStatus.LOSE:
            reward -= 1800
        # if game_status == config.GameStatus.WIN:
        # reward += self.get_sun() * 5
        # reward -= 10

        return reward
        # reward += 25 * max(prev["plant_num"] - self._get_plant_num(), 0)
        # elif game_status == config.GameStatus.WIN:
        #     reward += self.get_sun()
        # reward = 1000 * (prev["brain_num"] - self._get_brain_num())

        # reward -= 10 * max(prev["zombie_num"] - self._get_zombie_num(), 0)
        # reward -= max(prev["sun"] - self.get_sun(), 0)

        # if game_status == config.GameStatus.WIN:
        #     reward += 5 * self.get_sun()

        # return reward

    def step(self, action):
        self.step_count += 1
        prev = {
            "sun": self.get_sun(),
            "zombie_num": self._get_zombie_num(),
            "plant_num": self._get_plant_num(),
            "brain_num": self._get_brain_num(),
        }

        self._take_action(action)
        for _ in range(self.step_length):
            self.world.update()
        self._sync_state_with_world()

        game_status = self._get_game_status()
        return (
            self._get_reward(prev, action, game_status),
            self.state,
            self.get_action_mask(),
            game_status,
        )

    def get_valid_actions(self, action_mask):
        actions = np.arange(config.ACTION_SIZE)
        return actions[action_mask]

    def get_action_mask(self):
        sun = self.get_sun()
        mask = np.zeros(config.ACTION_SIZE, dtype=bool)
        if self._get_zombie_num() > 0:
            mask[0] = True
        if sun >= 50:
            mask[1:6] = True
        if sun >= 125:
            mask[6:11] = True
        if sun >= 175:
            mask[11:16] = True
        for row in range(5):
            if self._brain_status[row] == 0:
                for i in range(3):
                    mask[i * 5 + row + 1] = False
        return mask

    def _sync_state_with_world(self):
        plant_hps = np.zeros(config.N_LANES * config.P_LANE_LENGTH, dtype=float)
        plant_type_nums = np.zeros(config.N_LANES * config.P_LANE_LENGTH, dtype=float)
        zombie_hps = np.zeros(config.N_LANES * config.LANE_LENGTH, dtype=float)
        zombie_type_nums = np.zeros(config.N_LANES * config.LANE_LENGTH, dtype=float)
        sun = np.array([self.get_sun() / config.SUN_MAX])

        self._plant_count = 0
        for plant in self.world.scene.plants:
            if plant.status == PlantStatus.squash_crushed:
                continue

            self._plant_count += 1

            plant_hps[plant.row * config.P_LANE_LENGTH + plant.col] = (
                plant.hp / config.P_MAX_HP
            )
            plant_type_nums[plant.row * config.P_LANE_LENGTH + plant.col] = (
                self._plant_type_to_plant_num(plant.type) / config.N_PLANT_TYPE
            )

        self._zombie_count = 0
        for zombie in self.world.scene.zombies:
            # Update brain status for each row
            if zombie.x < EAT_BRAIN_X_THRES:
                self._brain_status[zombie.row] = 0
                continue

            if zombie.is_dead or not zombie.is_not_dying:
                continue

            self._zombie_count += 1
            zombie_hp = zombie.hp * 0.66 + zombie.accessory_1.hp

            zombie_hps[
                zombie.row * config.LANE_LENGTH + self._zombie_x_to_col(zombie.x)
            ] += (zombie_hp / config.Z_MAX_HP)
            zombie_type_nums[
                zombie.row * config.LANE_LENGTH + self._zombie_x_to_col(zombie.x)
            ] = (self._zombie_type_to_zombie_num(zombie.type) / config.N_ZOMBIE_TYPE)

        self.state = np.concatenate(
            [
                plant_hps,
                plant_type_nums,
                zombie_hps,
                zombie_type_nums,
                sun,
                self._brain_status,
            ]
        )

    def _take_action(self, action):
        if action > 0:
            action -= 1
            z_idx = action // config.N_LANES
            row = action % config.N_LANES
            col = 4
            sun = self.get_sun() - zombie_deck[z_idx][1]
            assert sun >= 0
            self.world.zombie_factory.create(zombie_deck[z_idx][0], row, col)
            self.world.scene.set_sun(sun)

    def _zombie_x_to_col(self, x):
        col = int((x + 40) // 80)
        col = max(col, 0)
        col = min(col, config.LANE_LENGTH - 1)
        return col

    def get_sun(self):
        return self.world.scene.sun.sun

    def _get_brain_num(self):
        return np.count_nonzero(self._brain_status == 1)

    def _get_zombie_num(self):
        return self._zombie_count

    def _get_plant_num(self):
        return self._plant_count

    def _plant_num_to_str(self, plant_num):
        plant_num = int(round(plant_num))
        if plant_num == 1:
            return "sun"
        if plant_num == 2:
            return "pea"
        if plant_num == 3:
            return "sqa"
        if plant_num == 4:
            return "sno"
        return "___"

    def _plant_type_to_plant_num(self, plant_type):
        if plant_type == PlantType.sunflower:
            return 1
        if plant_type == PlantType.pea_shooter:
            return 2
        if plant_type == PlantType.squash:
            return 3
        if plant_type == PlantType.snow_pea:
            return 4
        return 0

    def _zombie_num_to_str(self, zombie_num):
        zombie_num = int(round(zombie_num))
        if zombie_num == 1:
            return "Z"
        if zombie_num == 2:
            return "B"
        if zombie_num == 3:
            return "F"
        return "_"

    def _zombie_type_to_zombie_num(self, zombie_type):
        if zombie_type == ZombieType.zombie:
            return 1
        if zombie_type == ZombieType.buckethead:
            return 2
        if zombie_type == ZombieType.football:
            return 3
        return 0

    def print_human_readable_state(self, highlight=None):
        state = self.state

        print(f"Allowed actions: {np.where(state[-config.ACTION_SIZE:] > 0.5)[0]}")

        print("==Plant HP==")
        for row in range(5):
            print(
                f"row {row+1}: {state[row * config.P_LANE_LENGTH  :(row+1)*config.P_LANE_LENGTH ] * config.P_MAX_HP}"
            )

        state = state[5 * config.P_LANE_LENGTH :]
        print("==Plant Type==")
        for row in range(5):
            plant_nums = (
                state[row * config.P_LANE_LENGTH : (row + 1) * config.P_LANE_LENGTH]
                * config.N_PLANT_TYPE
            )
            print(f"row {row+1}: {[self._plant_num_to_str(p) for p in plant_nums]}")

        state = state[5 * config.P_LANE_LENGTH :]
        print("==Zombie HP==")
        for row in range(5):
            print(
                f"row {row+1}: {state[row * config.LANE_LENGTH:(row+1)*config.LANE_LENGTH] * config.Z_MAX_HP}"
            )

        state = state[5 * config.LANE_LENGTH :]
        print("==Zombie Type==")
        highlight_row, highlight_col = (-1, -1) if highlight is None else highlight
        for row in range(5):
            zombie_nums = (
                state[row * config.LANE_LENGTH : (row + 1) * config.LANE_LENGTH]
                * config.N_ZOMBIE_TYPE
            )
            zombie_strs = [self._zombie_num_to_str(z) for z in zombie_nums]
            if row == highlight_row:
                zombie_strs[highlight_col] = f"[{zombie_strs[highlight_col]}]"
            print(f"row {row+1}: {zombie_strs}")

        print(
            f"Step: {self.step_count}; Sun: {self.get_sun()}; Brains: {self._get_brain_num()}; Game status: {self._get_game_status().name} "
        )
