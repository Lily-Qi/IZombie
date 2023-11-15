import numpy as np

# pylint: disable=E0611
from pvzemu import (
    World,
    SceneType,
    ZombieType,
    PlantType,
    # Scene,
    # SunData,
    # PlantList,
    # ZombieList,
    IZObservation,
)
from .config import (
    GameStatus,
    NUM_ZOMBIES,
    ZOMBIE_SIZE,
    NUM_PLANTS,
    PLANT_SIZE,
    BRAIN_BASE,
    N_LANES,
    LANE_LENGTH,
    P_LANE_LENGTH,
    ACTION_SIZE,
)


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


class IZenv:
    def __init__(self, step_length=50, max_step=None, fix_rand=False):
        self.step_length = step_length
        self.max_step = max_step
        self.fix_rand = fix_rand

        self.ob_factory = IZObservation(NUM_ZOMBIES, NUM_PLANTS)
        self.state = []
        self.zombie_count, self.plant_count, self.brains = 0, 0, [0, 1, 2, 3, 4]

        self.step_count = 0
        self.world = World(SceneType.night)
        self._reset_world()

    def reset(self) -> None:
        self.zombie_count, self.plant_count, self.brains = 0, 0, [0, 1, 2, 3, 4]
        self.step_count = 0
        self._reset_world()

    def get_state_and_mask(self):
        return self.state, self.get_action_mask()

    def step(self, action):
        self.step_count += 1
        prev = {
            "sun_before_action": self.get_sun(),
            "zombie_count": self.zombie_count,
            "plant_count": self.plant_count,
            "brain_count": len(self.brains),
        }

        self._take_action(action)
        prev["sun_after_action"] = self.get_sun()

        for _ in range(self.step_length):
            self.world.update()
        self._update_state()

        game_status = self._get_game_status()
        return (
            self._get_reward(prev, action, game_status),
            self.state,
            self.get_action_mask(),
            game_status,
        )

    def get_valid_actions(self, action_mask):
        actions = np.arange(ACTION_SIZE)
        return actions[action_mask]

    def get_action_mask(self):
        sun = self.get_sun()
        mask = np.zeros(ACTION_SIZE, dtype=bool)
        if self.zombie_count > 0:
            mask[0] = True
        if sun >= 50:
            mask[1:6] = True
        if sun >= 125:
            mask[6:11] = True
        if sun >= 175:
            mask[11:16] = True
        for row in range(5):
            if not row in self.brains:
                for i in range(3):
                    mask[i * 5 + row + 1] = False
        return mask

    def get_sun(self):
        return self.world.scene.sun.sun

    def _get_game_status(self):
        if self.max_step is not None and self.step_count >= self.max_step:
            return GameStatus.TIMEUP
        if len(self.brains) == 0:
            return GameStatus.WIN
        if self.get_sun() < 50 and self.zombie_count == 0:
            return GameStatus.LOSE
        return GameStatus.CONTINUE

    def _get_reward(self, prev, action, game_status):
        earned_sun = self.get_sun() - prev["sun_after_action"]
        eaten_plant_num = prev["plant_count"] - self.plant_count
        eaten_brain_num = prev["brain_count"] - len(self.brains)

        reward = earned_sun / 25 + eaten_plant_num * 2 + eaten_brain_num * 10

        if game_status == GameStatus.WIN:
            reward += self.get_sun() / 2

        if game_status == GameStatus.LOSE:
            reward -= 300

        return reward
        # reward = self.get_sun() - prev["sun_before_action"]
        # if game_status == GameStatus.LOSE:
        #     reward -= 1800
        # return reward

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
                index // P_LANE_LENGTH,
                index % P_LANE_LENGTH,
            )
        self._update_state()

    def _take_action(self, action):
        if action > 0:
            action -= 1
            z_idx = action // N_LANES
            row = action % N_LANES
            col = 4
            sun = self.get_sun() - zombie_deck[z_idx][1]
            assert sun >= 0
            self.world.zombie_factory.create(zombie_deck[z_idx][0], row, col)
            self.world.scene.set_sun(sun)

    def _update_state(self):
        state, self.zombie_count, self.plant_count = self.ob_factory.create(self.world)
        self.state = list(state)

        self.brains = []
        for i, b in enumerate(self.state[BRAIN_BASE : BRAIN_BASE + 5]):
            if b > 0.5:
                self.brains.append(i)

    def print_human_readable_state(self, highlight=None):
        def plant_str(plant_type):
            if plant_type == 1:
                return "sun"
            if plant_type == 2:
                return "pea"
            if plant_type == 3:
                return "sqa"
            if plant_type == 4:
                return "sno"
            return "---"

        def zombie_x_to_col(x):
            col = int((x + 40) // 80)
            return min(max(col, 0), LANE_LENGTH - 1)

        def zombie_str(zombie_type):
            if zombie_type == 1:
                return "Z"
            if zombie_type == 2:
                return "B"
            if zombie_type == 3:
                return "F"
            return "."

        def acc1_hp_max(zombie_type):
            if zombie_type == 2:
                return 1100
            if zombie_type == 3:
                return 1400
            return 0

        plant_hps = [[0 for _ in range(P_LANE_LENGTH)] for _ in range(N_LANES)]
        plant_types = [["---" for _ in range(P_LANE_LENGTH)] for _ in range(N_LANES)]
        zombie_hps = [[0 for _ in range(LANE_LENGTH)] for _ in range(N_LANES)]
        zombie_types = [["." for _ in range(LANE_LENGTH)] for _ in range(N_LANES)]

        state = self.state

        for i in range(NUM_PLANTS):
            base = NUM_ZOMBIES * ZOMBIE_SIZE + i * PLANT_SIZE
            if state[base] != 0:
                plant_type = int(round(state[base] * 4))
                hp = state[base + 1]
                row = int(round(state[base + 2] * 5))
                col = int(round(state[base + 3] * 9))

                plant_hps[row][col] += hp * 300
                plant_types[row][col] = plant_str(plant_type)

        for i in range(NUM_ZOMBIES):
            base = i * ZOMBIE_SIZE
            if state[base] != 0:
                zombie_type = int(round(state[base] * 3))
                x = state[base + 1] * 650
                row = int(round(state[base + 2] * 5))
                hp = state[base + 3]
                acc1_hp = state[base + 4]
                col = zombie_x_to_col(x)

                zombie_hps[row][col] += hp * 270 + acc1_hp * acc1_hp_max(zombie_type)
                zombie_types[row][col] = zombie_str(zombie_type)

        print("==Plant HP==")
        for row in range(N_LANES):
            print(f"row {row+1}: ", end="")
            for col in range(P_LANE_LENGTH):
                print(f"{plant_hps[row][col]:.2f}\t", end="")
            print()

        print("==Plant Type==")
        for row in range(N_LANES):
            print(f"row {row+1}: ", end="")
            for col in range(P_LANE_LENGTH):
                print(f"{plant_types[row][col]}\t", end="")
            print()

        print("==Zombie HP==")
        for row in range(N_LANES):
            print(f"row {row+1}: ", end="")
            for col in range(LANE_LENGTH):
                print(f"{zombie_hps[row][col]:.2f}\t", end="")
            print()

        print("==Zombie Type==")
        highlight_row, highlight_col = (-1, -1) if highlight is None else highlight
        for row in range(N_LANES):
            print(f"row {row+1}: ", end="")
            for col in range(LANE_LENGTH):
                out = f"{zombie_types[row][col]}"
                if row == highlight_row and col == highlight_col:
                    out = f"[{out}]"
                out += "\t"
                print(out, end="")
            print()

        print(
            f"Step: {self.step_count}; Sun: {self.get_sun()}; Brains: {len(self.brains)}; Game status: {self._get_game_status().name} "
        )
