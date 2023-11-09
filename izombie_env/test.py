from env import IZenv

env = IZenv()

env.get_valid_actions()
env.world.scene.set_sun(200)
env.get_valid_actions()
env.step(40)
env.get_valid_actions()

# env.world.update()
# print(env.world.get_json())
# print(env.zombie_x_to_col(410.1))
