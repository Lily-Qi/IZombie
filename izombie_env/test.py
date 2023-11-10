from env import IZenv

env = IZenv()

state = env.get_state()
print(env.get_valid_actions(state))
env.step(1)
state = env.get_state()
print(env.get_valid_actions(state))
 
# env.world.update()
# print(env.world.get_json())
# print(env.zombie_x_to_col(410.1))
