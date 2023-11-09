from env import IZenv

env = IZenv()

env.step(30)

for _ in range(30):
    env.step(0)

# print(env.world.get_json())
