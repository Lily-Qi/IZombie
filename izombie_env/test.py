from env import IZenv

env = IZenv()
for _ in range(100):
    env.world.update()

print(env.world.get_json())
