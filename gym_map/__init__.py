from gym.envs.registration import register

register(
    id='map-bro-v0',
    entry_point='gym_map.envs:broEnv',
)

register(
    id='map-hd-v0',
    entry_point='gym_map.envs:HDEnv',
)