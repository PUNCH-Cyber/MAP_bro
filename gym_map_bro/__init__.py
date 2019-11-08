from gym.envs.registration import register

register(
    id='map-bro-v0',
    entry_point='gym_map_bro.envs:broEnv',
)