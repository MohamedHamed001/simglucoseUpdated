try:
    from gymnasium.envs.registration import register
    register(
        id='simglucose-v0',
        entry_point='simglucose.envs:T1DSimEnv',
    )
except ImportError:
    # Gymnasium not available, skip registration
    pass
