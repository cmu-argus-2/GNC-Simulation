sim_config = {
    "timestep": 60.0,
    "method_order": [
        "step_position_and_velocity",
        "step_attitude_and_angular_velocity"
    ],
    "methods": {
        "step_position_and_velocity": {
            "enabled": True
        },
        "step_attitude_and_angular_velocity": {
            "enabled": True
        }
    },
    "members": {
        "position_and_velocity": {
            "len": 6,
            "enabled": True
        },
        "attitude_and_angular_velocity": {
            "len": 7,
            "enabled": True
        }
    },
}
