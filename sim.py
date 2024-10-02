import numpy as np

MAX_TIME = 1000
sim_dt = 0.01
controller_dt = 0.1
estimator_dt = 1

current_time = 0
last_controller_update = 0
last_estimator_update = 0


def controller():
    return np.array([0, 0, 0, 0, 0, 1])


def estimator(w):
    pass


while current_time <= MAX_TIME:
    if current_time >= last_controller_update + controller_dt:
        controller_output_torque = np.array([0, 0, 1])
        # apply_torque(controller_output_torque)
        last_controller_update = current_time
        print(f"Controller update: {current_time}")
    if current_time >= last_estimator_update + estimator_dt:
        # w = get_gyro_measurement()
        # estimator(w)
        last_estimator_update = current_time
        print(f"Estimator update: {current_time}")

    current_time += controller_dt
