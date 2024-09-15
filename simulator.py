import numpy as np
import matplotlib.pyplot as plt

from simulationSettings.SimSetup import SimSetup
from data.logger.Logger import Logger

from physics.Dynamics import Dynamics

if __name__ == "__main__":
    
    sim_params_file = 'simulationSettings/sim_params.yaml'
    sim_params = SimSetup(sim_params_file)
    logger = Logger(sim_params)
    
    physics_agent = Dynamics(sim_params)
    
    while physics_agent.time <= sim_params.final_time:
        physics_agent.step([0,0,0])
        
        # Add state info into logger
        record = [physics_agent.time,
                  physics_agent.state[0:3],
                  physics_agent.state[3:6],
                  physics_agent.state[6:10],
                  physics_agent.state[10:14],
                  physics_agent.control_inputs,
                  physics_agent.spherical_gravitational_acceleration,
                  physics_agent.J2_perturbation_acceleration,
                  physics_agent.drag_acceleration,
                  physics_agent.srp_acceleration]
        logger.log(record)
        print(physics_agent.time)
        
    
    # Move visualization to another folder as and when necessary
    '''
    #Earth
    # Generate the sphere coordinates
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    x = sim_params.R_earth*np.outer(np.cos(u), np.sin(v))
    y = sim_params.R_earth*np.outer(np.sin(u), np.sin(v))
    z = sim_params.R_earth*np.outer(np.ones(np.size(u)), np.cos(v))
    
    
    # Orbit Trace
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    ax.plot_wireframe(x, y, z, color='b', alpha=0.2)
    ax.plot(r[:,0], r[:,1], r[:,2], color='r')
    plt.show()
    
    # Kinetic Energy over time
    fig = plt.figure()
    plt.plot([10*i for i in range(len(vel))], np.linalg.norm(vel, axis=1))
    plt.show()
    '''