import numpy as np
from argusim.world.physics.geometry import Surface, Body, Sensor, SolarPanel
from copy import deepcopy
import yaml


def generate_lookup_tables():
    # Define the surfaces of a 1U CubeSat
    X_DIM = 0.1
    Y_DIM = 0.1
    Z_DIM = 0.1
    # Define the tilt angles for the solar panels (in degrees)
    # +X, -X, +Y, -Y
    solar_panel_tilt_angles = np.deg2rad(np.array([90, 90, 90, 90]))

    PXSurface = Surface([[X_DIM/2, -Y_DIM/2, -Z_DIM/2], 
                         [X_DIM/2, Y_DIM/2, -Z_DIM/2], 
                         [X_DIM/2, Y_DIM/2, Z_DIM/2], 
                         [X_DIM/2, -Y_DIM/2, Z_DIM/2]])

    MXSurface = Surface([[-X_DIM/2, -Y_DIM/2, -Z_DIM/2], 
                         [-X_DIM/2, Y_DIM/2, -Z_DIM/2], 
                         [-X_DIM/2, Y_DIM/2, Z_DIM/2], 
                         [-X_DIM/2, -Y_DIM/2, Z_DIM/2]])

    PYSurface = Surface([[-X_DIM/2, Y_DIM/2, -Z_DIM/2], 
                         [X_DIM/2, Y_DIM/2, -Z_DIM/2], 
                         [X_DIM/2, Y_DIM/2, Z_DIM/2], 
                         [-X_DIM/2, Y_DIM/2, Z_DIM/2]])

    MYSurface = Surface([[-X_DIM/2, -Y_DIM/2, -Z_DIM/2], 
                         [X_DIM/2, -Y_DIM/2, -Z_DIM/2], 
                         [X_DIM/2, -Y_DIM/2, Z_DIM/2], 
                         [-X_DIM/2, -Y_DIM/2, Z_DIM/2]])

    PZSurface = Surface([[-X_DIM/2, -Y_DIM/2, Z_DIM/2], 
                         [X_DIM/2, -Y_DIM/2, Z_DIM/2], 
                         [X_DIM/2, Y_DIM/2, Z_DIM/2], 
                         [-X_DIM/2, Y_DIM/2, Z_DIM/2]])

    MZSurface = Surface([[-X_DIM/2, -Y_DIM/2, -Z_DIM/2], 
                         [X_DIM/2, -Y_DIM/2, -Z_DIM/2], 
                         [X_DIM/2, Y_DIM/2, -Z_DIM/2], 
                         [-X_DIM/2, Y_DIM/2, -Z_DIM/2]])

    PXSPSurface = deepcopy(PXSurface)
    PXSPSurface.rotate_around_axis([X_DIM/2, 0, Z_DIM/2], [0, 1, 0], solar_panel_tilt_angles[0])
    MXSPSurface = deepcopy(MXSurface)
    MXSPSurface.rotate_around_axis([-X_DIM/2, 0, Z_DIM/2], [0, -1, 0], solar_panel_tilt_angles[1])
    PYSPSurface = deepcopy(PYSurface)
    PYSPSurface.rotate_around_axis([0, Y_DIM/2, Z_DIM/2], [-1, 0, 0], solar_panel_tilt_angles[2])
    MYSPSurface = deepcopy(MYSurface)
    MYSPSurface.rotate_around_axis([0, -Y_DIM/2, Z_DIM/2], [1, 0, 0], solar_panel_tilt_angles[3])

    surfaces = [
        PXSurface,   # Right face
        MXSurface,   # Left face
        PYSurface,   # Top face
        MYSurface,   # Bottom face
        PZSurface,   # Front face
        MZSurface,   # Back face
        MXSPSurface, # Top SP -X
        PXSPSurface, # Top SP +X
        MYSPSurface, # Top SP -Y
        PYSPSurface  # Top SP +Y
    ]

    # Parameters for the sun sensors
    half_angle = 90  # Half angle of the visibility cone in degrees

    sensors = [
        Sensor(direction=[1, 0, 0], position=[X_DIM/2, 0, 0], half_angle=half_angle),
        Sensor(direction=[-1, 0, 0], position=[-X_DIM/2, 0, 0], half_angle=half_angle),
        Sensor(direction=[0, 1, 0], position=[0, Y_DIM/2, 0], half_angle=half_angle),
        Sensor(direction=[0, -1, 0], position=[0, -Y_DIM/2, 0], half_angle=half_angle),
        Sensor(direction=[0, 0, -1], position=[0, 0, -Z_DIM/2], half_angle=half_angle),
        Sensor(direction=[0.7071, 0, 0.7071], position=[0, 0, Z_DIM/2], half_angle=half_angle),
        Sensor(direction=[0, 0.7071, 0.7071], position=[0, 0, Z_DIM/2], half_angle=half_angle),
        Sensor(direction=[-0.7071, 0, 0.7071], position=[0, 0, Z_DIM/2], half_angle=half_angle),
        Sensor(direction=[0, -0.7071, 0.7071], position=[0, 0, Z_DIM/2], half_angle=half_angle)
    ]

    SolarPanels = [
        SolarPanel(PXSurface.vertices, [1,0,0]),   # +X face
        SolarPanel(MXSurface.vertices, [-1,0,0]),  # -X face
        SolarPanel(PYSurface.vertices, [0,1,0]),   # +Y face
        SolarPanel(MYSurface.vertices, [0,-1,0]),  # -Y face
        SolarPanel(PZSurface.vertices, [0,0,1]),   # +Z face
        SolarPanel(MXSPSurface.vertices, [0,0,1]), # Top SP -X, +Z facing
        SolarPanel(PXSPSurface.vertices, [0,0,1]), # Top SP +X, +Z facing
        SolarPanel(MYSPSurface.vertices, [0,0,1]), # Top SP -Y, +Z facing
        SolarPanel(PYSPSurface.vertices, [0,0,1]),  # Top SP +Y, +Z facing
        SolarPanel(MXSPSurface.vertices, [0,0,-1]), # Top SP -X, -Z facing
        SolarPanel(PXSPSurface.vertices, [0,0,-1]), # Top SP +X, -Z facing
        SolarPanel(MYSPSurface.vertices, [0,0,-1]), # Top SP -Y, -Z facing
        SolarPanel(PYSPSurface.vertices, [0,0,-1])  # Top SP +Y, -Z facing
        ]

    SC_Body = Body(surfaces, solar_panels=SolarPanels, sensors=sensors)

    NS = len(sensors)
    # Create a grid of azimuth and elevation angles
    NA = 100
    NE = 100
    azimuth = np.linspace(0, 360, NA)  # Azimuth angles from 0 to 360 degrees
    elevation = np.linspace(-90, 90, NE)  # Elevation angles from -90 to 90 degrees
    azimuth_grid, elevation_grid = np.meshgrid(azimuth, elevation)

    # Convert azimuth and elevation to Cartesian coordinates
    x = np.cos(np.radians(elevation_grid)) * np.cos(np.radians(azimuth_grid))
    y = np.cos(np.radians(elevation_grid)) * np.sin(np.radians(azimuth_grid))
    z = np.sin(np.radians(elevation_grid))

    # Initialize visibility matrix
    visibility         = np.zeros((NS, NE, NA), dtype=bool)
    effective_sp_area  = np.zeros((NE, NA))    # solar panel cross-sectional, unoccluded area
    aero_torque_fac    = np.zeros((NE, NA, 3))
    aero_force_fac     = np.zeros((NE, NA, 3)) # 
    effective_sc_area  = np.zeros((NE, NA))    # whole sc cross-sectional, unoccluded area

    # Calculate visibility for each sensor direction
    for i in range(NE):
        for j in range(NA):
            direction = np.array([x[i, j], y[i, j], z[i, j]])
            visibility[:,i,j]                           = SC_Body.compute_sensor_visibility(direction)
            effective_sp_area[i, j]                     = SC_Body.compute_unoccluded_effective_area(direction, SC_Body)
            aero_torque_fac[i, j], aero_force_fac[i, j], effective_sc_area[i, j] = SC_Body.compute_center_of_pressure(direction, SC_Body)
            if (i * NE + j) % (NA * NE // 10) == 0:
                print(f'Progress: {100 * (i * NE + j) / (NA * NE):.1f}%')

    # Sum the visibility flags to get the number of sensors capturing each vector
    visibility_sum = np.sum(visibility, axis=0)

    visible_sensors = np.sum(visibility_sum > 2)

    # Save the Power_gen array to a file
    data_to_save = {
        "azimuth": azimuth_grid.tolist(),
        "elevation": elevation_grid.tolist(),
        "visibility_sum": visibility_sum.tolist(),
        "effective_sp_area": effective_sp_area.tolist(),
        "effective_sc_area": effective_sc_area.tolist(),
        "visibility": visible_sensors.tolist(),
        "aero_torque_fac": aero_torque_fac.tolist(),
        "aero_force_fac": aero_force_fac.tolist(),
        "NE": NE,
        "NA": NA,
        "NS": NS
    }

    with open('./lookup_tables.yaml', 'w') as f:
        yaml.dump(data_to_save, f)


