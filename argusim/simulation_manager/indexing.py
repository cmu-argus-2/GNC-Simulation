



class STATES:
    def __init__(self, num_RWs, num_MTBs):
        self.NSTATES   = 22 + num_RWs + num_MTBs + 4
        self.ECI_POS   = slice(0, 3)
        self.ECI_VEL   = slice(3, 6)
        self.TRANS     = slice(0, 6)
        self.QUAT      = slice(6, 10)
        self.ANG_VEL   = slice(10, 13)
        self.ROT       = slice(6, 13)
        self.SUN_POS   = slice(13, 16)
        self.MAG_FIELD = slice(16, 19)
        self.I_MTB     = slice(19, 19+num_MTBs)
        self.RW_SPEED  = slice(19+num_MTBs, 19 + num_MTBs + num_RWs)
        self.GYRO_BIAS = slice(19 + num_MTBs + num_RWs,22 + num_MTBs + num_RWs)
        self.BAT       = slice(22 + num_MTBs + num_RWs, 26 + num_MTBs + num_RWs)
        self.BAT_SOC   = 22 + num_MTBs + num_RWs
        self.BAT_TEMP  = 23 + num_MTBs + num_RWs
        self.BAT_VOLT  = 24 + num_MTBs + num_RWs
        self.BAT_CUR   = 25 + num_MTBs + num_RWs


class SENSORS:
    def __init__(self, num_stk, num_photodiodes, num_rws, num_MTBs, num_panels, num_deploy_sensors):
        self.GPS            = slice(0, 6)
        self.GPS_POS        = slice(0, 3)
        self.GPS_VEL        = slice(3, 6)
        self.GYRO           = slice(6, 9)
        self.MAG            = slice(9, 12)
        ny = 12
        self.STK_QUAT       = slice(ny, ny + num_stk)
        ny = ny + num_stk
        self.PHOTODIODES    = slice(ny, ny + num_photodiodes)
        ny = ny + num_photodiodes
        self.RTC            = slice(ny, ny + 1)
        ny = ny + 1
        self.RW_OMEGA       = slice(ny, ny + num_rws)
        ny = ny + num_rws
        self.MTB_POW        = slice(ny, ny + num_MTBs)
        ny = ny + num_MTBs
        self.SOL_POW        = slice(ny, ny +  num_panels)
        ny = ny+num_panels
        self.BATTERY        = slice(ny, ny + 11)
        self.BAT_SOC        = slice(ny, ny + 1)
        self.BAT_CAP        = slice(ny + 1, ny + 2)
        self.BAT_CUR        = slice(ny + 2, ny + 3)
        self.BAT_VOL        = slice(ny + 3, ny +4)
        self.BAT_MIDVOL     = slice(ny + 4, ny + 5)
        self.BAT_TTE        = slice(ny + 5, ny + 6)
        self.BAT_TTF        = slice(ny + 6, ny + 7)
        self.BAT_TEMP       = slice(ny + 7, ny + 8)
        self.BAT_TEMP_AIN1  = slice(ny + 8, ny + 9)
        self.BAT_TEMP_AIN2  = slice(ny + 9, ny + 10)
        self.BAT_TEMP_DIE   = slice(ny + 10, ny + 11)
        ny = ny + 11
        self.JET_POW        = slice(ny, ny + 1)
        ny = ny+1
        self.DEPLOY         = slice(ny, ny + num_deploy_sensors)
        ny = ny+num_deploy_sensors
        self.NSENSORS       = ny + num_deploy_sensors


class CONTROLS:
    def __init__(self, num_RWs, num_MTBs):
        self.NCONTROLS   = num_RWs + num_MTBs + 1
        self.MTB_VOLT    = slice(0, num_MTBs)
        self.RW_TORQUE   = slice(num_MTBs, num_RWs + num_MTBs)
        self.JETSON_ON   = num_RWs + num_MTBs


class IDX:
    def __init__(self, num_RWs, num_MTBs, num_stk, num_photodiodes, 
                 num_panels, num_deploy_sensors):
        
        self.STATES = STATES(num_RWs, num_MTBs)
        self.SENSORS = SENSORS(num_stk, num_photodiodes, num_RWs, num_MTBs, num_panels, num_deploy_sensors)
        self.CONTROLS = CONTROLS(num_RWs, num_MTBs)

        self.NSTATES   = self.STATES.NSTATES
        self.NSENSORS  = self.SENSORS.NSENSORS
        self.NCONTROLS = self.CONTROLS.NCONTROLS
        self.NRWS      = num_RWs
        self.NMTBS     = num_MTBs
        self.NSTK      = num_stk
        self.NPHOTODIODES = num_photodiodes
        self.NPANELS   = num_panels
        self.NDEPLOYS  = num_deploy_sensors
        