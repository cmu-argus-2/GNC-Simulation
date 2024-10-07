class Control:
    N_RW = 1
    N_MTB = 3
    NU = N_RW + N_MTB
    RW_TORQUE = slice(0, N_RW)
    MTB_TORQUE = slice(N_RW, N_RW + N_MTB)


class State:
    NX = 19
    ECI_POS = slice(0, 3)
    ECI_VEL = slice(3, 6)
    TRANS = slice(0, 6)
    QUAT = slice(6, 10)
    ANG_VEL = slice(10, 13)
    ROT = slice(6, 13)
    SUN_POS = slice(13, 16)
    MAG_FIELD = slice(16, 19)
    # RW speed should be a state because it depends on the torque applied and needs to be propagated
    RW_SPEED = slice(19, 19 + Control.N_RW)
