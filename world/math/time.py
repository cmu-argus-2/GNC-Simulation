import brahe
from brahe.epoch import Epoch


def increment_epoch(epoch: Epoch, dt: float) -> Epoch:
    """
    Increments the current epoch by the given time step

    :param epoch: The current epoch as an instance of brahe's Epoch class.
    :param dt: The amount of time to increment the epoch by, in seconds.
    """
    return Epoch(
            *brahe.time.jd_to_caldate(
                epoch.jd()
                + dt / (24 * 60 * 60)
            )
        )
