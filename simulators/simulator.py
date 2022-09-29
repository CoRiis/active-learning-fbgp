from simulators.analytical_simulators import *


def oracle_simulator(args):
    """
    Get simulator based on the arguments
    :param args: arguments
    :return: simulator
    """
    sim = args.simulator
    if sim == "motorcycle": return Motorcycle()
    elif sim == "gramacy1d": return GramacyAndLee1d(noise_sigma=0.1)
    elif sim == "gramacy2d": return GramacyAndLee2d(noise_sigma=0.05)
    elif sim == "higdon1d": return Higdon1d(noise_sigma=0.1)
    elif sim == "branin2d": return Branin2d(noise_sigma=11.32)
    elif sim == "ishigami3d": return Ishigami3d(noise_sigma=0.187)
    elif sim == "hartmann6d": return Hartmann6d(noise_sigma=0.0192)

    else:
        raise NotImplementedError(f'Simulator {sim} is not defined.')
