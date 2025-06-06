
from controllers.abstract_oscillator_controller import AbstractOscillatorController
from util.mp_util import sweep_1d
from util.rw import save_object
from metrics import compute_neural_metrics
import os
from tqdm import tqdm
from scipy.integrate import ode


def pretty(d, indent=1):
    """ print dictionary d in a pretty format """
    for key, value in d.items():
        print('\t' * indent + str(key))
        if isinstance(value, dict):
            pretty(value, indent+1)
        else:
            print('\t' * (indent+1) + str(value))


def step_rk(time, state, timestep, f):
    """Step"""
    k1 = timestep*f(time, state)
    k2 = timestep*f(time + timestep / 2, state + k1 / 2)
    k3 = timestep*f(time + timestep / 2, state + k2 / 2)
    k4 = timestep*f(time + timestep, state + k3)
    return state + 1 / 6 * (k1 + 2 * k2 + 2 * k3 + k4)


def step_ode(time, solver, timestep):
    """Step"""
    return solver.integrate(time+timestep)


def run_single(pars):
    """
    Run simulation
    Parameters
    ----------
    pars: <SimulationParameters class>
        class of simulation parameters
    Returns
    -------
    network: the FiringRateController() class
    """

    network = AbstractOscillatorController(pars)

    # Run network ODE
    _iterator = (
        tqdm(range(network.n_iterations-1))
        if network.pars.show_progress
        else range(network.n_iterations-1)
    )

    for i in _iterator:

        if network.pars.entraining_signals is None:
            pos = None
        else:
            pos = network.pars.entraining_signals[i]

        network.step_euler(
            iteration=i,
            timestep=network.timestep,
            pos=pos,
        )

    # Compute metrics
    if pars.compute_metrics in ['neural', 'all']:
        network.metrics = compute_neural_metrics(network)
    if pars.print_metrics:
        pretty(network.metrics)

    # Save
    if pars.log_path != "":
        os.makedirs(pars.log_path, exist_ok=True)
        save_object(
            network,
            '{}controller{}'.format(
                pars.log_path,
                pars.simulation_i))

    return network


def run_multiple(pars_list, num_process=6):
    """
    Run multiple simulation in parallel
    Parameters
    ----------
    pars_list: list of <SimulationParameters class>
        list of of simulation parameter classes
    Returns
    -------
    network_list: list of AbstractOscillatorController() classes
    """
    return sweep_1d(run_single, pars_list, num_process=num_process)

