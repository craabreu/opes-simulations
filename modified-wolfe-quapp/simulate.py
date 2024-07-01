import argparse
import multiprocessing as mp
from functools import partial

import numpy as np
import openmm as mm
from openmm import app, unit

from openmmopes import OPES


def write(file, *args):
    file.write(",".join([str(x) for x in args]) + "\n")


def run_opes(index: int, method: str, nstep: int = 200000000) -> None:

    if method not in ["metad", "opes", "opes-explore"]:
        raise ValueError("Invalid method")

    modified_wolfe_quapp_potential = (
        "1.34549*x^4"
        "+ 1.90211*x^3*y"
        "+ 3.92705*x^2*y^2"
        "- 6.44246*x^2"
        "- 1.90211*x*y^3"
        "+ 5.58721*x*y"
        "+ 1.33481*x"
        "+ 1.34549*y^4"
        "- 5.55754*y^2"
        "+ 0.904586*y"
        "+ 18.5598"
    )

    mass = 1.0
    tstep = 0.005
    kb = unit.MOLAR_GAS_CONSTANT_R.value_in_unit_system(unit.md_unit_system)
    temperature = 1.0 / kb
    friction = 10.0
    reference_position = (-1.88, 0.784)

    height = 1
    pace = 500
    sigma = 0.1
    bias_factor = 10
    variance_pace = 25

    grid_min = -3
    grid_max = 3
    grid_bin = 151

    system = mm.System()
    system.addParticle(mass)
    force = mm.CustomExternalForce(modified_wolfe_quapp_potential)
    force.addParticle(0, [])
    system.addForce(force)

    bias_force = mm.CustomExternalForce("x")
    bias_force.addParticle(0, [])

    bias_variable = app.BiasVariable(
        bias_force, grid_min, grid_max, sigma, False, grid_bin
    )

    explore = method == "opes-explore"
    if method.startswith("opes"):
        sampler = OPES(
            system,
            [bias_variable],
            temperature,
            bias_factor,
            pace,
            explore,
            variance_pace,
        )
    elif method == "metad":
        sampler = app.Metadynamics(
            system, [bias_variable], temperature, bias_factor, height, pace
        )
    else:
        raise ValueError("Invalid method")

    topology = app.Topology()
    topology.addAtom("ATOM", None, topology.addResidue("MOL", topology.addChain()))

    platform = mm.Platform.getPlatformByName("Reference")
    integrator = mm.LangevinMiddleIntegrator(temperature, friction, tstep)
    simulation = app.Simulation(topology, system, integrator, platform)
    context = simulation.context
    initial_position = np.random.normal(reference_position, sigma, 2)
    context.setPositions([(*initial_position, 0)])
    context.setVelocitiesToTemperature(temperature)

    num_cycles = nstep // pace
    filename = f"{method}_{index:02d}.csv"
    percentage = 0
    n = 75
    z = 0
    with open(filename, "w", encoding="utf-8") as file:
        file.write("step,x,y,variance,delta_f,z\n")
        print("step, x, y, variance, delta_f, z, percentage")
        for cycle in range(num_cycles):
            sampler.step(simulation, pace)
            state = context.getState(getPositions=True)
            positions = state.getPositions(asNumpy=True).value_in_unit(unit.nanometer)
            x, y, _ = positions.flatten()
            var = sigma**2 if method == "metad" else sampler.getVariance().item()
            fes = sampler.getFreeEnergy()
            delta_f = np.logaddexp.reduce(-fes[:n]) - np.logaddexp.reduce(-fes[n:])
            if method != "metad":
                z = np.exp(-sampler._KDE.getLogMeanInvDensity().item())
            write(file, cycle * pace, *map(np.float32, [x, y, var, delta_f, z]))
            if (cycle + 1) % (num_cycles // 100) == 0:
                percentage += 1
                print(index, cycle + 1, x, y, var, delta_f, z, f"{percentage}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run OPES workflow")
    parser.add_argument("method", type=str, help="Method to use in the workflow")
    parser.add_argument(
        "--steps", type=int, default=200000000, help="Number of steps to run"
    )
    parser.add_argument(
        "--np", type=int, default=1, help="Total number of parallel processes"
    )
    args = parser.parse_args()

    if args.np == 1:
        run_opes(0, args.method, args.steps)
    else:
        parallel_run_opes = partial(run_opes, method=args.method, nstep=args.steps)
        with mp.Pool(processes=args.np) as pool:
            pool.map(parallel_run_opes, range(args.np))

    print("Done!")
