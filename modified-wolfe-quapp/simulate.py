import os
from timeit import default_timer as timer

import gzip
import numpy as np
import openmm as mm
from openmm import app, unit

from opes import OPES

VARIANCE_PACE: int = 50


def modified_wolfe_quapp(index: int, method: str, directory: str = ".") -> float:
    start = timer()
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

    nstep = 30000000
    mass = 1.0
    tstep = 0.005
    kb = unit.MOLAR_GAS_CONSTANT_R.value_in_unit_system(unit.md_unit_system)
    temperature = 1.0 / kb
    friction = 10.0
    reference_position = (-1.88, 0.784)

    height = 1
    pace = 500
    sigma = 0.185815
    bias_factor = 10
    variance_pace = VARIANCE_PACE

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
            variance_pace,
            exploreMode=explore,
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
    os.makedirs(directory, exist_ok=True)
    filename = f"{directory}/{method}_{index:02d}.csv.gz"
    percentage = 0
    half = grid_bin // 2
    z = n = 0
    var = sigma**2
    with gzip.open(filename, "wt") as file:
        file.write("time,x,y,variance,z,n,delta_f\n")
        print("proc, time, x, y, variance, z, n, delta_f, percentage")
        for cycle in range(num_cycles):
            sampler.step(simulation, pace)
            time = (cycle + 1) * pace * tstep
            state = context.getState(getPositions=True)
            positions = state.getPositions(asNumpy=True).value_in_unit(unit.nanometer)
            x, y, _ = positions.flatten()
            fe = sampler.getFreeEnergy() / unit.kilojoules_per_mole
            delta_f = np.logaddexp.reduce(-fe[:half]) - np.logaddexp.reduce(-fe[half:])
            if method != "metad":
                z = sampler.getAverageDensity()
                n = sampler.getNumKernels()
                var = sampler.getVariance().item()
            values = tuple(
                map(np.float32, [time, x, y, var, z, n, delta_f])
            )
            file.write(",".join(map(str, values)) + "\n")
            if (cycle + 1) % (num_cycles // 100) == 0:
                percentage += 1
                print(index, *values, f"{percentage}%")

    filename = f"{directory}/profile_{method}_{index:02d}.csv"
    fes = sampler.getFreeEnergy() / unit.kilojoules_per_mole
    fes -= fes.min()
    with open(filename, "w", encoding="utf-8") as file:
        file.write("x,fes\n")
        for x, f in zip(np.linspace(grid_min, grid_max, grid_bin), fes):
            file.write(",".join(map(str, [x, f])) + "\n")

    return timer() - start