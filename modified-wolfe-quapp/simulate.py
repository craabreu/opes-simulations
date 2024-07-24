import os
from timeit import default_timer as timer

import gzip
import numpy as np
import openmm as mm
import pandas as pd
from openmm import app, unit

import opes

VARIANCE_PACE: int = 50


def compute_delta_f(sampler, half):
    fes = sampler.getFreeEnergy() / unit.kilojoules_per_mole
    return np.logaddexp.reduce(-fes[:half]) - np.logaddexp.reduce(-fes[half:])


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
        sampler = opes.OPES(
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
        titles = ["delta_f"]
        if method == "opes-explore":
            titles += [
                "delta_f_unreweighted",
                "delta_f_uncorrected",
                "delta_f_from_bias",
            ]
        file.write(f"time,x,y,variance,z,n,{','.join(titles)}\n")
        print(f"proc, time, x, y, variance, z, n, {', '.join(titles)}, percentage")
        for cycle in range(num_cycles):
            sampler.step(simulation, pace)
            time = (cycle + 1) * pace * tstep
            state = context.getState(getPositions=True)
            positions = state.getPositions(asNumpy=True).value_in_unit(unit.nanometer)
            x, y, _ = positions.flatten()
            delta_f = [compute_delta_f(sampler, half)]
            if method == "opes-explore":
                opes.REWEIGHTED_FES = False
                delta_f += [compute_delta_f(sampler, half)]
                opes.CORRECTED_OPES_EXPLORE = False
                delta_f += [compute_delta_f(sampler, half)]
                opes.USE_PDF_OPES_EXPLORE = False
                delta_f += [compute_delta_f(sampler, half)]
                opes.REWEIGHTED_FES = True
                opes.CORRECTED_OPES_EXPLORE = True
                opes.USE_PDF_OPES_EXPLORE = True
            if method != "metad":
                z = sampler.getAverageDensity()
                n = sampler.getNumKernels()
                var = sampler.getVariance().item()
            values = tuple(map(np.float32, [time, x, y, var, z, n, *delta_f]))
            file.write(",".join(map(str, values)) + "\n")
            if (cycle + 1) % (num_cycles // 100) == 0:
                percentage += 1
                print(index, *values, f"{percentage}%")

    df = pd.DataFrame({"x": np.linspace(grid_min, grid_max, grid_bin)})
    fes = sampler.getFreeEnergy() / unit.kilojoules_per_mole
    df["fes"] = fes - fes.min()
    if method == "opes-explore":
        opes.REWEIGHTED_FES = False
        fes = sampler.getFreeEnergy() / unit.kilojoules_per_mole
        df["fes_unreweighted"] = fes - fes.min()
        opes.CORRECTED_OPES_EXPLORE = False
        fes = sampler.getFreeEnergy() / unit.kilojoules_per_mole
        df["fes_uncorrected"] = fes - fes.min()
        opes.USE_PDF_OPES_EXPLORE = False
        fes = sampler.getFreeEnergy() / unit.kilojoules_per_mole
        df["fes_from_bias"] = fes - fes.min()
    df.to_csv(f"{directory}/profile_{method}_{index:02d}.csv", index=False)

    return timer() - start
