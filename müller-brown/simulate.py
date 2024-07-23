import os
from timeit import default_timer as timer

import gzip
import numpy as np
import openmm as mm
from openmm import app, unit

from opes import OPES

VARIANCE_PACE: int = 50
KB: float = unit.MOLAR_GAS_CONSTANT_R.value_in_unit_system(unit.md_unit_system)


class LangevinIntegrator(mm.CustomIntegrator):
    def __init__(self, temperature, friction, tstep):
        super().__init__(tstep)
        lscale = np.exp(-0.5 * tstep * friction)
        self.addGlobalVariable("lscale", lscale)
        self.addGlobalVariable("lrand", np.sqrt((1.0 - lscale**2) * KB * temperature))
        self.addComputePerDof("v", "lscale*v + lrand*gaussian + 0.5*dt*f/m")
        self.addComputePerDof("x", "x + dt*v")
        self.addComputePerDof("v", "lscale*(v + 0.5*dt*f/m) + lrand*gaussian")


def muller_brown(index: int, method: str, directory: str = ".") -> float:
    start = timer()
    if method not in ["metad", "opes", "opes-explore"]:
        raise ValueError("Invalid method")

    muller_brown_potential_str = (
        "0.15*("
        "  -200*exp(-(x-1)^2-10*y^2)"
        "  -100*exp(-x^2-10*(y-0.5)^2)"
        "  -170*exp(-6.5*(0.5+x)^2+11*(x+0.5)*(y-1.5)-6.5*(y-1.5)^2)"
        "  +15*exp(0.7*(1+x)^2+0.6*(x+1)*(y-1)+0.7*(y-1)^2)"
        "  +146.7"
        ")"
    )

    wall_kappa = 1000
    xmin = -1.3
    xmax = 1.0
    wall_potential_str = (
        f"{wall_kappa}*((step(dmin)*dmin)^2+(step(dmax)*dmax)^2)"
        f";dmin={xmin}-x"
        f";dmax=x-{xmax}"
    )

    nstep = 200000000
    mass = 1.0
    tstep = 0.005
    temperature = 1.0 / KB
    friction = 10.0
    reference_position = (-0.5582, 1.442)

    height = 1
    pace = 500
    sigma = 0.1
    bias_factor = 20
    variance_pace = VARIANCE_PACE

    grid_min = -2
    grid_max = 2
    grid_bin = 201

    system = mm.System()
    system.addParticle(mass)
    force = mm.CustomExternalForce(f"{muller_brown_potential_str}+{wall_potential_str}")
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
    integrator = LangevinIntegrator(temperature, friction, tstep)
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
            values = tuple(map(np.float32, [time, x, y, var, z, n, delta_f]))
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
