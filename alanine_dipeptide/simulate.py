import os

from timeit import default_timer as timer

import gzip
import numpy as np
import openmm as mm
from openmm import app, unit

from opes import OPES

VARIANCE_PACE: int = 20


def alanine_dipeptide(index: int, method: str, directory: str = ".") -> float:
    start = timer()
    if method not in ["metad", "opes", "opes-explore"]:
        raise ValueError("Invalid method")

    pdb = app.PDBFile("alanine_dipeptide.pdb")
    force_field = app.ForceField("amber14-all.xml")
    system = force_field.createSystem(
        pdb.topology, nonbondedMethod=app.NoCutoff, constraints=mm.app.HBonds
    )
    atoms = [(a.name, a.residue.name) for a in pdb.topology.atoms()]
    phi_atoms = [("C", "ACE"), ("N", "ALA"), ("CA", "ALA"), ("C", "ALA")]
    psi_atoms = [("N", "ALA"), ("CA", "ALA"), ("C", "ALA"), ("N", "NME")]

    total_time = 40 * unit.nanoseconds
    temp = 300 * unit.kelvin
    friction = 1 / unit.picosecond
    dt = 0.004 * unit.picoseconds
    nstep = np.rint(total_time / dt).astype(int)

    height = 1.2 * unit.kilojoules_per_mole
    bias_width = 0.15 * unit.radians
    barrier = 50 * unit.kilojoules_per_mole
    bias_factor = 10
    pace = 200
    grid_width = 101
    limits = (-np.pi * unit.radian, np.pi * unit.radian)
    variance_pace = VARIANCE_PACE

    cvs = []
    for torsion in (phi_atoms, psi_atoms):
        force = mm.CustomTorsionForce("theta")
        force.addTorsion(*[atoms.index(atom) for atom in torsion], [])
        cvs.append(app.BiasVariable(force, *limits, bias_width, True, grid_width))

    explore = method == "opes-explore"
    if method.startswith("opes"):
        sampler = OPES(
            system, cvs, temp, barrier, pace, variance_pace, bias_factor, explore
        )
    elif method == "metad":
        sampler = app.Metadynamics(system, cvs, temp, bias_factor, height, pace)
    else:
        raise ValueError("Invalid method")

    platform = mm.Platform.getPlatformByName("Reference")
    integrator = mm.LangevinMiddleIntegrator(temp, friction, dt)
    simulation = app.Simulation(pdb.topology, system, integrator, platform)
    context = simulation.context
    context.setPositions(pdb.positions)
    context.setVelocitiesToTemperature(temp)

    num_cycles = nstep // pace
    os.makedirs(directory, exist_ok=True)
    filename = f"{directory}/{method}_{index:02d}.csv.gz"
    percentage = 0
    var = [(bias_width / unit.radian) ** 2] * 2
    tstep = dt / unit.nanoseconds
    with gzip.open(filename, "wt") as file:
        file.write("time,phi,psi,varphi,varpsi\n")
        print("proc, time, phi, psi, varphi, varpsi, percentage")
        for cycle in range(num_cycles):
            sampler.step(simulation, pace)
            time = (cycle + 1) * pace * tstep
            angles = sampler.getCollectiveVariables(simulation)
            if method != "metad":
                var = sampler.getVariance().tolist()
            values = tuple(map(np.float32, [time, *angles, *var]))
            file.write(",".join(map(str, values)) + "\n")
            if (cycle + 1) % (num_cycles // 100) == 0:
                percentage += 1
                print(index, *values, f"{percentage}%")

    return timer() - start
