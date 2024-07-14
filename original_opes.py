import os
import pickle
import re
import warnings

import numpy as np
import openmm as mm
from openmm import unit
from openmm.app.metadynamics import _LoadedBias

from online_kde import CVSpace, OnlineKDE

STATS_WINDOW_SIZE = 10
UNCORRECTED_OPES_EXPLORE = False
REWEIGHTED_FES = True


class OPES:
    """Performs On-the-fly Probability Enhanced Sampling (OPES).

    This class implements the On-the-fly Probability Enhanced Sampling (OPES) method,
    as described in two papers by Invernizzi and Parrinello:

    1. Rethinking Metadynamics: From Bias Potentials to Probability Distributions
       https://doi.org/10.1021/acs.jpclett.0c00497

    2. Exploration vs Convergence Speed in Adaptive-Bias Enhanced Sampling
       https://doi.org/10.1021/acs.jctc.2c00152

    Parameters
    ----------
    system
        The System to simulate. A CustomCVForce implementing the bias is created and
        added to the System.
    variables
        The collective variables whose sampling should be enhanced.
    temperature
        The temperature at which the simulation will be run.
    barrier
        The energy barrier that the simulation should overcome.
    frequency
        The interval in time steps at which to add a new kernel to the probability
        distribution estimate.
    exploreMode
        Whether to apply the OPES-Explore variant.
    """

    def __init__(
        self,
        system,
        variables,
        temperature,
        barrier,
        frequency,
        exploreMode=False,
        varianceFrequency=None,
        saveFrequency=None,
        biasDir=None,
    ):
        if not unit.is_quantity(temperature):
            temperature = temperature * unit.kelvin
        if not unit.is_quantity(barrier):
            barrier = barrier * unit.kilojoules_per_mole
        self.variables = variables
        self.temperature = temperature
        self.barrier = barrier
        self.frequency = frequency
        self.exploreMode = exploreMode
        self.varianceFrequency = varianceFrequency
        self.saveFrequency = saveFrequency
        self.biasDir = biasDir

        d = len(variables)
        kbt = unit.MOLAR_GAS_CONSTANT_R * temperature
        biasFactor = barrier / kbt
        numPeriodics = sum(v.periodic for v in variables)
        freeGroups = set(range(32)) - set(f.getForceGroup() for f in system.getForces())
        self._validate(d, biasFactor, numPeriodics, freeGroups)

        prefactor = (1 - 1 / biasFactor) * kbt
        if exploreMode:
            prefactor *= biasFactor
        self._kbt = kbt.in_units_of(unit.kilojoules_per_mole)
        self._biasFactor = biasFactor
        self._prefactor = prefactor / unit.kilojoules_per_mole
        self._logEpsilon = -barrier / prefactor

        self._kde = {}
        varScale = 1.0 / biasFactor
        self._cvSpace = CVSpace(variables)
        self._cases = ("total",) + ("self",) * bool(saveFrequency)
        for case in self._cases:
            self._kde[case] = OnlineKDE(self._cvSpace, 1.0 if exploreMode else varScale)
            if exploreMode and REWEIGHTED_FES:
                self._kde[case + "_reweighted"] = OnlineKDE(self._cvSpace, varScale)
        if saveFrequency:
            self._id = np.random.randint(0x7FFFFFFF)
            self._saveIndex = 0
            self._loadedBiases = {}
            self._syncWithDisk()

        self._adaptiveVariance = varianceFrequency is not None
        self._interval = varianceFrequency or frequency
        if self._adaptiveVariance:
            self._tau = STATS_WINDOW_SIZE * frequency // varianceFrequency
            self._counter = 0
            self._sampleMean = np.zeros(d)
        else:
            sqdev = np.array([cv.biasWidth**2 for cv in variables])
            for kde in self._kde.values():
                kde.updateVariance(sqdev)

        gridWidths = [cv.gridWidth for cv in variables]
        self._widths = [] if d == 1 else gridWidths
        self._limits = sum(([cv.minValue, cv.maxValue] for cv in variables), [])

        energyFunction = "table(" + ",".join(f"cv{i}" for i in range(d)) + ")"
        self._force = mm.CustomCVForce(energyFunction)
        for i, var in enumerate(variables):
            self._force.addCollectiveVariable(f"cv{i}", var.force)
        table = getattr(mm, f"Continuous{d}DFunction")(
            *self._widths,
            np.full(np.prod(gridWidths), -barrier / unit.kilojoules_per_mole),
            *self._limits,
            numPeriodics == d,
        )
        self._force.addTabulatedFunction("table", table)
        self._force.setForceGroup(max(freeGroups))
        system.addForce(self._force)

    def _validate(self, d, biasFactor, numPeriodics, freeGroups):
        if self.varianceFrequency and (self.frequency % self.varianceFrequency != 0):
            raise ValueError("varianceFrequency must be a divisor of frequency")
        if (self.saveFrequency is None) != (self.biasDir is None):
            raise ValueError("Must specify both saveFrequency and biasDir")
        if self.saveFrequency and (self.saveFrequency % self.frequency != 0):
            raise ValueError("saveFrequency must be a multiple of frequency")
        if biasFactor <= 1.0:
            raise ValueError("OPES barrier must be greater than 1 kT")
        if numPeriodics not in [0, d]:
            raise ValueError("OPES cannot handle mixed periodic/non-periodic variables")
        if not 1 <= d <= 3:
            raise ValueError("OPES requires 1, 2, or 3 collective variables")
        if not freeGroups:
            raise RuntimeError("OPES requires a free force group, but all are in use.")

    def _getBias(self):
        kde = self._kde["total"]
        return self._prefactor * np.logaddexp(
            kde.getLogPDF() - kde.getLogMeanDensity(), self._logEpsilon
        )

    def _updateSampleStats(self, values):
        """Update the sample mean and variance of the collective variables."""
        self._counter += 1
        delta = self._cvSpace.displacement(self._sampleMean, values)
        x = 1 / min(self._tau, self._counter)
        self._sampleMean = self._cvSpace.endpoint(self._sampleMean, x * delta)
        sqdev = delta * self._cvSpace.displacement(self._sampleMean, values)
        for kde in self._kde.values():
            kde.updateVariance(sqdev)

    def _addKernel(self, values, energy, context):
        """Add a kernel to the PDF estimate and update the bias potential."""
        logWeight = energy / self._kbt
        for case in self._cases:
            self._kde[case].update(values, 0 if self.exploreMode else logWeight)
            if self.exploreMode and REWEIGHTED_FES:
                self._kde[f"{case}_reweighted"].update(values, logWeight)
        self._force.getTabulatedFunction(0).setFunctionParameters(
            *self._widths, self._getBias().ravel(), *self._limits
        )
        self._force.updateParametersInContext(context)

    def _syncWithDisk(self):
        """
        Save biases to disk, and check for updated files created by other processes.
        """

        oldName = os.path.join(self.biasDir, f"kde_{self._id}_{self._saveIndex}.pkl")
        self._saveIndex += 1
        tempName = os.path.join(self.biasDir, f"temp_{self._id}_{self._saveIndex}.pkl")
        fileName = os.path.join(self.biasDir, f"kde_{self._id}_{self._saveIndex}.pkl")
        data = {"self": self._kde["self"]}
        if self.exploreMode and REWEIGHTED_FES:
            data["self_reweighted"] = self._kde["self_reweighted"]
        with open(tempName, "wb") as file:
            pickle.dump(data, file)
        os.rename(tempName, fileName)
        if os.path.exists(oldName):
            os.remove(oldName)

        fileLoaded = False
        pattern = re.compile(r"kde_(.*)_(.*)\.pkl")
        for filename in os.listdir(self.biasDir):
            match = pattern.match(filename)
            if match is not None:
                matchId = int(match.group(1))
                matchIndex = int(match.group(2))
                if matchId != self._id and (
                    matchId not in self._loadedBiases
                    or matchIndex > self._loadedBiases[matchId].index
                ):
                    try:
                        with open(os.path.join(self.biasDir, filename), "rb") as file:
                            data = pickle.load(file)
                        self._loadedBiases[matchId] = _LoadedBias(
                            matchId, matchIndex, data
                        )
                        fileLoaded = True
                    except IOError:
                        warnings.warn(
                            f"The file {filename} seems to have been deleted. Using"
                            "the lastest loaded data from the same process."
                        )

        if fileLoaded:
            self._kde["total"] = self._kde["self"].copy()
            if self.exploreMode and REWEIGHTED_FES:
                self._kde["total_reweighted"] = self._kde["self_reweighted"].copy()
            for bias in self._loadedBiases.values():
                self._kde["total"] += bias.bias["self"]
                if self.exploreMode and REWEIGHTED_FES:
                    self._kde["total_reweighted"] += bias.bias["self_reweighted"]

    def getFreeEnergy(self):
        """
        Get the free energy of the system as a function of the collective variables.

        The result is returned as a N-dimensional NumPy array, where N is the number
        of collective variables. The values are in kJ/mole. The i'th position along an
        axis corresponds to minValue + i*(maxValue-minValue)/gridWidth.
        """
        if self.exploreMode and REWEIGHTED_FES:
            return -self._kbt * self._kde["total_reweighted"].getLogPDF()
        freeEnergy = -self._kbt * self._kde["total"].getLogPDF()
        if not self.exploreMode:
            return freeEnergy
        if UNCORRECTED_OPES_EXPLORE:
            return freeEnergy * self._biasFactor
        return freeEnergy - self._getBias() * unit.kilojoules_per_mole

    def getAverageDensity(self):
        """
        Get the average density of the system as a function of the collective variables.
        """
        return np.exp(self._kde["total"].getLogMeanDensity())

    def getCollectiveVariables(self, simulation):
        """
        Get the current values of all collective variables in a Simulation.

        Parameters
        ----------
        simulation
            The Simulation to query.
        """
        return self._force.getCollectiveVariableValues(simulation.context)

    def step(self, simulation, steps):
        """
        Advance the simulation by integrating a specified number of time steps.

        Parameters
        ----------
        simulation:
            The Simulation to advance.
        steps
            The number of time steps to integrate.
        """
        stepsToGo = steps
        groups = {self._force.getForceGroup()}
        while stepsToGo > 0:
            nextSteps = stepsToGo
            nextSteps = min(
                nextSteps, self._interval - simulation.currentStep % self._interval
            )
            simulation.step(nextSteps)
            if simulation.currentStep % self._interval == 0:
                position = self.getCollectiveVariables(simulation)
                if self._adaptiveVariance:
                    self._updateSampleStats(position)
                if simulation.currentStep % self.frequency == 0:
                    state = simulation.context.getState(getEnergy=True, groups=groups)
                    energy = state.getPotentialEnergy()
                    self._addKernel(position, energy, simulation.context)
                    if (
                        self.saveFrequency is not None
                        and simulation.currentStep % self.saveFrequency == 0
                    ):
                        self._syncWithDisk()
            stepsToGo -= nextSteps

    def setVariance(self, variance):
        """
        Set the variance of the probability distribution estimate.

        Parameters
        ----------
        variance
            The variance of the probability distribution estimate.
        """
        self._sampleVariance = variance

    def getVariance(self):
        """Get the variance of the probability distribution estimate."""
        return self._kde["total"].getVariance()
