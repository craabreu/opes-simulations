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
CORRECTED_OPES_EXPLORE = True
USE_PDF_OPES_EXPLORE = True
REWEIGHTED_FES = True


class RunningAverage:
    """Class to handle running average calculations."""

    def __init__(self, numDimensions=None):
        self._numSamples = 0
        self._sumSamples = 0 if numDimensions is None else np.zeros(numDimensions)

    def __getstate__(self):
        return {"numSamples": self._numSamples, "sumSamples": self._sumSamples}

    def __setstate__(self, state):
        self._numSamples = state["numSamples"]
        self._sumSamples = state["sumSamples"]

    def __iadd__(self, other):
        self._sumSamples += other._sumSamples
        self._numSamples += other._numSamples
        return self

    def copy(self):
        d = None if np.isscalar(self._sumSamples) else self._sumSamples.shape[0]
        new = self.__class__(d)
        new._numSamples = self._numSamples
        new._sumSamples = self._sumSamples
        return new

    def update(self, sample):
        """Update the running average with a new sample."""
        self._numSamples += 1
        self._sumSamples += sample

    def get(self):
        """Get the running average."""
        if self._numSamples == 0:
            return np.zeros_like(self._sumSamples)
        return self._sumSamples / self._numSamples


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
    varianceFrequency
        The interval in time steps at which to update the variance of the collective
        variables.
    biasFactor
        The bias factor to use. If None, then barrier / kT is used.
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
        varianceFrequency,
        biasFactor=None,
        exploreMode=False,
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
        self.varianceFrequency = varianceFrequency
        self.biasFactor = biasFactor
        self.exploreMode = exploreMode
        self.saveFrequency = saveFrequency
        self.biasDir = biasDir

        d = len(variables)
        kbt = unit.MOLAR_GAS_CONSTANT_R * temperature
        biasFactor = barrier / kbt if biasFactor is None else biasFactor
        numPeriodics = sum(v.periodic for v in variables)
        freeGroups = set(range(32)) - set(f.getForceGroup() for f in system.getForces())
        self._validate(d, biasFactor, numPeriodics, freeGroups)

        prefactor = (1 - 1 / biasFactor) * kbt
        if exploreMode:
            prefactor *= biasFactor
        self._kbt = kbt.in_units_of(unit.kilojoules_per_mole)
        self._biasFactor = biasFactor
        self._prefactor = prefactor
        self._logEpsilon = -barrier / prefactor

        varScale = 1.0 / biasFactor
        self._cvSpace = CVSpace(variables)
        self._cases = ("total",) + ("self",) * bool(saveFrequency)
        self._kde = {}
        for case in self._cases:
            if exploreMode:
                self._kde[case] = OnlineKDE(self._cvSpace, 1.0, False)
                if REWEIGHTED_FES:
                    self._kde[f"{case}_rw"] = OnlineKDE(self._cvSpace, varScale, True)
            else:
                self._kde[case] = OnlineKDE(self._cvSpace, varScale, True)

        self._adaptiveVariance = varianceFrequency is not None
        self._interval = varianceFrequency or frequency
        self._variance = {case: RunningAverage(d) for case in self._cases}
        if self._adaptiveVariance:
            self._tau = STATS_WINDOW_SIZE * frequency // varianceFrequency
            self._counter = 0
            self._sampleMean = np.zeros(d)
        else:
            sqdev = np.array([cv.biasWidth**2 for cv in variables])
            for case in self._cases:
                self._variance[case].update(sqdev)

        if saveFrequency:
            self._id = np.random.RandomState().randint(0x7FFFFFFF)
            self._saveIndex = 0
            self._loadedBiases = {}
            self._syncWithDisk()

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

    def _updateSampleStats(self, values):
        """Update the sample mean and variance of the collective variables."""
        self._counter += 1
        delta = self._cvSpace.displacement(self._sampleMean, values)
        x = 1 / min(self._tau, self._counter)
        self._sampleMean = self._cvSpace.endpoint(self._sampleMean, x * delta)
        sqdev = delta * self._cvSpace.displacement(self._sampleMean, values)
        for case in self._cases:
            self._variance[case].update(sqdev)

    def _syncWithDisk(self):
        """
        Save biases to disk, and check for updated files created by other processes.
        """

        oldName = os.path.join(self.biasDir, f"kde_{self._id}_{self._saveIndex}.pkl")
        self._saveIndex += 1
        tempName = os.path.join(self.biasDir, f"temp_{self._id}_{self._saveIndex}.pkl")
        fileName = os.path.join(self.biasDir, f"kde_{self._id}_{self._saveIndex}.pkl")
        data = {"kde": self._kde["self"], "var": self._variance["self"]}
        if self.exploreMode and REWEIGHTED_FES:
            data["kde_rw"] = self._kde["self_rw"]
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
            self._variance["total"] = self._variance["self"].copy()
            if self.exploreMode and REWEIGHTED_FES:
                self._kde["total_rw"] = self._kde["self_rw"].copy()
            for bias in self._loadedBiases.values():
                self._kde["total"] += bias.bias["kde"]
                self._variance["total"] += bias.bias["var"]
                if self.exploreMode and REWEIGHTED_FES:
                    self._kde["total_rw"] += bias.bias["kde_rw"]

    def getNumKernels(self):
        """Get the number of kernels in the kernel density estimator."""
        return self._kde["total"].getNumKernels()

    def getBias(self):
        """Get the OPES bias potential evaluated on the grid."""
        kde = self._kde["total"]
        return self._prefactor * np.logaddexp(
            kde.getLogPDF() - kde.getLogMeanDensity(), self._logEpsilon
        )

    def getFreeEnergy(self):
        """
        Get the free energy of the system as a function of the collective variables.

        The result is returned as a N-dimensional NumPy array, where N is the number
        of collective variables. The values are in kJ/mole. The i'th position along an
        axis corresponds to minValue + i*(maxValue-minValue)/gridWidth.
        """
        if self.exploreMode and REWEIGHTED_FES:
            return -self._kbt * self._kde["total_rw"].getLogPDF()
        freeEnergy = -self._kbt * self._kde["total"].getLogPDF()
        if not self.exploreMode:
            return freeEnergy
        if not CORRECTED_OPES_EXPLORE:
            if not USE_PDF_OPES_EXPLORE:
                return -self._kbt * self.getBias() * self._biasFactor / self._prefactor
            return freeEnergy * self._biasFactor
        return (
            freeEnergy
            - self.getBias()
            + self._kde["total"].getLogNormConstRatio() * unit.kilojoules_per_mole
        )

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

    def updateContext(self, context):
        """Update the collective variables in the context."""
        bias = self.getBias().value_in_unit(unit.kilojoules_per_mole)
        self._force.getTabulatedFunction(0).setFunctionParameters(
            *self._widths, bias.ravel(), *self._limits
        )
        self._force.updateParametersInContext(context)

    def addKernel(self, values, biasEnergy, variance=None):
        """Add a kernel to the PDF estimate and update the bias potential."""
        if not isinstance(biasEnergy, unit.Quantity):
            biasEnergy = biasEnergy * unit.kilojoules_per_mole
        if variance is None:
            variance = self._variance["total"].get()
        logWeight = biasEnergy / self._kbt
        for kde in self._kde.values():
            kde.update(values, logWeight, variance)

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
                    self.addKernel(position, energy)
                    self.updateContext(simulation.context)
                    if (
                        self.saveFrequency is not None
                        and simulation.currentStep % self.saveFrequency == 0
                    ):
                        self._syncWithDisk()
            stepsToGo -= nextSteps

    def getVariance(self):
        """Get the variance of the probability distribution estimate."""
        return self._variance["total"].get()
