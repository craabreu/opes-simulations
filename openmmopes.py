"""
.. module:: opes
   :platform: Linux, MacOS
   :synopsis: On-the-fly Probability Enhanced Sampling with OpenMM

.. moduleauthor:: Charlles Abreu <craabreu@gmail.com>
"""

import functools
import os
import pickle
import re

import numpy as np
import openmm as mm
from openmm import unit
from openmm.app.metadynamics import _LoadedBias

LOG2PI = np.log(2 * np.pi)
STATS_WINDOW: int = 10
GLOBAL_VARIANCE: bool = True


class KernelDensityEstimate:
    """
    A kernel density estimate on a grid.

    Parameters
    ----------
    shape
        The shape of the grid.

    Attributes
    ----------
    d
        The number of dimensions.
    shape
        The shape of the grid.
    counter
        The number of times the kernel density estimate has been updated.
    logSumW
        The logarithm of the sum of the weights.
    logSumW2
        The logarithm of the sum of the squared weights.
    logAccInvDens
        The logarithm of the accumulated inverse probability density.
    logSumWID
        The logarithm of the sum of the weights added to the accumulated inverse
        probability density.
    logAccGaussian
        The logarithm of the accumulated Gaussians on a grid.
    """

    def __init__(self, shape):
        self.d = len(shape)
        self.shape = np.array(shape)
        self.counter = 0
        self.logSumW = self.logSumW2 = self.logSumWID = self.logAccInvDens = -np.inf
        self.logAccGaussian = np.full(np.flip(shape), -np.inf)

    def __iadd__(self, other):
        self.logSumW = np.logaddexp(self.logSumW, other.logSumW)
        self.logSumW2 = np.logaddexp(self.logSumW2, other.logSumW2)
        self.logSumWID = np.logaddexp(self.logSumWID, other.logSumWID)
        self.logAccInvDens = np.logaddexp(self.logAccInvDens, other.logAccInvDens)
        self.logAccGaussian = np.logaddexp(self.logAccGaussian, other.logAccGaussian)
        return self

    def copy(self):
        """Create a copy of the kernel density estimate."""
        result = KernelDensityEstimate(self.shape)
        result.counter = self.counter
        result.logSumW = self.logSumW
        result.logSumW2 = self.logSumW2
        result.logSumWID = self.logSumWID
        result.logAccInvDens = self.logAccInvDens
        result.logAccGaussian = np.copy(self.logAccGaussian)
        return result

    def getLogPDF(self):
        """
        Get the logarithm of the probability density function evaluated on the grid.
        """
        return self.logAccGaussian - self.logSumW

    def getLogMeanInvDensity(self):
        """
        Get the logarithm of the mean inverse probability density.
        """
        return self.logAccInvDens - self.logSumWID

    def getBias(self, prefactor, logEpsilon):
        """
        Get the bias potential evaluated on the grid.

        Parameters
        ----------
        prefactor
            The prefactor of the bias potential.
        logEpsilon
            The logarithm of the minimum value of the bias potential.
        """
        return prefactor * np.logaddexp(
            self.getLogMeanInvDensity() + self.getLogPDF(), logEpsilon
        )

    def update(self, logWeight, axisSquaredDistances, variances, indices):
        """
        Update the kernel density estimate with a new Gaussian kernel.

        Parameters
        ----------
        logWeight
            The logarithm of the weight assigned to the kernel.
        axisSquaredDistances
            The squared distances from the kernel center to the grid points along each
            axis.
        variances
            The variances of the variables along each axis.
        indices
            The indices of the grid point closest to the kernel center.
        """
        self.counter += 1
        self.logSumW = np.logaddexp(self.logSumW, logWeight)
        self.logSumW2 = np.logaddexp(self.logSumW2, 2 * logWeight)
        neff = np.exp(2 * self.logSumW - self.logSumW2)
        bandwidth = (neff * (self.d + 2) / 4) ** (-2 / (self.d + 4)) * variances
        exponents = [
            -0.5 * x2 / sigma2 for x2, sigma2 in zip(axisSquaredDistances, bandwidth)
        ]
        logHeight = logWeight - 0.5 * (self.d * LOG2PI + np.log(bandwidth).sum())
        logGaussian = logHeight + functools.reduce(np.add.outer, reversed(exponents))
        self.logAccGaussian = np.logaddexp(self.logAccGaussian, logGaussian)
        if np.any(indices < 0) or np.any(indices >= self.shape):
            return
        self.logSumWID = np.logaddexp(self.logSumWID, logWeight)
        logDensity = self.logAccGaussian[tuple(reversed(indices))] - self.logSumW
        self.logAccInvDens = np.logaddexp(self.logAccInvDens, logWeight - logDensity)


class OPES:
    """Performs OPES."""

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
        varNames = [f"cv{i}" for i in range(d)]

        self._d = d
        self._lengths = np.array([v.maxValue - v.minValue for v in variables])
        self._lbounds = np.array([v.minValue for v in variables])
        self._widths = np.array([v.gridWidth for v in variables])
        self._limits = sum(([v.minValue, v.maxValue] for v in variables), [])
        self._periodic = numPeriodics == d
        self._grid = [
            np.linspace(v.minValue, v.maxValue, v.gridWidth) for v in variables
        ]

        self._prefactor = prefactor.value_in_unit(unit.kilojoules_per_mole)
        self._logEpsilon = -barrier / prefactor
        self._biasFactor = biasFactor
        self._kbt = kbt.in_units_of(unit.kilojoules_per_mole)

        self._adaptiveVariance = varianceFrequency is not None
        self._interval = varianceFrequency or frequency
        self._tau = STATS_WINDOW * frequency // varianceFrequency
        self._sampleMean = np.zeros(d)
        self._sampleVariance = np.array([v.biasWidth**2 for v in variables])
        self._sampleCounter = 0

        self._KDE = self._rwKDE = KernelDensityEstimate(self._widths)
        if exploreMode:
            self._rwKDE = KernelDensityEstimate(self._widths)
        if saveFrequency:
            self._selfKDE = self._selfRwKDE = KernelDensityEstimate(self._widths)
            if exploreMode:
                self._selfRwKDE = KernelDensityEstimate(self._widths)

        self._id = np.random.randint(0x7FFFFFFF)
        self._saveIndex = 0
        self._loadedBiases = {}
        self._syncWithDisk()

        self._force = mm.CustomCVForce(f"table({', '.join(varNames)})")
        self._table = getattr(mm, f"Continuous{d}DFunction")(
            *(self._widths if d > 1 else []),
            np.full(np.prod(self._widths), -barrier / unit.kilojoules_per_mole),
            *self._limits,
            self._periodic,
        )
        for name, var in zip(varNames, variables):
            self._force.addCollectiveVariable(name, var.force)
        self._force.addTabulatedFunction("table", self._table)
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
        self._sampleCounter += 1
        delta = values - self._sampleMean
        if self._periodic:
            delta -= self._lengths * np.rint(delta / self._lengths)
        x = 1 / min(self._tau, self._sampleCounter)
        self._sampleMean += x * delta
        if self._periodic:
            self._sampleMean = (
                self._lbounds + (self._sampleMean - self._lbounds) % self._lengths
            )
        variance = (1 - x) * delta**2
        if GLOBAL_VARIANCE:
            x = 1 / (self._tau + self._sampleCounter)
        self._sampleVariance += x * (variance - self._sampleVariance)

    def step(self, simulation, steps):
        """Advance the simulation by integrating a specified number of time steps.

        Parameters
        ----------
        simulation: Simulation
            the Simulation to advance
        steps: int
            the number of time steps to integrate
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
                    self._addGaussian(position, energy, simulation.context)
                    if (
                        self.saveFrequency is not None
                        and simulation.currentStep % self.saveFrequency == 0
                    ):
                        self._syncWithDisk()
            stepsToGo -= nextSteps

    def getFreeEnergy(self, reweighted=False, original=False):
        """
        Get the free energy of the system as a function of the collective variables.

        The result is returned as a N-dimensional NumPy array, where N is the number of
        collective variables.  The values are in kJ/mole.  The i'th position along an
        axis corresponds to minValue + i*(maxValue-minValue)/gridWidth.
        """
        if reweighted or not self.exploreMode:
            return -self._kbt * self._rwKDE.getLogPDF()
        biasedFreeEnergy = -self._kbt * self._KDE.getLogPDF()
        if original:
            return biasedFreeEnergy * self._biasFactor
        biasPotential = self._KDE.getBias(self._prefactor, self._logEpsilon)
        variation = self._kbt * (self._rwKDE.logSumW - np.log(self._rwKDE.counter))
        return biasedFreeEnergy - biasPotential * unit.kilojoules_per_mole + variation

    def getAverageDensity(self):
        """
        Get the average density of the system as a function of the collective variables.
        """
        return np.exp(-self._KDE.getLogMeanInvDensity())

    def getCollectiveVariables(self, simulation):
        """Get the current values of all collective variables in a Simulation."""
        return self._force.getCollectiveVariableValues(simulation.context)

    def setVariance(self, variance) -> None:
        """
        Set the variance of the probability distribution estimate.

        Parameters
        ----------
        variance
            The variance of the probability distribution estimate.
        """
        self._sampleVariance = np.array(variance)

    def getVariance(self) -> np.ndarray:
        """
        Get the variance of the probability distribution estimate.
        """
        return self._sampleVariance

    def _addGaussian(self, values, energy, context):
        """Add a Gaussian to the bias function."""
        scaled = (np.array(values) - self._lbounds) / self._lengths
        if self._periodic:
            scaled %= 1
        indices = np.rint(scaled * (self._widths - 1)).astype(int)

        axisSqDistances = []
        for value, nodes, length in zip(values, self._grid, self._lengths):
            distances = nodes - value
            if self._periodic:
                distances -= length * np.rint(distances / length)
                distances[-1] = distances[0]
            axisSqDistances.append(distances**2)

        if self.exploreMode:
            self._KDE.update(0, axisSqDistances, self._sampleVariance, indices)
            if self.saveFrequency:
                self._selfKDE.update(0, axisSqDistances, self._sampleVariance, indices)

        logWeight = energy / self._kbt
        variance = self._sampleVariance / self._biasFactor
        self._rwKDE.update(logWeight, axisSqDistances, variance, indices)
        if self.saveFrequency:
            self._selfRwKDE.update(logWeight, axisSqDistances, variance, indices)

        self._table.setFunctionParameters(
            *(self._widths if self._d > 1 else []),
            self._KDE.getBias(self._prefactor, self._logEpsilon).ravel(),
            *self._limits,
        )
        self._force.updateParametersInContext(context)

    def _syncWithDisk(self):
        """
        Save biases to disk, and check for updated files created by other processes.
        """
        if self.biasDir is None:
            return

        # Use a safe save to write out the biases to disk, then delete the older file.

        oldName = os.path.join(self.biasDir, f"bias_{self._id}_{self._saveIndex}.pkl")
        self._saveIndex += 1
        tempName = os.path.join(self.biasDir, f"temp_{self._id}_{self._saveIndex}.pkl")
        fileName = os.path.join(self.biasDir, f"bias_{self._id}_{self._saveIndex}.pkl")
        data = [self._selfKDE]
        if self.exploreMode:
            data.append(self._selfRwKDE)
        with open(tempName, "wb") as file:
            pickle.dump(data, file)
        os.rename(tempName, fileName)
        if os.path.exists(oldName):
            os.remove(oldName)

        # Check for any files updated by other processes.

        fileLoaded = False
        pattern = re.compile(r"bias_(.*)_(.*)\.pkl")
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
                        # data = np.load(os.path.join(self.biasDir, filename))
                        with open(os.path.join(self.biasDir, filename), "rb") as file:
                            data = pickle.load(file)
                        self._loadedBiases[matchId] = _LoadedBias(
                            matchId, matchIndex, data
                        )
                        fileLoaded = True
                    except IOError:
                        # There's a tiny chance the file could get deleted by another
                        # process between when we check the directory and when we try
                        # to load it.  If so, just ignore the error and keep using
                        # whatever version of that process' biases we last loaded.
                        pass

        # If we loaded any files, recompute the total bias from all processes.

        if fileLoaded:
            self._KDE = self._selfKDE.copy()
            self._rwKDE = self._selfRwKDE.copy() if self.exploreMode else self._KDE
            for bias in self._loadedBiases.values():
                self._KDE += bias.bias[0]
                if self.exploreMode:
                    self._rwKDE += bias.bias[1]
