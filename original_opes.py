import functools
import os
import pickle
import re
import warnings
from collections import namedtuple

import numpy as np
import openmm as mm
from openmm import unit
from openmm.app.metadynamics import _LoadedBias

COMPRESSION_THRESHOLD = 1.0
STATS_WINDOW_SIZE = 10
LIMITED_SUPPORT = False
UNCORRECTED_OPES_EXPLORE = False
REWEIGHTED_FES = True


_CV = namedtuple("_CV", ["minValue", "maxValue", "gridWidth", "periodic"])


def logsubexp(x, y):
    """Compute log(exp(x) - exp(y)) in a numerically stable way."""
    array1 = np.full_like(x, -np.inf)
    mask1 = y < x
    array2 = -np.exp(y[mask1] - x[mask1])
    mask2 = array2 > -1.0
    array2[mask2] = np.log1p(array2[mask2])
    array2[~mask2] = -np.inf
    array1[mask1] = array2 + x[mask1]
    return array1


class CVSpace:
    """A class to represent the space of collective variables."""

    def __init__(self, variables):
        self.variables = [
            _CV(cv.minValue, cv.maxValue, cv.gridWidth, cv.periodic) for cv in variables
        ]
        self._periodic = any(cv.periodic for cv in variables)
        self._grid = [
            np.linspace(cv.minValue, cv.maxValue, cv.gridWidth) for cv in variables
        ]
        if self._periodic:
            self._pdims = [i for i, cv in enumerate(variables) if cv.periodic]
            self._lbounds = np.array([variables[i].minValue for i in self._pdims])
            ubounds = np.array([variables[i].maxValue for i in self._pdims])
            self._lengths = ubounds - self._lbounds

    def __getstate__(self):
        return {"variables": [cv._asdict() for cv in self.variables]}

    def __setstate__(self, state):
        self.__init__([_CV(**kwargs) for kwargs in state["variables"]])

    @property
    def gridShape(self):
        """Return the shape of the CV space grid."""
        return tuple(cv.gridWidth for cv in reversed(self.variables))

    @property
    def numDimensions(self):
        """Return the number of dimensions in the CV space."""
        return len(self.variables)

    def displacement(self, position, endpoint):
        """Compute the displacement between two points in the CV space."""
        disp = endpoint - position
        if self._periodic:
            disp[..., self._pdims] -= self._lengths * np.rint(
                disp[..., self._pdims] / self._lengths
            )
        return disp

    def endpoint(self, position, displacement):
        """Compute the endpoint given a starting position and a displacement."""
        end = position + displacement
        if self._periodic:
            end[..., self._pdims] = (
                self._lbounds + (end[..., self._pdims] - self._lbounds) % self._lengths
            )
        return end

    def gridDistances(self, position):
        """Compute the distances from a position to all points on a regular grid."""
        distances = [points - x for points, x in zip(self._grid, position)]
        if self._periodic:
            for dim, length in zip(self._pdims, self._lengths):
                distances[dim] -= length * np.rint(distances[dim] / length)
                distances[dim][-1] = distances[dim][0]
        return distances


class Kernel:
    """A multivariate kernel function with zero covariance."""

    def __init__(self, cvSpace, position, bandwidth, logWeight):
        self.cvSpace = cvSpace
        self.position = np.array(position)
        self.bandwidth = np.array(bandwidth)
        self.logWeight = logWeight
        self.logHeight = self._computeLogHeight()

    def _computeLogHeight(self):
        if np.any(self.bandwidth == 0):
            return -np.inf
        const = np.log(559872 / 35) if LIMITED_SUPPORT else np.log(2 * np.pi) / 2
        d = self.cvSpace.numDimensions
        return self.logWeight - d * const - np.sum(np.log(self.bandwidth))

    def _scaledDistances(self, points):
        return self.cvSpace.displacement(self.position, points) / self.bandwidth

    @staticmethod
    def _exponents(x):
        if LIMITED_SUPPORT:
            values = 9 - x**2
            mask = values > 0
            values[mask] = 4 * np.log(values[mask])
            values[~mask] = -np.inf
            return values
        return -0.5 * x**2

    def findNearest(self, points, ignore=()):
        """
        Given a list of points in space, return the index of the nearest one and the
        squared Mahalanobis distance to it. Optionally ignore some points.
        """
        if points.size == 0:
            return -1, np.inf
        sqMahalanobisDistances = np.sum(self._scaledDistances(points) ** 2, axis=-1)
        if ignore:
            sqMahalanobisDistances[ignore] = np.inf
        index = np.argmin(sqMahalanobisDistances)
        return index, sqMahalanobisDistances[index]

    def merge(self, other):
        """Change this kernel by merging it with another one."""
        logSumWeights = np.logaddexp(self.logWeight, other.logWeight)
        w1 = np.exp(self.logWeight - logSumWeights)
        w2 = np.exp(other.logWeight - logSumWeights)
        disp = self.cvSpace.displacement(self.position, other.position)
        self.position = self.cvSpace.endpoint(self.position, w2 * disp)
        self.bandwidth = np.sqrt(
            w1 * self.bandwidth**2 + w2 * other.bandwidth**2 + w1 * w2 * disp**2
        )
        self.logWeight = logSumWeights
        self.logHeight = self._computeLogHeight()

    def evaluate(self, points):
        """Evaluate the logarithm of the kernel at the given point or points."""
        return self.logHeight + np.sum(
            self._exponents(self._scaledDistances(points)), axis=-1
        )

    def evaluateOnGrid(self):
        """Evaluate the logarithms of the kernel on a regular grid."""
        distances = self.cvSpace.gridDistances(self.position)
        exponents = [
            self._exponents(dist / sigma)
            for dist, sigma in zip(distances, self.bandwidth)
        ]
        return self.logHeight + functools.reduce(np.add.outer, reversed(exponents))


class OnlineKDE:
    """Online Kernel Density Estimation (KDE) for collective variables."""

    def __init__(self, cvSpace, varianceScale=1.0):
        self._kernels = []
        self._cvSpace = cvSpace
        self._numSamples = 0
        self._logSumW = -np.inf
        self._logSumWSq = -np.inf
        self._logPK = np.empty(0)
        self._logPG = np.full(cvSpace.gridShape, -np.inf)
        self._d = len(cvSpace.gridShape)
        self._varianceScale = varianceScale
        self._numVarianceSamples = 0
        self._sumVariance = np.zeros(self._d)

    def __getstate__(self):
        if self._kernels:
            kernelData = [
                np.stack([k.position for k in self._kernels]),
                np.stack([k.bandwidth for k in self._kernels]),
                np.array([k.logWeight for k in self._kernels]),
            ]
        else:
            kernelData = []
        return {
            "cvSpace": self._cvSpace,
            "numSamples": self._numSamples,
            "sumW": self._logSumW,
            "sumWSq": self._logSumWSq,
            "logPK": self._logPK,
            "logPG": self._logPG,
            "varianceScale": self._varianceScale,
            "numVarianceSamples": self._numVarianceSamples,
            "sumVariance": self._sumVariance,
            "kernelData": kernelData,
        }

    def __setstate__(self, state):
        self.__init__(state["cvSpace"])
        self._numSamples = state["numSamples"]
        self._logSumW = state["sumW"]
        self._logSumWSq = state["sumWSq"]
        self._logPK = state["logPK"]
        self._logPG = state["logPG"]
        self._varianceScale = state["varianceScale"]
        self._numVarianceSamples = state["numVarianceSamples"]
        self._sumVariance = state["sumVariance"]
        self._kernels = [
            Kernel(self._cvSpace, *data) for data in zip(*state["kernelData"])
        ]

    def __iadd__(self, other):
        if not np.isclose(other._varianceScale, self._varianceScale):
            raise ValueError("Cannot add KDEs with incompatible variance scales")
        self._numVarianceSamples += other._numVarianceSamples
        self._sumVariance += other._sumVariance.copy()
        for k in other._kernels:
            self._addKernel(k.position, k.bandwidth, k.logWeight, False)
        return self

    def _addKernel(self, position, bandwidth, logWeight, adjustBandwidth):
        """Update the KDE by depositing a new kernel."""
        self._numSamples += 1
        self._logSumW = np.logaddexp(self._logSumW, logWeight)
        self._logSumWSq = np.logaddexp(self._logSumWSq, 2 * logWeight)
        if adjustBandwidth:
            neff = np.exp(2 * self._logSumW - self._logSumWSq)
            silverman = (neff * (self._d + 2) / 4) ** (-1 / (self._d + 4))
            bandwidth = bandwidth * silverman
        newKernel = Kernel(self._cvSpace, position, bandwidth, logWeight)
        if self._kernels:
            points = np.stack([k.position for k in self._kernels])
            index, minSqDist = newKernel.findNearest(points)
            toRemove = []
            while minSqDist <= COMPRESSION_THRESHOLD**2:
                toRemove.append(index)
                newKernel.merge(self._kernels[index])
                index, minSqDist = newKernel.findNearest(points, toRemove)
            self._logPK = np.logaddexp(self._logPK, newKernel.evaluate(points))
            self._logPG = np.logaddexp(self._logPG, newKernel.evaluateOnGrid())
            if toRemove:
                toRemove = sorted(toRemove, reverse=True)
                for index in toRemove:
                    k = self._kernels.pop(index)
                    self._logPK = logsubexp(self._logPK, k.evaluate(points))
                    self._logPG = logsubexp(self._logPG, k.evaluateOnGrid())
                self._logPK = np.delete(self._logPK, toRemove)
            self._kernels.append(newKernel)
            logNewP = [k.evaluate(newKernel.position) for k in self._kernels]
            self._logPK = np.append(self._logPK, np.logaddexp.reduce(logNewP))
        else:
            self._kernels = [newKernel]
            self._logPG = newKernel.evaluateOnGrid()
            self._logPK = np.array([newKernel.logHeight])

    def copy(self):
        new = OnlineKDE(self._cvSpace)
        new._kernels = self._kernels.copy()
        new._logSumW = self._logSumW
        new._logSumWSq = self._logSumWSq
        new._numSamples = self._numSamples
        new._logPK = self._logPK.copy()
        new._logPG = self._logPG.copy()
        new._varianceScale = self._varianceScale
        new._numVarianceSamples = self._numVarianceSamples
        new._sumVariance = self._sumVariance
        return new

    def getVariance(self):
        """Get the variance of the sampled variables."""
        return self._sumVariance / self._numVarianceSamples

    def getLogPDF(self):
        """Get the logarithm of the probability density function (PDF) on the grid."""
        return self._logPG - self._logSumW

    def getLogMeanDensity(self):
        """Get the logarithm of the mean density."""
        n = len(self._kernels)
        return np.logaddexp.reduce(self._logPK) - np.log(n) - self._logSumW

    def getLogNormalizingConstantRatio(self):
        """Get the logarithm of the ratio of the normalizing constants."""
        return self._logSumW - np.log(self._counter)

    def updateVariance(self, squaredDeviationFromMean):
        """Update the variance of the sampled variables."""
        self._numVarianceSamples += 1
        self._sumVariance += squaredDeviationFromMean

    def update(self, position, logWeight):
        """Update the KDE by depositing a new kernel."""
        bandwidth = np.sqrt(self._varianceScale * self.getVariance())
        self._addKernel(position, bandwidth, logWeight, True)


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
