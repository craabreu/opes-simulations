import functools
from collections import namedtuple

import numpy as np

COMPRESSION_THRESHOLD = 1.0
BOUNDED_KERNELS = False
UNCOMPRESSED_KDE = False
USE_EXISTING_BANDWIDTHS = True


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
            self._CV(cv.minValue, cv.maxValue, cv.gridWidth, cv.periodic)
            for cv in variables
        ]
        self._periodic = any(cv.periodic for cv in variables)
        self._grid = [
            np.linspace(cv.minValue, cv.maxValue, cv.gridWidth) for cv in variables
        ]
        self._widths = np.array([cv.gridWidth for cv in variables])
        self._lbounds = np.array([cv.minValue for cv in variables])
        ubounds = np.array([cv.maxValue for cv in variables])
        self._lengths = ubounds - self._lbounds
        if self._periodic:
            self._pdims = tuple(i for i, cv in enumerate(variables) if cv.periodic)
            self._plbounds = self._lbounds[self._pdims]
            self._plengths = self._lengths[self._pdims]

    _CV = namedtuple("_CV", ["minValue", "maxValue", "gridWidth", "periodic"])

    def __getstate__(self):
        return {"variables": [cv._asdict() for cv in self.variables]}

    def __setstate__(self, state):
        self.__init__([self._CV(**kwargs) for kwargs in state["variables"]])

    @property
    def gridShape(self):
        """Return the shape of the CV space grid."""
        return tuple(reversed(self._widths))

    @property
    def numDimensions(self):
        """Return the number of dimensions in the CV space."""
        return len(self.variables)

    def displacement(self, position, endpoint):
        """Compute the displacement between two centers in the CV space."""
        disp = endpoint - position
        if self._periodic:
            disp[..., self._pdims] -= self._plengths * np.rint(
                disp[..., self._pdims] / self._plengths
            )
        return disp

    def endpoint(self, position, displacement):
        """Compute the endpoint given a starting position and a displacement."""
        end = position + displacement
        if self._periodic:
            end[..., self._pdims] = (
                self._plbounds
                + (end[..., self._pdims] - self._plbounds) % self._plengths
            )
        return end

    def gridDistances(self, position):
        """Compute the distances from a position to all centers on a regular grid."""
        distances = [centers - x for centers, x in zip(self._grid, position)]
        if self._periodic:
            for dim, length in zip(self._pdims, self._plengths):
                distances[dim] -= length * np.rint(distances[dim] / length)
                distances[dim][-1] = distances[dim][0]
        return distances

    def closestNode(self, position):
        """Find the closest node to a position in the CV space."""
        indices = np.rint(
            (self._widths - 1) * (position - self._lbounds) / self._lengths
        ).astype(int)
        if self._periodic:
            indices[self._pdims] %= self._widths[self._pdims]
        indices = np.clip(indices, 0, self._widths - 1)
        return tuple(reversed(indices))


class Kernel:
    """A multivariate kernel function with zero covariance."""

    def __init__(self, cvSpace, position, bandwidth, logWeight, fractions=(1.0,)):
        self.cvSpace = cvSpace
        self.position = np.array(position)
        self.bandwidth = np.array(bandwidth)
        self.logWeight = logWeight
        if not np.isclose(np.sum(fractions), 1.0):
            raise ValueError("fractions must sum to 1.0")
        self.fractions = fractions
        self.logHeight = self._computeLogHeight()

    def _computeLogHeight(self):
        if np.any(self.bandwidth == 0):
            return -np.inf
        const = np.log(559872 / 35) if BOUNDED_KERNELS else np.log(2 * np.pi) / 2
        d = self.cvSpace.numDimensions
        return self.logWeight - d * const - np.sum(np.log(self.bandwidth))

    def _scaledDistances(self, centers, bandwidths):
        return self.cvSpace.displacement(self.position, centers) / bandwidths

    def _logFraction(self, label):
        if label is None:
            return 0.0
        if self.fractions[label] == 0.0:
            return -np.inf
        return np.log(self.fractions[label])

    @staticmethod
    def _exponents(x):
        if BOUNDED_KERNELS:
            values = 9 - x**2
            mask = values > 0
            values[mask] = 4 * np.log(values[mask])
            values[~mask] = -np.inf
            return values
        return -0.5 * x**2

    def findNearest(self, centers, bandwidths, ignore=()):
        """
        Given a list of centers and their corresponding bandwidths, return the index of
        the nearest center and the squared Mahalanobis distance to it. Optionally ignore
        some centers.
        """
        if centers.size == 0:
            return -1, np.inf
        sqMahalanobisDistances = np.sum(
            self._scaledDistances(centers, bandwidths) ** 2, axis=-1
        )
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
        self.fractions += w2 * (other.fractions - self.fractions)
        self.logWeight = logSumWeights
        self.logHeight = self._computeLogHeight()

    def evaluate(self, centers, label=None):
        """Evaluate the logarithm of the kernel at the given point or centers."""
        return (
            self._logFraction(label)
            + self.logHeight
            + np.sum(
                self._exponents(self._scaledDistances(centers, self.bandwidth)), axis=-1
            )
        )

    def evaluateOnGrid(self, label=None):
        """Evaluate the logarithms of the kernel on a regular grid."""
        distances = self.cvSpace.gridDistances(self.position)
        exponents = [
            self._exponents(dist / sigma)
            for dist, sigma in zip(distances, self.bandwidth)
        ]
        return (
            self.logHeight
            + self._logFraction(label)
            + functools.reduce(np.add.outer, reversed(exponents))
        )


class OnlineKDE:
    """Online Kernel Density Estimation (KDE) for collective variable sampling."""

    def __init__(self, cvSpace, varianceScale=1.0, reweighting=False, numLabels=1):
        self._kernels = []
        self._cvSpace = cvSpace
        self._numSamples = 0
        self._logSumW = -np.inf
        self._logSumWSq = -np.inf
        self._logPK = np.empty(0)
        self._logPG = np.full(cvSpace.gridShape, -np.inf)
        self._maskPG = np.full(cvSpace.gridShape, False) if UNCOMPRESSED_KDE else None
        self._d = len(cvSpace.gridShape)
        self._varianceScale = varianceScale
        self._reweighting = reweighting
        self._numLabels = numLabels

    def __getstate__(self):
        if self._kernels:
            kernelData = [
                np.stack([k.position for k in self._kernels]),
                np.stack([k.bandwidth for k in self._kernels]),
                np.array([k.logWeight for k in self._kernels]),
            ]
            if self._numLabels > 1:
                kernelData.append(np.stack([k.fractions for k in self._kernels]))
        else:
            kernelData = []
        return {
            "cvSpace": self._cvSpace,
            "numSamples": self._numSamples,
            "sumW": self._logSumW,
            "sumWSq": self._logSumWSq,
            "logPK": self._logPK,
            "logPG": self._logPG if UNCOMPRESSED_KDE else None,
            "maskPG": self._maskPG if UNCOMPRESSED_KDE else None,
            "varianceScale": self._varianceScale,
            "reweighting": self._reweighting,
            "numLabels": self._numLabels,
            "kernelData": kernelData,
        }

    def __setstate__(self, state):
        self.__init__(state["cvSpace"])
        self._numSamples = state["numSamples"]
        self._logSumW = state["sumW"]
        self._logSumWSq = state["sumWSq"]
        self._logPK = state["logPK"]
        self._logPG = state["logPG"]
        self._maskPG = state["maskPG"]
        self._varianceScale = state["varianceScale"]
        self._reweighting = state["reweighting"]
        self._numLabels = state["numLabels"]
        self._kernels = [
            Kernel(self._cvSpace, *data) for data in zip(*state["kernelData"])
        ]
        if self._logPG is None:
            self._logPG = functools.reduce(
                np.logaddexp, (k.evaluateOnGrid() for k in self._kernels)
            )

    def __iadd__(self, other):
        if other._reweighting != self._reweighting:
            raise ValueError("Cannot add KDEs with incompatible reweighting flags")
        if not np.isclose(other._varianceScale, self._varianceScale):
            raise ValueError("Cannot add KDEs with incompatible variance scales")
        if other._numLabels != self._numLabels:
            raise ValueError("Cannot add KDEs with incompatible numbers of labels")
        if UNCOMPRESSED_KDE:
            self._numSamples += other._numSamples
            self._logSumW = np.logaddexp(self._logSumW, other._logSumW)
            self._logSumWSq = np.logaddexp(self._logSumWSq, other._logSumWSq)
            self._logPG = np.logaddexp(self._logPG, other._logPG)
            self._maskPG = np.logical_or(self._maskPG, other._maskPG)
        else:
            for k in other._kernels:
                self._addKernel(
                    k.position,
                    k.bandwidth,
                    k.logWeight,
                    k.fractions,
                    adjustBandwidth=False,
                )
        return self

    def __bool__(self):
        return self._numSamples > 0

    def _logDenominator(self):
        return self._logSumW if self._reweighting else np.log(self._numSamples)

    def _addKernel(self, position, bandwidth, logWeight, fractions, adjustBandwidth):
        """Update the KDE by depositing a new kernel."""
        self._numSamples += 1
        self._logSumW = np.logaddexp(self._logSumW, logWeight)
        self._logSumWSq = np.logaddexp(self._logSumWSq, 2 * logWeight)
        if adjustBandwidth:
            neff = np.exp(2 * self._logSumW - self._logSumWSq)
            silverman = (neff * (self._d + 2) / 4) ** (-1 / (self._d + 4))
            bandwidth = bandwidth * silverman
        newKernel = Kernel(
            self._cvSpace,
            position,
            bandwidth,
            logWeight if self._reweighting else 0.0,
            fractions,
        )
        if UNCOMPRESSED_KDE:
            self._maskPG[self._cvSpace.closestNode(position)] = True
            self._logPG = np.logaddexp(self._logPG, newKernel.evaluateOnGrid())
        elif self._kernels:
            centers = np.stack([k.position for k in self._kernels])
            if USE_EXISTING_BANDWIDTHS:
                bandwidths = np.stack([k.bandwidth for k in self._kernels])
            else:
                bandwidths = bandwidth
            index, minSqDist = newKernel.findNearest(centers, bandwidths)
            toRemove = []
            while minSqDist <= COMPRESSION_THRESHOLD**2:
                toRemove.append(index)
                newKernel.merge(self._kernels[index])
                index, minSqDist = newKernel.findNearest(centers, bandwidths, toRemove)
            self._logPK = np.logaddexp(self._logPK, newKernel.evaluate(centers))
            self._logPG = np.logaddexp(self._logPG, newKernel.evaluateOnGrid())
            if toRemove:
                toRemove = sorted(toRemove, reverse=True)
                for index in toRemove:
                    k = self._kernels.pop(index)
                    self._logPK = logsubexp(self._logPK, k.evaluate(centers))
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
        new = self.__class__(self._cvSpace)
        new._kernels = self._kernels.copy()
        new._logSumW = self._logSumW
        new._logSumWSq = self._logSumWSq
        new._numSamples = self._numSamples
        new._logPK = self._logPK.copy()
        new._logPG = self._logPG.copy()
        new._maskPG = self._maskPG.copy()
        new._varianceScale = self._varianceScale
        new._reweighting = self._reweighting
        new._numLabels = self._numLabels
        return new

    def getNumKernels(self):
        """Get the number of kernels in the kernel density estimator."""
        return len(self._kernels)

    def getLogPDF(self, label=None):
        """Get the logarithm of the probability density function (PDF) on the grid."""
        if label is None:
            logP = self._logPG
        else:
            logP = functools.reduce(
                np.logaddexp, (k.evaluateOnGrid(label) for k in self._kernels)
            )
        return logP - self._logDenominator()

    def getLogMeanDensity(self):
        """Get the logarithm of the mean density."""
        if UNCOMPRESSED_KDE:
            logP = np.logaddexp.reduce(self._logPG[self._maskPG])
            n = self._maskPG.sum()
        else:
            logP = np.logaddexp.reduce(self._logPK)
            n = len(self._kernels)
        return logP - np.log(n) - self._logDenominator()

    def getLogNormConstRatio(self):
        """Get the logarithm of the ratio of the normalizing constants."""
        return self._logSumW - np.log(self._numSamples)

    def update(self, position, logWeight, variance, label=0):
        """Update the KDE by depositing a new kernel."""
        bandwidth = np.sqrt(self._varianceScale * variance)
        fractions = np.zeros(self._numLabels)
        fractions[label] = 1
        self._addKernel(position, bandwidth, logWeight, fractions, True)

    def evaluate(self, point):
        """Evaluate the logarithm of the kernel at the given point."""
        if UNCOMPRESSED_KDE:
            logP = self._logPG[self._cvSpace.closestNode(point)]
        else:
            logP = np.logaddexp.reduce([k.evaluate(point) for k in self._kernels])
        return logP - self._logDenominator()
