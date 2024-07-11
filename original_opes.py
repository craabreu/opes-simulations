from functools import reduce

import numpy as np
import openmm as mm
from openmm import unit


COMPRESSION_THRESHOLD = 1.0
FINITE_SUPPORT = False


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


class Kernel:
    """
    A multivariate Gaussian kernel with diagonal bandwidth matrix.

    Parameters
    ----------
    variables
        The collective variables that define the multidimensional domain of the kernel.
    position
        The point in space where the kernel is centered.
    bandwidth
        The bandwidth (standard deviation) of the kernel in each direction.
    logWeight
        The logarithm of the weight assigned to the kernel.

    Attributes
    ----------
    position
        The point in space where the kernel is centered.
    bandwidth
        The bandwidth (standard deviation) of the kernel in each direction.
    logWeight
        The logarithm of the weight assigned to the kernel.
    logHeight
        The logarithm of the kernel's height.
    """

    def __init__(self, variables, position, bandwidth, logWeight):
        ndims = len(variables)
        assert len(position) == len(bandwidth) == ndims
        self.position = np.asarray(position)
        self.bandwidth = np.asarray(bandwidth)
        self.logWeight = logWeight
        self._periodic = any(cv.periodic for cv in variables)
        if self._periodic:
            self._pdims = [i for i, cv in enumerate(variables) if cv.periodic]
            self._lbounds = np.array([variables[i].minValue for i in self._pdims])
            ubounds = np.array([variables[i].maxValue for i in self._pdims])
            self._lengths = ubounds - self._lbounds
        self.logHeight = self._computeLogHeight()

    def _computeLogHeight(self):
        if np.any(self.bandwidth == 0):
            return -np.inf
        ndims = len(self.bandwidth)
        log_height = self.logWeight - np.log(self.bandwidth).sum()
        if FINITE_SUPPORT:
            log_height += ndims * (np.log(35) - np.log(559872))
        else:
            log_height -= ndims * np.log(2 * np.pi) / 2
        return log_height

    def _squareMahalanobisDistances(self, points):
        return np.square(self.displacement(points) / self.bandwidth).sum(axis=-1)

    def displacement(self, endpoint):
        """
        Compute the displacement vector from the kernel's position to a given endpoint,
        taking periodicity into account.

        Parameters
        ----------
        endpoint
            The endpoint to which the displacement vector is computed.

        Returns
        -------
        np.ndarray
            The displacement vector from the kernel's position to the endpoint.
        """
        disp = endpoint - self.position
        if self._periodic:
            disp[..., self._pdims] -= self._lengths * np.round(
                disp[..., self._pdims] / self._lengths
            )
        return disp

    def endpoint(self, displacement):
        """
        Compute the endpoint of a displacement vector from the kernel's position

        Parameters
        ----------
        displacement
            The displacement vector from the kernel's position.

        Returns
        -------
        np.ndarray
            The endpoint of the displacement vector from the kernel's position.
        """
        end = self.position + displacement
        if self._periodic:
            end[..., self._pdims] = (
                self._lbounds + (end[..., self._pdims] - self._lbounds) % self._lengths
            )
        return end

    def findNearest(self, points, ignore=()):
        """
        Given a list of points in space, return the index of the nearest one and the
        squared Mahalanobis distance to it. Optionally ignore some points.

        Parameters
        ----------
        points
            The list of points to compare against. The shape of this array must be
            :math:`(N, d)`, where :math:`N` is the number of points and :math:`d` is
            the dimensionality of the kernel.
        ignore
            The indices of points to ignore.

        Returns
        -------
        int
            The index of the point (or -1 if no points are given)
        float
            The squared Mahalanobis distance to the closest point (or infinity if
            no points are given)
        """
        if points.size == 0:
            return -1, np.inf
        sq_mahalanobis_distances = self._squareMahalanobisDistances(points)
        if ignore:
            sq_mahalanobis_distances[ignore] = np.inf
        index = np.argmin(sq_mahalanobis_distances)
        return index, sq_mahalanobis_distances[index]

    def merge(self, other):
        """
        Change this kernel by merging it with another one.

        Parameters
        ----------
        other
            The kernel to merge with.
        """
        log_sum_weights = np.logaddexp(self.logWeight, other.logWeight)
        w1 = np.exp(self.logWeight - log_sum_weights)
        w2 = np.exp(other.logWeight - log_sum_weights)

        displacement = self.displacement(other.position)
        mean_position = self.endpoint(w2 * displacement)
        mean_squared_bandwidth = w1 * self.bandwidth**2 + w2 * other.bandwidth**2

        self.logWeight = log_sum_weights
        self.position = mean_position
        self.bandwidth = np.sqrt(mean_squared_bandwidth + w1 * w2 * displacement**2)
        self.logHeight = self._computeLogHeight()

    @staticmethod
    def _finiteSupportLogs(x):
        values = 9 - x**2
        mask = values > 0
        values[mask] = 4 * np.log(values[mask])
        values[~mask] = -np.inf
        return values

    def evaluate(self, points):
        """
        Compute the natural logarithm of the kernel evaluated at the given point or
        points.

        Parameters
        ----------
        point
            The point or points at which to evaluate the kernel. The shape of this
            array must be either :math:`(d,)` or :math:`(N, d)`, where :math:`d` is
            the dimensionality of the kernel and :math:`N` is the number of points.

        Returns
        -------
        float
            The logarithm of the kernel evaluated at the given point or points.
        """
        if FINITE_SUPPORT:
            return self.logHeight + self._finiteSupportLogs(
                self.displacement(points) / self.bandwidth
            ).sum(axis=-1)
        return self.logHeight - 0.5 * self._squareMahalanobisDistances(points)

    def evaluateOnGrid(self, gridMarks):
        """
        Compute the natural logarithms of the kernel evaluated on a rectilinear grid.

        Parameters
        ----------
        gridMarks
            The points in each dimension used to define the rectilinear grid. The length
            of this list must match the dimensionality :math:`d` of the kernel. The size
            :math:`N_i` of each array :math:`i` is arbitrary. For periodic dimensions,
            it is assumed that the grid spans the entire periodic length, i.e. that the
            last point differs from the first by the periodic length.

        Returns
        -------
        np.ndarray
            The logarithm of the kernel evaluated on the grid points. The shape of this
            array is :math:`(N_d, \\ldots, N_2, N_1)`, which makes it compatible with
            OpenMM's ``TabulatedFunction`` convention.
        """
        distances = [points - x for points, x in zip(gridMarks, self.position)]
        if self._periodic:
            for dim, length in zip(self._pdims, self._lengths):
                distances[dim] -= length * np.round(distances[dim] / length)
                distances[dim][-1] = distances[dim][0]
        if FINITE_SUPPORT:
            exponents = [
                self._finiteSupportLogs(distance / sigma)
                for distance, sigma in zip(distances, self.bandwidth)
            ]
        else:
            exponents = [
                -0.5 * (distance / sigma) ** 2
                for distance, sigma in zip(distances, self.bandwidth)
            ]
        return self.logHeight + reduce(np.add.outer, reversed(exponents))


class OPESForce(mm.CustomCVForce):
    """
    A custom force implementation for the On-The-Fly Probability Enhanced Sampling
    (OPES) method.

    Parameters
    ----------
    variables
        The collective variables to sample.
    barrier
        The energy barrier to overcome.
    prefactor
        The prefactor of the bias potential.

    Raises
    ------
    ValueError
        If the number of periodic variables is not 0 or equal to the total number of
        variables.
    RuntimeError
        If all 32 force groups in the system are already in use.
    """

    def __init__(self, variables, barrier, prefactor):
        barrier = barrier.value_in_unit(unit.kilojoules_per_mole)
        prefactor = prefactor.value_in_unit(unit.kilojoules_per_mole)
        num_vars = len(variables)
        num_periodics = sum(cv.periodic for cv in variables)
        if num_periodics not in [0, num_vars]:
            raise ValueError("OPES cannot handle mixed periodic/non-periodic variables")
        self._prefactor = prefactor
        self._logEpsilon = -barrier / prefactor
        grid_widths = [cv.gridWidth for cv in variables]
        self._widths = [] if num_vars == 1 else grid_widths
        self._limits = sum(([cv.minValue, cv.maxValue] for cv in variables), [])
        self._table = getattr(mm, f"Continuous{num_vars}DFunction")(
            *self._widths,
            np.full(np.prod(grid_widths), -barrier),
            *self._limits,
            num_periodics == num_vars,
        )
        var_names = [f"cv{i}" for i in range(len(variables))]
        super().__init__(f"table({', '.join(var_names)})")
        for name, var in zip(var_names, variables):
            self.addCollectiveVariable(name, var.force)
        self.addTabulatedFunction("table", self._table)

    def setUniqueForceGroup(self, system):
        """
        Set the force group to the unused group with the highest index in the system.

        Parameters
        ----------
        system
            The System to which the force will be added.

        Raises
        ------
        RuntimeError
            If all 32 force groups in the system are already in use.
        """
        free_groups = set(range(32)) - {f.getForceGroup() for f in system.getForces()}
        if not free_groups:
            raise RuntimeError("All 32 force groups are already in use.")
        self.setForceGroup(max(free_groups))

    def getEnergy(self, context):
        """
        Get the energy of the bias potential.

        Parameters
        ----------
        context
            The Context in which to evaluate the energy.
        """
        state = context.getState(getEnergy=True, groups={self.getForceGroup()})
        return state.getPotentialEnergy()

    def update(self, kde, context):
        """
        Update the tabulated function with new values of the bias potential.

        Parameters
        ----------
        logP
            The new values of the bias potential.
        logZ
            The logarithm of the partition function.
        context
            The Context in which to apply the bias.
        """
        self._table.setFunctionParameters(
            *self._widths,
            kde.getBias(self._prefactor, self._logEpsilon).flatten(),
            *self._limits,
        )
        self.updateParametersInContext(context)


class OnlineKDE:
    def __init__(self, variables):
        self.variables = variables
        self.d = len(variables)
        self._kernels = []
        self._grid = [
            np.linspace(cv.minValue, cv.maxValue, cv.gridWidth) for cv in variables
        ]
        self._logSumW = self._logSumWSq = -np.inf
        self._logPK = np.empty(0)
        self._logPG = np.full([cv.gridWidth for cv in reversed(variables)], -np.inf)

    def update(self, log_weight, values, variance):
        self._logSumW = np.logaddexp(self._logSumW, log_weight)
        self._logSumWSq = np.logaddexp(self._logSumWSq, 2 * log_weight)
        neff = np.exp(2 * self._logSumW - self._logSumWSq)
        silverman = (neff * (self.d + 2) / 4) ** (-1 / (self.d + 4))
        new_kernel = Kernel(self.variables, values, silverman * variance, log_weight)
        if self._kernels:
            points = np.stack([k.position for k in self._kernels])
            index, min_sq_dist = new_kernel.findNearest(points)
            to_remove = []
            while min_sq_dist <= COMPRESSION_THRESHOLD**2:
                to_remove.append(index)
                new_kernel.merge(self._kernels[index])
                index, min_sq_dist = new_kernel.findNearest(points, to_remove)
            self._logPK = np.logaddexp(self._logPK, new_kernel.evaluate(points))
            self._logPG = np.logaddexp(
                self._logPG, new_kernel.evaluateOnGrid(self._grid)
            )
            if to_remove:
                to_remove = sorted(to_remove, reverse=True)
                for index in to_remove:
                    k = self._kernels.pop(index)
                    self._logPK = logsubexp(self._logPK, k.evaluate(points))
                    self._logPG = logsubexp(self._logPG, k.evaluateOnGrid(self._grid))
                self._logPK = np.delete(self._logPK, to_remove)
            self._kernels.append(new_kernel)
            log_p_new = [k.evaluate(new_kernel.position) for k in self._kernels]
            self._logPK = np.append(self._logPK, np.logaddexp.reduce(log_p_new))
        else:
            self._kernels = [new_kernel]
            self._logPG = new_kernel.evaluateOnGrid(self._grid)
            self._logPK = log_p_new = np.array([new_kernel.logHeight])

    def getLogZ(self):
        return (
            np.logaddexp.reduce(self._logPK)
            - np.log(len(self._kernels))
            - self._logSumW
        )

    def getBias(self, prefactor, logEpsilon):
        logZ = np.logaddexp.reduce(self._logPK) - np.log(len(self._kernels))
        return prefactor * np.logaddexp(self._logPG - logZ, logEpsilon)


class OPES:  # pylint: disable=too-many-instance-attributes
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
        exploreMode,
        varianceFrequency,
    ):
        if not unit.is_quantity(temperature):
            temperature *= unit.kelvin
        if not unit.is_quantity(barrier):
            barrier *= unit.kilojoules_per_mole
        self.variables = variables
        self.temperature = temperature
        self.frequency = frequency
        self.barrier = barrier
        self.exploreMode = exploreMode

        num_vars = len(variables)
        if not 1 <= num_vars <= 3:
            raise ValueError("OPES requires 1, 2, or 3 collective variables")

        kbt = unit.MOLAR_GAS_CONSTANT_R * temperature
        if barrier <= kbt:
            raise ValueError(f"barrier must be greater than {kbt}")
        self._kbt = kbt.in_units_of(unit.kilojoules_per_mole)
        self._kde = OnlineKDE(variables)

        gamma = barrier / kbt
        prefactor = (gamma - 1) * kbt if exploreMode else (1 - 1 / gamma) * kbt
        self._force = OPESForce(variables, barrier, prefactor)
        self._bias_factor = gamma
        if not exploreMode:
            self._force.setUniqueForceGroup(system)
        system.addForce(self._force)

        self._tau = 10 * frequency
        self._movingKernel = Kernel(variables, *[np.zeros(num_vars)] * 2, 0.0)
        self._counter = 0
        self._bwFactor = 1.0 if exploreMode else 1.0 / np.sqrt(gamma)

        self._log_acc_inv_density = -np.inf

    def getFreeEnergy(self, corrected=True):
        """
        Get the free energy of the system as a function of the collective variables.

        The result is returned as a N-dimensional NumPy array, where N is the number
        of collective variables. The values are in kJ/mole. The i'th position along an
        axis corresponds to minValue + i*(maxValue-minValue)/gridWidth.
        """
        free_energy = -self._kbt * (self._kde._logPG - self._kde._logSumW)
        if self.exploreMode:
            if corrected:
                free_energy -= (
                    self._kde.getBias(self._force._prefactor, self._force._logEpsilon)
                    * unit.kilojoules_per_mole
                )
            else:
                free_energy *= self._bias_factor
        return free_energy

    def getAverageDensity(self):
        """
        Get the average density of the system as a function of the collective variables.
        """
        return np.exp(self._kde.getLogZ())

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
        while steps:
            next_steps = min(
                steps, self.frequency - simulation.currentStep % self.frequency
            )
            for _ in range(next_steps):
                simulation.step(1)
                values = self._force.getCollectiveVariableValues(simulation.context)
                self._updateMovingKernel(values)
            if simulation.currentStep % self.frequency == 0:
                self._addKernel(values, simulation.context)
            steps -= next_steps

    def setVariance(self, variance):
        """
        Set the variance of the probability distribution estimate.

        Parameters
        ----------
        variance
            The variance of the probability distribution estimate.
        """
        self._movingKernel.bandwidth = np.sqrt(variance)

    def getVariance(self):
        """
        Get the variance of the probability distribution estimate.
        """
        return self._movingKernel.bandwidth**2

    def _updateMovingKernel(self, values):
        """
        Update the moving kernel used to estimate the bandwidth of the

        Parameters
        ----------
        values
            The current values of the collective variables.
        """
        self._counter += 1
        kernel = self._movingKernel
        delta = kernel.displacement(values)
        x = 1 / min(self._tau, self._counter)
        kernel.position = kernel.endpoint(x * delta)
        delta *= kernel.displacement(values)
        n = self._tau + self._counter
        variance = (n - 1) * kernel.bandwidth**2 + delta
        kernel.bandwidth = np.sqrt(variance / n)

    def _addKernel(self, values, context):
        """
        Add a kernel to the probability distribution estimate and update the bias
        potential.

        Parameters
        ----------
        values
            The current values of the collective variables
        context
            The Context in which to apply the bias.
        """
        if self.exploreMode:
            log_weight = 0
        else:
            log_weight = self._force.getEnergy(context) / self._kbt
        variance = self._bwFactor * self._movingKernel.bandwidth
        self._kde.update(log_weight, values, variance)
        self._force.update(self._kde, context)
