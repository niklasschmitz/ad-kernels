import jax.numpy as jnp
from jax.config import config
from jax import jit, vmap
from gdml_jax.kernels import rbf, AtomicDoubleSumKernel, GlobalSymmetryKernel


from jax_md import space
from functools import partial
import numpy as np

# enable double precision
config.update("jax_enable_x64", True)


####################################################
# FCHL19 atom centred symmetry functions
# https://arxiv.org/pdf/1909.01946.pdf
####################################################


def f_cutoff(x, rcut):
    c = jnp.cos(x / rcut * jnp.pi / 2) ** 2
    return jnp.where(x < rcut, c, 0.0)


def radial_symmetry_functions(
    displacement_or_metric, species, nRs2=24, eta2=0.32, rcut=8.0, two_body_decay=1.8
):
    """Returns a function that computes radial symmetry functions.
    
    Modified from the JAX-MD library:
    https://github.com/google/jax-md/blob/6318f7347d0d0f644c08799de180e67ef9250b8f/jax_md/nn.py
    
    Args:
    displacement: A function that produces an `[N_atoms, M_atoms,
    spatial_dimension]` of particle displacements from particle positions
      specified as an `[N_atoms, spatial_dimension] and `[M_atoms,
      spatial_dimension]` respectively.
    species: An `[N_atoms]` that contains the species of each particle.
    nRs2: Number of radial symmetry functions per species.
    rcut: Neighbors whose distance is larger than rcut do
      not contribute to each others symmetry functions. The contribution of a
      neighbor to the symmetry function and its derivative goes to zero at this
      distance.
    Returns:
    A function that computes the radial symmetry function from input `[N_atoms,
    spatial_dimension]` and returns `[N_atoms, N_types * nRs2]` where nRs2 is
    the number of twobody symmetry functions per species, N_types is the number of types of particles 
    in the system.
    """
    metric = space.canonicalize_displacement_or_metric(displacement_or_metric)

    Rs2 = jnp.linspace(0, rcut, 1 + nRs2)[1:]

    def compute_fun(R, **kwargs):
        _metric = partial(metric, **kwargs)
        _metric = space.map_product(_metric)

        def radial_fn(dr_, rs2):
            dr = jnp.where(dr_ > 0, dr_, 1.0)
            xi2 = 1.0 / dr ** two_body_decay
            tmp = 1.0 + eta2 / dr ** 2
            mu = jnp.log(dr / jnp.sqrt(tmp))
            sig = jnp.sqrt(jnp.log(tmp))
            fcut = f_cutoff(dr, rcut)
            radial = xi2 * fcut * 1.0 / (sig * jnp.sqrt(2 * jnp.pi) * rs2)
            radial *= jnp.exp(-((jnp.log(rs2) - mu) ** 2) / (2.0 * sig ** 2))
            return jnp.where(dr_ > 0, radial, jnp.zeros_like(radial))

        def return_radial(atom_type):
            """Returns the radial symmetry functions for neighbor type atom_type."""
            R_neigh = R[species == atom_type, :]
            dr = _metric(R, R_neigh)

            radial = vmap(radial_fn, (None, 0))(dr, Rs2)
            return jnp.sum(radial, axis=1).T

        return jnp.hstack([return_radial(atom_type) for atom_type in np.unique(species)])

    return compute_fun


def single_pair_angular_symmetry_function(
    dR12,
    dR13,
    eta3,
    three_body_decay,
    three_body_weight,
    nFourier,
    zeta,
    Rs3,
    cutoff_distance,
):
    """Computes the angular symmetry function due to one pair of neighbors."""

    # TODO might reuse values from twobody terms
    dR23 = dR12 - dR13
    dr12 = space.distance(dR12)
    dr13 = space.distance(dR13)
    dr23 = space.distance(dR23)

    # prevent nan gradients
    cond = dr12 * dr13 * dr23 > 0
    dR12 = jnp.where(cond, dR12, jnp.ones_like(dR12))
    dR13 = jnp.where(cond, dR13, jnp.ones_like(dR13))
    dR23 = jnp.where(cond, dR12, jnp.ones_like(dR23))
    dr12 = jnp.where(cond, dr12, 1.0)
    dr13 = jnp.where(cond, dr13, 1.0)
    dr23 = jnp.where(cond, dr23, 1.0)

    fcut = lambda x: f_cutoff(x, cutoff_distance)
    triplet_cutoff = fcut(dr12) * fcut(dr13) * fcut(dr23)

    cos_angle1 = jnp.dot(dR12, dR13) / dr12 / dr13
    cos_angle2 = -jnp.dot(dR12, dR23) / dr12 / dr23
    cos_angle3 = jnp.dot(dR13, dR23) / dr13 / dr23

    xi3 = 1.0 + 3.0 * cos_angle1 * cos_angle2 * cos_angle3
    xi3 *= three_body_weight
    xi3 /= (dr12 * dr13 * dr23) ** three_body_decay
    radial = jnp.exp(-eta3 * (0.5 * (dr12 + dr13) - Rs3) ** 2)
    radial *= xi3 * triplet_cutoff

    angle = jnp.arccos(cos_angle1)
    odd_expansion_orders = jnp.arange(1, nFourier + 1, 2)
    angular_exp = jnp.exp(-((zeta * odd_expansion_orders) ** 2) / 2)
    angular_cos = 2.0 * jnp.cos(odd_expansion_orders * angle) * angular_exp
    angular_sin = 2.0 * jnp.sin(odd_expansion_orders * angle) * angular_exp

    result = jnp.concatenate(
        [jnp.outer(radial, angular_cos), jnp.outer(radial, angular_sin)]
    ).flatten()

    result = jnp.where(cond, result, jnp.zeros_like(result))
    return result


def angular_symmetry_functions(
    displacement,
    species,
    nRs3=20,
    nFourier=1,
    eta3=2.7,
    zeta=jnp.pi,
    acut=8.0,
    two_body_decay=1.8,
    three_body_decay=0.57,
    three_body_weight=13.4,
):
    """Returns a function that computes angular symmetry functions.
    
    Modified from the JAX-MD library:
    https://github.com/google/jax-md/blob/6318f7347d0d0f644c08799de180e67ef9250b8f/jax_md/nn.py
    
    Args:
    displacement: A function that produces an `[N_atoms, M_atoms,
    spatial_dimension]` of particle displacements from particle positions
      specified as an `[N_atoms, spatial_dimension] and `[M_atoms,
      spatial_dimension]` respectively.
    species: An `[N_atoms]` that contains the species of each particle.
    eta: Parameter of angular symmetry function that controls the spatial
      extension.
    lam:
    zeta:
    cutoff_distance: Neighbors whose distance is larger than cutoff_distance do
      not contribute to each others symmetry functions. The contribution of a
      neighbor to the symmetry function and its derivative goes to zero at this
      distance.
    Returns:
    A function that computes the angular symmetry function from input `[N_atoms,
    spatial_dimension]` and returns `[N_atoms, N_types * (N_types + 1) / 2]`
    where N_types is the number of types of particles in the system.
    """
    Rs3 = jnp.linspace(0, acut, 1 + nRs3)[1:]

    _angular_fn = single_pair_angular_symmetry_function

    _batched_angular_fn = lambda dR12, dR13: _angular_fn(
        dR12, dR13, eta3, three_body_decay, three_body_weight, nFourier, zeta, Rs3, acut
    )
    _all_pairs_angular = vmap(vmap(vmap(_batched_angular_fn, (0, None)), (None, 0)), 0)

    def compute_fun(R, **kwargs):
        D_fn = partial(displacement, **kwargs)
        D_fn = space.map_product(D_fn)
        D_different_types = [
            D_fn(R[species == atom_type, :], R) for atom_type in np.unique(species)
        ]
        out = []
        atom_types = np.unique(species)
        for i in range(len(atom_types)):
            for j in range(i, len(atom_types)):
                out += [
                    jnp.sum(
                        _all_pairs_angular(D_different_types[i], D_different_types[j]),
                        axis=[1, 2],
                    )
                ]
        return jnp.hstack(out)

    return compute_fun


def FCHL19Representation(
    z,
    nRs2=24,
    nRs3=20,
    nFourier=1,
    eta2=0.32,
    eta3=2.7,
    zeta=jnp.pi,
    rcut=8.0,
    acut=8.0,
    two_body_decay=1.8,
    three_body_decay=0.57,
    three_body_weight=13.4,
):
    """Returns a function that computes the FCHL19 Representation."""

    displacement_fn, shift_fn = space.free()
    three_body_weight = jnp.sqrt(eta3 / jnp.pi) * three_body_weight
    twobody_fn = radial_symmetry_functions(
        displacement_fn,
        z,
        nRs2=nRs2,
        eta2=eta2,
        rcut=rcut,
        two_body_decay=two_body_decay,
    )
    threebody_fn = angular_symmetry_functions(
        displacement_fn,
        z,
        nRs3=nRs3,
        nFourier=nFourier,
        eta3=eta3,
        zeta=zeta,
        acut=acut,
        two_body_decay=two_body_decay,
        three_body_decay=three_body_decay,
        three_body_weight=three_body_weight,
    )

    def compute_fn(R):
        return jnp.hstack((twobody_fn(R), threebody_fn(R)))

    return compute_fn


def FCHL19Kernel(z, kappa=rbf):
    """Returns a function that computes the scalar FCHL19 Kernel between two molecules."""
    return AtomicDoubleSumKernel(z, FCHL19Representation(z), kappa)
