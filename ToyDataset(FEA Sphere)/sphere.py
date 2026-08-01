"""
Toy Dataset Generation for PINN Point Set Registration
=======================================================
Generates source and target point clouds for a single sphere
under uniform radial pressure (BC1 only).

Uses analytic linear elasticity solution — no FEA library needed.
Output: sphere_uniform_pressure.npy of shape (2, 1024, 3)
  - index 0 = source (undeformed sphere)
  - index 1 = target (deformed sphere)
Compatible with the existing prostateset_v2 dataloader.

Material properties (uniform, linear elastic):
    lame1 (lambda) = 82.21  kPa  (soft tissue, PZ values from paper)
    lame2 (mu)     = 1.68   kPa
"""

import numpy as np
import os

# ------------------------------------------------------------------ #
#  Material properties — matching the paper's PZ values
# ------------------------------------------------------------------ #
LAMBDA = 82.21   # Lame parameter lambda (kPa)
MU     = 1.68    # Lame parameter mu / shear modulus (kPa)

# Derived quantities
E  = MU * (3*LAMBDA + 2*MU) / (LAMBDA + MU)   # Young's modulus
NU = LAMBDA / (2*(LAMBDA + MU))                # Poisson's ratio

print(f"Material properties:")
print(f"  lambda = {LAMBDA:.2f} kPa")
print(f"  mu     = {MU:.2f} kPa")
print(f"  E      = {E:.2f} kPa")
print(f"  nu     = {NU:.4f}")

# ------------------------------------------------------------------ #
#  Parameters
# ------------------------------------------------------------------ #
N_POINTS   = 1024   # number of points per point cloud
R          = 1.0    # sphere radius
P          = 0.1    # applied pressure magnitude (kPa)
SEED       = 42
OUTPUT_DIR = "toy_datasets"
os.makedirs(OUTPUT_DIR, exist_ok=True)

np.random.seed(SEED)

# ------------------------------------------------------------------ #
#  Point sampling
# ------------------------------------------------------------------ #

def sample_sphere_surface(n, radius=1.0):
    """
    Uniformly sample n points on sphere surface using Fibonacci lattice.
    Returns array of shape (n, 3).
    """
    golden = (1 + np.sqrt(5)) / 2          # golden ratio ~ 1.618
    i      = np.arange(n)                  # indices [0, 1, ..., n-1]

    # polar angle: maps indices evenly from north pole (theta=0) to south pole (theta=pi)
    theta = np.arccos(1 - 2*(i+0.5)/n)

    # azimuthal angle: irrational golden ratio spacing prevents clustering
    phi = 2 * np.pi * i / golden

    # convert spherical to cartesian coordinates
    x = radius * np.sin(theta) * np.cos(phi)
    y = radius * np.sin(theta) * np.sin(phi)
    z = radius * np.cos(theta)

    return np.stack([x, y, z], axis=1)     # (n, 3)


def sample_sphere_volume(n, radius=1.0):
    """
    Uniformly sample n points inside sphere volume using rejection sampling.
    Returns array of shape (n, 3).
    """
    points = []
    while len(points) < n:
        # generate random points in bounding cube
        pts  = np.random.uniform(-radius, radius, (n*2, 3))
        # keep only points inside the sphere
        mask = np.linalg.norm(pts, axis=1) <= radius
        points.extend(pts[mask].tolist())
    return np.array(points[:n])            # (n, 3)


# ------------------------------------------------------------------ #
#  Analytic displacement: uniform radial pressure
# ------------------------------------------------------------------ #
# Lame's solution for sphere under uniform external pressure P:
#
#   u_r = -P * r / (3K)
#
# where K = lambda + 2*mu/3 is the bulk modulus.
# Every point moves radially inward, proportional to its distance from origin.
# ------------------------------------------------------------------ #

def sphere_uniform_pressure_displacement(points, P, lam, mu):
    """
    Compute analytic displacement field for sphere under uniform external pressure.

    Args:
        points: (N, 3) array of point coordinates
        P:      applied pressure magnitude (kPa)
        lam:    Lame parameter lambda (kPa)
        mu:     Lame parameter mu / shear modulus (kPa)

    Returns:
        u: (N, 3) array of displacement vectors
    """
    K     = lam + 2*mu/3                                        # bulk modulus

    r     = np.linalg.norm(points, axis=1, keepdims=True)       # (N,1) distance from origin
    r     = np.where(r == 0, 1e-10, r)                          # avoid division by zero at origin

    r_hat = points / r                                           # (N,3) unit radial vector

    u_r   = -P * r / (3 * K)                                    # (N,1) radial displacement magnitude
                                                                 # negative = moves inward

    u     = u_r * r_hat                                          # (N,3) displacement vector
    return u


# ------------------------------------------------------------------ #
#  Generate dataset
# ------------------------------------------------------------------ #

# 512 surface points + 512 interior points = 1024 total
# matching the paper's convention of boundary + internal points
sphere_surface = sample_sphere_surface(512, R)
sphere_volume  = sample_sphere_volume(512, R)
sphere_source  = np.vstack([sphere_surface, sphere_volume])      # (1024, 3)

# compute analytic displacements for every point
u_sphere = sphere_uniform_pressure_displacement(sphere_source, P, LAMBDA, MU)

# target = source + displacement
sphere_target = sphere_source + u_sphere                         # (1024, 3)

# stack into (2, 1024, 3) — format expected by prostateset_v2
dataset = np.array([sphere_source, sphere_target])               # (2, 1024, 3)

# save dataset
dataset_path = os.path.join(OUTPUT_DIR, "sphere_uniform_pressure.npy")
np.save(dataset_path, dataset)

# save ground truth displacements separately for validation
gt_path = os.path.join(OUTPUT_DIR, "sphere_uniform_pressure_ground_truth.npy")
np.save(gt_path, u_sphere)

# ------------------------------------------------------------------ #
#  Summary
# ------------------------------------------------------------------ #
print(f"\nDataset generated:")
print(f"  Source shape:           {sphere_source.shape}")
print(f"  Target shape:           {sphere_target.shape}")
print(f"  Max displacement:       {np.max(np.linalg.norm(u_sphere, axis=1)):.6f}")
print(f"  Mean displacement:      {np.mean(np.linalg.norm(u_sphere, axis=1)):.6f}")
print(f"  Dataset saved to:       {dataset_path}")
print(f"  Ground truth saved to:  {gt_path}")

print(f"\nTo load in the PINN pipeline:")
print(f"  data = np.load('{dataset_path}', allow_pickle=True)")
print(f"  source = data[0]  # shape (1024, 3) — undeformed sphere")
print(f"  target = data[1]  # shape (1024, 3) — deformed sphere")

print(f"\nTo load ground truth for validation:")
print(f"  gt = np.load('{gt_path}', allow_pickle=True)")
print(f"  # gt[i] = true displacement vector for point i")
print(f"  # compare against network predicted displacements after training")
