import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import skew
from scipy.optimize import fsolve
import time
import os
import argparse

# Number of simulated electrons
NUM_ELECTRONS = 100_000

# Source and plate geometry
SOURCE_RADIUS = 0.25e-3         # [m]
PLATE_DISTANCE = 0.3e-2         # [m]
PLATE_RADIUS = 2e-2             # [m]
SCREEN_DISTANCE = 0.15          # [m]
V_SUPPLY = 3e3                  # [V]

# Particle properties (electron)
ELECTRON_MASS = 9.11e-31        # [kg]
ELECTRON_CHARGE = 1.6e-19       # [C]

# Vacuum permittivity
epsilon_0 = 8.85e-12            # [F/m]

# Initial kinematics
V0_VOLTAGE = 100                # [V]
VEL_Z = np.sqrt((2*ELECTRON_CHARGE*V_SUPPLY / ELECTRON_MASS))    # [m/s]

# Characteristic times
DEFLECTION_TIME = 2 * PLATE_RADIUS / VEL_Z
TOTAL_TIME = SCREEN_DISTANCE / VEL_Z
LENS_TIME = TOTAL_TIME - DEFLECTION_TIME

# Electric field
MEAN_EFIELD = V0_VOLTAGE / PLATE_DISTANCE     # [V/m]
SIGMA_EFIELD = MEAN_EFIELD * 0.01             # [V/m]

# Other system characteristics
K_LENS = VEL_Z * ELECTRON_MASS / (ELECTRON_CHARGE*LENS_TIME*(SCREEN_DISTANCE - 2*PLATE_RADIUS))  # [Kg/C*m^2]
SIGMA_PHOSPHOR = 7e-6           # [m]

def compute_lens_k(m, q, v_z, t_lens, L, R_p):
    """Function to numerically estimate k and omega values"""
    # Required focal length
    f_target = L - 2*R_p

    # Equation to solve for omega
    def eq(omega):
        return v_z / (omega * np.tan(omega * t_lens)) - f_target

    # Initial guess for omega
    omega_guess = np.pi / (2*t_lens) * 0.9

    omega_lens, = fsolve(eq, omega_guess)

    # Compute k_lens
    k_lens = (m/q) * omega_lens**2
    return k_lens, omega_lens

K_LENS, OMEGA_LENS = compute_lens_k(ELECTRON_MASS, ELECTRON_CHARGE, VEL_Z, LENS_TIME, SCREEN_DISTANCE, PLATE_RADIUS)

# Seed for reproducibility and integration step
np.random.seed(seed=23)
DT = 1e-12

def print_constants():
    #Print of important constants of the simulation
    print("\n=== CONSTANTS AND SIMULATION PARAMETERS ===\n")
    print(f"MEAN ELECTRIC FIELD : {MEAN_EFIELD:.3e} V/m")
    print(f"SIGMA ELECTRIC FIELD: {SIGMA_EFIELD:.3e} V/m")
    print(f"VELOCITY ALONG Z    : {VEL_Z:.3e} m/s\n")
    print(f"DEFLECTION TIME : {DEFLECTION_TIME:.3e} s")
    print(f"LENS TIME       : {LENS_TIME:.3e} s")
    print(f"TOTAL TIME      : {TOTAL_TIME:.3e} s")
    print(f"N deflection steps : {int(DEFLECTION_TIME/DT)}\n")
    print(f"SIGMA_PHOSPHOR  : {SIGMA_PHOSPHOR:.3e} m")
    print(f"K_LENS          : {K_LENS:.3e} kg/(C*m²)")
    print("\n===========================================\n")


def print_statistical_results(points, description):
    """Prints statistics about the distribution of xs and ys in mm for a cloud of points."""
    sigma = np.std(points, axis=0) * 1e3
    print("\n===========================================\n")
    print(f"{description}")
    print("\nSpot width:")
    print(f"    -> sigma_x = {sigma[0]:.4f} mm")
    print(f"    -> sigma_y = {sigma[1]:.4f} mm\n")

    # Statistics
    y_vals = points[:, 1]
    print(f"Skewness: {skew(y_vals, bias=False):.4f}")
    mean_y = np.mean(y_vals) * 1e3
    print(f"Simulated mean y: {mean_y:.4f} mm | Theoretical: {y_t*1e3:.4f} mm | Delta = {y_t*1e3 - mean_y:+.4f} mm")


def generate_initial_positions(radius, n):
    """Distributes n electrons uniformly in a disk of given radius."""
    theta = np.random.uniform(0, 2 * np.pi, n)
    r = radius * np.sqrt(np.random.uniform(0, 1, n))
    return r * np.cos(theta), r * np.sin(theta)


def generate_gaussian_numbers(mu, sigma, size=1):
    """Generates Gaussian random numbers using Box-Muller."""
    u1 = np.random.rand(size)
    u2 = np.random.rand(size)
    R = np.sqrt(-2 * np.log(u1))
    theta = 2 * np.pi * u2
    return mu + sigma * R * np.cos(theta)


def theoretical_electron_y(use_lens):
    """Computes the theoretical final y-position of a point-like beam under uniform field."""
    ay = ELECTRON_CHARGE * MEAN_EFIELD / ELECTRON_MASS
    vy = ay * DEFLECTION_TIME
    y_defl = 0.5 * ay * DEFLECTION_TIME**2

    if use_lens:
        ay_lens = -ELECTRON_CHARGE * K_LENS * y_defl / ELECTRON_MASS
        y_fin = y_defl + vy * LENS_TIME + 0.5 * ay_lens * LENS_TIME**2
    else:
        y_fin = y_defl + vy * (TOTAL_TIME - DEFLECTION_TIME)

    return y_fin


def apply_lens(x, y, vx, vy):
    """Applies electrostatic lens to electron beams."""
    ax = -ELECTRON_CHARGE * K_LENS * x / ELECTRON_MASS
    ay = -ELECTRON_CHARGE * K_LENS * y / ELECTRON_MASS

    vx_l = vx + ax * LENS_TIME
    vy_l = vy + ay * LENS_TIME

    x_l = x + vx * LENS_TIME + 0.5 * ax * LENS_TIME**2
    y_l = y + vy * LENS_TIME + 0.5 * ay * LENS_TIME**2

    return x_l, y_l


def add_phosphor_diffusion(points, sigma_phosphor):
    """Adds Gaussian diffusion to simulated points to model the phosphorus response."""
    points_diffused = points.copy()
    points_diffused[:, 0] += generate_gaussian_numbers(0, sigma_phosphor, len(points))
    points_diffused[:, 1] += generate_gaussian_numbers(0, sigma_phosphor, len(points))
    return points_diffused


def simulate_convolution_model(x0, y0):
    """Simulation using the convolution model. Returns positions and velocities."""
    n = len(x0)
    efield = generate_gaussian_numbers(MEAN_EFIELD, SIGMA_EFIELD, n)
    ay = ELECTRON_CHARGE * efield / ELECTRON_MASS
    vy = ay * DEFLECTION_TIME
    vx = np.zeros_like(vy)

    y_defl = y0 + 0.5 * ay * DEFLECTION_TIME**2
    x_defl = x0

    return x_defl, y_defl, vx, vy


def simulate_first_order_model(x0, y0):
    """Simulation using the first-order model. Returns positions and velocities at the edge of the plates."""
    n = len(x0)
    sigma = epsilon_0 * MEAN_EFIELD
    v_z = VEL_Z

    x, y = np.copy(x0), np.copy(y0)
    z = np.full(n, -PLATE_RADIUS)
    vx, vy = np.zeros(n), np.zeros(n)

    n_step = int(DEFLECTION_TIME / DT)

    for _ in range(n_step):
        z += v_z * DT
        rho = np.sqrt(x**2 + z**2)
        yc = y + PLATE_DISTANCE / 2

        Ey = (sigma / epsilon_0) * (1 - (3 * yc * (PLATE_DISTANCE - yc)) / (2 * PLATE_RADIUS**2))
        Erho = (3 * sigma) / (2 * epsilon_0) * (rho * (PLATE_DISTANCE - 2 * yc) * yc * (PLATE_DISTANCE - yc)) / (PLATE_RADIUS**4)

        ay = ELECTRON_CHARGE * Ey / ELECTRON_MASS
        ax = ELECTRON_CHARGE * Erho * (x / rho) / ELECTRON_MASS

        vy += ay * DT
        y += vy * DT
        vx += ax * DT
        x += vx * DT

    return x, y, vx, vy


def propagate_to_screen(x, y, vx, vy, use_lens):
    """Simulation of the motion of electrons after the plates"""
    if use_lens:
        return apply_lens(x, y, vx, vy)
    else:
        x_final = x + vx * (TOTAL_TIME - DEFLECTION_TIME)
        y_final = y + vy * (TOTAL_TIME - DEFLECTION_TIME)
        return x_final, y_final


def simulate_electron_batch(x0, y0, model):
    """Dispatcher for the different simulation models. Returns positions and velocities."""
    if model == 'convolution':
        return simulate_convolution_model(x0, y0)
    elif model == 'first_order':
        return simulate_first_order_model(x0, y0)
    else:
        raise ValueError("Unrecognized model")


def plot_full_results(points, model, y_theoretical=None, sigma_phosphor=None, save_dir="plots"):
    """Creates X/Y histograms, scatter plots of impact points and saves the image to file."""

    # Create folder if it does not exist
    os.makedirs(save_dir, exist_ok=True)

    points_to_show = add_phosphor_diffusion(points, sigma_phosphor) if sigma_phosphor else points

    fig, axs = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)
    fig.suptitle(
        f"Simulation model: {model.upper()}" +
        (f" + phosphor diffusion " if sigma_phosphor else ""),
        fontsize=16, weight='bold'
    )

    axs[0].hist(points_to_show[:, 0]*1e3, bins=50,
                color='dodgerblue' if sigma_phosphor is None else 'orange',
                edgecolor='black', alpha=0.8)
    axs[0].set_title('X Distribution')
    axs[0].set_xlabel('x [mm]')
    axs[0].set_ylabel('Count')

    axs[1].hist(points_to_show[:, 1]*1e3, bins=50,
                color='crimson' if sigma_phosphor is None else 'firebrick',
                edgecolor='black', alpha=0.8)
    axs[1].set_title('Y Distribution')
    axs[1].set_xlabel('y [mm]')
    axs[1].set_ylabel('Count')

    x_mm = points_to_show[:, 0]*1e3
    y_mm = points_to_show[:, 1]*1e3
    axs[2].scatter(x_mm, y_mm, s=2, alpha=0.3,
                   color='purple' if sigma_phosphor is None else 'darkorange')
    axs[2].set_title('2D Impact Points')
    axs[2].set_xlabel('x [mm]')
    axs[2].set_ylabel('y [mm]')
    axs[2].set_aspect('equal', adjustable='datalim')

    if y_theoretical is not None and sigma_phosphor is None:
        axs[2].scatter(0, y_theoretical * 1e3, color='black', marker='x', s=100, label='Theoretical position')
        axs[2].legend(loc='upper right')

    for ax in axs:
        ax.grid(True, linestyle='--', alpha=0.5)
        for spine in ax.spines.values():
            spine.set_visible(False)

    description = model.upper() + " with_diffusion" if sigma_phosphor else model.upper() + " without_diffusion"
    print_statistical_results(points_to_show, description)

    # Clear and readable filename
    filename = f"{description.replace(' ', '_')}.png"
    filepath = os.path.join(save_dir, filename)

    plt.savefig(filepath, dpi=300)
    plt.close(fig) 


if __name__ == "__main__":
    # Parser
    parser = argparse.ArgumentParser(description="Electron beam simulator")
    parser.add_argument("--model", choices=["convolution", "first_order", "all"], default="all",
                        help="Select the simulation model ('all' is default)")
    parser.add_argument("--lens", choices=["with", "without", "all"], default="all",
                        help="Select whether to use the lens ('all' is default)")
    parser.add_argument(
        "--diffusione", choices=["with", "without", "all"], default="all",
        help="Select simulation with phosphor diffusion, without, or both (default: all)")
    parser.add_argument("--outdir", default="plots", help="Folder to save results (default is 'plots')")
    args = parser.parse_args()

    # Print constants
    print_constants()

    # Generate initial particle positions in the x-y plane
    x0, y0 = generate_initial_positions(SOURCE_RADIUS, NUM_ELECTRONS)

    # Define models and lens states to simulate
    models = ["convolution", "first_order"] if args.model == "all" else [args.model]
    lens_states = [False, True] if args.lens == "all" else [args.lens == "with"]
    diffusion_states = [False, True] if args.diffusione == "all" else [args.diffusione == "with"]

    # Simulation loop
    for model in models:
        # Simulate electron batch once per model
        x_defl, y_defl, vx, vy = simulate_electron_batch(x0, y0, model)
        
        for use_lens in lens_states:
            lens_state = "with_lens" if use_lens else "without_lens"
            
            # Apply lens if needed
            x_final, y_final = propagate_to_screen(x_defl, y_defl, vx, vy, use_lens)
            
            y_t = theoretical_electron_y(use_lens)

            for use_diffusion in diffusion_states:
                diffusion_label = "with_diffusion" if use_diffusion else "without_diffusion"
                sigma = SIGMA_PHOSPHOR if use_diffusion else None

                points = np.column_stack((x_final, y_final))
                
                plot_full_results(
                    points,
                    model=f"{model} ({lens_state})",
                    y_theoretical=y_t if not use_diffusion else None,
                    sigma_phosphor=sigma,
                    save_dir=args.outdir
                )



