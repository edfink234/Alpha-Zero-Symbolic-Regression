#!/usr/bin/env python3
"""Correct a symbolic Swift--Hohenberg field with Newton--Krylov.

The stationary PDE is exactly the one used in the accompanying SR code:

    0 = mu*u + nu*u**2 - u**3 - (1 + Laplacian)**2 u.

To avoid discretizing a fourth derivative directly, introduce

    q = (1 + Laplacian)u,

and solve the equivalent mixed system

    q - u - Laplacian(u) = 0,
    mu*u + nu*u**2 - u**3 - q - Laplacian(q) = 0.

Theta derivatives are Fourier-spectral; radial derivatives are second-order
finite differences.  The grid is an annulus [r_min, r_max] because r_min > 0.

Examples
--------
Start small:

    python sh_newton_krylov.py --mu 5 --nu 5 --nr 64 --ntheta 128

Then check resolution:

    python sh_newton_krylov.py --mu 5 --nu 5 --nr 128 --ntheta 256

By default the radial boundary conditions are the common no-flux conditions
u_r = q_r = 0.  ``--bc match_sr`` instead fixes the SR field and its radial
derivative at both radial boundaries; that is useful diagnostically, but it is
not an independently specified physical boundary condition.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import optimize


def sech(x):
    return 1.0 / np.cosh(x)


def sr_field(r, theta, mu, nu):
    """Current R=10, derivative-threshold=0.1 symbolic champion.

    Replace only this function to test another discovered expression.  The
    inputs ``r`` and ``theta`` are broadcast-compatible NumPy arrays; ``mu``
    and ``nu`` are scalars.
    """
    return (
        -0.00195350866881836
        * mu
        * (
            0.667308909317341 * mu
            - 0.332691090682659 * nu
            + np.tanh(np.sin(np.sin(nu)))
            + 5.29374763584655
        )
        + 0.031476477934005 * mu
        + 0.0012356406210145 * nu * np.sin(mu)
        - 0.0102329377521325 * nu
        + (0.0052880833181294 * mu + 0.0052880833181294 * nu)
        * sech(mu + sech(r))
        + (
            -0.0334313806017774 * mu
            + 0.0334313806017774 * nu
            + 0.581565092300317 * np.sin(mu + 2.76775203006243)
            + 0.522444772545678
        )
        * np.sin(
            -2.95106781036787 * mu
            + 0.724380679097289 * nu
            + np.sin(0.916852121490251 * nu + 0.299548075966314)
            + np.sin(
                0.16511117129636 * mu
                + 1.90597661788442 * nu
                + 1.45179723755797
            )
            - 4.18494386826759 * np.tanh(np.sin(mu - 0.960735448003189))
            + 1.9051529156941
        )
        * sech(
            1.0168985896321 * nu * np.sin(np.sin(1.33763310087043 * nu))
            - 1.6971170215574 * nu
            + (1.71623897187253 * mu - 0.0209548095369095)
            * np.sin(0.954250660077747 * mu)
            - 3.34087138239956
        )
        + (
            0.000477731295359431 * mu**2
            + 0.000955462590718861 * mu
            + 0.000477731295359431 * nu
            - 0.218146374751491
        )
        * np.sin(r)
        * np.sin(
            0.440995297287277 * mu * nu * (nu + 5.48026851580401)
            + nu
            - 0.733331112935436 * r * np.sin(theta)
            - theta
            + 2.0 * np.sin(mu)
            + sech(mu)
            + 1.87998639400335
        )
        - 0.111379367583707
        * np.tanh(
            mu
            + 1.86972108084505 * np.cos(0.087233085059265 * nu)
            - 2.77980207884371
        )
        + 0.00508122882766174 * np.tanh(sech(sech(r)))
        + 0.0336184380857523
    )


class PolarOperators:
    def __init__(self, r: np.ndarray, ntheta: int):
        self.r = np.asarray(r, dtype=float)
        self.nr = self.r.size
        self.ntheta = int(ntheta)
        if self.nr < 5:
            raise ValueError("nr must be at least 5")
        self.dr = self.r[1] - self.r[0]
        if not np.allclose(np.diff(self.r), self.dr):
            raise ValueError("this implementation requires uniform radial spacing")

        # With theta_j = 2*pi*j/ntheta, fftfreq(..., d=1/ntheta)
        # returns the integer angular wave numbers.
        self.k_theta = np.fft.fftfreq(self.ntheta, d=1.0 / self.ntheta)

    def dr1(self, u: np.ndarray) -> np.ndarray:
        out = np.empty_like(u)
        h = self.dr
        out[1:-1] = (u[2:] - u[:-2]) / (2.0 * h)
        out[0] = (-3.0 * u[0] + 4.0 * u[1] - u[2]) / (2.0 * h)
        out[-1] = (3.0 * u[-1] - 4.0 * u[-2] + u[-3]) / (2.0 * h)
        return out

    '''
    f'(x) =  (f(x+h) - f(x-h))/2h

    f''(x) = (f'(x+h) - f'(x-h))/2h = (  ((f(x+h+h) - f(x-h+h))/2h) - ((f(x+h-h) - f(x-h-h))/2h) ) /2h
                                    = ((f(x+2h) - f(x))/2h) - ((f(x) - f(x-2h))/2h) )/2h
                                    = (f(x+2h) - 2f(x) - f(x-2h))/(4h^2)
    f'(x) =  (f(x+h) - f(x))/2h

    f''(x) = (f'(x+h) - f'(x-h))/2h = ( ((f(x+2h) - f(x+h))/2h) - ((f(x) - f(x-h))/2h))/2h
                                    = (f(x+2h) - f(x+h) - f(x) + f(x-h))/(4h^2)
    '''
    def dr2(self, u: np.ndarray) -> np.ndarray:
        out = np.empty_like(u)
        h2 = self.dr**2
        out[1:-1] = (u[2:] - 2.0 * u[1:-1] + u[:-2]) / h2
        out[0] = (2.0 * u[0] - 5.0 * u[1] + 4.0 * u[2] - u[3]) / h2
        out[-1] = (
            2.0 * u[-1] - 5.0 * u[-2] + 4.0 * u[-3] - u[-4]
        ) / h2
        return out

    def dtheta2(self, u: np.ndarray) -> np.ndarray:
        u_hat = np.fft.fft(u, axis=1)
        return np.fft.ifft(
            -(self.k_theta[None, :] ** 2) * u_hat, axis=1
        ).real

    def laplacian(self, u: np.ndarray) -> np.ndarray:
        return (
            self.dr2(u)
            + self.dr1(u) / self.r[:, None]
            + self.dtheta2(u) / self.r[:, None] ** 2
        )

    def laplacian_neumann(self, u: np.ndarray) -> np.ndarray:
        """Laplacian with homogeneous radial Neumann BC via ghost points."""
        radial_first = self.dr1(u)
        radial_second = self.dr2(u)
        # Centered ghost-point identities u[-1]=u[1] and u[N]=u[N-2]
        # impose u_r=0 while retaining the PDE at the boundary nodes.
        radial_first[0] = 0.0
        radial_first[-1] = 0.0
        radial_second[0] = 2.0 * (u[1] - u[0]) / self.dr**2
        radial_second[-1] = 2.0 * (u[-2] - u[-1]) / self.dr**2
        return (
            radial_second
            + radial_first / self.r[:, None]
            + self.dtheta2(u) / self.r[:, None] ** 2
        )


class MixedSwiftHohenberg:
    def __init__(
        self,
        ops: PolarOperators,
        mu: float,
        nu: float,
        u_sr: np.ndarray,
        bc_mode: str,
    ):
        self.ops = ops
        self.mu = float(mu)
        self.nu = float(nu)
        self.u_sr = np.asarray(u_sr, dtype=float)
        self.bc_mode = bc_mode
        self.q_sr = self.u_sr + self._laplacian(self.u_sr)
        self.ur_sr = self.ops.dr1(self.u_sr)
        self.shape = self.u_sr.shape
        self.size = self.u_sr.size

    def _laplacian(self, u: np.ndarray) -> np.ndarray:
        if self.bc_mode == "natural":
            return self.ops.laplacian_neumann(u)
        return self.ops.laplacian(u)

    def pack(self, u: np.ndarray, q: np.ndarray) -> np.ndarray:
        return np.concatenate((u.ravel(), q.ravel()))

    def unpack(self, z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        u = z[: self.size].reshape(self.shape)
        q = z[self.size :].reshape(self.shape)
        return u, q

    def initial_vector(self) -> np.ndarray:
        return self.pack(self.u_sr, self.q_sr)

    def residual(self, z: np.ndarray) -> np.ndarray:
        u, q = self.unpack(z)
        lap_u = self._laplacian(u)
        lap_q = self._laplacian(q)

        f1 = q - u - lap_u
        f2 = self.mu * u + self.nu * u**2 - u**3 - q - lap_q

        # The PDE equations apply at radial interior points.  At each radial
        # edge, one row from each mixed equation is replaced by one BC.
        if self.bc_mode == "natural":
            # The ghost-point Laplacians already impose u_r=q_r=0 while the
            # two mixed equations remain enforced at every grid node.
            pass
        elif self.bc_mode == "match_sr":
            # Preserve the SR value and normal derivative at each edge.
            # This is a diagnostic boundary completion, not a physical BC.
            ur = self.ops.dr1(u)
            f1[0] = u[0] - self.u_sr[0]
            f1[-1] = u[-1] - self.u_sr[-1]
            f2[0] = ur[0] - self.ur_sr[0]
            f2[-1] = ur[-1] - self.ur_sr[-1]
        elif self.bc_mode == "clamped":
            ur = self.ops.dr1(u)
            f1[0], f1[-1] = u[0], u[-1]
            f2[0], f2[-1] = ur[0], ur[-1]
        else:
            raise ValueError(f"unknown bc_mode={self.bc_mode!r}")

        return self.pack(f1, f2)

    def pde_residual(self, u: np.ndarray) -> np.ndarray:
        """Original fourth-order PDE residual, evaluated through q."""
        q = u + self._laplacian(u)
        return self.mu * u + self.nu * u**2 - u**3 - q - self._laplacian(q)


def area_rms(a: np.ndarray, r: np.ndarray, radial_slice=slice(None)) -> float:
    values = a[radial_slice]
    weights = np.broadcast_to(r[radial_slice, None], values.shape)
    return float(np.sqrt(np.sum(weights * values**2) / np.sum(weights)))


def save_comparison(
    path: Path,
    r: np.ndarray,
    theta: np.ndarray,
    u0: np.ndarray,
    u1: np.ndarray,
) -> None:
    # Close only the plotting seam; the numerical grid itself has no duplicate.
    theta_closed = np.r_[theta, 2.0 * np.pi]
    u0_closed = np.c_[u0, u0[:, 0]]
    u1_closed = np.c_[u1, u1[:, 0]]
    rr, tt = np.meshgrid(r, theta_closed, indexing="ij")

    vmax = max(np.max(np.abs(u0_closed)), np.max(np.abs(u1_closed)))
    fig, axes = plt.subplots(
        1,
        3,
        figsize=(15, 4.5),
        constrained_layout=True,
        subplot_kw={"projection": "polar"},
    )
    panels = (
        (u0_closed, "symbolic initial field", -vmax, vmax, "coolwarm"),
        (u1_closed, "Newton-corrected field", -vmax, vmax, "coolwarm"),
        (u1_closed - u0_closed, "correction", None, None, "coolwarm"),
    )
    for ax, (field, title, vmin, vmax_i, cmap) in zip(axes, panels):
        if vmin is None:
            correction_limit = np.max(np.abs(field))
            vmin, vmax_i = -correction_limit, correction_limit
        mesh = ax.pcolormesh(
            tt, rr, field, shading="auto", cmap=cmap, vmin=vmin, vmax=vmax_i
        )
        ax.set_title(title)
        fig.colorbar(mesh, ax=ax, shrink=0.8)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mu", type=float, default=5.0)
    p.add_argument("--nu", type=float, default=5.0)
    p.add_argument("--r-min", type=float, default=0.01)
    p.add_argument("--r-max", type=float, default=10.0)
    p.add_argument("--nr", type=int, default=64)
    p.add_argument("--ntheta", type=int, default=128)
    p.add_argument(
        "--bc",
        choices=("natural", "match_sr", "clamped"),
        default="natural",
    )
    p.add_argument("--f-tol", type=float, default=1.0e-8)
    p.add_argument("--maxiter", type=int, default=500)
    p.add_argument("--inner-maxiter", type=int, default=50)
    p.add_argument("--output-prefix", default="sh_newton_correction")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    r = np.linspace(args.r_min, args.r_max, args.nr)
    theta = 2.0 * np.pi * np.arange(args.ntheta) / args.ntheta
    rr, tt = np.meshgrid(r, theta, indexing="ij")

    u0 = np.asarray(sr_field(rr, tt, args.mu, args.nu), dtype=float)
    if u0.shape != rr.shape:
        u0 = np.broadcast_to(u0, rr.shape).copy()
    if not np.all(np.isfinite(u0)):
        raise FloatingPointError("the symbolic initial field contains NaN or inf")

    ops = PolarOperators(r, args.ntheta)
    problem = MixedSwiftHohenberg(ops, args.mu, args.nu, u0, args.bc)
    z0 = problem.initial_vector()

    initial_full = problem.pde_residual(u0)
    # Excluding two radial layers makes this diagnostic comparable for all BC
    # modes and avoids letting boundary stencils dominate a bulk PDE metric.
    interior = slice(2, -2)
    initial_coord_rms = float(np.sqrt(np.mean(initial_full[interior] ** 2)))
    initial_area_rms = area_rms(initial_full, r, interior)
    print(f"grid                 = {args.nr} x {args.ntheta}")
    print(f"mu, nu               = {args.mu}, {args.nu}")
    print(f"radial BC             = {args.bc}")
    print(f"initial coordinate RMS= {initial_coord_rms:.12g}")
    print(f"initial area RMS      = {initial_area_rms:.12g}")

    converged = True
    try:
        z1 = optimize.newton_krylov(
            problem.residual,
            z0,
            method="lgmres",
            inner_maxiter=args.inner_maxiter,
            maxiter=args.maxiter,
            f_tol=args.f_tol,
            line_search="armijo",
            verbose=1,
        )
    except optimize.NoConvergence as exc:
        converged = False
        z1 = np.asarray(exc.args[0])
        print("WARNING: Newton--Krylov did not meet the requested tolerance.")
        print("The last iterate will still be diagnosed and saved.")

    u1, q1 = problem.unpack(np.asarray(z1))
    final_full = problem.pde_residual(u1)
    final_coord_rms = float(np.sqrt(np.mean(final_full[interior] ** 2)))
    final_area_rms = area_rms(final_full, r, interior)
    correction_rms = area_rms(u1 - u0, r)
    initial_field_rms = area_rms(u0, r)
    final_field_rms = area_rms(u1, r)
    relative_correction = correction_rms / max(final_field_rms, np.finfo(float).tiny)
    nonlinear_residual = np.max(np.abs(problem.residual(problem.pack(u1, q1))))

    print(f"converged             = {converged}")
    print(f"final coordinate RMS  = {final_coord_rms:.12g}")
    print(f"final area RMS        = {final_area_rms:.12g}")
    print(f"max mixed residual    = {nonlinear_residual:.12g}")
    print(f"initial field RMS     = {initial_field_rms:.12g}")
    print(f"final field RMS       = {final_field_rms:.12g}")
    print(f"relative correction   = {relative_correction:.12g}")

    if final_field_rms < 0.1 * initial_field_rms:
        print("WARNING: the solve appears to have approached the trivial branch.")

    prefix = Path(args.output_prefix)
    np.savez_compressed(
        prefix.with_suffix(".npz"),
        r=r,
        theta=theta,
        mu=args.mu,
        nu=args.nu,
        bc=args.bc,
        converged=converged,
        u_symbolic=u0,
        u_corrected=u1,
        q_corrected=q1,
        residual_symbolic=initial_full,
        residual_corrected=final_full,
        relative_correction=relative_correction,
    )
    save_comparison(prefix.with_suffix(".png"), r, theta, u0, u1)
    print(f"saved                 = {prefix.with_suffix('.npz')}")
    print(f"saved                 = {prefix.with_suffix('.png')}")


if __name__ == "__main__":
    main()
