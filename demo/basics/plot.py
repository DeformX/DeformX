"""Plot an existing beginner recording without loading CUDA."""

import argparse
import json
from pathlib import Path

from .run import sha256


def plot(run_dir, out):
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle
    from matplotlib.ticker import MaxNLocator

    report = json.loads((run_dir / "report.json").read_text())
    trajectory = run_dir / "trajectory.npz"
    if sha256(trajectory) != report["trajectory_sha256"]:
        raise ValueError("trajectory does not match its report")
    with np.load(trajectory, allow_pickle=False) as data:
        t, p = data["t"], data["positions"]
        settings = report["settings"]
        case = settings["case"]
        fig, (left, right) = plt.subplots(1, 2, figsize=(9, 4.2), layout="constrained")
        contact_off = settings.get("contact_enabled") is False
        title = case.replace("_", " ").title()
        if case in ("obstacle", "self_contact"):
            contact = "Obstacle contact" if case == "obstacle" else "Self-contact"
            title = f"{contact} off (control)" if contact_off else f"{contact} on"
        if case == "tip_target":
            policy = "Zero-action policy (control)" if settings["policy"] == "zero" else "Feedback policy"
            title = f"{title} | {policy}"
        status = "Recorded solver states" if report["complete"] else "Incomplete recording"
        fig.suptitle(f"{title} | {status}", fontsize=14)
        if case == "tip_target":
            target = data["target"]
            left.plot(*p[:, 0, 0, :2].T, color="#c96325", lw=2.5, label="Driven support")
            left.plot(*p[:, 0, -1, :2].T, color="#14788b", ls="--", lw=2.0,
                      marker="o", markevery=[-1], ms=4, label="Free tip")
            left.scatter(*target[:2], color="#20242d", marker="x", s=70, label="Target")
            left.set(xlabel="x (m)", ylabel="y (m)")
            left.set_box_aspect(1)
            left.set_aspect("equal", adjustable="datalim")
            right.plot(t, data["error_m"], color="#14788b", lw=2)
            right.axhline(settings["tolerance_m"], color="#6d737e", ls="--", label="Success tolerance")
            right.set(ylabel="Tip error (m)")
            right.legend(frameon=False)
        else:
            colors = ("#14788b", "#c96325")
            for r in range(p.shape[1]):
                tag = ("Lower EI", "Higher EI")[r] if case == "materials" else "Rod"
                left.plot(p[0, r, :, 0], p[0, r, :, 2], ls="--", color=colors[r],
                          alpha=0.55, label="Start" if r == 0 else None)
                last_label = " at end" if report["complete"] else " at last sample"
                left.plot(p[-1, r, :, 0], p[-1, r, :, 2], color=colors[r], lw=2, label=tag + last_label)
                left.scatter(*p[-1, r, settings["pinned_vertices"][r]][:, [0, 2]].T,
                             marker="s", color=colors[r], s=14, zorder=3)
            left.set(xlabel="x (m)", ylabel="z (m)", aspect="equal")
            if case == "obstacle":
                obj = settings["obstacle"]
                # Projection of the sphere; the rod can deflect out of the x-z plane.
                label = "Sphere (reference)" if contact_off else "Sphere projection"
                left.add_patch(Circle((obj["pos"][0], obj["pos"][2]), obj["radius"],
                                      color="#abb7be", alpha=0.5, label=label))
                right.plot(t, np.linalg.norm(data["mesh_wrench"][:, :3], axis=-1), color=colors[0])
                right.set(ylabel="Solver sphere reaction (N)")
            elif case == "self_contact":
                left.axhline(0, color="#6d737e", lw=1, label="Floor")
                right.plot(t, data["nonlocal_distance_m"], color=colors[0])
                right.axhline(2*settings["radius_m"], color="#6d737e", ls="--",
                              label="Rod diameter reference" if contact_off else "Barrier onset")
                right.set(ylabel="Closest nonlocal distance (m)")
                right.legend(frameon=False)
            elif case == "endpoints":
                displacement = np.linalg.norm(p[:, 0] - p[0, 0], axis=-1)
                right.plot(t, displacement[:, 0], color=colors[1], label="Driven support")
                right.plot(t, displacement[:, p.shape[2]//2], color=colors[0], label="Rod midpoint")
                right.set(ylabel="Displacement from start (m)")
                right.legend(frameon=False)
            else:
                for r in range(p.shape[1]):
                    label = f"EI = {settings['EI_Nm2'][r]:g} N m²" if p.shape[1] > 1 else None
                    right.plot(t, p[:, r, -1, 2], color=colors[r], label=label)
                right.set(ylabel="Free-tip height (m)")
                if p.shape[1] > 1:
                    right.legend(frameon=False)
        left.xaxis.set_major_locator(MaxNLocator(nbins=5))
        left.legend(frameon=False, fontsize=9, loc="lower center",
                    bbox_to_anchor=(0.5, 1.01), ncol=2)
        right.set(xlabel="Simulated time (s)")
        for ax in (left, right):
            ax.spines[["top", "right"]].set_visible(False)
            ax.grid(alpha=0.20)
        with out.open("xb") as dst:
            fig.savefig(dst, format=out.suffix[1:], dpi=180)
        plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path, help="directory with report.json and trajectory.npz")
    parser.add_argument("--out", type=Path, help="new PNG or SVG path (default: RUN_DIR/plot.png)")
    args = parser.parse_args(argv)
    out = args.out if args.out is not None else args.run_dir / "plot.png"
    if out.suffix not in (".png", ".svg"):
        parser.error("--out must end in .png or .svg")
    try:
        plot(args.run_dir, out)
    except (OSError, ValueError, ImportError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
