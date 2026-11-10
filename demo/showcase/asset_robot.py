"""Kinematics from imported USD joint frames; no hand-built robot dimensions."""
import numpy as np


def load_profile(path):
    import json
    from pathlib import Path
    profile = json.loads(Path(path).read_text())
    return validate_profile(profile)


def validate_profile(profile):
    import hashlib
    from pathlib import Path
    if profile.get("schema_version") != 1:
        raise ValueError("Unsupported asset profile version")
    for name in ("robot", "rack"):
        asset = profile[name]
        if asset["meters_per_unit"] != 1 or asset["up_axis"] != "Z":
            raise ValueError("Expected meter/Z-up asset profile")
        if not asset["files"] or asset["path"] not in [x["path"] for x in asset["files"]]:
            raise ValueError("Root asset is absent from dependency manifest")
        for item in asset["files"]:
            if hashlib.sha256(Path(item["path"]).read_bytes()).hexdigest() != item["sha256"]:
                raise ValueError("Asset dependency checksum mismatch: " + item["path"])
    Kinematics(profile["robot"]["kinematics"])
    return profile


def motion(axis, value, kind):
    t = np.eye(4)
    axis = np.asarray(axis, float)
    if kind == "prismatic":
        t[:3, 3] = axis * value
    elif kind == "revolute":
        x, y, z = axis
        cross = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
        t[:3, :3] = np.eye(3) + np.sin(value) * cross + (1 - np.cos(value)) * cross @ cross
    return t


class Kinematics:
    def __init__(self, model):
        self.model = model
        self.placement = np.asarray(model["placement"], float)
        self.roots = {key: np.asarray(value, float) for key, value in model["roots"].items()}
        known = set(self.roots)
        pending = list(model["joints"])
        self.joints = []
        while pending:
            ready = [j for j in pending if j["parent"] in known]
            if not ready:
                raise ValueError("Unresolved joint tree: " + str([j["name"] for j in pending]))
            for joint in ready:
                if joint["child"] in known:
                    raise ValueError("Duplicate link in joint tree: " + joint["child"])
                self.joints.append((joint, np.asarray(joint["parent_frame"]), np.linalg.inv(joint["child_frame"])))
                known.add(joint["child"])
                pending.remove(joint)
        self.names = model["control_joints"]
        self.tcp_transform = np.asarray(model["tcp_transform"], float)

    def forward(self, q):
        if len(q) != len(self.names) or not np.isfinite(q).all():
            raise ValueError("Invalid robot joint vector")
        values = dict(self.model.get("preset", {}))
        values.update(zip(self.names, q))
        frames = {key: self.placement @ value for key, value in self.roots.items()}
        for joint, parent, child_inverse in self.joints:
            frames[joint["child"]] = (frames[joint["parent"]] @ parent
                @ motion(joint["axis"], values.get(joint["name"], 0), joint["kind"]) @ child_inverse)
        tcp = frames[self.model["tcp_link"]] @ self.tcp_transform
        return frames, tcp


def forward(q, model):
    return Kinematics(model).forward(q)


def solve_path(targets, model):
    from scipy.optimize import least_squares
    k = Kinematics(model)
    joint_map = {j["name"]: j for j in model["joints"]}
    lower = np.array([joint_map[n]["lower"] for n in k.names])
    upper = np.array([joint_map[n]["upper"] for n in k.names])
    desired = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]], float)
    q = np.clip(np.array(model.get("seed", [0] * len(k.names))), lower + 1e-5, upper - 1e-5)
    records, vectors, errors, angles = [], [], [], []
    link_names = list(k.forward(q)[0])
    for index, target in enumerate(targets):
        prior = q.copy()

        def residual(x):
            _, tcp = k.forward(x)
            return np.r_[4 * (tcp[:3, 3] - target), (tcp[:3, :3] - desired).ravel(), 1e-5 * (x - prior)]

        seeds = [q]
        if index == 0:
            rng = np.random.default_rng(41)
            seeds += [rng.uniform(np.maximum(lower, -np.pi), np.minimum(upper, np.pi)) for _ in range(8)]
        best = None
        for seed in seeds:
            fit = least_squares(residual, seed, bounds=(lower, upper), max_nfev=250,
                                ftol=1e-10, xtol=1e-10, gtol=1e-10)
            if best is None or np.linalg.norm(residual(fit.x)) < np.linalg.norm(residual(best.x)):
                best = fit
            if np.linalg.norm(residual(best.x)) < 1e-4:
                break
        q = best.x
        frames, tcp = k.forward(q)
        error = float(np.linalg.norm(tcp[:3, 3] - target))
        angle = float(np.arccos(np.clip((np.trace(desired.T @ tcp[:3, :3]) - 1) / 2, -1, 1)))
        if error > 0.001 or angle > 0.01:
            raise RuntimeError(f"Asset IK failed at frame {index}: {error:.5f} m, {angle:.5f} rad")
        records.append([frames[n] for n in link_names])
        vectors.append(q.copy())
        errors.append(error)
        angles.append(angle)
    return dict(joints=np.asarray(vectors), link_transforms=np.asarray(records),
                link_names=np.asarray(link_names), ik_error=np.asarray(errors), orientation_error=np.asarray(angles))
