"""Normal-speed performance followed by one explicitly labeled slow replay."""
import math

from .contact_loop_strum_path import loop_contract, strokes


def loop_replay_spec(motion):
    version, count, duration, _ = loop_contract(motion.get("pick_preset"), motion.get("probe", False))
    if (motion.get("version") != version or motion.get("duration_s") != duration
            or motion.get("probe") or motion.get("record_hz") != 480):
        raise ValueError("Loop edit requires a complete named 480 Hz performance")
    hz = motion["record_hz"]
    pick = motion["loop_strum"]
    sweep = pick["strokes"]
    if sweep != strokes(count):
        raise ValueError("Loop replay must cover the complete frozen reciprocal schedule")

    def index(t):
        if not math.isfinite(t) or abs(t*hz-round(t*hz)) > 1e-8:
            raise ValueError("Replay boundary is not a recorded source timestamp")
        return round(t*hz)

    windows = [(index(row["window_s"][0]), index(row["window_s"][1])) for row in sweep]
    first, last = index(pick["visible_start_s"]), index(motion["duration_s"])
    if not 0 <= first <= windows[0][0] < windows[-1][1] <= last or (last-first) % 16:
        raise ValueError("Normal chapter cannot cover the full visible source at exactly 1x")
    return dict(chapter_order=["normal", "slow_replay"], record_hz=hz, fps=30,
                normal_source_indices_inclusive=[first,last], normal_source_stride=16,
                normal_view="performance", slow_pair_indices=[0,1],
                slow_source_indices_inclusive=[windows[0][0],windows[1][1]],
                slow_source_stride=1, slow_view="right")


def build_loop_edit(motion):
    spec = loop_replay_spec(motion)
    rows = []
    for name, key, replay in (("normal", "normal", False), ("slow_replay", "slow", True)):
        first, last = spec[key+"_source_indices_inclusive"]
        step = spec[key+"_source_stride"]
        rows.extend(dict(trajectory_index=i, view=spec[key+"_view"],
                         playback_speed=step*30/spec["record_hz"],
                         chapter=name, source_replay=replay)
                    for i in range(first,last+1,step))
    return rows


def verify_loop_edit(motion, frames, spec):
    expected = loop_replay_spec(motion)
    if spec != expected:
        raise ValueError("Loop edit specification differs from the accepted motion")
    first, last = expected["normal_source_indices_inclusive"]
    normal_indices = list(range(first,last+1,16))
    first, last = expected["slow_source_indices_inclusive"]
    slow_indices = list(range(first,last+1))
    if len(frames) != len(normal_indices) + len(slow_indices):
        raise ValueError("Loop edit omitted or duplicated part of a chapter")
    split = len(normal_indices)
    for chapter, part, indices, view, speed, replay in (
            ("normal", frames[:split], normal_indices, "performance", 1., False),
            ("slow_replay", frames[split:], slow_indices, "right", 1/16, True)):
        if [row["trajectory_index"] for row in part] != indices:
            raise ValueError("Source cadence or order changed within " + chapter)
        if any(row.get("chapter") != chapter or row.get("source_replay") is not replay
               or row["view"] != view or row["playback_speed"] != speed for row in part):
            raise ValueError("Camera, speed or replay label changed within " + chapter)
    for i,row in enumerate(frames):
        if (row["phase"] != "HOLD" or row["video_frame"] != i
                or abs(row["video_time_s"]-i/30) > 1e-10
                or abs(row["simulation_time_s"]-row["trajectory_index"]/480) > 1e-10):
            raise ValueError("Loop edit contains an invalid measured source frame")
