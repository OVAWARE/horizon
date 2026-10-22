#!/usr/bin/env python3
"""Far terrain benchmark: how fast the distant view fills in and what it costs.

Opens a fresh seeded creative world (seed 1234) on a headless X server, stands
on a mountain looking over the land, and for each LOD distance measures:

- fill: seconds until every target tile of the far view is sampled, and the
  tile counts and GPU memory when it is;
- still: frame times looking at the view, after it has filled;
- spyglass: the same with a spyglass raised, and how long its finer detail
  takes to arrive;
- flight: frame times flying 32 blocks a second (elytra speed) across it.

    python3 tools/farview_bench.py --label base --distances 128,512,1024
    python3 tools/farview_bench.py --compare base mine

The game renders offscreen on the machine's GPU (`--offscreen`), so the numbers
are the real GPU's while no window shows on the desktop. Screenshots of each
view are written beside the JSON, under benchmarks/farview/<label>/.
"""
import argparse
import datetime
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

try:
    from . import lab
except ImportError:
    import lab

ROOT = Path(__file__).resolve().parent.parent
SEED = "1234"
WORLD = "FarViewBench"
# On the mountain of tools/graphics_bench.py, looking west over the land.
VISTA = (560.0, 262.0, 860.0)
YAW, PITCH = 90.0, 2.0
FLIGHT = ((560.0, 262.0, 860.0), (176.0, 262.0, 860.0))
OPTIONS = """version:4790
enableVsync:false
maxFps:260
inactivityFpsLimit:minimized
renderDistance:{near}
simulationDistance:8
fov:0.0
guiScale:0
"""


class Game:
    def __init__(self, ctl, socket, env, cwd):
        self.ctl, self.socket, self.env, self.cwd = ctl, socket, env, cwd

    def run(self, *words, check=True, timeout=900):
        out = subprocess.run(
            [str(self.ctl), "--socket", str(self.socket), *map(str, words)],
            cwd=self.cwd, env=self.env, capture_output=True, text=True, timeout=timeout,
        )
        line = out.stdout.strip().splitlines()[-1] if out.stdout.strip() else "{}"
        reply = json.loads(line) if line.startswith("{") else {"error": out.stderr.strip()}
        if check and "error" in reply:
            raise RuntimeError(f"{' '.join(map(str, words))}: {reply} {out.stderr.strip()}")
        return reply

    def lod(self):
        return self.run("status").get("lod", {})

    def wait_filled(self, limit):
        """Seconds until every target tile is sampled and the active set stops
        changing for a second; the last status too. `limit` caps the wait."""
        if self.distance == 0:
            return 0.0, self.lod()
        start = time.monotonic()
        stable_since, last_active = None, None
        while True:
            lod = self.lod()
            elapsed = time.monotonic() - start
            target, sampled = lod.get("target_tiles", 0), lod.get("sampled_target_tiles", 0)
            active = lod.get("active_tiles", 0)
            done = target > 0 and sampled >= target
            if done and active == last_active:
                stable_since = stable_since or elapsed
                if elapsed - stable_since >= 1.0:
                    return stable_since, lod
            else:
                stable_since = None
            last_active = active
            if elapsed >= limit:
                return None, lod
            time.sleep(0.25)


def frame_line(name, s):
    return (f"  {name:<9} {s.get('avg_fps', 0):7.1f} fps  1% low {s.get('low_1pct_fps', 0):6.1f}"
            f"  p99 {s.get('p99_ms', 0):6.2f} ms  cpu {((s.get('cpu') or {}).get('avg_ms') or 0):5.2f} ms"
            f"  gpu {((s.get('gpu') or {}).get('avg_ms') or 0):5.2f} ms")


def run_distance(args, ctl, env, cwd, distance, shots):
    (cwd / "options.txt").write_text(OPTIONS.format(near=args.near))
    graphics = cwd / "horizon-graphics.txt"
    graphics.unlink(missing_ok=True)
    if args.graphics:
        graphics.write_text("".join(line + "\n" for line in args.graphics.split(",")))
    socket = cwd / "control.sock"
    game = Game(ctl, socket, env, cwd)
    game.distance = distance
    launch = [
        "launch", "--world", WORLD, "--seed", SEED, "--creative",
        "--render-distance", args.near, "--resources", args.resources,
        "--no-pause", "--weather", "clear", "--time", "6000", "--offscreen",
    ]
    game.run(*launch)
    result = {}
    try:
        game.run("wait-ready", 300)
        game.run("tp", *VISTA)
        game.run("look", YAW, PITCH)
        game.run("wait-ready", 300)
        game.run("lod-distance", distance)
        seconds, lod = game.wait_filled(args.limit)
        status = game.run("status")
        r = status.get("renderer") or {}
        result["fill_seconds"] = seconds
        result["lod"] = lod
        result["renderer"] = {k: r.get(k) for k in (
            "adapter", "backend", "drawn_sections", "section_draws", "arena_bytes",
            "section_vertex_bytes", "gpu_allocated_bytes", "resident_sections", "lod_levels")}
        print(f"  lod {distance:>4}: filled in {seconds if seconds is None else round(seconds, 1)} s,"
              f" {lod.get('active_tiles')} tiles, arena {(r.get('arena_bytes') or 0) / 2**20:.0f} MiB",
              flush=True)
        time.sleep(args.settle)
        result["still"] = game.run("bench", args.seconds)
        print(frame_line("still", result["still"]), flush=True)
        game.run("screenshot", shots / f"lod{distance}-view.png")

        game.run("give", "minecraft:spyglass")
        game.run("hotbar", 1)
        game.run("mouse", "right", "down")
        time.sleep(1.0)
        seconds, lod = game.wait_filled(args.limit)
        result["spyglass_fill_seconds"] = seconds
        result["spyglass_lod"] = lod
        time.sleep(args.settle)
        result["spyglass"] = game.run("bench", args.seconds)
        print(f"  spyglass refined in {seconds if seconds is None else round(seconds, 1)} s,"
              f" {lod.get('active_tiles')} tiles", flush=True)
        print(frame_line("spyglass", result["spyglass"]), flush=True)
        game.run("screenshot", shots / f"lod{distance}-spyglass.png")
        game.run("mouse", "right", "up")
        game.wait_filled(args.limit)

        (a, b) = FLIGHT
        result["flight"] = game.run("fly", *a, *b, args.flight, PITCH)
        print(frame_line("flight", result["flight"]), flush=True)
    finally:
        game.run("quit", check=False)
        for _ in range(100):
            if not socket.exists():
                break
            time.sleep(0.2)
        log = socket.with_suffix(".log")
        if log.exists():
            shutil.copy(log, shots / f"lod{distance}.log")
    return result


def compare(before, after):
    load = lambda label: json.loads((ROOT / "benchmarks/farview" / f"{label}.json").read_text())
    a, b = load(before), load(after)
    arrow = lambda x, y, f: f"{f(x)} → {f(y)}"
    secs = lambda v: "—" if v is None else f"{v:.1f}"
    fps = lambda v: f"{v:.0f}"
    print("| LOD | fill s | still FPS | still 1% low | spyglass fill s | spyglass FPS | flight FPS | flight 1% low |")
    print("| --- | --- | --- | --- | --- | --- | --- | --- |")
    for d, old in a["distances"].items():
        new = b["distances"].get(d)
        if not new:
            continue
        g = lambda r, k, f="avg_fps": (r.get(k) or {}).get(f, 0)
        print(f"| {d} | {arrow(old.get('fill_seconds'), new.get('fill_seconds'), secs)} "
              f"| {arrow(g(old, 'still'), g(new, 'still'), fps)} "
              f"| {arrow(g(old, 'still', 'low_1pct_fps'), g(new, 'still', 'low_1pct_fps'), fps)} "
              f"| {arrow(old.get('spyglass_fill_seconds'), new.get('spyglass_fill_seconds'), secs)} "
              f"| {arrow(g(old, 'spyglass'), g(new, 'spyglass'), fps)} "
              f"| {arrow(g(old, 'flight'), g(new, 'flight'), fps)} "
              f"| {arrow(g(old, 'flight', 'low_1pct_fps'), g(new, 'flight', 'low_1pct_fps'), fps)} |")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--label", default="run")
    p.add_argument("--distances", default="128,512,1024")
    p.add_argument("--near", type=int, default=12, help="full-detail render distance")
    p.add_argument("--seconds", type=float, default=8.0, help="seconds per still measurement")
    p.add_argument("--flight", type=float, default=12.0, help="seconds of flight (384 blocks)")
    p.add_argument("--settle", type=float, default=1.0)
    p.add_argument("--limit", type=float, default=240.0, help="longest wait for the view to fill")
    p.add_argument("--graphics", help="comma-separated horizon-graphics.txt lines, e.g. resolutionScale:50")
    p.add_argument("--resources", default=str(lab.RESOURCES))
    p.add_argument("--build", default=str(ROOT / "target/release"))
    p.add_argument("--compare", nargs=2, metavar=("BEFORE", "AFTER"))
    args = p.parse_args()
    if args.compare:
        compare(*args.compare)
        return
    ctl = Path(args.build).resolve() / "horizon-ctl"
    if not ctl.exists():
        sys.exit(f"{ctl} missing: cargo build --release -p horizon-client --bins")
    cwd = ROOT / ".local/farview-bench"
    shutil.rmtree(cwd / "saves" / WORLD, ignore_errors=True)
    cwd.mkdir(parents=True, exist_ok=True)
    shots = ROOT / "benchmarks/farview" / args.label
    shots.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, HORIZON_NO_AUDIO="1")
    xvfb = None
    if not args.__dict__.get("display"):
        xvfb = subprocess.Popen(["Xvfb", ":77", "-screen", "0", "1920x1080x24", "-nolisten", "tcp"],
                                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        env["DISPLAY"] = ":77"
        env.pop("WAYLAND_DISPLAY", None)
        time.sleep(1.0)
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                            capture_output=True, text=True).stdout.strip()
    report = {
        "label": args.label, "date": datetime.date.today().isoformat(), "commit": commit,
        "host": os.uname().nodename, "seed": SEED, "near": args.near,
        "vista": VISTA, "yaw": YAW, "pitch": PITCH, "distances": {},
        "load_before": os.getloadavg(),
    }
    try:
        for d in [int(x) for x in args.distances.split(",")]:
            shutil.rmtree(cwd / "saves" / WORLD, ignore_errors=True)
            report["distances"][str(d)] = run_distance(args, ctl, env, cwd, d, shots)
    finally:
        if xvfb:
            xvfb.terminate()
    report["load_after"] = os.getloadavg()
    out = ROOT / "benchmarks/farview" / f"{args.label}.json"
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(out)


if __name__ == "__main__":
    main()
