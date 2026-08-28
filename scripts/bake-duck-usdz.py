#!/usr/bin/env python3
"""Bake a walking robot into a USDZ for iPhone AR Quick Look.

iOS has no WebXR, but it has AR Quick Look: Safari opens a `.usdz` in the
native AR viewer, which places it on a detected surface at true scale. This
tool produces that file with the walking baked in — it rolls the real
`alpha_walking.onnx` policy out in MuJoCo (the same control pipeline the web
viewer runs), records every body's world transform at 25 fps while the robot
walks one full circle, and writes the visual meshes plus those transforms as
a time-sampled USD animation. Quick Look loops it: a real-size microduck
walking laps on your desk.

    scripts/bake-duck-usdz.py path/to/robot_walk.xml [out.usdz]

The MJCF must be the app repository's robot dir (its `scene.xml` and STL
assets are used for physics and visuals). Needs numpy, mujoco, onnxruntime
and usd-core. The web viewer links to the committed output — regenerate and
re-commit when the model or gait changes.
"""

import re
import struct
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent

FPS = 25.0
CONTROL_HZ = 50.0

# duck-control/src/model.rs DEFAULT_POSITION minus the mouth (policy order).
HOME = np.array([
    0.0, -0.0873, -0.4579, -0.0049, 0.4530,
    0.3491, 0.3491, 0.0, 0.0,
    0.0, 0.0873, 0.4579, 0.0049, -0.4530,
])

# The walk command: forward plus yaw so the loop closes on itself — a circle
# ends where it began, which is what makes Quick Look's looping seamless-ish.
CMD = (0.25, 0.0, 0.5)


def load_stl(path: Path) -> np.ndarray:
    data = path.read_bytes()
    (n,) = struct.unpack_from("<I", data, 80)
    tris = np.frombuffer(data, dtype=np.uint8, offset=84).reshape(n, 50)
    return tris[:, 12:48].copy().view("<f4").reshape(n, 3, 3)


def decimate(tris: np.ndarray, budget: int) -> np.ndarray:
    """Vertex clustering, as bake-duck-mesh.py: Quick Look wants a phone-sized
    mesh, not the CAD export."""
    lo = tris.min(axis=(0, 1))
    cell = max(float(np.linalg.norm(tris.max(axis=(0, 1)) - lo)) / 96.0, 1e-5)
    while True:
        flat = tris.reshape(-1, 3)
        keys = np.floor((flat - lo) / cell).astype(np.int64)
        uniq, inverse = np.unique(keys, axis=0, return_inverse=True)
        faces = inverse.reshape(-1, 3)
        keep = ((faces[:, 0] != faces[:, 1]) & (faces[:, 1] != faces[:, 2])
                & (faces[:, 0] != faces[:, 2]))
        faces = faces[keep]
        if len(faces):
            faces = np.unique(np.sort(faces, axis=1), axis=0)
        if len(faces) <= budget:
            break
        cell *= 1.3
    verts = np.zeros((len(uniq), 3))
    counts = np.bincount(inverse, minlength=len(uniq)).astype(float)
    for axis in range(3):
        verts[:, axis] = np.bincount(inverse, weights=flat[:, axis].astype(float),
                                     minlength=len(uniq))
    verts /= counts[:, None]
    return verts[faces].astype(np.float32)


def parse_visual(mjcf_path: Path) -> tuple[list[dict], dict]:
    """The visual model: named bodies and their mesh parts, as in duck-viewer."""
    root = ET.parse(mjcf_path).getroot()
    files = {}
    for m in root.iter("mesh"):
        f = m.get("file")
        files[m.get("name") or Path(f).stem] = f
    materials = {
        m.get("name"): [float(x) for x in (m.get("rgba") or "0.5 0.5 0.5 1").split()]
        for m in root.iter("material")
    }
    bodies = []

    def fl(text, default):
        return [float(x) for x in (text or default).split()]

    def walk(el):
        bodies.append({
            "name": el.get("name"),
            "parts": [
                {
                    "mesh": g.get("mesh"),
                    "rgba": materials.get(g.get("material"), [0.5, 0.5, 0.5, 1]),
                    "pos": fl(g.get("pos"), "0 0 0"),
                    "quat": fl(g.get("quat"), "1 0 0 0"),
                }
                for g in el.findall("geom")
                if g.get("class") == "visual" and g.get("type") == "mesh"
            ],
        })
        for child in el.findall("body"):
            walk(child)

    for top in root.find("worldbody").findall("body"):
        walk(top)
    return bodies, files


def rollout(robot_dir: Path, bodies: list[dict]):
    """Walk one circle in MuJoCo with the real policy; return per-frame world
    poses (pos, quat) for every visual body, sampled at FPS."""
    import mujoco
    import onnxruntime as ort

    m = mujoco.MjModel.from_xml_path(str(robot_dir / "scene.xml"))
    m.opt.timestep = 0.005
    d = mujoco.MjData(m)
    decim = round(1.0 / CONTROL_HZ / m.opt.timestep)

    body_ids = []
    for b in bodies:
        bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, b["name"])
        if bid < 0:
            raise SystemExit(f"visual body {b['name']!r} not in the physics model")
        body_ids.append(bid)

    jadr = [m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)]
            for n in ["left_hip_yaw", "left_hip_roll", "left_hip_pitch", "left_knee",
                      "left_ankle", "neck_pitch", "head_pitch", "head_yaw", "head_roll",
                      "right_hip_yaw", "right_hip_roll", "right_hip_pitch",
                      "right_knee", "right_ankle"]]
    jadr = np.array(jadr)

    fj = m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "trunk_base_freejoint")]
    d.qpos[fj + 2] = 0.125
    d.qpos[fj + 3:fj + 7] = [1, 0, 0, 0]
    d.qpos[jadr] = HOME
    d.ctrl[:] = HOME
    mujoco.mj_forward(m, d)

    sess = ort.InferenceSession(str(REPO / "policies" / "alpha_walking.onnx"),
                                providers=["CPUExecutionProvider"])
    la = np.zeros(14, np.float32)

    frames = []
    yaw_acc, prev_yaw = 0.0, None
    record_from = 3.0  # gait settled
    per_frame = int(round(CONTROL_HZ / FPS))
    max_ticks = int(60 * CONTROL_HZ)
    for tick in range(max_ticks):
        t = tick / CONTROL_HZ
        r = min(1.0, max(0.0, t - 0.5))
        q = d.qpos[fj + 3:fj + 7]
        w, x, y, z = q
        R = np.array([
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ])
        obs = np.concatenate([
            d.qvel[3:6], R.T @ [0, 0, -1.0],
            d.qpos[jadr] - HOME, d.qvel[6:20],
            la, [CMD[0] * r, CMD[1] * r, CMD[2] * r], [0, 0, 0, 0], [0, 0, 0, 0, 0, 0],
        ]).astype(np.float32)[None, :]
        a = sess.run(None, {"obs": obs})[0][0]
        la = a.copy()
        d.ctrl[:] = HOME + a.astype(np.float64)
        for _ in range(decim):
            mujoco.mj_step(m, d)

        yaw = np.arctan2(2 * (q[0] * q[3] + q[1] * q[2]), 1 - 2 * (q[2] ** 2 + q[3] ** 2))
        if prev_yaw is not None:
            dy = yaw - prev_yaw
            if dy > np.pi:
                dy -= 2 * np.pi
            if dy < -np.pi:
                dy += 2 * np.pi
            if t >= record_from:
                yaw_acc += dy
        prev_yaw = yaw

        if 1 - 2 * (q[1] ** 2 + q[2] ** 2) < 0.3:
            raise SystemExit(f"the rollout fell over at t={t:.1f}s — retune CMD")
        if t >= record_from and tick % per_frame == 0:
            frames.append([(d.xpos[b].copy(), d.xquat[b].copy()) for b in body_ids])
            if abs(yaw_acc) >= 2 * np.pi:
                break
    else:
        print(f"note: circle did not close in {max_ticks / CONTROL_HZ:.0f}s "
              f"(yaw {np.degrees(yaw_acc):.0f}°); looping what we have")
    return frames


def sanitize(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]", "_", name)


def build_usdz(bodies, files, mesh_dir: Path, frames, out_path: Path) -> None:
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade, UsdUtils, Vt

    usdc = out_path.with_suffix(".usdc")
    stage = Usd.Stage.CreateNew(str(usdc))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.y)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    stage.SetStartTimeCode(0)
    stage.SetEndTimeCode(len(frames) - 1)
    stage.SetTimeCodesPerSecond(FPS)
    stage.SetFramesPerSecond(FPS)

    root = UsdGeom.Xform.Define(stage, "/Duck")
    stage.SetDefaultPrim(root.GetPrim())
    # MJCF is z-up; Quick Look wants y-up. Row-vector convention: a point is
    # p·M_body·M_root, so the root op carries the up-axis change.
    tilt = Gf.Matrix4d().SetRotate(Gf.Rotation(Gf.Vec3d(1, 0, 0), -90))
    root.AddTransformOp().Set(tilt)

    # One material per distinct rgba.
    mat_cache = {}

    def material(rgba):
        key = tuple(round(c, 4) for c in rgba)
        if key not in mat_cache:
            name = f"/Duck/Materials/m{len(mat_cache)}"
            mat = UsdShade.Material.Define(stage, name)
            shader = UsdShade.Shader.Define(stage, name + "/pbr")
            shader.CreateIdAttr("UsdPreviewSurface")
            shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(
                Gf.Vec3f(float(key[0]), float(key[1]), float(key[2])))
            shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.55)
            shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
            if key[3] < 0.99:
                shader.CreateInput("opacity", Sdf.ValueTypeNames.Float).Set(float(key[3]))
            mat.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
            mat_cache[key] = mat
        return mat_cache[key]

    mesh_cache = {}

    def mesh_tris(name):
        if name not in mesh_cache:
            tris = load_stl(mesh_dir / files[name])
            diag_mm = float(np.linalg.norm(tris.max(axis=(0, 1)) - tris.min(axis=(0, 1)))) * 1000
            budget = int(np.clip(diag_mm * 40, 800, 7000))
            mesh_cache[name] = decimate(tris, budget)
        return mesh_cache[name]

    total = 0
    for i, b in enumerate(bodies):
        xform = UsdGeom.Xform.Define(stage, f"/Duck/{sanitize(b['name'])}")
        op = xform.AddTransformOp()
        for f, frame in enumerate(frames):
            pos, quat = frame[i]
            mat = Gf.Matrix4d().SetTransform(
                Gf.Rotation(Gf.Quatd(float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3]))),
                Gf.Vec3d(float(pos[0]), float(pos[1]), float(pos[2])))
            op.Set(mat, Usd.TimeCode(f))
        for j, p in enumerate(b["parts"]):
            tris = mesh_tris(p["mesh"])
            total += len(tris)
            prim = UsdGeom.Mesh.Define(stage, f"/Duck/{sanitize(b['name'])}/part{j}")
            local = Gf.Matrix4d().SetTransform(
                Gf.Rotation(Gf.Quatd(*p["quat"])), Gf.Vec3d(*p["pos"]))
            UsdGeom.Xformable(prim).AddTransformOp().Set(local)
            pts = tris.reshape(-1, 3)
            prim.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(pts.astype(np.float32)))
            prim.CreateFaceVertexCountsAttr([3] * len(tris))
            prim.CreateFaceVertexIndicesAttr(list(range(len(pts))))
            n = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
            n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)
            prim.CreateNormalsAttr(Vt.Vec3fArray.FromNumpy(np.repeat(n, 3, axis=0).astype(np.float32)))
            prim.SetNormalsInterpolation(UsdGeom.Tokens.faceVarying)
            prim.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
            # CAD STL winding is not consistent; Quick Look culls back faces by
            # default, which shows up as holes in the shells. Draw both sides.
            prim.CreateDoubleSidedAttr(True)
            prim.CreateDisplayColorAttr([Gf.Vec3f(*[float(c) for c in p["rgba"][:3]])])
            UsdShade.MaterialBindingAPI.Apply(prim.GetPrim()).Bind(material(p["rgba"]))

    stage.Save()

    # Sanity: USD's composed world pose of the trunk at frame 0 must equal the
    # recorded MuJoCo pose, tilted into y-up. Catches convention mistakes.
    cache = UsdGeom.XformCache(Usd.TimeCode(0))
    world = cache.GetLocalToWorldTransform(
        stage.GetPrimAtPath(f"/Duck/{sanitize(bodies[0]['name'])}"))
    got = np.array(world.ExtractTranslation())
    p0 = frames[0][0][0]
    want = np.array([p0[0], p0[2], -p0[1]])
    if not np.allclose(got, want, atol=1e-5):
        raise SystemExit(f"transform convention error: trunk at {got}, expected {want}")

    ok = UsdUtils.CreateNewUsdzPackage(str(usdc), str(out_path))
    if not ok:
        raise SystemExit("usdz packaging failed")
    usdc.unlink()
    print(f"{out_path}: {out_path.stat().st_size / 1e6:.1f} MB, "
          f"{len(frames)} frames ({len(frames) / FPS:.1f}s), {total} triangles")


def main() -> None:
    args = sys.argv[1:]
    if not args or not args[0].endswith(".xml"):
        raise SystemExit(__doc__)
    mjcf = Path(args[0]).expanduser()
    out = Path(args[1]) if len(args) > 1 else REPO / "web" / "duck-walk.usdz"
    bodies, files = parse_visual(mjcf)
    frames = rollout(mjcf.parent, bodies)
    build_usdz(bodies, files, mjcf.parent / "assets", frames, out)


if __name__ == "__main__":
    main()
