#!/usr/bin/env python3
"""Emit a standalone HTML viewer of the robot's assembly model — optionally with a
physics simulation the robot's real policies can walk around in.

The page is a self-contained web view of the robot: a Three.js scene with orbit
controls, a centimetre grid and ruler, real-world scale references (a drink can,
a bottle, a phone), an x-ray mode that reveals the parts inside the shells, and
per-joint pose sliders limited to the real servo ranges — so anyone with a
browser can judge the robot's size and articulation without MuJoCo or CAD tools.

Three tiers, by what is available at build time:

    scripts/duck-viewer.py [out.html]

falls back to `robotctl/assets/duck.bin`, the terminal monitor's decimated bake
committed here — coarse, but needs nothing else.

    scripts/duck-viewer.py path/to/robot_walk.xml [out.html]

embeds the app repository's MJCF visual model at full CAD resolution (~430k
triangles; needs numpy, like bake-duck-mesh.py).

    scripts/duck-viewer.py path/to/robot_walk.xml [out.html] --sim

additionally embeds a browser-side physics simulation: MuJoCo compiled to
WebAssembly steps the same collision model the policies were trained on, and
`policies/alpha_walking.onnx` + `alpha_stand.onnx` run as plain JavaScript
(they are small MLPs), reproducing robotd's control pipeline at 50 Hz — the
duck walks, turns, falls when pushed, and gets back up, all client-side.
The MJCF's directory must be the app repository's robot dir (it also needs
`robot_allcollisions.xml` and `assets/*.stl`), and --sim needs:

    numpy, onnx        (pip install numpy onnx)
    mujoco wasm dist   (npm install mujoco; or --mujoco-dist DIR with
                        mujoco.js + mujoco.wasm from the `mujoco` npm package)

The observation layout, action scaling and walk/stand switching mirror
`duck-control` (obs.rs, policy.rs) and were validated against MuJoCo before
this page existed; if those contracts move, this script must move with them.

Writes `duck-viewer.html` next to this script by default. Everything is
embedded (gzip+base64; the page inflates it with DecompressionStream) and only
Three.js loads from a CDN — serve it, mail it, or open it from disk.
"""

import base64
import gzip
import json
import struct
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

# The wire order of `duck_ipc_proto::JOINT_NAMES`, which is also the joint index
# space of duck.bin (see bake-duck-mesh.py). `mouth` has no MJCF joint.
JOINT_NAMES = [
    "left_hip_yaw", "left_hip_roll", "left_hip_pitch", "left_knee", "left_ankle",
    "neck_pitch", "head_pitch", "head_yaw", "head_roll", "mouth",
    "right_hip_yaw", "right_hip_roll", "right_hip_pitch", "right_knee", "right_ankle",
]

# Parts enclosed by the shells: batteries, PCBs, brackets. Invisible normally,
# the whole point of the viewer's x-ray mode.
INNER = {
    "np_f970",
    "pcb__raspberry_pi_zero_2_w",
    "elec_rpi_robot_hat_pcb",
    "power_support",
    "banana_pcb_locker",
    "motor_support",
}

# Meshes the full-collision model collides with. Everything else is visual and
# is stripped from the physics MJCF.
COLLISION_MESHES = {
    "np_f970", "hip_l", "leg", "sole_left", "sole_right",
    "top_head_shell", "jaw", "bottom_head_shell", "power_support",
}

# duck-control/src/model.rs DEFAULT_POSITION, minus the mouth: the home pose the
# policies observe joints relative to, in 14-wide policy order.
HOME_POSE = [
    0.0, -0.0873, -0.4579, -0.0049, 0.4530,
    0.3491, 0.3491, 0.0, 0.0,
    0.0, 0.0873, 0.4579, 0.0049, -0.4530,
]


def joint_ranges(mjcf: Path) -> dict[str, list[float]]:
    """Joint name → [lo, hi] radians, straight from the model the policies use."""
    root = ET.parse(mjcf).getroot()
    out = {}
    for joint in root.iter("joint"):
        name, rng = joint.get("name"), joint.get("range")
        if name in JOINT_NAMES and rng:
            lo, hi = (float(x) for x in rng.split())
            out[name] = [lo, hi]
    return out


def parse_floats(text: str | None, default: str) -> list[float]:
    return [float(x) for x in (text or default).split()]


# ---- source: full-resolution MJCF + STL -------------------------------------

def load_stl(path: Path):
    """Binary STL → (n, 3, 3) float32 triangle corners. Meters, as exported."""
    import numpy as np
    data = path.read_bytes()
    (n,) = struct.unpack_from("<I", data, 80)
    if len(data) != 84 + n * 50:
        raise SystemExit(f"{path}: not the binary STL this expects")
    tris = np.frombuffer(data, dtype=np.uint8, offset=84).reshape(n, 50)
    # Each record: normal (12 bytes, ignored — recomputed at render), 3 corners, pad.
    return tris[:, 12:48].copy().view("<f4").reshape(n, 3, 3)


def weld(tris):
    """STL soup → indexed (verts (v, 3) f32, faces (f, 3) int).

    Exact-duplicate welding only (to a 1 µm grid): the geometry stays the CAD's
    own, this just stops every triangle from carrying three private vertices.
    """
    import numpy as np
    flat = tris.reshape(-1, 3)
    keys = np.round(flat * 1e6).astype(np.int64)
    uniq, index, inverse = np.unique(keys, axis=0, return_index=True, return_inverse=True)
    verts = flat[index].astype(np.float32)
    faces = inverse.reshape(-1, 3)
    if len(verts) > 0xFFFF:
        raise SystemExit(f"mesh has {len(verts)} welded vertices; grow the index width")
    return verts, faces


def decimate(tris, budget: int):
    """Vertex-cluster down to at most `budget` triangles (bake-duck-mesh.py's
    method). For collision shells only — contact cares about the envelope, not
    the finish, and every embedded byte is paid for twice in base64."""
    import numpy as np
    lo = tris.min(axis=(0, 1))
    cell = max(float(np.linalg.norm(tris.max(axis=(0, 1)) - lo)) / 64.0, 1e-5)
    while True:
        flat = tris.reshape(-1, 3)
        keys = np.floor((flat - lo) / cell).astype(np.int64)
        uniq, inverse = np.unique(keys, axis=0, return_inverse=True)
        faces = inverse.reshape(-1, 3)
        keep = (
            (faces[:, 0] != faces[:, 1])
            & (faces[:, 1] != faces[:, 2])
            & (faces[:, 0] != faces[:, 2])
        )
        faces = faces[keep]
        if len(faces):
            faces = np.unique(np.sort(faces, axis=1), axis=0)
        if len(faces) <= budget:
            break
        cell *= 1.3
    verts = np.zeros((len(uniq), 3))
    counts = np.bincount(inverse, minlength=len(uniq)).astype(float)
    for axis in range(3):
        verts[:, axis] = np.bincount(
            inverse, weights=flat[:, axis].astype(float), minlength=len(uniq)
        )
    verts /= counts[:, None]
    return verts[faces].astype(np.float32)


def write_stl_bytes(tris) -> bytes:
    import numpy as np
    out = bytearray(b"\0" * 80 + struct.pack("<I", len(tris)))
    for t in tris:
        n = np.cross(t[1] - t[0], t[2] - t[0])
        l = np.linalg.norm(n)
        out += struct.pack("<3f", *(n / l if l > 0 else n))
        for v in t:
            out += struct.pack("<3f", *v)
        out += b"\0\0"
    return bytes(out)


def model_from_mjcf(mjcf_path: Path) -> dict:
    root = ET.parse(mjcf_path).getroot()
    mesh_dir = mjcf_path.parent / root.find("compiler").get("meshdir", ".")

    files = {}  # asset name → STL file (name defaults to the file's stem)
    for m in root.iter("mesh"):
        file = m.get("file")
        files[m.get("name") or Path(file).stem] = file
    materials = {
        m.get("name"): parse_floats(m.get("rgba"), "0.5 0.5 0.5 1")
        for m in root.iter("material")
    }

    meshes: dict[str, tuple] = {}  # insertion order is the baked index space

    def mesh_index(name: str) -> int:
        if name not in meshes:
            meshes[name] = weld(load_stl(mesh_dir / files[name]))
        return list(meshes).index(name)

    bodies: list[dict] = []
    parts: list[dict] = []

    def walk(body: ET.Element, parent: int) -> None:
        index = len(bodies)
        joint = body.find("joint")
        bodies.append(
            {
                "parent": parent,
                "pos": parse_floats(body.get("pos"), "0 0 0"),
                "quat": parse_floats(body.get("quat"), "1 0 0 0"),
                "joint": JOINT_NAMES.index(joint.get("name")) if joint is not None else -1,
                "axis": parse_floats(joint.get("axis"), "0 0 1") if joint is not None else [0, 0, 0],
            }
        )
        for geom in body.findall("geom"):
            if geom.get("class") != "visual" or geom.get("type") != "mesh":
                continue
            name = geom.get("mesh")
            parts.append(
                {
                    "body": index,
                    "mesh": mesh_index(name),
                    "rgba": materials.get(geom.get("material"), [0.5, 0.5, 0.5, 1]),
                    "inner": name in INNER,
                    "pos": parse_floats(geom.get("pos"), "0 0 0"),
                    "quat": parse_floats(geom.get("quat"), "1 0 0 0"),
                }
            )
        for child in body.findall("body"):
            walk(child, index)

    for top in root.find("worldbody").findall("body"):
        walk(top, -1)

    return {"meshes": list(meshes.values()), "bodies": bodies, "parts": parts}


# ---- source: the committed terminal bake ------------------------------------

def model_from_duck_bin(blob: bytes) -> dict:
    """Parse bake-duck-mesh.py's FORMAT_VERSION 1 (see robotctl/src/duck.rs)."""
    magic, version, n_mesh, n_body, n_part = struct.unpack_from("<4sIHHH", blob, 0)
    if magic != b"DUCK" or version != 1:
        raise SystemExit("duck.bin is not the format this viewer understands; "
                         "update this script alongside bake-duck-mesh.py")
    at = 14
    meshes = []
    for _ in range(n_mesh):
        nv, nf = struct.unpack_from("<HH", blob, at)
        at += 4
        verts = [struct.unpack_from("<3f", blob, at + i * 12) for i in range(nv)]
        at += nv * 12
        faces = [struct.unpack_from("<3H", blob, at + i * 6) for i in range(nf)]
        at += nf * 6
        meshes.append((verts, faces))
    bodies = []
    for _ in range(n_body):
        parent, joint = struct.unpack_from("<hh", blob, at)
        vals = struct.unpack_from("<10f", blob, at + 4)
        at += 44
        bodies.append({"parent": parent, "joint": joint,
                       "pos": vals[0:3], "quat": vals[3:7], "axis": vals[7:10]})
    parts = []
    for _ in range(n_part):
        body, mesh, r, g, b, _pad = struct.unpack_from("<HHBBBB", blob, at)
        vals = struct.unpack_from("<7f", blob, at + 8)
        at += 36
        parts.append({"body": body, "mesh": mesh, "rgba": [r / 255, g / 255, b / 255, 1],
                      "inner": False, "pos": vals[0:3], "quat": vals[3:7]})
    return {"meshes": meshes, "bodies": bodies, "parts": parts}


# ---- serialize for the page -------------------------------------------------
# "DUK2": the viewer's own wire format, independent of duck.bin. Per mesh a u32
# vertex/face count, f32 vertices, u16 indices; bodies and parts as in the v1
# bake but with rgba and an "inner" flag on parts.

def serialize(model: dict) -> bytes:
    out = bytearray()
    out += struct.pack("<4sIHHH", b"DUK2", 2,
                       len(model["meshes"]), len(model["bodies"]), len(model["parts"]))
    for verts, faces in model["meshes"]:
        out += struct.pack("<II", len(verts), len(faces))
        if hasattr(verts, "astype"):  # numpy, from the MJCF path
            out += verts.astype("<f4").tobytes()
            out += faces.astype("<u2").tobytes()
        else:
            for v in verts:
                out += struct.pack("<3f", *v)
            for f in faces:
                out += struct.pack("<3H", *f)
    for b in model["bodies"]:
        out += struct.pack("<hh3f4f3f", b["parent"], b["joint"],
                           *b["pos"], *b["quat"], *b["axis"])
    for p in model["parts"]:
        rgba = [min(255, max(0, round(c * 255))) for c in p["rgba"]]
        out += struct.pack("<HH4B B 3x 3f 4f", p["body"], p["mesh"], *rgba,
                           1 if p["inner"] else 0, *p["pos"], *p["quat"])
    return bytes(out)


# ---- the physics bundle (--sim) ---------------------------------------------

def physics_xml(robot_dir: Path) -> tuple[str, dict[str, bytes]]:
    """The full-collision MJCF, stripped to what physics needs, plus its meshes.

    Visual geoms carry no dynamics (mass and inertia are explicit <inertial>
    tags), so they are dropped wholesale; the collision meshes are decimated —
    MuJoCo collides against their hulls and the page pays for every byte twice
    in base64. timestep 0.005 × decimation 4 is the training cadence (50 Hz).
    """
    tree = ET.parse(robot_dir / "robot_allcollisions.xml")
    root = tree.getroot()

    def prune(el):
        for child in list(el):
            if child.tag == "geom" and child.get("class") == "visual":
                el.remove(child)
            else:
                prune(child)
    prune(root)

    asset = root.find("asset")
    meshes: dict[str, bytes] = {}
    for m in list(asset):
        name = None
        if m.tag == "mesh":
            name = m.get("name") or Path(m.get("file")).stem
        if name in COLLISION_MESHES:
            budget = 400 if "sole" in name else 200
            tris = decimate(load_stl(robot_dir / "assets" / f"{name}.stl"), budget)
            meshes[f"coll_{name}.stl"] = write_stl_bytes(tris)
            m.set("name", name)
            m.set("file", f"coll_{name}.stl")
        else:
            asset.remove(m)
    for g in root.iter("geom"):
        g.attrib.pop("material", None)

    compiler = root.find("compiler")
    compiler.set("meshdir", ".")
    # Single-threaded compile: the wasm build spawns a ThreadPool for 2+ meshes,
    # and browser pages without cross-origin isolation have no SharedArrayBuffer
    # to back it — the compile dies in the thread constructor.
    compiler.set("usethread", "false")
    ET.SubElement(root, "option").set("timestep", "0.005")
    wb = root.find("worldbody")
    wb.insert(0, ET.Element("geom", dict(name="floor", type="plane", size="0 0 0.05")))

    # The 14 hinges must sit in policy order behind the freejoint: qpos[7+i],
    # qvel[6+i] and ctrl[i] are indexed positionally in the page's control loop.
    hinges = [j.get("name") for j in root.iter("joint") if j.get("name")]
    expected = [n for n in JOINT_NAMES if n != "mouth"]
    if hinges != expected:
        raise SystemExit(f"physics joint order {hinges} != policy order {expected}")

    return ET.tostring(root, encoding="unicode"), meshes


def policy_blob(path: Path) -> tuple[bytes, list[dict]]:
    """ONNX MLP → flat f32 blob + layout. The alpha policies are all
    normalizer → 3×(Gemm+Elu) → Gemm; anything else must fail here, loudly,
    rather than run garbage in the page."""
    import onnx
    mdl = onnx.load(str(path))
    ops = sorted({n.op_type for n in mdl.graph.node})
    if ops != ["Div", "Elu", "Gemm", "Sub"]:
        raise SystemExit(f"{path}: unexpected ops {ops}; the page's inference "
                         "implements only the alpha MLP shape")
    weights = {i.name: onnx.numpy_helper.to_array(i) for i in mdl.graph.initializer}
    order = ["obs_normalizer._mean", "onnx::Div_24",
             "mlp.0.weight", "mlp.0.bias", "mlp.2.weight", "mlp.2.bias",
             "mlp.4.weight", "mlp.4.bias", "mlp.6.weight", "mlp.6.bias"]
    blob = bytearray()
    meta = []
    for name in order:
        a = weights[name].astype("<f4")
        meta.append({"name": name, "shape": list(a.shape), "offset": len(blob) // 4})
        blob += a.tobytes()
    return bytes(blob), meta


def gz_b64(data: bytes) -> str:
    return base64.b64encode(gzip.compress(data, 9)).decode()


def build_sim_assets(robot_dir: Path, mujoco_dist: Path) -> str:
    """The __SIM_ASSETS__ substitution: everything the in-page sim needs."""
    xml, meshes = physics_xml(robot_dir)

    walk_blob, walk_meta = policy_blob(REPO / "policies" / "alpha_walking.onnx")
    stand_blob, stand_meta = policy_blob(REPO / "policies" / "alpha_stand.onnx")

    glue = (mujoco_dist / "mujoco.js").read_text()
    marker = "export default loadMujoco;"
    if marker not in glue:
        raise SystemExit(f"{mujoco_dist}/mujoco.js: no '{marker}' to patch — "
                         "is this the `mujoco` npm package?")
    glue = glue.replace(marker, "globalThis.loadMujoco = loadMujoco;")

    assets = {
        "wasm": gz_b64((mujoco_dist / "mujoco.wasm").read_bytes()),
        "xml": base64.b64encode(xml.encode()).decode(),
        "meshes": {k: base64.b64encode(v).decode() for k, v in meshes.items()},
        "walk": {"blob": gz_b64(walk_blob), "meta": walk_meta},
        "stand": {"blob": gz_b64(stand_blob), "meta": stand_meta},
    }
    return json.dumps(assets), glue


def main() -> None:
    args = [a for a in sys.argv[1:]]
    sim = "--sim" in args
    if sim:
        args.remove("--sim")
    mujoco_dist = Path(__file__).parent / "node_modules" / "mujoco"
    if "--mujoco-dist" in args:
        i = args.index("--mujoco-dist")
        mujoco_dist = Path(args[i + 1])
        del args[i:i + 2]
    mjcf = Path(args.pop(0)).expanduser() if args and args[0].endswith(".xml") else None
    out_path = Path(args[0]) if args else Path(__file__).parent / "duck-viewer.html"

    if mjcf is not None:
        try:
            import numpy  # noqa: F401 — load_stl/weld need it; fail before parsing
        except ImportError:
            raise SystemExit("full-resolution mode needs numpy (pip install numpy)")
        model = model_from_mjcf(mjcf)
        ranges = joint_ranges(mjcf)
        tris = sum(len(f) for _, f in model["meshes"])
        source_note = f"フルCADメッシュ（{tris / 1000:.0f}k三角形） — {mjcf.name}"
        dims_note = "寸法は現在のポーズの外接寸法（CAD原寸）。床グリッドは5cm。"
    else:
        if sim:
            raise SystemExit("--sim needs the MJCF path (the physics model and "
                             "collision STLs live next to it)")
        model = model_from_duck_bin((REPO / "robotctl" / "assets" / "duck.bin").read_bytes())
        ranges = joint_ranges(REPO / "kinematics" / "assets" / "alpha" / "robot_walk.xml")
        source_note = "robotctl/assets/duck.bin — 端末モニタ用の間引きメッシュ"
        dims_note = "寸法は現在のポーズの外接寸法（メッシュはデシメート済みのため±数mm）。床グリッドは5cm。"

    if sim:
        sim_assets, glue = build_sim_assets(mjcf.parent, mujoco_dist)
    else:
        sim_assets, glue = "null", ""

    html = TEMPLATE
    html = html.replace("__DUCK_GZ_B64__", gz_b64(serialize(model)))
    html = html.replace("__JOINT_NAMES__", json.dumps(JOINT_NAMES))
    html = html.replace("__JOINT_RANGES__", json.dumps(ranges))
    html = html.replace("__HOME_POSE__", json.dumps(HOME_POSE))
    html = html.replace("__SOURCE_NOTE__", source_note)
    html = html.replace("__DIMS_NOTE__", dims_note)
    html = html.replace("__MUJOCO_GLUE__", glue)
    html = html.replace("__SIM_ASSETS__", sim_assets)
    # A visible build tag (in the AR診断 line) so "which version am I looking
    # at?" is answerable from a phone screenshot.
    import datetime
    html = html.replace("__BUILD_TAG__", datetime.date.today().isoformat())
    out_path.write_text(html)
    print(f"{out_path}: {out_path.stat().st_size / 1024 / 1024:.1f} MB"
          + (" (with physics sim)" if sim else ""))


# The page itself. Kept as one template string so the tool stays a single file;
# tokens (__DUCK_GZ_B64__ etc.) are substituted above.
TEMPLATE = r"""<meta charset="utf-8">
<title>Microduck Assy Viewer</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Chakra+Petch:wght@500;600&family=IBM+Plex+Sans+JP:wght@400;500&family=IBM+Plex+Mono:wght@400;500&display=swap">
<style>
  :root {
    --bg: #eef0f2;
    --surface: rgba(255, 255, 255, 0.82);
    --surface-border: rgba(27, 34, 40, 0.10);
    --ink: #1b2228;
    --muted: #5d6a74;
    --accent: #d99011;
    --accent-ink: #8a5a05;
    --track: rgba(27, 34, 40, 0.14);
    --grid-major: #9aa4ac;
    --grid-minor: #c2c9cf;
    --shadow: 0 10px 30px rgba(27, 34, 40, 0.12);
  }
  @media (prefers-color-scheme: dark) {
    :root:not([data-theme="light"]) {
      --bg: #14181c;
      --surface: rgba(24, 30, 35, 0.85);
      --surface-border: rgba(232, 234, 236, 0.10);
      --ink: #e8eaec;
      --muted: #98a2ab;
      --accent: #f0a82a;
      --accent-ink: #f0a82a;
      --track: rgba(232, 234, 236, 0.16);
      --grid-major: #4d565e;
      --grid-minor: #2b3238;
      --shadow: 0 10px 30px rgba(0, 0, 0, 0.4);
    }
  }
  :root[data-theme="dark"] {
    --bg: #14181c;
    --surface: rgba(24, 30, 35, 0.85);
    --surface-border: rgba(232, 234, 236, 0.10);
    --ink: #e8eaec;
    --muted: #98a2ab;
    --accent: #f0a82a;
    --accent-ink: #f0a82a;
    --track: rgba(232, 234, 236, 0.16);
    --grid-major: #4d565e;
    --grid-minor: #2b3238;
    --shadow: 0 10px 30px rgba(0, 0, 0, 0.4);
  }
  * { margin: 0; box-sizing: border-box; }
  html, body { height: 100%; }
  body {
    background: var(--bg);
    color: var(--ink);
    font-family: "IBM Plex Sans JP", "Hiragino Sans", "Noto Sans JP", sans-serif;
    font-size: 14px;
    overflow: hidden;
  }
  #arlayer { position: fixed; inset: 0; z-index: 0; display: none; background: #000; }
  #arlayer video, #arlayer img {
    width: 100%; height: 100%; object-fit: cover; display: none;
  }
  #stage { position: fixed; inset: 0; z-index: 1; display: block; touch-action: none; cursor: grab; }
  #stage:active { cursor: grabbing; }

  .panel {
    position: fixed;
    z-index: 5;
    background: var(--surface);
    border: 1px solid var(--surface-border);
    border-radius: 10px;
    box-shadow: var(--shadow);
    backdrop-filter: blur(10px);
    -webkit-backdrop-filter: blur(10px);
  }

  #head { top: 16px; left: 16px; padding: 14px 18px 12px; max-width: min(320px, calc(100vw - 32px)); }
  #head h1 {
    font-family: "Chakra Petch", "IBM Plex Sans JP", sans-serif;
    font-size: 18px; font-weight: 600; letter-spacing: 0.02em;
    text-wrap: balance;
  }
  #head .sub { color: var(--muted); font-size: 12px; margin-top: 2px; }

  #dims {
    display: grid; grid-template-columns: repeat(2, auto);
    gap: 8px 22px; margin-top: 12px;
  }
  #dims .cell .k {
    color: var(--muted); font-size: 10.5px; letter-spacing: 0.09em;
    text-transform: uppercase;
  }
  #dims .cell .v {
    font-family: "Chakra Petch", sans-serif;
    font-size: 21px; font-weight: 600; font-variant-numeric: tabular-nums;
    color: var(--accent-ink);
  }
  #dims .cell .v small { font-size: 12px; font-weight: 500; color: var(--muted); margin-left: 2px; }
  #head .note { color: var(--muted); font-size: 11px; margin-top: 10px; }

  #chips {
    bottom: 16px; left: 16px; right: auto;
    display: flex; flex-wrap: wrap; gap: 8px;
    padding: 10px; z-index: 5;
    /* stay clear of the right-hand panel so chips are always clickable */
    max-width: calc(100vw - 326px);
  }
  @media (max-width: 760px) {
    #chips { max-width: calc(100vw - 32px); }
  }
  .chip {
    font: inherit; font-size: 12.5px; color: var(--muted);
    background: transparent; border: 1px solid var(--surface-border);
    border-radius: 999px; padding: 5px 12px; cursor: pointer;
  }
  .chip[aria-pressed="true"] {
    color: var(--ink); border-color: var(--accent);
    box-shadow: inset 0 0 0 1px var(--accent);
  }
  .chip:disabled { opacity: 0.5; cursor: wait; }
  .chip:focus-visible, input:focus-visible, .chip-reset:focus-visible, .simbtn:focus-visible {
    outline: 2px solid var(--accent); outline-offset: 2px;
  }

  #joints, #simpanel {
    top: 16px; right: 16px; width: 262px;
    max-height: calc(100vh - 32px);
    display: flex; flex-direction: column;
  }
  .bar {
    display: flex; align-items: center; justify-content: space-between;
    padding: 10px 14px;
  }
  .bar h2 { font-size: 12px; font-weight: 500; letter-spacing: 0.08em; color: var(--muted); }
  .chip-reset {
    font: inherit; font-size: 11.5px; color: var(--muted);
    background: transparent; border: 1px solid var(--surface-border);
    border-radius: 6px; padding: 3px 9px; cursor: pointer;
  }
  #joints .body, #simpanel .body { overflow-y: auto; padding: 0 14px 12px; }
  #joints h3, #simpanel h3 {
    font-size: 10.5px; font-weight: 500; letter-spacing: 0.09em;
    text-transform: uppercase; color: var(--muted);
    margin: 12px 0 4px;
  }
  .row { display: grid; grid-template-columns: 1fr 44px; align-items: center; gap: 8px; padding: 3px 0; }
  .row label { font-size: 12px; display: block; }
  input[type="range"] { width: 100%; accent-color: var(--accent); height: 18px; }
  .deg {
    font-family: "IBM Plex Mono", monospace; font-size: 11px;
    font-variant-numeric: tabular-nums; text-align: right; color: var(--muted);
  }

  #simpanel { display: none; }
  #simstatus {
    display: grid; grid-template-columns: auto 1fr; gap: 4px 12px;
    font-size: 12px; padding: 2px 0 6px;
  }
  #simstatus .k { color: var(--muted); }
  #simstatus .v { font-family: "IBM Plex Mono", monospace; font-variant-numeric: tabular-nums; text-align: right; }
  #simstate { font-weight: 500; }
  .padgrid {
    display: grid; grid-template-columns: repeat(3, 1fr); gap: 6px; margin: 6px 0;
  }
  .simbtn {
    font: inherit; font-size: 12.5px; color: var(--ink);
    background: transparent; border: 1px solid var(--surface-border);
    border-radius: 8px; padding: 9px 4px; cursor: pointer;
    user-select: none; -webkit-user-select: none; touch-action: none;
  }
  .simbtn:active, .simbtn.held, .simbtn[aria-pressed="true"] { border-color: var(--accent); box-shadow: inset 0 0 0 1px var(--accent); }
  .simbtn.wide { grid-column: span 3; }
  #simpanel .hintline { color: var(--muted); font-size: 11px; margin: 8px 0 2px; }
  #ar-escape a.simbtn { display: block; text-align: center; text-decoration: none; margin: 6px 0; }
  .urlrow { display: flex; gap: 6px; align-items: center; margin: 4px 0; }
  #ar-url {
    flex: 1; min-width: 0; font-family: "IBM Plex Mono", monospace; font-size: 10.5px;
    color: var(--muted); background: transparent;
    border: 1px solid var(--surface-border); border-radius: 6px; padding: 5px 7px;
  }
  #fallnote {
    display: none; color: var(--accent-ink); font-size: 12px; font-weight: 500;
    margin-top: 6px;
  }

  #hint {
    position: fixed; bottom: 16px; right: 16px;
    color: var(--muted); font-size: 11.5px; text-align: right;
    pointer-events: none;
  }
  #fallback {
    position: fixed; inset: 0; display: none;
    place-content: center; text-align: center; color: var(--muted); padding: 24px;
  }
  @media (max-width: 760px) {
    #joints { display: none; }
    #hint { display: none; }
  }
  @media (prefers-reduced-motion: reduce) {
    * { transition: none !important; }
  }
</style>

<div id="arlayer">
  <video id="arvideo" playsinline muted autoplay></video>
  <img id="arimg" alt="AR背景">
</div>
<canvas id="stage"></canvas>

<div class="panel" id="head">
  <h1>Microduck Assy Viewer</h1>
  <div class="sub">__SOURCE_NOTE__</div>
  <div id="dims">
    <div class="cell"><div class="k">全高</div><div class="v" id="d-h">—<small>mm</small></div></div>
    <div class="cell"><div class="k">質量</div><div class="v">約800<small>g</small></div></div>
    <div class="cell"><div class="k">全幅</div><div class="v" id="d-w">—<small>mm</small></div></div>
    <div class="cell"><div class="k">奥行</div><div class="v" id="d-d">—<small>mm</small></div></div>
  </div>
  <div class="note">__DIMS_NOTE__</div>
</div>

<div class="panel" id="chips" role="group" aria-label="表示オプション">
  <button class="chip" data-ref="can" aria-pressed="true">350ml缶</button>
  <button class="chip" data-ref="bottle" aria-pressed="false">500mlペットボトル</button>
  <button class="chip" data-ref="phone" aria-pressed="false">スマートフォン</button>
  <button class="chip" data-ref="ruler" aria-pressed="true">スケール（cm）</button>
  <button class="chip" id="xray" aria-pressed="false">透視</button>
  <button class="chip" id="simtoggle" aria-pressed="false" hidden>歩行シミュレーション</button>
</div>

<div class="panel" id="joints">
  <div class="bar">
    <h2>関節ポーズ</h2>
    <button class="chip-reset" id="reset">リセット</button>
  </div>
  <div class="body" id="joint-rows"></div>
</div>

<div class="panel" id="simpanel">
  <div class="bar">
    <h2>歩行シミュレーション</h2>
    <button class="chip-reset" id="simreset">リセット</button>
  </div>
  <div class="body">
    <div id="simstatus">
      <span class="k">状態</span><span class="v" id="simstate">—</span>
      <span class="k">前進速度</span><span class="v" id="simvx">0.00 m/s</span>
      <span class="k">旋回速度</span><span class="v" id="simwz">0.00 rad/s</span>
      <span class="k">移動距離</span><span class="v" id="simdist">0.00 m</span>
    </div>
    <div class="padgrid">
      <button class="simbtn" data-cmd="turnl">⟲ 左旋回</button>
      <button class="simbtn" data-cmd="fwd">▲ 前進</button>
      <button class="simbtn" data-cmd="turnr">⟳ 右旋回</button>
      <button class="simbtn" data-cmd="left">◀ 左移動</button>
      <button class="simbtn" data-cmd="stop">■ 停止</button>
      <button class="simbtn" data-cmd="right">▶ 右移動</button>
      <button class="simbtn wide" data-cmd="push">押す（いたずら）</button>
    </div>
    <div id="fallnote">転倒！ 指令を止めて自動復帰中…</div>
    <h3>AR合成</h3>
    <div class="padgrid">
      <button class="simbtn" id="ar-shot" aria-pressed="false">撮影</button>
      <button class="simbtn" id="ar-photo" aria-pressed="false">写真</button>
      <button class="simbtn" id="ar-cam" aria-pressed="false">ライブ</button>
      <button class="simbtn wide" id="ar-off" aria-pressed="true">AR オフ</button>
      <button class="simbtn wide" id="ar-open" hidden>別タブで開く（ライブカメラ用）</button>
      <button class="simbtn wide" id="ar-xr" hidden>WebXR AR（実寸でその場に置く）</button>
      <button class="simbtn wide" id="ar-grid" aria-pressed="true">床グリッド表示</button>
    </div>
    <div id="ar-escape" hidden>
      <a class="simbtn wide" id="ar-link" target="_blank" rel="noopener">リンクとして単独ページを開く</a>
      <div class="urlrow">
        <input id="ar-url" readonly aria-label="単独ページのURL">
        <button class="chip-reset" id="ar-copy">コピー</button>
      </div>
    </div>
    <div class="hintline" id="ar-diag"></div>
    <input type="file" id="ar-file" accept="image/*" hidden>
    <input type="file" id="ar-capture" accept="image/*" capture="environment" hidden>
    <div class="hintline">撮影＝その場でカメラ撮影して重ねる（権限不要・スマホ推奨） ／ 写真＝保存済み画像 ／ ライブ＝カメラ映像（ブラウザの権限が必要）</div>
    <div class="hintline" id="ar-hint" style="display:none">端末や写真を固定し、ドラッグ／ホイールで視点を実写の床に合わせてください。グリッドが合ったら非表示に。</div>
    <h3>頭コマンド</h3>
    <div id="head-rows"></div>
    <div class="hintline">キー操作: W/↑ 前進 ・ A/D 横移動 ・ ←/→ 旋回 ・ Space 停止</div>
    <div class="hintline">ボタンとキーは押している間だけ効きます。倒れたら手を離すと自力で起き上がります。実機と同じ学習済みポリシー（alpha_walking / alpha_stand）がMuJoCo (WebAssembly) 上で50Hz動作しています。</div>
  </div>
</div>

<div id="hint">ドラッグ：回転 ／ ホイール・ピンチ：ズーム ／ Shift+ドラッグ：移動</div>
<div id="fallback">3D表示を初期化できませんでした。WebGL対応のブラウザでご覧ください。</div>

<script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
<script type="module">__MUJOCO_GLUE__</script>
<script type="module">
"use strict";

// ---- embedded data ----------------------------------------------------------
const JOINT_NAMES = __JOINT_NAMES__;
const JOINT_RANGES = __JOINT_RANGES__;
const HOME = __HOME_POSE__;          // 14-wide policy order (no mouth)
const SIM_ASSETS = __SIM_ASSETS__;   // null when built without --sim
const DUCK_GZ = "__DUCK_GZ_B64__";

function b64bytes(b64) {
  return Uint8Array.from(atob(b64), (c) => c.charCodeAt(0));
}
async function gunzip(b64) {
  const ds = new DecompressionStream("gzip");
  const resp = new Response(new Blob([b64bytes(b64)]).stream().pipeThrough(ds));
  return new Uint8Array(await resp.arrayBuffer());
}

// "DUK2", written by scripts/duck-viewer.py — see serialize() there.
function parseDuck(bytes) {
  const dv = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  let at = 0;
  const u16 = () => { const v = dv.getUint16(at, true); at += 2; return v; };
  const i16 = () => { const v = dv.getInt16(at, true); at += 2; return v; };
  const u32 = () => { const v = dv.getUint32(at, true); at += 4; return v; };
  const f32 = () => { const v = dv.getFloat32(at, true); at += 4; return v; };
  const vec3 = () => [f32(), f32(), f32()];
  const quat = () => [f32(), f32(), f32(), f32()]; // MJCF order: w x y z

  if (String.fromCharCode(...bytes.slice(0, 4)) !== "DUK2") throw new Error("bad magic");
  at = 4;
  if (u32() !== 2) throw new Error("bad version");
  const nMesh = u16(), nBody = u16(), nPart = u16();

  const base = bytes.byteOffset;
  const meshes = [];
  for (let m = 0; m < nMesh; m++) {
    const nv = u32(), nf = u32();
    // Copy out (slice) rather than view: offsets are not element-aligned.
    const verts = new Float32Array(bytes.buffer.slice(base + at, base + at + nv * 12)); at += nv * 12;
    const faces = new Uint16Array(bytes.buffer.slice(base + at, base + at + nf * 6)); at += nf * 6;
    meshes.push({ verts, faces });
  }
  const bodies = [];
  for (let b = 0; b < nBody; b++) {
    bodies.push({ parent: i16(), joint: i16(), pos: vec3(), quat: quat(), axis: vec3() });
  }
  const parts = [];
  for (let p = 0; p < nPart; p++) {
    const body = u16(), mesh = u16();
    const rgba = [bytes[at], bytes[at + 1], bytes[at + 2], bytes[at + 3]];
    const inner = bytes[at + 4] === 1;
    at += 8;
    parts.push({ body, mesh, rgba, inner, pos: vec3(), quat: quat() });
  }
  return { meshes, bodies, parts };
}

// ---- scene ------------------------------------------------------------------
const canvas = document.getElementById("stage");
let renderer;
try {
  // alpha: the AR modes composite the scene over a camera feed or photo.
  renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: true });
} catch (e) {
  document.getElementById("fallback").style.display = "grid";
  throw e;
}
renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
renderer.shadowMap.type = THREE.PCFSoftShadowMap;

const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(32, 1, 0.01, 40);

scene.add(new THREE.HemisphereLight(0xffffff, 0x8899aa, 0.85));
const sun = new THREE.DirectionalLight(0xffffff, 0.75);
sun.position.set(0.6, 1.2, 0.8);
scene.add(sun);
scene.add(sun.target);
const fill = new THREE.DirectionalLight(0xffffff, 0.25);
fill.position.set(-0.8, 0.4, -0.6);
scene.add(fill);

// Grids live in Three's y-up world; the robot group is rotated from MJCF z-up.
const themed = []; // objects recoloured/redrawn when the theme flips
let arMode = "off"; // "off" | "cam" | "photo" — the sim's AR compositing mode
function cssColor(name) {
  return new THREE.Color(getComputedStyle(document.documentElement).getPropertyValue(name).trim());
}
let gridMajor = null, gridMinor = null;
function buildGrids(span) {
  if (gridMajor) { scene.remove(gridMajor, gridMinor); }
  gridMajor = new THREE.GridHelper(span, Math.round(span / 0.05)); // 5 cm
  gridMinor = new THREE.GridHelper(span, Math.round(span / 0.01)); // 1 cm
  gridMinor.material.transparent = true;
  gridMinor.material.opacity = 0.45;
  scene.add(gridMajor, gridMinor);
  applyTheme();
}

function applyTheme() {
  // In an AR mode the canvas is transparent and the camera feed shows through.
  renderer.setClearColor(cssColor("--bg"), arMode === "off" ? 1 : 0);
  if (gridMajor) {
    gridMajor.material.color = cssColor("--grid-major");
    gridMinor.material.color = cssColor("--grid-minor");
  }
  for (const fn of themed) fn();
}
matchMedia("(prefers-color-scheme: dark)").addEventListener("change", applyTheme);
new MutationObserver(applyTheme).observe(document.documentElement, { attributes: true, attributeFilter: ["data-theme"] });
buildGrids(0.6);

// ---- robot ------------------------------------------------------------------
const duck = parseDuck(await gunzip(DUCK_GZ));

// simWorld is the placeable origin: WebXR AR moves it onto a real surface, and
// everything that must land there with the duck (its contact shadow) rides in it.
const simWorld = new THREE.Group();
scene.add(simWorld);

const robotRoot = new THREE.Group();
robotRoot.rotation.x = -Math.PI / 2; // MJCF z-up → Three y-up
simWorld.add(robotRoot);

// Contact shadow for the AR composite: an invisible plane that only shows shade.
const shadowPlane = new THREE.Mesh(
  new THREE.PlaneGeometry(4, 4),
  new THREE.ShadowMaterial({ opacity: 0.3 }),
);
shadowPlane.rotation.x = -Math.PI / 2;
shadowPlane.receiveShadow = true;
shadowPlane.visible = false;
simWorld.add(shadowPlane);
sun.shadow.mapSize.set(1024, 1024);
sun.shadow.camera.left = -0.8; sun.shadow.camera.right = 0.8;
sun.shadow.camera.top = 0.8; sun.shadow.camera.bottom = -0.8;
sun.shadow.camera.near = 0.1; sun.shadow.camera.far = 5;

const geometries = duck.meshes.map(({ verts, faces }) => {
  let g = new THREE.BufferGeometry();
  g.setAttribute("position", new THREE.BufferAttribute(verts, 3));
  g.setIndex(new THREE.BufferAttribute(faces, 1));
  g = g.toNonIndexed(); // flat facets: CAD's own faces, no invented smoothing
  g.computeVertexNormals();
  return g;
});

// Parents precede children in the source's order, so one forward pass builds the tree.
const bodyGroups = [];
for (const b of duck.bodies) {
  const group = new THREE.Group();
  group.position.set(...b.pos);
  (b.parent < 0 ? robotRoot : bodyGroups[b.parent]).add(group);
  bodyGroups.push(group);
}

const outerMats = [], innerMats = [];
for (const p of duck.parts) {
  const mat = new THREE.MeshPhongMaterial({
    color: new THREE.Color(p.rgba[0] / 255, p.rgba[1] / 255, p.rgba[2] / 255),
    shininess: 22,
  });
  if (p.rgba[3] < 250) {
    mat.transparent = true;
    mat.opacity = p.rgba[3] / 255;
  }
  (p.inner ? innerMats : outerMats).push(mat);
  const mesh = new THREE.Mesh(geometries[p.mesh], mat);
  mesh.castShadow = true; // only drawn once shadowMap is enabled (AR modes)
  mesh.position.set(...p.pos);
  mesh.quaternion.set(p.quat[1], p.quat[2], p.quat[3], p.quat[0]);
  bodyGroups[p.body].add(mesh);
}

// X-ray: the shells go to glass so the electronics inside read at a glance.
const baseOpacity = outerMats.map((m) => (m.transparent ? m.opacity : 1));
document.getElementById("xray").addEventListener("click", (e) => {
  const on = e.currentTarget.getAttribute("aria-pressed") !== "true";
  e.currentTarget.setAttribute("aria-pressed", String(on));
  outerMats.forEach((m, i) => {
    m.transparent = on || baseOpacity[i] < 1;
    m.opacity = on ? Math.min(0.28, baseOpacity[i]) : baseOpacity[i];
    m.depthWrite = !on;
    m.needsUpdate = true;
  });
});

const angles = new Array(JOINT_NAMES.length).fill(0);
const tmpQ = new THREE.Quaternion();
const tmpAxis = new THREE.Vector3();
function pose() {
  duck.bodies.forEach((b, i) => {
    const q = bodyGroups[i].quaternion;
    q.set(b.quat[1], b.quat[2], b.quat[3], b.quat[0]);
    if (b.joint >= 0) {
      tmpAxis.set(...b.axis);
      q.multiply(tmpQ.setFromAxisAngle(tmpAxis, angles[b.joint]));
    }
  });
  if (!simActive) updateDims();
}

// ---- dimensions -------------------------------------------------------------
const bbox = new THREE.Box3();
function updateDims() {
  robotRoot.updateMatrixWorld(true);
  bbox.setFromObject(robotRoot);
  const size = bbox.getSize(new THREE.Vector3());
  const mm = (v) => `${Math.round(v * 1000)}<small>mm</small>`;
  // The z-up→y-up rotation sends MJCF y (left-right) to world z and keeps
  // MJCF x (front-back) on world x.
  document.getElementById("d-h").innerHTML = mm(size.y);
  document.getElementById("d-w").innerHTML = mm(size.z);
  document.getElementById("d-d").innerHTML = mm(size.x);
}

// ---- scale references -------------------------------------------------------
// Everyday objects with well-known sizes, drawn as quiet grey ghosts.
function ghostMaterial() {
  const mat = new THREE.MeshPhongMaterial({ transparent: true, opacity: 0.55, shininess: 10 });
  themed.push(() => { mat.color = cssColor("--grid-major"); });
  return mat;
}
const refs = {};
{
  // 350 ml can: ⌀66 × 122 mm.
  const can = new THREE.Mesh(new THREE.CylinderGeometry(0.033, 0.033, 0.122, 40), ghostMaterial());
  can.position.set(-0.16, 0.061, 0.05);
  refs.can = can;

  // 500 ml bottle: ⌀65 mm body, ~205 mm tall with a shouldered neck.
  const bottle = new THREE.Group();
  const bmat = ghostMaterial();
  const bodyPart = new THREE.Mesh(new THREE.CylinderGeometry(0.0325, 0.0325, 0.15, 40), bmat);
  bodyPart.position.y = 0.075;
  const shoulder = new THREE.Mesh(new THREE.CylinderGeometry(0.014, 0.0325, 0.04, 40), bmat);
  shoulder.position.y = 0.17;
  const neck = new THREE.Mesh(new THREE.CylinderGeometry(0.014, 0.014, 0.015, 40), bmat);
  neck.position.y = 0.1975;
  bottle.add(bodyPart, shoulder, neck);
  bottle.position.set(0.18, 0, 0.03);
  refs.bottle = bottle;

  // A typical smartphone: 147 × 72 × 8 mm, stood upright.
  const phone = new THREE.Mesh(new THREE.BoxGeometry(0.072, 0.147, 0.008), ghostMaterial());
  phone.position.set(0.02, 0.0735, -0.14);
  refs.phone = phone;

  // Height ruler: a tick every cm, a label every 5 cm.
  const ruler = new THREE.Group();
  const rmat = new THREE.LineBasicMaterial();
  themed.push(() => { rmat.color = cssColor("--grid-major"); });
  const pts = [];
  const X = -0.13, H = 0.26;
  pts.push(new THREE.Vector3(X, 0, -0.12), new THREE.Vector3(X, H, -0.12));
  for (let cm = 0; cm <= 26; cm++) {
    const len = cm % 5 === 0 ? 0.012 : 0.006;
    pts.push(new THREE.Vector3(X, cm / 100, -0.12), new THREE.Vector3(X + len, cm / 100, -0.12));
  }
  ruler.add(new THREE.LineSegments(new THREE.BufferGeometry().setFromPoints(pts), rmat));
  for (let cm = 0; cm <= 25; cm += 5) {
    const c = document.createElement("canvas");
    c.width = 128; c.height = 64;
    const tex = new THREE.CanvasTexture(c);
    const draw = () => {
      const ctx = c.getContext("2d");
      ctx.clearRect(0, 0, 128, 64);
      ctx.font = "500 40px 'IBM Plex Mono', monospace";
      ctx.textAlign = "center"; ctx.textBaseline = "middle";
      ctx.fillStyle = getComputedStyle(document.documentElement).getPropertyValue("--muted").trim();
      ctx.fillText(String(cm), 64, 34);
      tex.needsUpdate = true;
    };
    draw();
    themed.push(draw);
    const sprite = new THREE.Sprite(new THREE.SpriteMaterial({ map: tex, transparent: true }));
    sprite.scale.set(0.028, 0.014, 1);
    sprite.position.set(X - 0.022, cm / 100, -0.12);
    ruler.add(sprite);
  }
  refs.ruler = ruler;
}
for (const [name, obj] of Object.entries(refs)) {
  obj.visible = name === "can" || name === "ruler";
  scene.add(obj);
}
for (const chip of document.querySelectorAll(".chip[data-ref]")) {
  chip.addEventListener("click", () => {
    const on = chip.getAttribute("aria-pressed") !== "true";
    chip.setAttribute("aria-pressed", String(on));
    refs[chip.dataset.ref].visible = on && !simActive;
  });
}
function refsVisible(show) {
  for (const chip of document.querySelectorAll(".chip[data-ref]")) {
    refs[chip.dataset.ref].visible = show && chip.getAttribute("aria-pressed") === "true";
  }
}

// ---- joint sliders ----------------------------------------------------------
const GROUPS = [
  ["頭・首", ["neck_pitch", "head_pitch", "head_yaw", "head_roll"]],
  ["左脚", ["left_hip_yaw", "left_hip_roll", "left_hip_pitch", "left_knee", "left_ankle"]],
  ["右脚", ["right_hip_yaw", "right_hip_roll", "right_hip_pitch", "right_knee", "right_ankle"]],
];
const LABELS = {
  hip_yaw: "股ヨー", hip_roll: "股ロール", hip_pitch: "股ピッチ", knee: "膝", ankle: "足首",
  neck_pitch: "首ピッチ", head_pitch: "頭ピッチ", head_yaw: "頭ヨー", head_roll: "頭ロール",
};
const rowsEl = document.getElementById("joint-rows");
const sliders = [];
for (const [title, names] of GROUPS) {
  const h = document.createElement("h3");
  h.textContent = title;
  rowsEl.appendChild(h);
  for (const name of names) {
    const idx = JOINT_NAMES.indexOf(name);
    const [lo, hi] = JOINT_RANGES[name];
    const row = document.createElement("div");
    row.className = "row";
    const label = document.createElement("label");
    label.textContent = LABELS[name.replace(/^(left|right)_/, "")] || name;
    const input = document.createElement("input");
    input.type = "range";
    input.min = lo; input.max = hi; input.step = 0.01; input.value = 0;
    input.setAttribute("aria-label", name);
    const deg = document.createElement("span");
    deg.className = "deg";
    deg.textContent = "0°";
    input.addEventListener("input", () => {
      angles[idx] = parseFloat(input.value);
      deg.textContent = `${Math.round(angles[idx] * 180 / Math.PI)}°`;
      pose();
    });
    label.appendChild(input);
    row.append(label, deg);
    rowsEl.appendChild(row);
    sliders.push({ input, deg, idx });
  }
}
document.getElementById("reset").addEventListener("click", () => {
  for (const s of sliders) { s.input.value = 0; angles[s.idx] = 0; s.deg.textContent = "0°"; }
  pose();
});

// ---- camera: a small hand-rolled orbit --------------------------------------
const target = new THREE.Vector3(0, 0.125, 0);
let yaw = 0.7, pitch = 0.32, dist = 0.72;
function placeCamera() {
  camera.position.set(
    target.x + dist * Math.cos(pitch) * Math.sin(yaw),
    target.y + dist * Math.sin(pitch),
    target.z + dist * Math.cos(pitch) * Math.cos(yaw),
  );
  camera.lookAt(target);
}
let drag = null;
canvas.addEventListener("pointerdown", (e) => {
  canvas.setPointerCapture(e.pointerId);
  drag = { x: e.clientX, y: e.clientY, pan: e.shiftKey || e.button === 2 };
});
canvas.addEventListener("pointermove", (e) => {
  if (!drag) return;
  const dx = e.clientX - drag.x, dy = e.clientY - drag.y;
  drag.x = e.clientX; drag.y = e.clientY;
  if (drag.pan) {
    const s = dist * 0.0012;
    const right = new THREE.Vector3().setFromMatrixColumn(camera.matrix, 0);
    const up = new THREE.Vector3().setFromMatrixColumn(camera.matrix, 1);
    target.addScaledVector(right, -dx * s).addScaledVector(up, dy * s);
  } else {
    yaw -= dx * 0.006;
    pitch = Math.min(1.45, Math.max(-0.2, pitch + dy * 0.006));
  }
  placeCamera();
});
canvas.addEventListener("pointerup", () => { drag = null; });
canvas.addEventListener("contextmenu", (e) => e.preventDefault());
canvas.addEventListener("wheel", (e) => {
  e.preventDefault();
  dist = Math.min(6, Math.max(0.15, dist * Math.exp(e.deltaY * 0.001)));
  placeCamera();
}, { passive: false });

// Pinch zoom: track two pointers.
const touches = new Map();
canvas.addEventListener("pointerdown", (e) => touches.set(e.pointerId, e));
canvas.addEventListener("pointermove", (e) => {
  if (!touches.has(e.pointerId)) return;
  const prev = [...touches.values()];
  touches.set(e.pointerId, e);
  if (touches.size === 2) {
    const now = [...touches.values()];
    const d0 = Math.hypot(prev[0].clientX - prev[1].clientX, prev[0].clientY - prev[1].clientY);
    const d1 = Math.hypot(now[0].clientX - now[1].clientX, now[0].clientY - now[1].clientY);
    if (d0 > 0) { dist = Math.min(6, Math.max(0.15, dist * d0 / d1)); placeCamera(); }
    drag = null; // two fingers zoom; don't also rotate
  }
});
canvas.addEventListener("pointerup", (e) => touches.delete(e.pointerId));
canvas.addEventListener("pointercancel", (e) => touches.delete(e.pointerId));

// ---- physics simulation (--sim builds only) ---------------------------------
// MuJoCo (wasm) steps the training collision model at 200 Hz; the real robot's
// walking and standing policies run as plain-JS MLPs at 50 Hz, with robotd's
// observation layout and walk/stand switching. See duck-control/src/obs.rs.
let simActive = false;
let sim = null;

function mlp(meta, blob) {
  const W = {};
  for (const m of meta) {
    const n = m.shape.reduce((a, b) => a * b, 1);
    W[m.name] = { a: blob.subarray(m.offset, m.offset + n), shape: m.shape };
  }
  const linear = (w, b, x) => {
    const [out, inn] = w.shape;
    const y = new Float32Array(out);
    for (let o = 0; o < out; o++) {
      let s = b.a[o];
      const row = o * inn;
      for (let i = 0; i < inn; i++) s += w.a[row + i] * x[i];
      y[o] = s;
    }
    return y;
  };
  const elu = (x) => { for (let i = 0; i < x.length; i++) if (x[i] < 0) x[i] = Math.exp(x[i]) - 1; return x; };
  return (obs) => {
    const x = new Float32Array(61);
    const mean = W["obs_normalizer._mean"].a, std = W["onnx::Div_24"].a;
    for (let i = 0; i < 61; i++) x[i] = (obs[i] - mean[i]) / std[i];
    let h = elu(linear(W["mlp.0.weight"], W["mlp.0.bias"], x));
    h = elu(linear(W["mlp.2.weight"], W["mlp.2.bias"], h));
    h = elu(linear(W["mlp.4.weight"], W["mlp.4.bias"], h));
    return linear(W["mlp.6.weight"], W["mlp.6.bias"], h);
  };
}

async function initSim() {
  const [wasm, walkBlob, standBlob] = await Promise.all([
    gunzip(SIM_ASSETS.wasm), gunzip(SIM_ASSETS.walk.blob), gunzip(SIM_ASSETS.stand.blob),
  ]);
  const mujoco = await loadMujoco({ wasmBinary: wasm });
  mujoco.FS.writeFile("/physics.xml", b64bytes(SIM_ASSETS.xml));
  for (const [name, b64] of Object.entries(SIM_ASSETS.meshes)) {
    mujoco.FS.writeFile("/" + name, b64bytes(b64));
  }
  const model = mujoco.MjModel.mj_loadXML("/physics.xml");
  const data = new mujoco.MjData(model);

  const nets = {
    walk: mlp(SIM_ASSETS.walk.meta, new Float32Array(walkBlob.buffer)),
    stand: mlp(SIM_ASSETS.stand.meta, new Float32Array(standBlob.buffer)),
  };

  let lastAction = new Float32Array(14);
  const obs = new Float32Array(61);

  function reset() {
    mujoco.mj_resetData(model, data);
    const qpos = data.qpos;
    qpos[2] = 0.125;
    qpos[3] = 1; qpos[4] = 0; qpos[5] = 0; qpos[6] = 0;
    for (let j = 0; j < 14; j++) { qpos[7 + j] = HOME[j]; data.ctrl[j] = HOME[j]; }
    lastAction = new Float32Array(14);
    mujoco.mj_forward(model, data);
  }
  reset();

  // One 50 Hz control tick: observation → policy → targets → 4 physics steps.
  function tick(cmd) {
    const qpos = data.qpos, qvel = data.qvel;
    const w = qpos[3], x = qpos[4], y = qpos[5], z = qpos[6];
    obs[0] = qvel[3]; obs[1] = qvel[4]; obs[2] = qvel[5]; // gyro, trunk frame
    // projected gravity: R^T (0,0,-1)
    obs[3] = -2 * (x * z - w * y);
    obs[4] = -2 * (y * z + w * x);
    obs[5] = -(1 - 2 * (x * x + y * y));
    for (let j = 0; j < 14; j++) {
      obs[6 + j] = qpos[7 + j] - HOME[j];
      obs[20 + j] = qvel[6 + j];
      obs[34 + j] = lastAction[j];
    }
    obs[48] = cmd.vx; obs[49] = cmd.vy; obs[50] = cmd.vyaw;
    obs[51] = cmd.head[0]; obs[52] = cmd.head[1]; obs[53] = cmd.head[2]; obs[54] = cmd.head[3];
    obs[55] = 0; obs[56] = 0; obs[57] = cmd.bodyZ; obs[58] = 0; obs[59] = 0; obs[60] = 0;

    // robotd's rule: below the standing threshold the standing net drives —
    // it is also the fall-recovery net, so zero command = get back up.
    const mag = Math.hypot(cmd.vx, cmd.vy, cmd.vyaw);
    const action = (mag <= 0.05 ? nets.stand : nets.walk)(obs);
    lastAction = action;
    for (let j = 0; j < 14; j++) data.ctrl[j] = HOME[j] + action[j];
    for (let k = 0; k < 4; k++) mujoco.mj_step(model, data);
  }

  function push() {
    const dir = Math.random() * Math.PI * 2;
    const mag = 1.0 + Math.random() * 0.4; // the training push magnitude
    data.qvel[0] += mag * Math.cos(dir);
    data.qvel[1] += mag * Math.sin(dir);
  }

  return {
    tick, reset, push,
    get qpos() { return data.qpos; },
    get qvel() { return data.qvel; },
    up() { const q = data.qpos; return 1 - 2 * (q[4] * q[4] + q[5] * q[5]); },
  };
}

// ---- sim UI -----------------------------------------------------------------
const simToggle = document.getElementById("simtoggle");
if (SIM_ASSETS && typeof DecompressionStream !== "undefined") simToggle.hidden = false;

const held = new Set();
const cmd = { vx: 0, vy: 0, vyaw: 0, head: [0, 0, 0, 0], bodyZ: 0 };
const cmdTarget = { vx: 0, vy: 0, vyaw: 0 };
let fallen = false;
let simDistance = 0;
let lastSimPos = null;

function computeTarget() {
  // Fallen: force zero twist so the standing net can do its recovery job.
  if (fallen) { cmdTarget.vx = 0; cmdTarget.vy = 0; cmdTarget.vyaw = 0; return; }
  cmdTarget.vx = held.has("fwd") ? 0.3 : held.has("back") ? -0.25 : 0;
  cmdTarget.vy = held.has("left") ? 0.2 : held.has("right") ? -0.2 : 0;
  cmdTarget.vyaw = held.has("turnl") ? 0.7 : held.has("turnr") ? -0.7 : 0;
}
const KEYMAP = {
  w: "fwd", arrowup: "fwd", s: "back", arrowdown: "back",
  a: "left", d: "right", arrowleft: "turnl", arrowright: "turnr",
};
addEventListener("keydown", (e) => {
  if (!simActive || e.repeat || e.target.tagName === "INPUT") return;
  const c = KEYMAP[e.key.toLowerCase()];
  if (c) { held.add(c); computeTarget(); e.preventDefault(); }
  if (e.key === " ") { held.clear(); computeTarget(); e.preventDefault(); }
});
addEventListener("keyup", (e) => {
  const c = KEYMAP[e.key.toLowerCase()];
  if (c) { held.delete(c); computeTarget(); }
});
for (const btn of document.querySelectorAll(".simbtn[data-cmd]")) {
  const c = btn.dataset.cmd;
  if (c === "stop") { btn.addEventListener("click", () => { held.clear(); computeTarget(); }); continue; }
  if (c === "push") { btn.addEventListener("click", () => { if (sim) sim.push(); }); continue; }
  btn.addEventListener("pointerdown", (e) => { btn.setPointerCapture(e.pointerId); btn.classList.add("held"); held.add(c); computeTarget(); });
  const release = () => { btn.classList.remove("held"); held.delete(c); computeTarget(); };
  btn.addEventListener("pointerup", release);
  btn.addEventListener("pointercancel", release);
}
document.getElementById("simreset").addEventListener("click", () => {
  if (!sim) return;
  sim.reset();
  simDistance = 0; lastSimPos = null;
  held.clear(); computeTarget();
});

// Head command sliders: ride the command block (obs.rs 51..55), so they work
// while standing and walking alike — the policy moves the head, not us.
{
  const rows = document.getElementById("head-rows");
  const HEADS = [["首ピッチ", 0, -0.5, 0.5], ["頭ピッチ", 1, -0.5, 0.5],
                 ["頭ヨー", 2, -1.0, 1.0], ["頭ロール", 3, -0.4, 0.4]];
  for (const [label, i, lo, hi] of HEADS) {
    const row = document.createElement("div");
    row.className = "row";
    const l = document.createElement("label");
    l.textContent = label;
    const input = document.createElement("input");
    input.type = "range"; input.min = lo; input.max = hi; input.step = 0.01; input.value = 0;
    input.setAttribute("aria-label", `head command ${label}`);
    const deg = document.createElement("span");
    deg.className = "deg"; deg.textContent = "0°";
    input.addEventListener("input", () => {
      cmd.head[i] = parseFloat(input.value);
      deg.textContent = `${Math.round(cmd.head[i] * 180 / Math.PI)}°`;
    });
    l.appendChild(input);
    row.append(l, deg);
    rows.appendChild(row);
  }
}

// ---- AR compositing ---------------------------------------------------------
// "cam"/"photo" put a live camera feed or a picked picture behind a transparent
// canvas: a fixed-shot composite the user lines up by hand (the hint says how).
// WebXR AR, where the device supports it, is the real thing: hit-test finds a
// surface, a tap drops the duck there at true scale.
const arLayer = document.getElementById("arlayer");
const arVideo = document.getElementById("arvideo");
const arImg = document.getElementById("arimg");
const arHint = document.getElementById("ar-hint");
const arGridBtn = document.getElementById("ar-grid");
let camStream = null;

function setShadows(on) {
  shadowPlane.visible = on;
  if (renderer.shadowMap.enabled !== on) {
    renderer.shadowMap.enabled = on;
    sun.castShadow = on;
    for (const m of [...outerMats, ...innerMats]) m.needsUpdate = true;
  }
}

function applyAr(mode) {
  arMode = mode;
  const on = mode !== "off";
  arLayer.style.display = on ? "block" : "none";
  arVideo.style.display = mode === "cam" ? "block" : "none";
  arImg.style.display = mode === "photo" ? "block" : "none";
  arHint.style.display = on ? "block" : "none";
  if (mode !== "cam" && camStream) {
    for (const t of camStream.getTracks()) t.stop();
    camStream = null;
    arVideo.srcObject = null;
  }
  setShadows(on);
  const showGrid = !on || arGridBtn.getAttribute("aria-pressed") === "true";
  gridMajor.visible = showGrid;
  gridMinor.visible = showGrid;
  document.getElementById("ar-cam").setAttribute("aria-pressed", String(mode === "cam"));
  document.getElementById("ar-photo").setAttribute("aria-pressed", String(mode === "photo"));
  document.getElementById("ar-shot").setAttribute("aria-pressed", String(mode === "photo"));
  document.getElementById("ar-off").setAttribute("aria-pressed", String(mode === "off"));
  applyTheme();
}

// The page is usually viewed inside an embedding frame that doesn't grant the
// camera permission — there getUserMedia can never succeed. But the artifact
// is served from its own origin, so its own URL opened as a top-level tab CAN
// prompt for the camera. The 別タブ button is that escape hatch.
const embedded = (() => { try { return window.self !== window.top; } catch { return true; } })();
const arOpen = document.getElementById("ar-open");
const arEscape = document.getElementById("ar-escape");
const arUrl = document.getElementById("ar-url");
if (embedded) {
  arOpen.hidden = false;
  arEscape.hidden = false;
  document.getElementById("ar-link").href = location.href;
  arUrl.value = location.href;
}
// A one-line diagnosis so "it doesn't work" conversations have something to
// quote: where the page thinks it runs, and whether the live API even exists.
document.getElementById("ar-diag").textContent =
  `build __BUILD_TAG__ ／ 状態: ${embedded ? "埋め込み表示（ライブカメラ不可の場合あり）" : "単独ページ"} ／ ` +
  `ライブAPI: ${navigator.mediaDevices && navigator.mediaDevices.getUserMedia ? "あり" : "なし"}`;
arOpen.addEventListener("click", async () => {
  const w = window.open(location.href, "_blank");
  if (!w) {
    arHint.style.display = "block";
    arHint.textContent = "ポップアップがブロックされました。下の「リンクとして単独ページを開く」を押すか、URL欄をコピーしてSafariに貼り付けてください。";
  }
});
document.getElementById("ar-copy").addEventListener("click", async () => {
  let ok = false;
  try { await navigator.clipboard.writeText(location.href); ok = true; } catch {}
  if (!ok) { arUrl.focus(); arUrl.select(); try { ok = document.execCommand("copy"); } catch {} }
  document.getElementById("ar-copy").textContent = ok ? "コピー済" : "手動で選択してコピー";
});

document.getElementById("ar-cam").addEventListener("click", async () => {
  if (arMode === "cam") return;
  arHint.style.display = "none";
  const fail = (msg) => {
    arHint.style.display = "block";
    arHint.textContent = msg + " 「撮影」ならカメラアプリで撮った写真にすぐ重ねられます（権限不要）。";
  };
  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    fail(embedded
      ? "この埋め込み表示ではライブカメラAPIを使えません。「別タブで開く」から単独ページとして開くと使えます。"
      : "このブラウザではライブカメラAPIを使えません。");
    return;
  }
  try {
    camStream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: { ideal: "environment" } }, audio: false,
    });
  } catch (e) {
    if (e && (e.name === "NotAllowedError" || e.name === "SecurityError")) {
      fail(embedded
        ? "埋め込み表示のためカメラ権限が下りませんでした。上の「別タブで開く」から単独ページとして開き、もう一度「ライブ」を押してください。"
        : "カメラ権限が拒否されています。iPhoneのSafariでは、アドレスバー左の「ぁあ」→「Webサイトの設定」→「カメラ」を「許可」にしてください。");
    } else if (e && e.name === "NotFoundError") {
      fail("カメラデバイスが見つかりませんでした。");
    } else {
      fail(`カメラを起動できませんでした（${e && e.name || "エラー"}）。`);
    }
    return;
  }
  arVideo.srcObject = camStream;
  try { await arVideo.play(); } catch {}
  applyAr("cam");
});
document.getElementById("ar-photo").addEventListener("click", () => {
  document.getElementById("ar-file").click();
});
// 撮影: a capture-flagged file input opens the camera app directly on phones —
// no getUserMedia, no permission prompt an embedding frame can strip away.
document.getElementById("ar-shot").addEventListener("click", () => {
  document.getElementById("ar-capture").click();
});
function useArPicture(e) {
  const file = e.target.files && e.target.files[0];
  if (!file) return;
  if (arImg.src.startsWith("blob:")) URL.revokeObjectURL(arImg.src);
  arImg.src = URL.createObjectURL(file);
  e.target.value = ""; // let the same picture be re-picked later
  applyAr("photo");
}
document.getElementById("ar-file").addEventListener("change", useArPicture);
document.getElementById("ar-capture").addEventListener("change", useArPicture);
document.getElementById("ar-off").addEventListener("click", () => applyAr("off"));
arGridBtn.addEventListener("click", () => {
  const on = arGridBtn.getAttribute("aria-pressed") !== "true";
  arGridBtn.setAttribute("aria-pressed", String(on));
  const showGrid = arMode === "off" || on;
  gridMajor.visible = showGrid;
  gridMinor.visible = showGrid;
});

// ---- WebXR AR (hit-test placement, true scale) ------------------------------
const xrBtn = document.getElementById("ar-xr");
let xrHitSource = null;
let xrPlaced = false;
const reticle = new THREE.Mesh(
  new THREE.RingGeometry(0.05, 0.065, 40),
  new THREE.MeshBasicMaterial({ color: 0xf0a82a, side: THREE.DoubleSide }),
);
reticle.rotation.x = -Math.PI / 2;
reticle.matrixAutoUpdate = false;
reticle.visible = false;
scene.add(reticle);

if (navigator.xr && SIM_ASSETS) {
  navigator.xr.isSessionSupported("immersive-ar")
    .then((ok) => { if (ok) xrBtn.hidden = false; })
    .catch(() => {});
}
// Taps on the DOM-overlaid panel must not double as placement taps.
document.getElementById("simpanel").addEventListener("beforexrselect", (e) => e.preventDefault());
xrBtn.addEventListener("click", async () => {
  try {
    const session = await navigator.xr.requestSession("immersive-ar", {
      requiredFeatures: ["hit-test"],
      optionalFeatures: ["dom-overlay"],
      domOverlay: { root: document.getElementById("simpanel") },
    });
    applyAr("off");
    renderer.xr.enabled = true;
    renderer.xr.setReferenceSpaceType("local");
    await renderer.xr.setSession(session);
    const viewer = await session.requestReferenceSpace("viewer");
    xrHitSource = await session.requestHitTestSource({ space: viewer });
    xrPlaced = false;
    gridMajor.visible = false;
    gridMinor.visible = false;
    setShadows(true);
    session.addEventListener("select", () => {
      if (reticle.visible) {
        simWorld.position.setFromMatrixPosition(reticle.matrix);
        if (!xrPlaced && sim) { sim.reset(); simDistance = 0; lastSimPos = null; }
        else if (sim) {
          // Re-placing after a walk: land the duck itself on the reticle, not
          // the origin it wandered away from (MJCF z-up → world x, -z).
          simWorld.position.x -= sim.qpos[0];
          simWorld.position.z -= -sim.qpos[1];
        }
        xrPlaced = true;
      } else if (xrPlaced && !session.domOverlayState) {
        // No DOM overlay to press buttons on: tapping toggles forward walking.
        if (held.has("fwd")) held.delete("fwd"); else held.add("fwd");
        computeTarget();
      }
    });
    session.addEventListener("end", () => {
      renderer.xr.enabled = false;
      xrHitSource = null;
      reticle.visible = false;
      simWorld.position.set(0, 0, 0);
      setShadows(arMode !== "off");
      gridMajor.visible = true;
      gridMinor.visible = true;
      resize();
      placeCamera();
    });
  } catch (e) {
    xrBtn.textContent = "WebXR ARを開始できませんでした";
    console.error(e);
  }
});

function xrFrameUpdate(frame) {
  if (!frame || !xrHitSource) return;
  const hits = frame.getHitTestResults(xrHitSource);
  if (hits.length) {
    const pose = hits[0].getPose(renderer.xr.getReferenceSpace());
    if (pose) {
      reticle.visible = !xrPlaced || !held.size; // once walking, stop flashing it
      reticle.matrix.fromArray(pose.transform.matrix);
      if (!xrPlaced) {
        simWorld.position.setFromMatrixPosition(reticle.matrix);
      }
    }
  } else {
    reticle.visible = false;
  }
}

let simInitPromise = null;
simToggle.addEventListener("click", async () => {
  const on = simToggle.getAttribute("aria-pressed") !== "true";
  if (on && !sim) {
    simToggle.disabled = true;
    simToggle.textContent = "読み込み中…";
    try {
      simInitPromise ??= initSim();
      sim = await simInitPromise;
    } catch (e) {
      simToggle.textContent = "シム初期化失敗";
      console.error(e);
      return;
    }
    simToggle.disabled = false;
    simToggle.textContent = "歩行シミュレーション";
  }
  simActive = on && !!sim;
  simToggle.setAttribute("aria-pressed", String(simActive));
  document.getElementById("joints").style.display = simActive ? "none" : "";
  document.getElementById("simpanel").style.display = simActive ? "flex" : "";
  document.getElementById("hint").style.display = simActive ? "none" : "";
  refs.ruler.visible = !simActive && document.querySelector('[data-ref="ruler"]').getAttribute("aria-pressed") === "true";
  refsVisible(!simActive);
  buildGrids(simActive ? 4.0 : 0.6);
  held.clear(); computeTarget();
  if (simActive) {
    sim.reset();
    simDistance = 0; lastSimPos = null;
    simAccum = 0; simPrev = null;
  } else {
    applyAr("off");
    const xrSession = renderer.xr.getSession && renderer.xr.getSession();
    if (xrSession) xrSession.end().catch(() => {});
    // Back to the assy view: home pose at the origin.
    for (const s of sliders) { angles[s.idx] = parseFloat(s.input.value); }
    bodyGroups[0].position.set(...duck.bodies[0].pos);
    pose();
    target.set(0, 0.125, 0);
    placeCamera();
    updateDims();
  }
});

// ---- sim loop ---------------------------------------------------------------
const CONTROL_DT = 0.02;
let simAccum = 0;
let simPrev = null;
const stateEl = document.getElementById("simstate");
const vxEl = document.getElementById("simvx");
const wzEl = document.getElementById("simwz");
const distEl = document.getElementById("simdist");
const fallnote = document.getElementById("fallnote");

function simFrame(now) {
  if (simPrev === null) simPrev = now;
  simAccum += Math.min((now - simPrev) / 1000, 0.1);
  simPrev = now;

  // Ramp the actual command toward the held target: sticks, not steps.
  const approach = (v, t, rate) => v + Math.sign(t - v) * Math.min(Math.abs(t - v), rate * CONTROL_DT);
  let ticks = 0;
  while (simAccum >= CONTROL_DT && ticks < 8) {
    cmd.vx = approach(cmd.vx, cmdTarget.vx, 0.5);
    cmd.vy = approach(cmd.vy, cmdTarget.vy, 0.4);
    cmd.vyaw = approach(cmd.vyaw, cmdTarget.vyaw, 1.6);
    sim.tick(cmd);
    simAccum -= CONTROL_DT;
    ticks++;
  }

  const qpos = sim.qpos, qvel = sim.qvel;
  const up = sim.up();
  const wasFallen = fallen;
  fallen = up < 0.55 || qpos[2] < 0.055;
  if (fallen !== wasFallen) { fallnote.style.display = fallen ? "block" : "none"; computeTarget(); }

  // Drive the visual model: joint angles + trunk pose from the physics state.
  for (let slot = 0; slot < 14; slot++) {
    const joint = slot < 9 ? slot : slot + 1; // skip the mouth slot
    angles[joint] = qpos[7 + slot];
  }
  pose();
  const trunk = bodyGroups[0];
  trunk.position.set(qpos[0], qpos[1], qpos[2]);
  trunk.quaternion.set(qpos[4], qpos[5], qpos[6], qpos[3]);

  // Camera follows the duck (MJCF z-up → world: x→x, z→y, y→-z) — but not in
  // the AR modes, where the camera must hold still against a fixed background
  // (or is the device's own pose under WebXR).
  if (arMode === "off" && !renderer.xr.isPresenting) {
    target.x += (qpos[0] - target.x) * 0.08;
    target.y += (qpos[2] - target.y) * 0.08;
    target.z += (-qpos[1] - target.z) * 0.08;
    placeCamera();
  }
  // Keep the shadow (and its light frustum) centred on the walking duck. The
  // plane is a child of simWorld; the light lives in the scene, so it needs
  // the simWorld offset (WebXR placement) added on.
  if (renderer.shadowMap.enabled) {
    shadowPlane.position.set(qpos[0], 0.001, -qpos[1]);
    const wx = simWorld.position.x + qpos[0], wz = simWorld.position.z - qpos[1];
    sun.position.set(wx + 0.6, simWorld.position.y + 1.2, wz + 0.8);
    sun.target.position.set(wx, simWorld.position.y, wz);
    sun.target.updateMatrixWorld();
  }

  // Status readout: body-frame forward speed and yaw rate, plus odometer.
  const w = qpos[3], x = qpos[4], y = qpos[5], z = qpos[6];
  const fwd = (1 - 2 * (y * y + z * z)) * qvel[0] + 2 * (x * y + w * z) * qvel[1] + 2 * (x * z - w * y) * qvel[2];
  if (lastSimPos) simDistance += Math.hypot(qpos[0] - lastSimPos[0], qpos[1] - lastSimPos[1]);
  lastSimPos = [qpos[0], qpos[1]];
  stateEl.textContent = fallen ? "転倒 → 復帰中"
    : Math.hypot(cmd.vx, cmd.vy, cmd.vyaw) <= 0.05 ? "立位 (alpha_stand)" : "歩行 (alpha_walking)";
  vxEl.textContent = `${fwd.toFixed(2)} m/s`;
  wzEl.textContent = `${qvel[5].toFixed(2)} rad/s`;
  distEl.textContent = `${simDistance.toFixed(2)} m`;
}

// ---- boot -------------------------------------------------------------------
function resize() {
  const w = innerWidth, h = innerHeight;
  renderer.setSize(w, h, false);
  camera.aspect = w / h;
  camera.updateProjectionMatrix();
}
addEventListener("resize", resize);
resize();
applyTheme();
pose();
placeCamera();
renderer.setAnimationLoop((now, frame) => {
  if (renderer.xr.isPresenting) xrFrameUpdate(frame);
  if (simActive && sim) simFrame(now);
  else simPrev = null;
  renderer.render(scene, camera);
});
</script>
"""


if __name__ == "__main__":
    main()
