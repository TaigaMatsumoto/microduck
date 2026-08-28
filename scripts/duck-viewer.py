#!/usr/bin/env python3
"""Emit a standalone HTML viewer of the robot's assembly model.

The page is a self-contained web view of the robot: a Three.js scene with orbit
controls, a centimetre grid and ruler, real-world scale references (a drink can,
a bottle, a phone), an x-ray mode that reveals the parts inside the shells, and
per-joint pose sliders limited to the real servo ranges — so anyone with a
browser can judge the robot's size and articulation without MuJoCo or CAD tools.

Two sources, by fidelity:

    scripts/duck-viewer.py path/to/robot_walk.xml [out.html]

embeds the app repository's MJCF visual model at full CAD resolution (~430k
triangles; needs numpy, like bake-duck-mesh.py). Without an MJCF,

    scripts/duck-viewer.py [out.html]

falls back to `robotctl/assets/duck.bin`, the terminal monitor's decimated bake
that is committed here — coarse, but always available.

Writes `duck-viewer.html` next to this script by default. The page embeds the
geometry as base64 and loads only Three.js from a CDN — serve it, mail it, or
open it from disk; there is nothing else to deploy.
"""

import base64
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


def main() -> None:
    args = sys.argv[1:]
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
        model = model_from_duck_bin((REPO / "robotctl" / "assets" / "duck.bin").read_bytes())
        ranges = joint_ranges(REPO / "kinematics" / "assets" / "alpha" / "robot_walk.xml")
        source_note = "robotctl/assets/duck.bin — 端末モニタ用の間引きメッシュ"
        dims_note = "寸法は現在のポーズの外接寸法（メッシュはデシメート済みのため±数mm）。床グリッドは5cm。"

    html = TEMPLATE
    html = html.replace("__DUCK_B64__", base64.b64encode(serialize(model)).decode())
    html = html.replace("__JOINT_NAMES__", json.dumps(JOINT_NAMES))
    html = html.replace("__JOINT_RANGES__", json.dumps(ranges))
    html = html.replace("__SOURCE_NOTE__", source_note)
    html = html.replace("__DIMS_NOTE__", dims_note)
    out_path.write_text(html)
    print(f"{out_path}: {out_path.stat().st_size / 1024 / 1024:.1f} MB")


# The page itself. Kept as one template string so the tool stays a single file;
# tokens (__DUCK_B64__ etc.) are substituted above.
TEMPLATE = r"""<title>Microduck Assy Viewer</title>
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
  #stage { position: fixed; inset: 0; display: block; touch-action: none; cursor: grab; }
  #stage:active { cursor: grabbing; }

  .panel {
    position: fixed;
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
    padding: 10px; max-width: calc(100vw - 32px);
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
  .chip:focus-visible, #joints input:focus-visible, .chip-reset:focus-visible {
    outline: 2px solid var(--accent); outline-offset: 2px;
  }

  #joints {
    top: 16px; right: 16px; width: 252px;
    max-height: calc(100vh - 32px);
    display: flex; flex-direction: column;
  }
  #joints .bar {
    display: flex; align-items: center; justify-content: space-between;
    padding: 10px 14px;
  }
  #joints .bar h2 { font-size: 12px; font-weight: 500; letter-spacing: 0.08em; color: var(--muted); }
  .chip-reset {
    font: inherit; font-size: 11.5px; color: var(--muted);
    background: transparent; border: 1px solid var(--surface-border);
    border-radius: 6px; padding: 3px 9px; cursor: pointer;
  }
  #joints .body { overflow-y: auto; padding: 0 14px 12px; }
  #joints h3 {
    font-size: 10.5px; font-weight: 500; letter-spacing: 0.09em;
    text-transform: uppercase; color: var(--muted);
    margin: 12px 0 4px;
  }
  #joints .row { display: grid; grid-template-columns: 1fr 44px; align-items: center; gap: 8px; padding: 3px 0; }
  #joints label { font-size: 12px; display: block; }
  #joints input[type="range"] { width: 100%; accent-color: var(--accent); height: 18px; }
  #joints .deg {
    font-family: "IBM Plex Mono", monospace; font-size: 11px;
    font-variant-numeric: tabular-nums; text-align: right; color: var(--muted);
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
</div>

<div class="panel" id="joints">
  <div class="bar">
    <h2>関節ポーズ</h2>
    <button class="chip-reset" id="reset">リセット</button>
  </div>
  <div class="body" id="joint-rows"></div>
</div>

<div id="hint">ドラッグ：回転 ／ ホイール・ピンチ：ズーム ／ Shift+ドラッグ：移動</div>
<div id="fallback">3D表示を初期化できませんでした。WebGL対応のブラウザでご覧ください。</div>

<script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
<script>
"use strict";

// ---- embedded model ---------------------------------------------------------
// "DUK2", written by scripts/duck-viewer.py — see serialize() there.
const JOINT_NAMES = __JOINT_NAMES__;
const JOINT_RANGES = __JOINT_RANGES__;
const DUCK_B64 = "__DUCK_B64__";

function parseDuck(b64) {
  const bytes = Uint8Array.from(atob(b64), (c) => c.charCodeAt(0));
  const dv = new DataView(bytes.buffer);
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

  const meshes = [];
  for (let m = 0; m < nMesh; m++) {
    const nv = u32(), nf = u32();
    // Copy out (slice) rather than view: offsets are not element-aligned.
    const verts = new Float32Array(bytes.buffer.slice(at, at + nv * 12)); at += nv * 12;
    const faces = new Uint16Array(bytes.buffer.slice(at, at + nf * 6)); at += nf * 6;
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
  renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
} catch (e) {
  document.getElementById("fallback").style.display = "grid";
  throw e;
}
renderer.setPixelRatio(Math.min(devicePixelRatio, 2));

const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(32, 1, 0.01, 20);

scene.add(new THREE.HemisphereLight(0xffffff, 0x8899aa, 0.85));
const sun = new THREE.DirectionalLight(0xffffff, 0.75);
sun.position.set(0.6, 1.2, 0.8);
scene.add(sun);
const fill = new THREE.DirectionalLight(0xffffff, 0.25);
fill.position.set(-0.8, 0.4, -0.6);
scene.add(fill);

// Grids live in Three's y-up world; the robot group is rotated from MJCF z-up.
const themed = []; // objects recoloured/redrawn when the theme flips
function cssColor(name) {
  return new THREE.Color(getComputedStyle(document.documentElement).getPropertyValue(name).trim());
}
const gridMajor = new THREE.GridHelper(0.6, 12); // 5 cm
const gridMinor = new THREE.GridHelper(0.6, 60); // 1 cm
gridMinor.material.transparent = true;
gridMinor.material.opacity = 0.45;
scene.add(gridMajor, gridMinor);

function applyTheme() {
  renderer.setClearColor(cssColor("--bg"));
  gridMajor.material.color = cssColor("--grid-major");
  gridMinor.material.color = cssColor("--grid-minor");
  for (const fn of themed) fn();
}
matchMedia("(prefers-color-scheme: dark)").addEventListener("change", applyTheme);
new MutationObserver(applyTheme).observe(document.documentElement, { attributes: true, attributeFilter: ["data-theme"] });

// ---- robot ------------------------------------------------------------------
const duck = parseDuck(DUCK_B64);

const robotRoot = new THREE.Group();
robotRoot.rotation.x = -Math.PI / 2; // MJCF z-up → Three y-up
scene.add(robotRoot);

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
  updateDims();
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
    refs[chip.dataset.ref].visible = on;
  });
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
  dist = Math.min(3, Math.max(0.15, dist * Math.exp(e.deltaY * 0.001)));
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
    if (d0 > 0) { dist = Math.min(3, Math.max(0.15, dist * d0 / d1)); placeCamera(); }
    drag = null; // two fingers zoom; don't also rotate
  }
});
canvas.addEventListener("pointerup", (e) => touches.delete(e.pointerId));
canvas.addEventListener("pointercancel", (e) => touches.delete(e.pointerId));

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
renderer.setAnimationLoop(() => renderer.render(scene, camera));
</script>
"""


if __name__ == "__main__":
    main()
