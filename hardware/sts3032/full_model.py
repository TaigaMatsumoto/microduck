#!/usr/bin/env python3
"""STS3032換装フルモデル (アダプタ方式の全身ASSY).

現行microduckのフルCAD (robot_walk.xml の視覚モデル) から、全XL330
インスタンスを「ケージ + STS3032 + ホーンアダプタ + テールスリーブ」の
換装ユニットに差し替えた全身モデルを生成する。アダプタ群は xl330.stl と
同一座標系 (軸=X, 原点=出力軸) で設計してあるため、各インスタンスの
ポーズにそのまま置換できる — これがアダプタ方式の意味そのもの。

出力:
  full_model_sts3032.stl   — 全身結合メッシュ (立位ホームポーズ)
  full_model_sts3032.usdz  — AR Quick Look用 (実寸)
  render_full_*.png        — 全身レンダ (通常 / シェル透過でユニット可視)

実行: python full_model.py <robot_dir>
       robot_dir = microduck_rl の robot/microduck ディレクトリ
"""

import math
import sys
from pathlib import Path

import numpy as np

import cadquery as cq
import trimesh

import adapter_cad as A
import unit_assy as U

HERE = Path(__file__).parent

HOME = {"left_hip_yaw": 0, "left_hip_roll": -0.0873, "left_hip_pitch": -0.4579,
        "left_knee": -0.0049, "left_ankle": 0.4530, "neck_pitch": 0.3491,
        "head_pitch": 0.3491, "head_yaw": 0, "head_roll": 0,
        "right_hip_yaw": 0, "right_hip_roll": 0.0873, "right_hip_pitch": 0.4579,
        "right_knee": 0.0049, "right_ankle": -0.4530}

UNIT_COLORS = {"cage": (0.85, 0.63, 0.23), "servo": (0.16, 0.18, 0.22),
               "horn": (0.42, 0.60, 0.75), "sleeve": (0.55, 0.70, 0.55)}


def cq_mesh(wp, tol=0.05) -> trimesh.Trimesh:
    tmp = HERE / "_t.stl"
    cq.exporters.export(wp, str(tmp), tolerance=tol)
    m = trimesh.load(tmp)
    tmp.unlink()
    return m


UNIT_BUDGET = {"cage": 1200, "servo": 600, "horn": 300, "sleeve": 250}


def _slim(name, m):
    tris = _decimate(m.vertices[m.faces], UNIT_BUDGET[name])
    return trimesh.Trimesh(vertices=tris.reshape(-1, 3),
                           faces=np.arange(len(tris) * 3).reshape(-1, 3),
                           process=False)


def unit_parts() -> list[tuple[str, trimesh.Trimesh]]:
    """換装ユニットの4部品 (xl330座標系, mm)。全インスタンスで共有。
    USDZ/レンダ容量のため一度だけ間引いてから複製する。"""
    out = [("cage", _slim("cage", cq_mesh(A.cage()))),
           ("servo", _slim("servo", cq_mesh(U.servo_mock())))]
    horn = _slim("horn", cq_mesh(A.horn_adapter()))
    horn.apply_translation([16, 0, 0])
    out.append(("horn", horn))
    sleeve = _slim("sleeve", cq_mesh(A.tail_sleeve()))
    sleeve.apply_transform(
        trimesh.transformations.translation_matrix([-14.6, 0, 0])
        @ trimesh.transformations.rotation_matrix(math.pi, [0, 0, 1]))
    out.append(("sleeve", sleeve))
    return out


def build(robot_dir: Path):
    """→ [(name, trimesh(m単位・ワールド), rgb, is_unit)] を立位ポーズで。"""
    import mujoco
    m = mujoco.MjModel.from_xml_path(str(robot_dir / "robot_walk.xml"))
    d = mujoco.MjData(m)
    fq = m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT,
                                          "trunk_base_freejoint")]
    d.qpos[fq + 2] = 0.125
    d.qpos[fq + 3] = 1
    for n, h in HOME.items():
        d.qpos[m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)]] = h
    mujoco.mj_forward(m, d)

    units = [(n, mesh.copy()) for n, mesh in unit_parts()]
    for _, mesh in units:
        mesh.apply_scale(0.001)  # mm → m

    out = []
    n_swapped = 0
    for g in range(m.ngeom):
        if m.geom_group[g] != 2 or m.geom_dataid[g] < 0:
            continue
        mid = m.geom_dataid[g]
        mesh_name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_MESH, mid)
        R = d.geom_xmat[g].reshape(3, 3)
        p = d.geom_xpos[g]
        M = np.eye(4)
        M[:3, :3] = R
        M[:3, 3] = p
        if mesh_name == "xl330":
            n_swapped += 1
            for pname, pm in units:
                inst = pm.copy()
                inst.apply_transform(M)
                out.append((pname, inst, UNIT_COLORS[pname], True))
        else:
            va, vn = m.mesh_vertadr[mid], m.mesh_vertnum[mid]
            fa, fn = m.mesh_faceadr[mid], m.mesh_facenum[mid]
            tm = trimesh.Trimesh(vertices=m.mesh_vert[va:va + vn].copy(),
                                 faces=m.mesh_face[fa:fa + fn].copy(), process=False)
            tm.apply_transform(M)
            mat = m.geom_matid[g]
            rgb = tuple(m.mat_rgba[mat][:3]) if mat >= 0 else tuple(m.geom_rgba[g][:3])
            out.append((mesh_name, tm, rgb, False))
    print(f"xl330 instances swapped for STS3032 units: {n_swapped}")
    return out


def render(parts, fname, elev, azim, xray):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    fig = plt.figure(figsize=(7.5, 7), dpi=115)
    ax = fig.add_subplot(111, projection="3d")
    order = sorted(parts, key=lambda t: t[3])  # 外装を先に (透過時の重なり対策)
    for name, mesh, rgb, is_unit in order:
        alpha = 1.0
        if xray and not is_unit:
            alpha = 0.16
        tris = mesh.vertices[mesh.faces]
        n = mesh.face_normals
        lum = 0.62 + 0.38 * np.clip(n @ np.array([0.45, 0.55, 0.7]), -1, 1)
        pc = Poly3DCollection(tris, edgecolor="none", alpha=alpha)
        pc.set_facecolor(np.clip(np.array(rgb)[None, :] * lum[:, None], 0, 1))
        ax.add_collection3d(pc)
    ax.set_xlim(-0.16, 0.16)
    ax.set_ylim(-0.16, 0.16)
    ax.set_zlim(0, 0.30)
    ax.set_box_aspect((0.32, 0.32, 0.30))
    ax.view_init(elev=elev, azim=azim)
    ax.set_axis_off()
    title = "microduck STS3032 full model" + (" — x-ray (units visible)" if xray else "")
    ax.set_title(title, fontsize=10)
    plt.tight_layout()
    plt.savefig(HERE / fname)
    plt.close()
    print(fname)


def _decimate(tris: np.ndarray, budget: int) -> np.ndarray:
    """頂点クラスタ間引き (bake-duck-mesh.py方式)。入力/出力とも(n,3,3), mm。"""
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


def usdz(parts) -> None:
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade, UsdUtils, Vt
    usdc = HERE / "full_model_sts3032.usdc"
    if usdc.exists():
        usdc.unlink()
    stage = Usd.Stage.CreateNew(str(usdc))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.y)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    root = UsdGeom.Xform.Define(stage, "/Duck")
    stage.SetDefaultPrim(root.GetPrim())
    root.AddTransformOp().Set(
        Gf.Matrix4d().SetRotate(Gf.Rotation(Gf.Vec3d(1, 0, 0), -90)))
    mats = {}
    for i, (name, mesh, rgb, is_unit) in enumerate(parts):
        # 外装は間引いて容量を抑える (接触品質は不要、見た目のみ)
        mm = mesh
        if len(mesh.faces) > 2500:
            tris = _decimate(mesh.vertices[mesh.faces] * 1000, 2500) / 1000
            mm = trimesh.Trimesh(vertices=tris.reshape(-1, 3),
                                 faces=np.arange(len(tris) * 3).reshape(-1, 3),
                                 process=False)
        # RealityKitはdoubleSided無視 → 両巻き複製
        tris = mm.vertices[mm.faces]
        tris = np.concatenate([tris, tris[:, ::-1]])
        prim = UsdGeom.Mesh.Define(stage, f"/Duck/g{i}")
        pts = tris.reshape(-1, 3).astype(np.float32)
        prim.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(pts))
        prim.CreateFaceVertexCountsAttr([3] * len(tris))
        prim.CreateFaceVertexIndicesAttr(list(range(len(pts))))
        nrm = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
        nrm /= np.maximum(np.linalg.norm(nrm, axis=1, keepdims=True), 1e-12)
        prim.CreateNormalsAttr(Vt.Vec3fArray.FromNumpy(
            np.repeat(nrm, 3, axis=0).astype(np.float32)))
        prim.SetNormalsInterpolation(UsdGeom.Tokens.faceVarying)
        prim.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
        key = tuple(round(c, 3) for c in rgb)
        if key not in mats:
            mat = UsdShade.Material.Define(stage, f"/Duck/Materials/m{len(mats)}")
            sh = UsdShade.Shader.Define(stage, f"/Duck/Materials/m{len(mats)}/pbr")
            sh.CreateIdAttr("UsdPreviewSurface")
            sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(
                Gf.Vec3f(*[float(c) for c in key]))
            sh.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.5)
            mat.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(), "surface")
            mats[key] = mat
        UsdShade.MaterialBindingAPI.Apply(prim.GetPrim()).Bind(mats[key])
    stage.Save()
    ok = UsdUtils.CreateNewUsdzPackage(str(usdc), str(HERE / "full_model_sts3032.usdz"))
    usdc.unlink()
    size = (HERE / "full_model_sts3032.usdz").stat().st_size / 1e6
    print(f"full_model_sts3032.usdz ({size:.1f} MB)" if ok else "usdz failed")


def main() -> None:
    robot_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    if not robot_dir or not (robot_dir / "robot_walk.xml").exists():
        raise SystemExit(__doc__)
    parts = build(robot_dir)
    combined = trimesh.util.concatenate([mesh for _, mesh, _, _ in parts])
    combined.export(HERE / "full_model_sts3032.stl")
    print(f"full_model_sts3032.stl ({len(combined.faces)} tris)")
    render(parts, "render_full_opaque.png", 12, -50, xray=False)
    render(parts, "render_full_xray.png", 12, -50, xray=True)
    render(parts, "render_full_xray_front.png", 4, 0, xray=True)
    usdz(parts)


if __name__ == "__main__":
    main()
