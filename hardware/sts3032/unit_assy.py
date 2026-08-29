#!/usr/bin/env python3
"""STS3032換装ユニットのASSYモデル生成.

adapter_cad.py の3部品に、カタログ寸法から起こしたSTS3032-C036のモック
モデルを加え、組付け状態のアセンブリを出力する:

  unit_assy.step        — 色付きSTEPアセンブリ (CADで開ける)
  unit_assy.stl         — 結合メッシュ (ビューア用)
  unit_assy.usdz        — AR Quick Look用 (iPhoneで実寸配置)
  render_unit*.png      — 組立/分解/XL330外形重ね合わせのレンダ

サーボのモックはカタログ公称 (32×12×27.5) + TODO_推定寸法で、界面検討の
可視化が目的。製造判断は実測更新後の adapter_cad.py に従うこと。

実行: python unit_assy.py   (要 cadquery, trimesh, numpy; usdzは usd-core)
"""

import math
from pathlib import Path

import cadquery as cq

import adapter_cad as A

HERE = Path(__file__).parent

# ── STS3032-C036 mock (カタログ+推定; 軸=X, ホーン座面 x=+13) ────────────────
SERVO_TOP_X = 13.0
SERVO_BOT_X = SERVO_TOP_X - 27.5          # -14.5 (尾側へ1.5mm突出)
HUB_DIA = 5.8                              # 出力ハブ ★要実測
HUB_LEN = 3.0
TAIL_STUB_LEN = 5.0                        # C036後軸の露出長 ★要実測


def servo_mock() -> cq.Workplane:
    zc = (A.STS_NOSE_Z + A.STS_TAIL_Z) / 2
    body = (cq.Workplane("XY")
            .box(27.5, A.STS_W, A.STS_L)
            .translate(((SERVO_TOP_X + SERVO_BOT_X) / 2, 0, zc))
            .edges("|X").fillet(1.2))
    hub = (cq.Workplane("YZ").workplane(offset=SERVO_TOP_X)
           .circle(HUB_DIA / 2).extrude(HUB_LEN))
    spline = (cq.Workplane("YZ").workplane(offset=SERVO_TOP_X + HUB_LEN)
              .circle(1.6).extrude(1.2))
    stub = (cq.Workplane("YZ").workplane(offset=SERVO_BOT_X)
            .circle(A.TODO_STS_TAIL_SHAFT / 2).extrude(-TAIL_STUB_LEN))
    # コネクタ/ケーブル口のモック (尾側端)
    conn = (cq.Workplane("XY")
            .box(6, 8, 3)
            .translate((SERVO_BOT_X + 4, 0, A.STS_TAIL_Z - 1.5)))
    return body.union(hub).union(spline).union(stub).union(conn)


def build(explode: float = 0.0) -> cq.Assembly:
    """explode>0 で分解図 (X方向オフセット)."""
    asm = cq.Assembly(name="sts3032_unit")
    asm.add(A.cage(), name="cage",
            color=cq.Color(0.85, 0.63, 0.23, 0.55 if explode == 0 else 1.0))
    asm.add(servo_mock(), name="sts3032_c036",
            loc=cq.Location(cq.Vector(0, 0, 0)),
            color=cq.Color(0.20, 0.23, 0.28, 1.0))
    # ホーンアダプタ: サーボホーン上 (x=+16) に着座
    asm.add(A.horn_adapter(), name="horn_adapter",
            loc=cq.Location(cq.Vector(16.0 + explode, 0, 0)),
            color=cq.Color(0.42, 0.60, 0.75, 1.0))
    # テールスリーブ: 後軸に挿入 (フランジ側を尾へ; 180°反転)
    asm.add(A.tail_sleeve(), name="tail_sleeve",
            loc=cq.Location(cq.Vector(-14.6 - explode, 0, 0),
                            cq.Vector(0, 0, 1), 180),
            color=cq.Color(0.55, 0.70, 0.55, 1.0))
    return asm


def export_all() -> None:
    asm = build()
    asm.save(str(HERE / "unit_assy.step"))
    print("unit_assy.step")

    # 結合STL + per-part transformed meshes (renders/usdz用)
    import numpy as np
    import trimesh
    parts = []
    for name, solid, loc, rgba in (
        ("cage", A.cage(), np.eye(4), (0.85, 0.63, 0.23, 1)),
        ("servo", servo_mock(), np.eye(4), (0.20, 0.23, 0.28, 1)),
        ("horn", A.horn_adapter(), trimesh.transformations.translation_matrix([16, 0, 0]), (0.42, 0.60, 0.75, 1)),
        ("sleeve", A.tail_sleeve(),
         trimesh.transformations.translation_matrix([-14.6, 0, 0])
         @ trimesh.transformations.rotation_matrix(math.pi, [0, 0, 1]), (0.55, 0.70, 0.55, 1)),
    ):
        tmp = HERE / f"_{name}.stl"
        cq.exporters.export(solid, str(tmp), tolerance=0.01)
        m = trimesh.load(tmp)
        m.apply_transform(loc)
        parts.append((name, m, rgba))
        tmp.unlink()

    combined = trimesh.util.concatenate([m for _, m, _ in parts])
    combined.export(HERE / "unit_assy.stl")
    print(f"unit_assy.stl ({len(combined.faces)} tris)")

    render(parts)
    usdz(parts)


def render(parts) -> None:
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import to_rgb
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    import trimesh

    def add(ax, m, color, alpha=1.0, shift=(0, 0, 0)):
        tris = (m.vertices + np.array(shift))[m.faces]
        lum = 0.55 + 0.45 * np.clip(m.face_normals @ np.array([0.5, 0.6, 0.62]), -1, 1)
        base = np.array(to_rgb(color))
        pc = Poly3DCollection(tris, alpha=alpha, edgecolor="none")
        pc.set_facecolor(np.clip(base[None, :] * lum[:, None], 0, 1))
        ax.add_collection3d(pc)

    colors = {"cage": "#d9a13a", "servo": "#343b47", "horn": "#6b99bf",
              "sleeve": "#8cb38c", "xl": "#c05050"}
    def scene(fname, entries, lim=30):
        fig = plt.figure(figsize=(7, 6), dpi=110)
        ax = fig.add_subplot(111, projection="3d")
        for name, m, alpha, shift in entries:
            add(ax, m, colors.get(name, "#888888"), alpha, shift)
        ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_zlim(-lim, lim)
        ax.set_box_aspect((1, 1, 1)); ax.view_init(elev=20, azim=-58)
        ax.set_axis_off(); plt.tight_layout()
        plt.savefig(HERE / fname); plt.close()
        print(fname)

    pm = {n: m for n, m, _ in parts}
    scene("render_unit_assembled.png",
          [("servo", pm["servo"], 1.0, (0, 0, 0)),
           ("horn", pm["horn"], 1.0, (0, 0, 0)),
           ("sleeve", pm["sleeve"], 1.0, (0, 0, 0)),
           ("cage", pm["cage"], 0.40, (0, 0, 0))])
    scene("render_unit_exploded.png",
          [("servo", pm["servo"], 1.0, (0, 0, 0)),
           ("horn", pm["horn"], 1.0, (14, 0, 0)),
           ("sleeve", pm["sleeve"], 1.0, (-12, 0, 0)),
           ("cage", pm["cage"], 0.40, (0, 24, 0))], lim=38)
    # XL330外形との重ね合わせ (app repoのメッシュが見つかれば)
    import os
    import trimesh as T
    xl_path = os.environ.get("XL330_STL", "")
    if xl_path and Path(xl_path).exists():
        m = T.load(xl_path)
        m.apply_scale(1000)
        scene("render_unit_vs_xl330.png",
              [("servo", pm["servo"], 1.0, (0, 0, 0)),
               ("cage", pm["cage"], 0.35, (0, 0, 0)),
               ("xl", m, 0.22, (0, 0, 0))])
    else:
        print("set XL330_STL=<path to xl330.stl> for the overlay render")


def usdz(parts) -> None:
    try:
        from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade, UsdUtils, Vt
    except ImportError:
        print("usd-core not installed; skipping usdz")
        return
    import numpy as np
    usdc = HERE / "unit_assy.usdc"
    stage = Usd.Stage.CreateNew(str(usdc))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.y)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    root = UsdGeom.Xform.Define(stage, "/Unit")
    stage.SetDefaultPrim(root.GetPrim())
    # mm → m、ホーン軸(X)を上(Y+)に立てて置く (行ベクトル規約: スケール→回転)。
    m4 = Gf.Matrix4d().SetScale(0.001) * \
         Gf.Matrix4d().SetRotate(Gf.Rotation(Gf.Vec3d(0, 0, 1), 90))
    root.AddTransformOp().Set(m4)
    for i, (name, m, rgba) in enumerate(parts):
        prim = UsdGeom.Mesh.Define(stage, f"/Unit/{name}")
        pts = m.vertices[m.faces].reshape(-1, 3).astype(np.float32)
        prim.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(pts))
        prim.CreateFaceVertexCountsAttr([3] * len(m.faces))
        prim.CreateFaceVertexIndicesAttr(list(range(len(pts))))
        n = np.repeat(m.face_normals.astype(np.float32), 3, axis=0)
        prim.CreateNormalsAttr(Vt.Vec3fArray.FromNumpy(n))
        prim.SetNormalsInterpolation(UsdGeom.Tokens.faceVarying)
        prim.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
        mat = UsdShade.Material.Define(stage, f"/Unit/Materials/m{i}")
        sh = UsdShade.Shader.Define(stage, f"/Unit/Materials/m{i}/pbr")
        sh.CreateIdAttr("UsdPreviewSurface")
        sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(
            Gf.Vec3f(float(rgba[0]), float(rgba[1]), float(rgba[2])))
        sh.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.5)
        mat.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(), "surface")
        UsdShade.MaterialBindingAPI.Apply(prim.GetPrim()).Bind(mat)
    stage.Save()
    ok = UsdUtils.CreateNewUsdzPackage(str(usdc), str(HERE / "unit_assy.usdz"))
    usdc.unlink()
    print("unit_assy.usdz" if ok else "usdz packaging failed")


if __name__ == "__main__":
    export_all()
