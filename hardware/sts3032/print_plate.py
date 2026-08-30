#!/usr/bin/env python3
"""STS3032系プリント部品の印刷用プレート生成 + 印刷性QA.

adapter_cad.py の3部品と hip_module.py のブラケット2部品を、それぞれ
サポート最小の向きに回して1枚のプレートに並べる。あわせて各部品の
水密性・体積・想定質量・オーバーハング率 (45°超の下向き面積) を検査して
印刷リスクを数字で出す。

出力:
  print/<part>.stl        — 印刷向きに回転済みの個別STL (Z0=ベッド)
  print_plate_sts3032.stl — 全部品を並べた1プレート
  render_print_plate.png  — プレート配置図
実行: python print_plate.py
"""

import math
from pathlib import Path

import numpy as np
import trimesh

import cadquery as cq
import adapter_cad as A
import hip_module as HM

HERE = Path(__file__).parent
OUT = HERE / "print"
OUT.mkdir(exist_ok=True)

PETG_DENSITY = 1.27  # g/cm3


def to_mesh(wp) -> trimesh.Trimesh:
    tmp = OUT / "_tmp.stl"
    cq.exporters.export(wp, str(tmp), tolerance=0.01)
    m = trimesh.load(tmp)
    tmp.unlink()
    return m


def orient(m: trimesh.Trimesh, rot_axis, rot_deg) -> trimesh.Trimesh:
    m = m.copy()
    if rot_deg:
        m.apply_transform(trimesh.transformations.rotation_matrix(
            math.radians(rot_deg), rot_axis))
    # ベッドに置く
    m.apply_translation([0, 0, -m.bounds[0][2]])
    m.apply_translation([-m.bounds.mean(axis=0)[0], -m.bounds.mean(axis=0)[1], 0])
    return m


def overhang_ratio(m: trimesh.Trimesh) -> float:
    """45°を超える下向き面の面積率 (ベッド接地面は除外)。"""
    n = m.face_normals
    areas = m.area_faces
    down = n[:, 2] < -math.sin(math.radians(45))
    # 接地面 (z≈0の面) はオーバーハングではない
    face_z = m.triangles[:, :, 2].max(axis=1)
    down &= face_z > 0.3
    return float(areas[down].sum() / areas.sum())


# 部品と印刷向き: (名前, ソリッド, 回転軸, 角度, 補足)
PARTS = [
    # ケージ: トンネル軸(モデルX)を鉛直に → ボルト穴もトンネルも垂直、サポート不要
    ("cage", A.cage(), [0, 1, 0], -90,
     "trunk側の面を上。サポート不要"),
    # ホーンアダプタ: 円盤を平置き (軸=モデルX→鉛直のまま)
    ("horn_adapter", A.horn_adapter(), [0, 1, 0], 90,
     "XLパターン面を下 (座ぐりが上向き)。サポート不要"),
    # テールスリーブ: フランジを下に
    ("tail_sleeve", A.tail_sleeve(), [0, 1, 0], -90,
     "フランジ面をベッドに。サポート不要"),
    # ヨーブラケット: 厚み18mmの一定断面L字 → 側面(横)を下にした平置きL
    ("yaw_bracket", HM.yaw_bracket(), [1, 0, 0], 90,
     "L字を寝かせて平置き。サポート不要・積層方向に強い"),
    # ロールブラケット: 円盤面を下に。seat板が浮くのでここだけサポート推奨
    ("roll_bracket", HM.roll_bracket(), [0, 1, 0], -90,
     "ホーン円盤面を下。seat板下面のみサポート推奨"),
]


def main() -> None:
    placed = []
    x_cursor = 0.0
    gap = 8.0
    print(f"{'part':14s} {'size XxYxZ (mm)':>22s} {'vol cm3':>8s} {'PETG g':>7s} "
          f"{'watertight':>10s} {'overhang%':>9s}")
    report = []
    for name, wp, ax, deg, note in PARTS:
        m = orient(to_mesh(wp), ax, deg)
        size = m.bounds[1] - m.bounds[0]
        vol = abs(m.volume) / 1000.0
        oh = overhang_ratio(m)
        wt = m.is_watertight
        print(f"{name:14s} {size[0]:6.1f}x{size[1]:5.1f}x{size[2]:5.1f} "
              f"{vol:8.2f} {vol*PETG_DENSITY:7.1f} {str(wt):>10s} {oh*100:8.1f}%")
        report.append((name, note))
        m.export(OUT / f"{name}.stl")
        mm = m.copy()
        mm.apply_translation([x_cursor - m.bounds[0][0], 0, 0])
        placed.append((name, mm))
        x_cursor += size[0] + gap
    plate = trimesh.util.concatenate([m for _, m in placed])
    plate.export(HERE / "print_plate_sts3032.stl")
    b = plate.bounds
    print(f"\nplate: {b[1][0]-b[0][0]:.0f} x {b[1][1]-b[0][1]:.0f} mm  "
          f"-> print_plate_sts3032.stl / print/*.stl")
    for name, note in report:
        print(f"  {name}: {note}")

    # 配置図
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import to_rgb
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    colors = ["#d9a13a", "#6b99bf", "#8cb38c", "#c58f5a", "#9a86b8"]
    fig = plt.figure(figsize=(10, 5), dpi=115)
    ax3 = fig.add_subplot(111, projection="3d")
    for (name, m), c in zip(placed, colors):
        tris = m.vertices[m.faces]
        n = m.face_normals
        lum = 0.6 + 0.4 * np.clip(n @ np.array([0.4, 0.5, 0.75]), -1, 1)
        base = np.array(to_rgb(c))
        pc = Poly3DCollection(tris, edgecolor="none")
        pc.set_facecolor(np.clip(base[None, :] * lum[:, None], 0, 1))
        ax3.add_collection3d(pc)
        ax3.text(m.bounds.mean(axis=0)[0], m.bounds[0][1] - 6, 0, name,
                 fontsize=8, ha="center")
    ax3.set_xlim(-10, x_cursor)
    ax3.set_ylim(-x_cursor / 2, x_cursor / 2)
    ax3.set_zlim(0, 44)
    ax3.set_box_aspect((x_cursor + 10, x_cursor, 44))
    ax3.view_init(elev=38, azim=-75)
    ax3.set_axis_off()
    ax3.set_title("STS3032 print plate (build orientation)", fontsize=10)
    plt.tight_layout()
    plt.savefig(HERE / "render_print_plate.png")
    print("render_print_plate.png")


if __name__ == "__main__":
    main()
