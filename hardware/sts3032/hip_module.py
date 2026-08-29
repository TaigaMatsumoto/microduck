#!/usr/bin/env python3
"""STS3032再レイアウト版・股関節モジュールの概念CAD.

小型化編 (README) の再レイアウト案を形にした、左股関節1脚ぶんの
ヨー→ロール→ピッチ 3軸クラスタ。レイアウト検討グレード (取付穴などは
TODO_寸法) であり、目的は「幅がどこまで詰まるか」を実体で確認すること。

配置の要点 (シム検証済みレイアウト narrow A/B に一致):
  - ヨー軸   : 鉛直。サーボの12mm薄がy(横)を向く → 左右間隔 26mm (現行35)
  - ロール軸 : 前後(x)。同じく12mm薄が横 → 横方向の追加張り出しゼロ
  - ピッチ軸 : 横(y)。ここはサーボ高27.5mmが横を向くため利得なし
               (XL330の26mmとほぼ同じ) → ロール→ピッチ横距離 16mm
  - 結果     : 脚(ピッチ軸)間隔 ≈ 70mm (現行85, -18%)。リンク長×0.88と
               組み合わせて既存ポリシーで歩行・復帰OK (scale_study参照)

出力: hip_module_assy.step / .stl, render_hip_module*.png
実行: python hip_module.py
"""

import math
from pathlib import Path

import cadquery as cq

HERE = Path(__file__).parent

# ── STS3032 mock (カタログ: 32L × 12W × 27.5H, 軸はH方向・ノーズから8) ───────
L, W, H = 32.0, 12.0, 27.5
AX2NOSE = 8.0        # ★要実測
HUB_D, HUB_L = 5.8, 3.0

# ── レイアウト (左脚, trunk中心 y=0) ─────────────────────────────────────────
YAW_Y = 13.0         # ヨー軸 y (左右間隔26)
ROLL_DROP = 18.0     # ヨーホーン面→ロール軸 落差
ROLL_X = 3.0         # ロール軸 x
PITCH_LAT = 22.0     # ロール軸→ピッチ軸 横距離 → ピッチ軸 y=35 (脚間隔70)
PITCH_DROP = 10.0    # ロール軸→ピッチ軸 落差
T = 3.0              # ブラケット板厚


def servo(axis: str) -> cq.Workplane:
    """軸方向 axis('z-','x+','y+') に出力を向けたSTS3032モック。原点=出力軸上、
    ホーン座面=原点。本体はホーンの反対側へ H。"""
    b = (cq.Workplane("XY").box(L, W, H).translate((0, 0, -H / 2)))
    b = b.union(cq.Workplane("XY").circle(HUB_D / 2).extrude(HUB_L))
    b = b.translate((-(L / 2 - AX2NOSE), 0, 0))  # 軸をノーズから8mmへ
    if axis == "z-":
        b = b.rotate((0, 0, 0), (1, 0, 0), 180)
    elif axis == "x+":
        b = b.rotate((0, 0, 0), (0, 1, 0), 90)
    elif axis == "y+":
        b = b.rotate((0, 0, 0), (1, 0, 0), -90)
    elif axis == "y-":
        b = b.rotate((0, 0, 0), (1, 0, 0), 90)
    return b


def yaw_bracket() -> cq.Workplane:
    """ヨーホーン(下面)から吊る L 字。ロールサーボを前向き(x+)に保持。"""
    top = (cq.Workplane("XY").box(26, W + 2 * T, T)
           .translate((0, 0, -T / 2)))
    side = (cq.Workplane("XY").box(T, W + 2 * T, ROLL_DROP + H / 2)
            .translate((-13 + T / 2, 0, -(ROLL_DROP + H / 2) / 2)))
    seat = (cq.Workplane("XY").box(H + T, W + 2 * T, T)
            .translate((-13 + (H + T) / 2, 0, -(ROLL_DROP + H / 2) - T / 2)))
    b = top.union(side).union(seat)
    for i in range(4):  # ヨーホーンねじ (★BC要実測)
        y = 4.0 * math.cos(math.radians(i * 90 + 45))
        x = 4.0 * math.sin(math.radians(i * 90 + 45))
        b = b.cut(cq.Workplane("XY").center(x, y).circle(1.1).extrude(-T - 1))
    return b


def roll_bracket() -> cq.Workplane:
    """ロールホーン(x+面)から外へ回り込み、ピッチサーボを横向きに保持するC字。"""
    face = (cq.Workplane("YZ").workplane(offset=ROLL_X + HUB_L)
            .circle(9).extrude(T))
    arm = (cq.Workplane("XY")
           .box(T, PITCH_LAT + T, 22)
           .translate((ROLL_X + HUB_L + T / 2, (PITCH_LAT + T) / 2, -PITCH_DROP / 2 - 3)))
    seat = (cq.Workplane("YZ").workplane(offset=ROLL_X + HUB_L)
            .center(PITCH_LAT, -PITCH_DROP)
            .rect(14, 14).extrude(T))
    b = face.union(arm).union(seat)
    for i in range(4):  # ロールホーンねじ (★BC要実測)
        z = 4.0 * math.cos(math.radians(i * 90 + 45))
        y = 4.0 * math.sin(math.radians(i * 90 + 45))
        b = b.cut(cq.Workplane("YZ").workplane(offset=ROLL_X + HUB_L - 1)
                  .center(y, z).circle(1.1).extrude(T + 2))
    return b


def build() -> cq.Assembly:
    asm = cq.Assembly(name="sts3032_hip_left")
    grey = cq.Color(0.22, 0.25, 0.30, 1)
    # ヨー: 軸鉛直・ホーン下向き、軸@ (0, YAW_Y, 0)
    asm.add(servo("z-"), name="yaw_servo",
            loc=cq.Location(cq.Vector(0, YAW_Y, 0)), color=grey)
    asm.add(yaw_bracket(), name="yaw_bracket",
            loc=cq.Location(cq.Vector(0, YAW_Y, -HUB_L)),
            color=cq.Color(0.85, 0.63, 0.23, 1))
    # ロール: 軸x+・ホーン前向き、軸@ (ROLL_X, YAW_Y, -ROLL_DROP-HUB_L)
    asm.add(servo("x+"), name="roll_servo",
            loc=cq.Location(cq.Vector(ROLL_X, YAW_Y, -ROLL_DROP - HUB_L)), color=grey)
    asm.add(roll_bracket(), name="roll_bracket",
            loc=cq.Location(cq.Vector(0, YAW_Y, -ROLL_DROP - HUB_L)),
            color=cq.Color(0.42, 0.60, 0.75, 1))
    # ピッチ: 軸は横・ホーン内向き(y-)、本体は外側(太もも)へ
    asm.add(servo("y-"), name="pitch_servo",
            loc=cq.Location(cq.Vector(ROLL_X + HUB_L + T, YAW_Y + PITCH_LAT,
                                      -ROLL_DROP - HUB_L - PITCH_DROP)), color=grey)
    return asm


def export() -> None:
    asm = build()
    asm.save(str(HERE / "hip_module_assy.step"))
    import numpy as np
    import trimesh
    meshes = []
    colors = {"yaw_servo": "#343b47", "roll_servo": "#343b47", "pitch_servo": "#343b47",
              "yaw_bracket": "#d9a13a", "roll_bracket": "#6b99bf"}
    parts = []
    for child in asm.children:
        tmp = HERE / f"_{child.name}.stl"
        w = cq.Workplane(obj=child.obj.val() if hasattr(child.obj, "val") else child.obj)
        cq.exporters.export(w, str(tmp), tolerance=0.02)
        m = trimesh.load(tmp)
        t = child.loc.toTuple()
        M = np.eye(4)
        M[:3, 3] = t[0]
        rx, ry, rz = [math.radians(a) for a in t[1]]
        M[:3, :3] = (trimesh.transformations.euler_matrix(rx, ry, rz)[:3, :3])
        m.apply_transform(M)
        parts.append((child.name, m))
        meshes.append(m)
        tmp.unlink()
    trimesh.util.concatenate(meshes).export(HERE / "hip_module_assy.stl")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import to_rgb
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    for fname, elev, azim in (("render_hip_module.png", 16, -55),
                              ("render_hip_module_front.png", 2, -90)):
        fig = plt.figure(figsize=(7, 6), dpi=115)
        ax = fig.add_subplot(111, projection="3d")
        for name, m in parts:
            tris = m.vertices[m.faces]
            n = m.face_normals
            lum = 0.6 + 0.4 * np.clip(n @ np.array([0.5, 0.55, 0.65]), -1, 1)
            base = np.array(to_rgb(colors[name]))
            pc = Poly3DCollection(tris, edgecolor="none")
            pc.set_facecolor(np.clip(base[None, :] * lum[:, None], 0, 1))
            ax.add_collection3d(pc)
            # 右脚ぶんをミラー (front view の幅感確認用)
            mm = m.copy()
            mm.vertices[:, 1] *= -1
            tris2 = mm.vertices[:, :][mm.faces][:, ::-1]
            pc2 = Poly3DCollection(tris2, edgecolor="none", alpha=0.45)
            pc2.set_facecolor(np.clip(base[None, :] * 0.8, 0, 1))
            ax.add_collection3d(pc2)
        ax.set_xlim(-40, 40)
        ax.set_ylim(-45, 45)
        ax.set_zlim(-70, 20)
        ax.set_box_aspect((0.8, 0.9, 0.9))
        ax.view_init(elev=elev, azim=azim)
        ax.set_axis_off()
        ax.text2D(0.08, 0.92, "STS3032 hip cluster (concept)  yaw sep 26mm / pitch sep ~70mm",
                  transform=ax.transAxes, fontsize=9)
        plt.tight_layout()
        plt.savefig(HERE / fname)
        plt.close()
        print(fname)
    print("hip_module_assy.step / .stl")


if __name__ == "__main__":
    export()
