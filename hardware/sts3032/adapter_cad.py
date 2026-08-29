#!/usr/bin/env python3
"""XL330 → STS3032 変換マウントのパラメトリックCAD (CadQuery).

microduckの構造部品はXL330の外形・取付界面を前提に設計されている。この
アダプタ群は Feetech STS3032 を「XL330の幽霊外形」に収め、既存部品を一切
変更せずにサーボだけを置き換えるためのもの。**両軸版 STS3032-C036 が前提**
（尾側ベアリング座はサーボ本体が軸線上を占有するためケージ側には作れず、
後軸スリーブで実現する）。3部品構成:

  cage          — XL330外形を再現する角筒ケージ。サーボは前面から挿入し、
                  既存のM2貫通ボルト4本の締結でそのまま挟み込まれる。
  horn_adapter  — STS3032ホーンに固定し、XL330ホーンの取付パターン
                  (4×M2下穴⌀1.6 @⌀12BC) を提供する⌀16円盤。
  tail_sleeve   — C036の後軸に接着/圧入する⌀16スリーブ。既存の16×22×4
                  ベアリングの内輪座になる（XL330の背面ボス⌀16の代替）。

■ 座標系: xl330.stl と同一 — 出力軸=X(+Xホーン側)、原点=軸上、
  幅y(±10)、長手z(-24.5〜+9.5)。
■ XL330側の寸法は xl330.stl の断面解析による実測値 (VERIFIED)。
■ STS3032側はカタログ公称(32×12×27.5, 20g)+推定。TODO_ 定数は
  印刷前に必ず実物をノギス実測して更新すること。

実行: pip install cadquery && python adapter_cad.py
"""

import math
from pathlib import Path

import cadquery as cq

# ── XL330 interface (VERIFIED from xl330.stl) ────────────────────────────────
XL_W = 20.0            # 幅 (y)
XL_D = 26.0            # 奥行 (x): ケース面 ±13.0
XL_Z_HEAD = 9.5        # ホーン側端 (z+)
XL_Z_TAIL = -24.5      # 尾側端 (z-)
XL_BOLT_Y = 8.0        # M2貫通ボルト y=±8
XL_BOLT_Z = (7.5, -22.5)   # 同 z (各端から2mm)
XL_BOLT_DIA = 2.2      # 実測⌀2.0 + クリアランス
XL_HORN_DIA = 16.0     # ホーン/背面ボス外径 (=ベアリング16x22x4の内輪径)
XL_HORN_BC = 12.0      # ホーン穴ボルトサークル
XL_HORN_PILOT = 1.6    # ホーン穴 (M2セルフタップ下穴)
XL_BOSS_PROUD = 1.6    # ボスのケース面からの突出

# ── STS3032 (公称32×12×27.5・20g。TODO_は要実測) ─────────────────────────────
STS_L = 32.0                    # 長手 (z方向に寝かせる)
STS_W = 12.0                    # 幅 (y)
TODO_STS_AXIS_TO_NOSE = 8.0     # 出力軸→前端面 ★要実測
TODO_STS_HORN_HUB = 6.4         # ホーンハブ外径 (受け座ぐり) ★要実測
TODO_STS_HORN_BC = 8.0          # ホーン止めねじBC ★要実測
TODO_STS_HORN_SCREW = 2.2       # 同 通し穴 ★要実測
TODO_STS_TAIL_SHAFT = 3.0       # C036後軸径 ★要実測
TODO_STS_CLAMP_Z = (6.0, -22.0) # 側面クランプ小ねじのz位置 ★タブ実測後調整
CLR = 0.25                      # 印刷クリアランス (片側)

# サーボ配置: 軸=X軸。長手32をz方向に、前端z=+8。幅12をyに。
STS_NOSE_Z = TODO_STS_AXIS_TO_NOSE
STS_TAIL_Z = STS_NOSE_Z - STS_L            # -24.0 → ケース内に収まる
TUN_Y = STS_W / 2 + CLR                    # トンネル半幅
TUN_Z_LO = STS_TAIL_Z - CLR
TUN_Z_HI = STS_NOSE_Z + CLR


def cage() -> cq.Workplane:
    zc = (XL_Z_HEAD + XL_Z_TAIL) / 2
    b = (cq.Workplane("XY")
         .box(XL_D, XL_W, XL_Z_HEAD - XL_Z_TAIL)
         .translate((0, 0, zc))
         .edges("|Z").fillet(2.0))
    # サーボトンネル (x方向貫通): 12.5幅 × (z: -24.25〜+8.25)
    tz = (TUN_Z_LO + TUN_Z_HI) / 2
    b = b.cut(cq.Workplane("XY")
              .box(XL_D + 6, 2 * TUN_Y, TUN_Z_HI - TUN_Z_LO)
              .translate((0, 0, tz)))
    # M2貫通ボルト×4 (x方向)
    for z in XL_BOLT_Z:
        for y in (XL_BOLT_Y, -XL_BOLT_Y):
            b = b.cut(cq.Workplane("YZ")
                      .center(y, z).circle(XL_BOLT_DIA / 2)
                      .extrude(XL_D + 8, both=True))
    # クランプ小ねじ下穴 (⌀1.6, y壁を貫通してサーボ側面/タブへ) ×2
    for z in TODO_STS_CLAMP_Z:
        b = b.cut(cq.Workplane("XZ")
                  .workplane(offset=XL_W / 2 + 1)
                  .center(0, z).circle(0.8)
                  .extrude(XL_W + 2))
    return b


def horn_adapter() -> cq.Workplane:
    t = 2.5   # 円盤厚 — XL330純正ホーンの座面高に合わせて要調整
    d = cq.Workplane("YZ").circle(XL_HORN_DIA / 2).extrude(t)
    # XL330側パターン: 4×⌀1.6 @⌀12BC (M2セルフタップ下穴, 深さ2.0)
    for i in range(4):
        y = XL_HORN_BC / 2 * math.cos(math.radians(i * 90))
        z = XL_HORN_BC / 2 * math.sin(math.radians(i * 90))
        d = d.cut(cq.Workplane("YZ").workplane(offset=t)
                  .center(y, z).circle(XL_HORN_PILOT / 2).extrude(-2.0))
    # STS側: ハブ受け座ぐり + ホーン止めねじ通し(45°振り) ★要実測
    d = d.cut(cq.Workplane("YZ").circle(TODO_STS_HORN_HUB / 2).extrude(1.4))
    for i in range(4):
        y = TODO_STS_HORN_BC / 2 * math.cos(math.radians(i * 90 + 45))
        z = TODO_STS_HORN_BC / 2 * math.sin(math.radians(i * 90 + 45))
        d = d.cut(cq.Workplane("YZ").center(y, z)
                  .circle(TODO_STS_HORN_SCREW / 2).extrude(t))
    return d


def tail_sleeve() -> cq.Workplane:
    # 16x22x4ベアリングの内輪座: OD16 × 幅5 (座4 + フランジ掛かり1)
    s = (cq.Workplane("YZ").circle(XL_HORN_DIA / 2).extrude(5.0)
         .faces(">X").workplane().circle(XL_HORN_DIA / 2 + 1.5).extrude(1.0))
    s = s.cut(cq.Workplane("YZ").workplane(offset=-1)
              .circle((TODO_STS_TAIL_SHAFT + 0.1) / 2).extrude(10))
    return s


if __name__ == "__main__":
    out = Path(__file__).parent
    for name, part in (("cage", cage()), ("horn_adapter", horn_adapter()),
                       ("tail_sleeve", tail_sleeve())):
        cq.exporters.export(part, str(out / f"{name}.step"))
        cq.exporters.export(part, str(out / f"{name}.stl"), tolerance=0.01)
        print(f"{name}: step+stl")
