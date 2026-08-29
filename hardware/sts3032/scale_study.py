#!/usr/bin/env python3
"""STS3032化にともなう機体スモール化のスケール検討.

robot_walk.xml / robot_allcollisions.xml から一様スケール s の縮小MJCFを
機械的に生成し (長さ×s, 質量×s^3, 慣性×s^5)、STS3032相当のアクチュエータ
(kp 0.35 N·m/rad, forcerange ±0.44 N·m) で既存ONNXポリシーを走らせて、
どのスケールまで「そのまま」歩けるかを測る。あわせて s=1.0 と縮小版の
サイズ比較レンダを出力する。

一様スケールは検討の下限を与える道具にすぎない (サーボ自体は縮まないので
実機の縮小はリンクの再レイアウトが主役)。だが「縮小したら今の歩行ポリシー
は使えるのか、再学習が要るのか」という一番効く問いには、これで直接答えが
出る。

実行: python scale_study.py <robot_dir>   (要 mujoco, onnxruntime, numpy)
       robot_dir = microduck_rl の robot/microduck ディレクトリ
"""

import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np
import onnxruntime as ort

REPO = Path(__file__).resolve().parent.parent.parent
HOME = np.array([0.0, -0.0873, -0.4579, -0.0049, 0.4530,
                 0.3491, 0.3491, 0.0, 0.0,
                 0.0, 0.0873, 0.4579, 0.0049, -0.4530])

STS_KP = 0.35        # N·m/rad — STS3032の等価剛性 (BAM同定までの近似)
STS_FRANGE = 0.44    # N·m — ストールトルク@6V


def scaled_xml(robot_dir: Path, s: float, out: Path) -> None:
    """robot_allcollisions.xml の一様スケール版 + 床 + STSアクチュエータ."""
    tree = ET.parse(robot_dir / "robot_allcollisions.xml")
    root = tree.getroot()

    def scale_attr(el, name, factor):
        v = el.get(name)
        if v is None:
            return
        el.set(name, " ".join(str(float(x) * factor) for x in v.split()))

    for el in root.iter():
        if el.tag in ("body", "geom", "site", "joint", "camera"):
            scale_attr(el, "pos", s)
        if el.tag == "geom" and el.get("type") in ("box", "sphere", "cylinder", "capsule"):
            scale_attr(el, "size", s)
        if el.tag == "inertial":
            scale_attr(el, "pos", s)
            scale_attr(el, "mass", s ** 3)
            scale_attr(el, "fullinertia", s ** 5)
            scale_attr(el, "diaginertia", s ** 5)
        if el.tag == "mesh":
            el.set("scale", f"{s} {s} {s}")
    # アクチュエータをSTS3032相当へ
    for pos in root.iter("position"):
        pos.set("kp", str(STS_KP))
        pos.attrib.pop("dampratio", None)
        pos.set("kv", "0.0")
        pos.set("forcerange", f"-{STS_FRANGE} {STS_FRANGE}")
    root.find("compiler").set("meshdir", str(robot_dir / "assets"))
    ET.SubElement(root, "option").set("timestep", "0.005")
    wb = root.find("worldbody")
    wb.insert(0, ET.Element("geom", dict(name="floor", type="plane", size="0 0 0.05")))
    tree.write(out)


class Runner:
    def __init__(self, xml: Path, s: float):
        self.m = mujoco.MjModel.from_xml_path(str(xml))
        self.d = mujoco.MjData(self.m)
        self.s = s
        self.fq = self.m.jnt_qposadr[
            mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_JOINT, "trunk_base_freejoint")]
        names = ["left_hip_yaw", "left_hip_roll", "left_hip_pitch", "left_knee",
                 "left_ankle", "neck_pitch", "head_pitch", "head_yaw", "head_roll",
                 "right_hip_yaw", "right_hip_roll", "right_hip_pitch",
                 "right_knee", "right_ankle"]
        self.jadr = np.array([self.m.jnt_qposadr[
            mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_JOINT, n)] for n in names])
        self.reset()

    def reset(self, tilt=0.0):
        mujoco.mj_resetData(self.m, self.d)
        self.d.qpos[self.fq + 2] = 0.125 * self.s
        self.d.qpos[self.fq + 3:self.fq + 7] = [np.cos(tilt / 2), 0, np.sin(tilt / 2), 0]
        self.d.qpos[self.jadr] = HOME
        self.d.ctrl[:] = HOME
        mujoco.mj_forward(self.m, self.d)
        self.la = np.zeros(14, np.float32)

    def tick(self, sess, cmd):
        d = self.d
        q = d.qpos[self.fq + 3:self.fq + 7]
        w, x, y, z = q
        R = np.array([
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])
        obs = np.concatenate([
            d.qvel[3:6], R.T @ [0, 0, -1.0],
            d.qpos[self.jadr] - HOME, d.qvel[6:20], self.la,
            cmd, [0, 0, 0, 0], [0, 0, 0, 0, 0, 0]]).astype(np.float32)[None, :]
        a = sess.run(None, {"obs": obs})[0][0]
        self.la = a.copy()
        d.ctrl[:] = HOME + a.astype(np.float64)
        for _ in range(4):
            mujoco.mj_step(self.m, self.d)

    def up(self):
        q = self.d.qpos[self.fq + 3:self.fq + 7]
        return 1 - 2 * (q[1] ** 2 + q[2] ** 2)


def study(robot_dir: Path) -> None:
    walk = ort.InferenceSession(str(REPO / "policies" / "alpha_walking.onnx"),
                                providers=["CPUExecutionProvider"])
    stand = ort.InferenceSession(str(REPO / "policies" / "alpha_stand.onnx"),
                                 providers=["CPUExecutionProvider"])
    out_dir = Path(__file__).parent
    print(f"{'scale':>6} {'stand 10s':>10} {'walk 20s':>16} {'recover':>10}")
    for s in (1.0, 0.9, 0.85, 0.8, 0.7):
        xml = out_dir / f"_scaled_{s}.xml"
        scaled_xml(robot_dir, s, xml)
        # stand
        r = Runner(xml, s)
        ok_stand = True
        for i in range(10 * 50):
            r.tick(stand, [0, 0, 0])
            if r.up() < 0.3:
                ok_stand = False
                break
        # walk (前進指令は脚長に合わせて×s)
        r = Runner(xml, s)
        res_walk = "OK"
        for i in range(20 * 50):
            t = i / 50
            ramp = min(1, max(0, t - 1))
            r.tick(walk, [0.3 * s * ramp, 0, 0])
            if r.up() < 0.3:
                res_walk = f"FELL {t:4.1f}s"
                break
        dist = float(np.hypot(r.d.qpos[r.fq], r.d.qpos[r.fq + 1]))
        # recovery from prone
        r = Runner(xml, s)
        r.reset(tilt=2.2)
        for i in range(12 * 50):
            r.tick(stand, [0, 0, 0])
        rec = "RECOVERED" if (r.up() > 0.9 and r.d.qpos[r.fq + 2] > 0.09 * s) else "down"
        print(f"{s:>6} {('OK' if ok_stand else 'FELL'):>10} "
              f"{res_walk + f' {dist:.2f}m':>16} {rec:>10}")
        xml.unlink()


if __name__ == "__main__":
    robot_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    if not robot_dir or not (robot_dir / "robot_allcollisions.xml").exists():
        raise SystemExit(__doc__)
    study(robot_dir)
