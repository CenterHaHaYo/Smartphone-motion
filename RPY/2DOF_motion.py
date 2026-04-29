# ============================================================
# Pan-Tilt 2-DOF Visualization
# Sensor : Acceleration with g (accX, accY, accZ)
# Pan    = Yaw   → BUT accelerometer cannot give yaw
#                  so Pan follows ROLL instead
#                  (tilting phone left/right = pan left/right)
# Tilt   = Pitch → tilt phone forward/back = tilt up/down
# One box only, with direction arrows
# ============================================================
    
from vpython import *
import requests
import numpy as np

URL = "http://172.20.10.1"
SAMPLE_RATE = 30
DT = 1.0 / SAMPLE_RATE
PAN_MIN, PAN_MAX = -90.0, 90.0
TILT_MIN, TILT_MAX = -45.0, 45.0

scene = canvas(
    title="Pan-Tilt 2DOF Simulator  |  Quaternion Raw",
    width=1000, height=560,
    background=vector(0.06, 0.06, 0.12),
)
scene.camera.pos = vector(8, 7, 14)
scene.camera.axis = vector(-8, -7, -14)

# ── Base ──
base = box(pos=vector(0, -1.5, 0), size=vector(2.0, 0.3, 2.0),
           color=vector(0.25, 0.25, 0.30))
pole = cylinder(pos=vector(0, -1.35, 0), axis=vector(0, 0.85, 0),
                radius=0.12, color=vector(0.35, 0.35, 0.40))

# ── Pan pivot (ศูนย์กลางการหมุน Pan) ──
pan_pivot = vector(0, -0.5, 0)

# ── Pan body + arms (หมุนรอบ Y ที่ pan_pivot) ──
pan_body = box(pos=pan_pivot, size=vector(1.4, 0.25, 1.0),
               color=vector(0.20, 0.45, 0.70))
arm_L = box(pos=pan_pivot + vector(-0.55, 0.40, 0),
            size=vector(0.15, 0.80, 0.20),
            color=vector(0.20, 0.45, 0.70))
arm_R = box(pos=pan_pivot + vector(0.55, 0.40, 0),
            size=vector(0.15, 0.80, 0.20),
            color=vector(0.20, 0.45, 0.70))

# ── Tilt pivot (อยู่เหนือ pan_pivot 0.80 หน่วย) ──
# ตำแหน่งเริ่มต้น: ตรงเหนือ pan_pivot
tilt_pivot = pan_pivot + vector(0, 0.80, 0)

# ── Tilt platform + camera (หมุนรอบ X ที่ tilt_pivot) ──
tilt_platform = box(pos=tilt_pivot, size=vector(0.90, 0.15, 0.70),
                    color=vector(0.70, 0.40, 0.15))
cam_body = box(pos=tilt_pivot + vector(0, 0.18, 0),
               size=vector(0.60, 0.36, 0.44),
               color=vector(0.12, 0.12, 0.18))
cam_lens = cylinder(pos=tilt_pivot + vector(0, 0.18, 0.22),
                    axis=vector(0, 0, 0.25), radius=0.14,
                    color=vector(0.05, 0.05, 0.22))
cam_lens_ring = cylinder(pos=tilt_pivot + vector(0, 0.18, 0.22),
                         axis=vector(0, 0, 0.26), radius=0.155,
                         color=vector(0.40, 0.40, 0.45))

pan_parts = [pan_body, arm_L, arm_R,
             tilt_platform, cam_body, cam_lens, cam_lens_ring]
tilt_parts = [tilt_platform, cam_body, cam_lens, cam_lens_ring]

# ── แกน XYZ ──
_TH = 0.04
arrow(pos=vector(0, 0, 0), axis=vector(3, 0, 0),
      color=color.orange, shaftwidth=_TH)
label(pos=vector(3.3, 0, 0), text="X", box=False,
      height=13, color=color.orange)
arrow(pos=vector(0, 0, 0), axis=vector(0, 3, 0),
      color=color.cyan, shaftwidth=_TH)
label(pos=vector(0, 3.3, 0), text="Y", box=False,
      height=13, color=color.cyan)
arrow(pos=vector(0, 0, 0), axis=vector(0, 0, 3),
      color=color.green, shaftwidth=_TH)
label(pos=vector(0, 0, 3.3), text="Z", box=False,
      height=13, color=color.green)

aim_ray = arrow(pos=tilt_pivot + vector(0, 0.18, 0),
                axis=vector(0, 0, 2.5),
                color=color.red, shaftwidth=0.06)

# ── HUD ──
lbl_title = label(pos=vector(0, 5.5, 0),
                  text="v1  Pure Quaternion (FIXED)",
                  box=False, height=14, color=vector(0.4, 0.8, 1.0))
lbl_pan = label(pos=vector(-5, 4.5, 0), text="",
                box=False, height=13, color=color.yellow)
lbl_tilt = label(pos=vector(-5, 3.9, 0), text="",
                 box=False, height=13, color=color.orange)
lbl_s1 = label(pos=vector(-5, 3.2, 0), text="",
               box=False, height=12, color=color.white)
lbl_s2 = label(pos=vector(-5, 2.6, 0), text="",
               box=False, height=12, color=color.white)
lbl_raw = label(pos=vector(-5, 1.8, 0), text="",
                box=False, height=11, color=color.gray(0.55))
lbl_cal = label(pos=vector(0, -3.5, 0),
                text="[C] or button → recalibrate",
                box=False, height=10, color=color.gray(0.45))
lbl_info = label(pos=vector(0, -4.2, 0), text="Waiting...",
                 box=False, height=11, color=color.white)

# ── กราฟ ──
g = graph(title="Pan & Tilt Angles", xtitle="Time (s)", ytitle="Degrees",
          width=620, height=220,
          background=color.black, foreground=color.white)
gc_pan = gcurve(graph=g, color=color.yellow, label="Pan", width=2)
gc_tilt = gcurve(graph=g, color=color.orange, label="Tilt", width=2)

MAX_PTS = SAMPLE_RATE * 15
data_pan, data_tilt = [], []

q0 = None

def reset_cal(src=None):
    global q0
    q0 = None
    lbl_cal.text = "✔ Recalibrated"
    lbl_cal.color = color.green

scene.append_to_caption("\n\n")
button(text="  ↺  Reset Calibration  ", bind=reset_cal,
       background=vector(0.15, 0.35, 0.65), color=color.white)
scene.bind("keydown", lambda e: reset_cal() if e.key.lower() == "c" else None)


def quat_conj(q):
    w, x, y, z = q
    return (w, -x, -y, -z)


def quat_mul(p, q):
    pw, px, py, pz = p
    qw, qx, qy, qz = q
    return (
        pw * qw - px * qx - py * qy - pz * qz,
        pw * qx + px * qw + py * qz - pz * qy,
        pw * qy - px * qz + py * qw + pz * qx,
        pw * qz + px * qy - py * qx + pz * qw,
    )


def quat_to_euler(q):
    w, x, y, z = q
    sinp = max(-1.0, min(1.0, 2 * (w * y - z * x)))
    pitch = np.degrees(np.arcsin(sinp))
    roll = np.degrees(np.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y)))
    yaw = np.degrees(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))
    return pitch, roll, yaw


def read_sensor():
    try:
        r = requests.get(f"{URL}/get?w&x&y&z&pitch&roll&yaw", timeout=1)
        buf = r.json()["buffer"]
        return (buf["w"]["buffer"][-1], buf["x"]["buffer"][-1],
                buf["y"]["buffer"][-1], buf["z"]["buffer"][-1],
                buf["pitch"]["buffer"][-1],
                buf["roll"]["buffer"][-1],
                buf["yaw"]["buffer"][-1])
    except Exception:
        return None


def rotate_about_Y(parts, pivot, angle_deg):
    """หมุนรอบแกน Y (Pan)"""
    a = np.radians(angle_deg)
    c, s = np.cos(a), np.sin(a)

    def rot_Y(v):
        return vector(c * v.x + s * v.z, v.y, -s * v.x + c * v.z)

    for obj in parts:
        offset = obj.pos - pivot
        obj.pos = pivot + rot_Y(offset)
        obj.axis = rot_Y(obj.axis)
        obj.up = rot_Y(obj.up)


def rotate_about_X(parts, pivot, angle_deg):
    """หมุนรอบแกน X (Tilt) — แก้จาก Z เป็น X"""
    a = np.radians(angle_deg)
    c, s = np.cos(a), np.sin(a)

    def rot_X(v):
        return vector(v.x, c * v.y - s * v.z, s * v.y + c * v.z)

    for obj in parts:
        offset = obj.pos - pivot
        obj.pos = pivot + rot_X(offset)
        obj.axis = rot_X(obj.axis)
        obj.up = rot_X(obj.up)


prev_pan = 0.0
prev_tilt = 0.0
t_now = 0.0
frame_n = 0

while True:
    rate(SAMPLE_RATE)

    raw = read_sensor()
    if raw is None:
        lbl_info.text = "⚠  No signal"
        lbl_info.color = color.red
        continue

    lbl_info.color = color.white
    qw_s, qx_s, qy_s, qz_s, pitch_s, roll_s, yaw_s = raw

    q_cur = (qw_s, -qx_s, qz_s, -qy_s)

    if q0 is None:
        q0 = q_cur
        prev_pan = prev_tilt = 0.0
        lbl_cal.text = "✔ Calibrated"
        lbl_cal.color = color.green

    q_rel = quat_mul(quat_conj(q0), q_cur)
    pitch_rel, roll_rel, yaw_rel = quat_to_euler(q_rel)

    pan_raw = yaw_rel
    tilt_raw = pitch_rel

    pan = max(PAN_MIN, min(PAN_MAX, pan_raw))
    tilt = max(TILT_MIN, min(TILT_MAX, tilt_raw))

    s1 = pan + 90.0
    s2 = tilt + 45.0

    # ── อัพเดท Pan ──
    delta_pan = pan - prev_pan
    if abs(delta_pan) > 0.1:
        rotate_about_Y(pan_parts, pan_pivot, delta_pan)
        # อัพเดท tilt_pivot ใหม่ (หมุนตาม Pan)
        # offset จาก pan_pivot = (0, 0.80, 0) ในกรอบ local
        # หลัง Pan หมุน a องศา:
        a = np.radians(pan)
        c, s = np.cos(a), np.sin(a)
        # offset (0, 0.80, 0) หมุนรอบ Y → (s*0.80, 0.80, c*0.80)? ไม่ใช่
        # แกน Y ไม่เปลี่ยน, แกน X,Z หมุน → offset_y = 0.80 คงที่
        # offset_x = 0, offset_z = 0 ในตอนเริ่มต้น
        # หลังหมุน: offset_new = rotate_Y(0, 0.80, 0) = (0, 0.80, 0) เหมือนเดิม!
        # แต่ถ้าเริ่มต้นมี offset จากแกนอื่นต้องคำนวณ
        # ในกรณีนี้ tilt_pivot เริ่มที่ pan_pivot + (0, 0.80, 0)
        # หลังหมุน pan_pivot ไม่เคลื่อนที่ (มันคือจุดหมุน)
        # offset (0, 0.80, 0) หมุนรอบ Y → ยังเป็น (0, 0.80, 0)
        # ดังนั้น tilt_pivot = pan_pivot + (0, 0.80, 0) เสมอ
        tilt_pivot = pan_pivot + vector(0, 0.80, 0)

    
    delta_tilt = tilt - prev_tilt
    if abs(delta_tilt) > 0.1:
        rotate_about_X(tilt_parts, tilt_pivot, delta_tilt)

    prev_pan = pan
    prev_tilt = tilt

    aim_ray.pos = cam_body.pos
    aim_ray.axis = cam_body.axis * 2.5 / mag(cam_body.axis)

    lbl_pan.text = f"Pan:   {pan:+6.1f}°  (raw Yaw: {pan_raw:+.1f}°)"
    lbl_tilt.text = f"Tilt:  {tilt:+6.1f}°  (raw Pitch: {tilt_raw:+.1f}°)"
    lbl_s1.text = f"Servo1 Pan  = {s1:.0f}°"
    lbl_s2.text = f"Servo2 Tilt = {s2:.0f}°"
    lbl_raw.text = f"Phyphox raw  P:{pitch_s:+.1f}  R:{roll_s:+.1f}  Y:{yaw_s:+.1f}"
    lbl_info.text = f"q_rel = ({q_rel[0]:.3f}, {q_rel[1]:.3f}, {q_rel[2]:.3f}, {q_rel[3]:.3f})"

    data_pan.append([t_now, pan])
    data_tilt.append([t_now, tilt])
    if len(data_pan) > MAX_PTS:
        data_pan = data_pan[-MAX_PTS:]
        data_tilt = data_tilt[-MAX_PTS:]

    if frame_n % 10 == 0:
        gc_pan.data = data_pan
        gc_tilt.data = data_tilt

    frame_n += 1
    t_now += DT