from vpython import *
import requests
import numpy as np
import os, csv, datetime


# ═══════════════════════════════════════════════════════════════════
#  CONFIGURATION
# ═══════════════════════════════════════════════════════════════════
URL         = "http://172.20.10.1:8080"  # iPhone hotspot
# URL = "http://192.168.1.43:8080"
SAMPLE_RATE = 30
DT          = 1.0 / SAMPLE_RATE
BUFFER_SIZE = 128       # FFT window ≈ 4.3 s → Δf ≈ 0.23 Hz

# ═══════════════════════════════════════════════════════════════════
#  3-D SCENE
# ═══════════════════════════════════════════════════════════════════
scene = canvas(
    title      = "📱 Smartphone Motion Visualizer  |  Kalman Filter",
    width      = 960,
    height     = 500,
    background = color.white,
)
scene.camera.pos  = vector(5, 4, 10)
scene.camera.axis = vector(-5, -4, -10)

# ──────────────────────────────────────────────────────────────────
#  PHONE 3-D MODEL
#
#  ใน compound frame:
#    +X = ขวาของโทรศัพท์  (phone.axis ชี้ทาง +X)
#    +Y = หน้าจอ           (phone.up  ชี้ทาง +Y)
#    +Z = ด้านบนของโทรศัพท์
# ──────────────────────────────────────────────────────────────────
_W, _T, _H = 1.40, 0.22, 3.00

_body   = box(size=vector(_W, _T, _H),
              color=vector(0.20, 0.22, 0.30))
_bezel  = box(pos=vector(0, _T/2+0.005, 0),
              size=vector(_W-0.12, 0.008, _H-0.10),
              color=vector(0.05, 0.05, 0.10))
_screen = box(pos=vector(0, _T/2+0.010, 0),
              size=vector(_W-0.22, 0.010, _H-0.22),
              color=vector(0.10, 0.30, 0.65))
_notch  = box(pos=vector(0, _T/2+0.012,  _H/2-0.12),
              size=vector(0.30, 0.010, 0.08),
              color=vector(0.05, 0.05, 0.10))
_cbump  = box(pos=vector(+0.28, -_T/2-0.025,  _H/2-0.35),
              size=vector(0.52, 0.045, 0.52),
              color=vector(0.15, 0.15, 0.22))
_clens1 = box(pos=vector(+0.18, -_T/2-0.042,  _H/2-0.22),
              size=vector(0.16, 0.030, 0.16),
              color=vector(0.05, 0.05, 0.20))
_clens2 = box(pos=vector(+0.38, -_T/2-0.042,  _H/2-0.22),
              size=vector(0.16, 0.030, 0.16),
              color=vector(0.05, 0.05, 0.20))
_clens3 = box(pos=vector(+0.28, -_T/2-0.042,  _H/2-0.42),
              size=vector(0.16, 0.030, 0.16),
              color=vector(0.05, 0.05, 0.20))
_flash  = box(pos=vector(+0.08, -_T/2-0.042,  _H/2-0.42),
              size=vector(0.09, 0.030, 0.09),
              color=vector(0.85, 0.75, 0.30))
_home   = box(pos=vector(0, _T/2+0.010, -_H/2+0.12),
              size=vector(0.38, 0.010, 0.07),
              color=vector(0.30, 0.30, 0.40))

phone = compound([_body, _bezel, _screen, _notch,
                  _cbump, _clens1, _clens2, _clens3, _flash, _home])

# ──────────────────────────────────────────────────────────────────
#  COORDINATE AXES  (สีเข้มสำหรับพื้นหลังขาว)
# ──────────────────────────────────────────────────────────────────
_TH = 0.05
label(pos=vector(0.15, 0.15, 0.15), text="O",
      box=False, height=14, color=color.black)

arrow(pos=vector(0,0,0), axis=vector(-2, -1, 0),
      color=vector(0.85, 0.40, 0.0), shaftwidth=_TH)
label(pos=vector(-2.5, -1.5, 0), text="X",
      box=False, height=15, color=vector(0.85, 0.40, 0.0))

arrow(pos=vector(0,0,0), axis=vector(2, 0, 0),
      color=vector(0.0, 0.55, 0.20), shaftwidth=_TH)
label(pos=vector(2.3, 0, 0), text="Y",
      box=False, height=15, color=vector(0.0, 0.55, 0.20))

arrow(pos=vector(0,0,0), axis=vector(0, 2.2, 0),
      color=vector(0.10, 0.30, 0.75), shaftwidth=_TH)
label(pos=vector(0, 2.6, 0), text="Z",
      box=False, height=15, color=vector(0.10, 0.30, 0.75))

# ──────────────────────────────────────────────────────────────────
#  HUD LABELS
# ──────────────────────────────────────────────────────────────────
lbl_info     = label(pos=vector(-1.5, 5.8, 0), text="Initialising...",
                     box=False, height=13, color=color.black)
lbl_kf_pitch = label(pos=vector(-1.5, 5.1, 0), text="",
                     box=False, height=12, color=vector(0.0, 0.40, 0.85))
lbl_kf_roll  = label(pos=vector(-1.5, 4.6, 0), text="",
                     box=False, height=12, color=vector(0.0, 0.55, 0.55))
lbl_kf_yaw   = label(pos=vector(-1.5, 4.1, 0), text="",
                     box=False, height=12, color=vector(0.60, 0.0, 0.60))
lbl_raw      = label(pos=vector(-1.5, 3.4, 0), text="",
                     box=False, height=11, color=color.gray(0.40))
lbl_servo    = label(pos=vector(-1.5, 2.8, 0), text="",
                     box=False, height=11, color=color.gray(0.30))

# FFT peak box ขวา — แสดงครบทั้ง 3 แกน
lbl_fft = label(pos=vector(5.5, 3.5, 0),
                text="FFT Peak\n────────────\nWaiting...",
                box=True, height=11, color=color.black,
                background=vector(0.95, 0.95, 0.92), opacity=0.8,
                linecolor=color.gray(0.5))

lbl_cal = label(pos=vector(0, -4.5, 0),
                text="[C] or button below to recalibrate",
                box=False, height=10, color=color.gray(0.45))

# ═══════════════════════════════════════════════════════════════════
#  KALMAN FILTER CLASS  (1-D per axis)
#
#  ใช้ filter Euler angles เพื่อกราฟและ HUD เท่านั้น
#  การหมุน 3D model จะใช้ quaternion ดิบเสมอ (ไม่มี Gimbal Lock)
#
#  ปรับ R_measure:
#    ค่าต่ำ  → ตอบสนองเร็ว แต่ noise มากกว่า
#    ค่าสูง  → smooth กว่า แต่ lag มากขึ้น
# ═══════════════════════════════════════════════════════════════════
class KalmanAngle:
    def __init__(self, Q_angle=0.001, Q_rate=0.003, R_measure=0.03):
        self.Q_angle   = Q_angle
        self.Q_rate    = Q_rate
        self.R_measure = R_measure
        self.angle = 0.0
        self.bias  = 0.0
        self.P     = [[0.0, 0.0], [0.0, 0.0]]

    def reset(self, init_angle=0.0):
        """Reset filter state — เรียกเมื่อ recalibrate"""
        self.angle = init_angle
        self.bias  = 0.0
        self.P     = [[0.0, 0.0], [0.0, 0.0]]

    def update(self, new_angle, dt):
        # Predict
        rate_est     = 0.0 - self.bias
        self.angle  += dt * rate_est
        self.P[0][0] += dt * (dt*self.P[1][1] - self.P[0][1]
                               - self.P[1][0] + self.Q_angle)
        self.P[0][1] -= dt * self.P[1][1]
        self.P[1][0] -= dt * self.P[1][1]
        self.P[1][1] += self.Q_rate * dt

        # Update
        S = self.P[0][0] + self.R_measure
        K = [self.P[0][0]/S, self.P[1][0]/S]
        y = new_angle - self.angle
        self.angle += K[0] * y
        self.bias  += K[1] * y
        P00, P01    = self.P[0][0], self.P[0][1]
        self.P[0][0] -= K[0] * P00
        self.P[0][1] -= K[0] * P01
        self.P[1][0] -= K[1] * P00
        self.P[1][1] -= K[1] * P01
        return self.angle


kf_pitch = KalmanAngle(Q_angle=0.001, Q_rate=0.003, R_measure=0.03)
kf_roll  = KalmanAngle(Q_angle=0.001, Q_rate=0.003, R_measure=0.01)
kf_yaw   = KalmanAngle(Q_angle=0.001, Q_rate=0.003, R_measure=0.05)

# ═══════════════════════════════════════════════════════════════════
#  COMPARISON GRAPHS  (Raw vs Kalman — Pitch / Roll / Yaw)
#  ใช้ gcurve.data เพื่อ update in-place → legend ไม่สะสม
# ═══════════════════════════════════════════════════════════════════
g_pitch = graph(title="Pitch — Raw vs Kalman",
                xtitle="Time (s)", ytitle="Degrees",
                width=580, height=200, background=color.white)
gc_pitch_raw    = gcurve(graph=g_pitch, color=color.red,
                         label="Raw Pitch",    width=2)
gc_pitch_kalman = gcurve(graph=g_pitch, color=color.blue,
                         label="Kalman Pitch", width=2)

g_roll = graph(title="Roll — Raw vs Kalman",
               xtitle="Time (s)", ytitle="Degrees",
               width=580, height=200, background=color.white)
gc_roll_raw    = gcurve(graph=g_roll, color=color.orange,
                        label="Raw Roll",    width=2)
gc_roll_kalman = gcurve(graph=g_roll, color=color.cyan,
                        label="Kalman Roll", width=2)

g_yaw = graph(title="Yaw — Raw vs Kalman",
              xtitle="Time (s)", ytitle="Degrees",
              width=580, height=200, background=color.white)
gc_yaw_raw    = gcurve(graph=g_yaw, color=color.green,
                       label="Raw Yaw",    width=2)
gc_yaw_kalman = gcurve(graph=g_yaw, color=color.magenta,
                       label="Kalman Yaw", width=2)

# ═══════════════════════════════════════════════════════════════════
#  CALIBRATION
# ═══════════════════════════════════════════════════════════════════
q0 = None

def reset_cal(src=None):
    """Reset both quaternion reference AND Kalman state พร้อมกัน"""
    global q0
    q0 = None
    # Kalman จะถูก reset ใน main loop ตอนที่ q0 = None อีกครั้ง
    lbl_cal.text  = "✔ Recalibrated — hold device flat & still"
    lbl_cal.color = color.green

scene.append_to_caption("\n\n")
button(text="  ↺  Reset Calibration  ",
       bind=reset_cal,
       background=vector(0.15, 0.35, 0.65),
       color=color.white)

def on_key(evt):
    if evt.key.lower() == "c":
        reset_cal()
scene.bind("keydown", on_key)

scene.append_to_caption("   ")
btn_rec = button(text="  ⏺  Record 10s  ",
                 bind=lambda s: start_recording(),
                 background=vector(0.55, 0.10, 0.10),
                 color=color.white)

lbl_rec = label(pos=vector(0, -5.2, 0),
                text="Ready to record",
                box=False, height=11, color=color.gray(0.5))

csv_file   = None
csv_writer = None

# ═══════════════════════════════════════════════════════════════════
#  MATH UTILITIES
# ═══════════════════════════════════════════════════════════════════
def quat_conj(q):
    w, x, y, z = q
    return (w, -x, -y, -z)

def quat_mul(p, q):
    pw, px, py, pz = p
    qw, qx, qy, qz = q
    return (
        pw*qw - px*qx - py*qy - pz*qz,
        pw*qx + px*qw + py*qz - pz*qy,
        pw*qy - px*qz + py*qw + pz*qx,
        pw*qz + px*qy - py*qx + pz*qw,
    )

def qrot(q, v):
    """Rotate vector v by quaternion q (sandwich product) — ไม่มี Gimbal Lock"""
    qw, qx, qy, qz = q
    vx, vy, vz = v
    tw = -qx*vx - qy*vy - qz*vz
    tx =  qw*vx + qy*vz - qz*vy
    ty =  qw*vy - qx*vz + qz*vx
    tz =  qw*vz + qx*vy - qy*vx
    return (
        tx*qw - tw*qx - ty*qz + tz*qy,
        ty*qw - tw*qy + tx*qz - tz*qx,
        tz*qw - tw*qz - tx*qy + ty*qx,
    )

def compute_fft(buf):
    """Hann-windowed FFT, ตัด DC bin ออก"""
    arr   = np.asarray(buf, dtype=float) - np.mean(buf)
    win   = np.hanning(len(arr))
    mags  = np.abs(np.fft.rfft(arr * win))
    freqs = np.fft.rfftfreq(len(arr), DT)
    return freqs[1:], mags[1:]

def start_recording():
    global recording, rec_start_time, csv_file, csv_writer
    if recording:
        return
    rec_start_time = t_now
    recording      = True
    lbl_rec.text   = "⏺ Recording..."
    lbl_rec.color  = color.red

    # เปิดไฟล์ทันที + เขียน header
    fname = os.path.join(
        save_folder,
        f"motion_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    )
    csv_file   = open(fname, "w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["time_s",
                         "roll_deg", "pitch_deg", "yaw_deg",
                         "raw_pitch_s", "raw_roll_s", "raw_yaw_s"])

# ═══════════════════════════════════════════════════════════════════
#  NETWORK
# ═══════════════════════════════════════════════════════════════════
def read_sensor():
    try:
        r   = requests.get(f"{URL}/get?w&x&y&z&pitch&roll&yaw", timeout=1)
        buf = r.json()["buffer"]
        return (buf["w"]["buffer"][-1],
                buf["x"]["buffer"][-1],
                buf["y"]["buffer"][-1],
                buf["z"]["buffer"][-1],
                buf["pitch"]["buffer"][-1],
                buf["roll"]["buffer"][-1],
                buf["yaw"]["buffer"][-1])
    except Exception:
        return None

# ═══════════════════════════════════════════════════════════════════
#  RUNTIME STATE
# ═══════════════════════════════════════════════════════════════════
# Rolling buffers สำหรับ FFT — เก็บทั้ง 3 แกน (raw)
buf_pitch = []
buf_roll  = []
buf_yaw   = []

# Data lists สำหรับกราฟ — จำกัดไว้ที่ 10 วินาทีล่าสุด
MAX_PTS = SAMPLE_RATE * 10
data_pitch_raw    = []
data_pitch_kalman = []
data_roll_raw     = []
data_roll_kalman  = []
data_yaw_raw      = []
data_yaw_kalman   = []

t_now   = 0.0
frame_n = 0

# ── CSV Recording ────────────────────────────────────────────────
RECORD_DURATION = 10.0   # second
recording       = False
rec_start_time  = 0.0
# rec_data        = []     # list of rows
save_folder = "dataset"
os.makedirs(save_folder,exist_ok=True)

# ═══════════════════════════════════════════════════════════════════
#  MAIN LOOP
# ═══════════════════════════════════════════════════════════════════
while True:
    rate(SAMPLE_RATE)

    raw = read_sensor()
    if raw is None:
        lbl_info.text  = "⚠ No signal — check Wi-Fi / server"
        lbl_info.color = color.red
        continue

    lbl_info.color = color.black
    qw_s, qx_s, qy_s, qz_s, pitch_s, roll_s, yaw_s = raw

    # ── Sensor-frame → VPython-frame quaternion remap ─────────────
    # (ทดสอบแล้วกับ iPhone 13 Pro Max)
    q_cur = (qw_s, -qx_s, qz_s, -qy_s)

    # ── Calibration: เก็บ q0 และ reset Kalman ทุกตัวพร้อมกัน ─────
    if q0 is None:
        q0 = q_cur
        kf_pitch.reset(pitch_s)
        kf_roll.reset(roll_s)
        kf_yaw.reset(yaw_s)
        lbl_cal.text  = "✔ Calibrated"
        lbl_cal.color = color.green

    # ── Relative quaternion  (q0^-1 * q_cur) ─────────────────────
    # นี่คือ rotation สัมพัทธ์จาก pose เริ่มต้น → ไม่มี Gimbal Lock
    q_rel = quat_mul(quat_conj(q0), q_cur)

    # ── หมุน 3D model ด้วย QUATERNION โดยตรง ─────────────────────
    # สำคัญมาก: ใช้ q_rel ตรงๆ ไม่ผ่าน Euler angles
    # → รองรับการเอียงทุกมุม รวมถึง pitch ±90° โดยไม่เพี้ยน
    phone.axis = vector(*qrot(q_rel, (1, 0, 0)))
    phone.up   = vector(*qrot(q_rel, (0, 1, 0)))

    # ── Raw Euler angles (Phyphox → display mapping ที่ทดสอบแล้ว) ─
    raw_pitch = pitch_s
    raw_roll  = roll_s
    raw_yaw   = yaw_s

    # ── Kalman filter — ใช้สำหรับ HUD และกราฟเท่านั้น ────────────
    # Kalman จะ smooth ค่าที่กระตุกหรือมี noise ออกไป
    # แต่ไม่มีผลกับ 3D model (ซึ่งใช้ quaternion แล้ว smooth เอง)
    k_pitch = kf_pitch.update(raw_pitch, DT)
    k_roll  = kf_roll.update(raw_roll,   DT)
    k_yaw   = kf_yaw.update(raw_yaw,     DT)

    # ── Servo mapping ─────────────────────────────────────────────
    pan  = max(-90.0, min(90.0,  k_yaw))
    tilt = max(-45.0, min(45.0,  k_pitch))
    s1   = pan  + 90.0    # 0–180°
    s2   = tilt + 45.0    # 0–90°

    # ── HUD ───────────────────────────────────────────────────────
    lbl_info.text     = "Kalman-filtered   |   3D model via Quaternion"
    lbl_kf_pitch.text = f"Pitch:  {k_pitch:+6.1f}°"
    lbl_kf_roll.text  = f"Roll:   {k_roll:+6.1f}°"
    lbl_kf_yaw.text   = f"Yaw:    {k_yaw:+6.1f}°"
    lbl_raw.text      = (f"Raw →  P:{raw_pitch:+.1f}°  "
                         f"R:{raw_roll:+.1f}°  Y:{raw_yaw:+.1f}°")
    lbl_servo.text    = (f"Servo1 (Pan) = {s1:.0f}°   "
                         f"Servo2 (Tilt) = {s2:.0f}°")

    # ── เก็บข้อมูลกราฟ ─────────────────────────────────────────
    data_pitch_raw.append(   [t_now, raw_pitch])
    data_pitch_kalman.append([t_now, k_pitch])
    data_roll_raw.append(    [t_now, raw_roll])
    data_roll_kalman.append( [t_now, k_roll])
    data_yaw_raw.append(     [t_now, raw_yaw])
    data_yaw_kalman.append(  [t_now, k_yaw])

    # จำกัดความยาวไว้ที่ 10 วินาที เพื่อไม่ให้กราฟช้าลงเมื่อรันนาน
    if len(data_pitch_raw) > MAX_PTS:
        data_pitch_raw    = data_pitch_raw[-MAX_PTS:]
        data_pitch_kalman = data_pitch_kalman[-MAX_PTS:]
        data_roll_raw     = data_roll_raw[-MAX_PTS:]
        data_roll_kalman  = data_roll_kalman[-MAX_PTS:]
        data_yaw_raw      = data_yaw_raw[-MAX_PTS:]
        data_yaw_kalman   = data_yaw_kalman[-MAX_PTS:]

    # ── FFT rolling buffers (ทุก frame) ──────────────────────────
    buf_pitch.append(raw_pitch)
    buf_roll.append(raw_roll)
    buf_yaw.append(raw_yaw)
    if len(buf_pitch) > BUFFER_SIZE:
        buf_pitch.pop(0)
        buf_roll.pop(0)
        buf_yaw.pop(0)

    # ── อัพเดทกราฟและ FFT ทุก ~10 frame ≈ 3 Hz ──────────────────
    if frame_n % 10 == 0:
        gc_pitch_raw.data    = data_pitch_raw
        gc_pitch_kalman.data = data_pitch_kalman
        gc_roll_raw.data     = data_roll_raw
        gc_roll_kalman.data  = data_roll_kalman
        gc_yaw_raw.data      = data_yaw_raw
        gc_yaw_kalman.data   = data_yaw_kalman

    # FFT ทุก ~30 frame ≈ 1 Hz (ต้องรอให้ buffer เต็มก่อน)
    if frame_n % 30 == 0 and len(buf_pitch) == BUFFER_SIZE:
        fr_P, mP = compute_fft(buf_pitch)
        fr_R, mR = compute_fft(buf_roll)
        fr_Y, mY = compute_fft(buf_yaw)

        pk_P = fr_P[np.argmax(mP)]
        pk_R = fr_R[np.argmax(mR)]
        pk_Y = fr_Y[np.argmax(mY)]

        lbl_fft.text = (f"FFT Peak\n"
                        f"────────────\n"
                        f"Pitch: {pk_P:.2f} Hz\n"
                        f"Roll:  {pk_R:.2f} Hz\n"
                        f"Yaw:   {pk_Y:.2f} Hz")

    frame_n += 1
    t_now   += DT
    
     # ── CSV recording logic ───────────────────────────────────────────
    if recording:
        elapsed = t_now - rec_start_time
        lbl_rec.text = f"⏺ Recording...  {elapsed:.1f} / {RECORD_DURATION:.0f} s"
        # rec_data.append([
        #     round(t_now, 4),
        #     round(real_roll,  2),
        #     round(real_pitch, 2),
        #     round(real_yaw,   2),
        #     round(pitch_s, 2),
        #     round(roll_s,  2),
        #     round(yaw_s,   2),
        # ])
        # ใหม่
        csv_writer.writerow([
            round(t_now, 4),
            round(k_roll,  2),
            round(k_pitch, 2),
            round(k_yaw,   2),
            round(pitch_s, 2),
            round(roll_s,  2),
            round(yaw_s,   2),
        ])
        csv_file.flush() 
        
        
        # if elapsed >= RECORD_DURATION:
        #     recording = False
        #     # save file
        #     import csv, datetime
        #     #fname = f"motion_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        #     fname = os.path.join(
        #     save_folder,
        #     f"motion_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
        #     with open(fname, "w", newline="") as f:
        #         w = csv.writer(f)
        #         w.writerow(["time_s",
        #                     "roll_deg", "pitch_deg", "yaw_deg",
        #                     "raw_pitch_s", "raw_roll_s", "raw_yaw_s"])
        #         w.writerows(rec_data)
        #     lbl_rec.text  = f"Saved → {fname}  ({len(rec_data)} rows)"
        #     lbl_rec.color = color.green
        if elapsed >= RECORD_DURATION:
            recording = False
            csv_file.close()   # ← ปิดไฟล์
            csv_file = None
            lbl_rec.text  = f"✔ Saved  ({int(RECORD_DURATION * SAMPLE_RATE)} rows)"
            lbl_rec.color = color.green