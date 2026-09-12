from vpython import *
import requests
import numpy as np
import os, csv, datetime


# ═══════════════════════════════════════════════════════════════════
#  CONFIGURATION
# ═══════════════════════════════════════════════════════════════════
# URL         = "http://172.20.10.1"
URL         = "http://192.168.1.43:8080"

SAMPLE_RATE = 30
DT          = 1.0 / SAMPLE_RATE
BUFFER_SIZE = 128       # FFT window ≈ 4.3 s → Δf ≈ 0.23 Hz
GRAPH_EVERY = 30        # graph refresh ≈ 1 Hz

# ═══════════════════════════════════════════════════════════════════
#  3-D SCENE
# ═══════════════════════════════════════════════════════════════════
scene = canvas(
    title      = "📱 Smartphone Motion Visualizer  |  Attitude Sensor (RPY)",
    width      = 960,
    height     = 500,
    background = vector(0.05, 0.05, 0.10),
)
scene.camera.pos  = vector(5, 4, 10)
scene.camera.axis = vector(-5, -4, -10)

# ──────────────────────────────────────────────────────────────────
#  Phone 3-D model  (portrait — taller than wide)
#  Local axes: +X = right side of phone, +Y = out of screen, +Z = top of phone
#  Camera island is on the BACK (−Y face), top-RIGHT (+X, +Z corner)
# ──────────────────────────────────────────────────────────────────
_W, _T, _H = 1.40, 0.22, 3.00   # width, thickness, height

_body   = box(size=vector(_W, _T, _H),
              color=vector(0.15, 0.15, 0.22))

_bezel  = box(pos=vector(0,  _T/2 + 0.005, 0),
              size=vector(_W - 0.12, 0.008, _H - 0.10),
              color=vector(0.03, 0.03, 0.08))

_screen = box(pos=vector(0,  _T/2 + 0.010, 0),
              size=vector(_W - 0.22, 0.010, _H - 0.22),
              color=vector(0.08, 0.25, 0.55))

# Dynamic Island — top-centre of screen
_notch  = box(pos=vector(0,  _T/2 + 0.012,  _H/2 - 0.12),
              size=vector(0.30, 0.010, 0.08),
              color=vector(0.03, 0.03, 0.08))

# Camera island — back face (−Y), top-RIGHT (+X, +Z corner)
_cbump  = box(pos=vector(+0.28, -_T/2 - 0.025,  _H/2 - 0.35),
              size=vector(0.52, 0.045, 0.52),
              color=vector(0.10, 0.10, 0.17))

_clens1 = box(pos=vector(+0.18, -_T/2 - 0.042,  _H/2 - 0.22),
              size=vector(0.16, 0.030, 0.16),
              color=vector(0.02, 0.02, 0.20))

_clens2 = box(pos=vector(+0.38, -_T/2 - 0.042,  _H/2 - 0.22),
              size=vector(0.16, 0.030, 0.16),
              color=vector(0.02, 0.02, 0.20))

_clens3 = box(pos=vector(+0.28, -_T/2 - 0.042,  _H/2 - 0.42),
              size=vector(0.16, 0.030, 0.16),
              color=vector(0.02, 0.02, 0.20))

_flash  = box(pos=vector(+0.08, -_T/2 - 0.042,  _H/2 - 0.42),
              size=vector(0.09, 0.030, 0.09),
              color=vector(0.85, 0.80, 0.45))

# Home bar — bottom-centre of screen
_home   = box(pos=vector(0,  _T/2 + 0.010, -_H/2 + 0.12),
              size=vector(0.38, 0.010, 0.07),
              color=vector(0.22, 0.22, 0.32))

phone = compound([_body, _bezel, _screen, _notch,
                  _cbump, _clens1, _clens2, _clens3, _flash, _home])

# ──────────────────────────────────────────────────────────────────
#  Coordinate axes
#  X → right (+X horizontal)   Y → back-left (horizontal)   Z → up
# ──────────────────────────────────────────────────────────────────
_TH = 0.05

label(pos=vector(0.15, 0.15, 0.15), text="O",
      box=False, height=14, color=color.white)

arrow(pos=vector(0,0,0), axis=vector(-2, -1, 0),
      color=color.orange, shaftwidth=_TH)
label(pos=vector(-2.5, -1.5,0), text="X",
      box=False, height=15, color=color.orange)

arrow(pos=vector(0,0,0), axis=vector(2, 0, 0),
      color=color.green, shaftwidth=_TH)
label(pos=vector(2.2, 0, 0), text="Y",
      box=False, height=15, color=color.green)

arrow(pos=vector(0,0,0), axis=vector(0, 2.2, 0),
      color=color.cyan, shaftwidth=_TH)
label(pos=vector(0, 2.6, 0), text="Z",
      box=False, height=15, color=color.cyan)

# ──────────────────────────────────────────────────────────────────
#  HUD labels
# ──────────────────────────────────────────────────────────────────

lbl_info  = label(pos=vector(-1.5, 5.8, 0), text="Initialising...",
                  box=False, height=13, color=color.white)
lbl_raw   = label(pos=vector(-1.5, 5.2, 0), text="",
                  box=False, height=11, color=color.gray(0.55))
lbl_pan   = label(pos=vector(-1.5, 4.6, 0), text="",
                  box=False, height=12, color=color.cyan)
lbl_tilt  = label(pos=vector(-1.5, 4.1, 0), text="",
                  box=False, height=12, color=color.green)
lbl_roll  = label(pos=vector(-1.5, 3.6, 0), text="",
                  box=False, height=12, color=color.orange)

# FFT peak box on the right
lbl_fft   = label(pos=vector(5.5, 3.0, 0),
                  text="FFT Peak\n────────────\nWaiting...",
                  box=True, height=11, color=color.yellow,
                  background=vector(0.05, 0.05, 0.05), opacity=0.6,
                  linecolor=color.gray(0.3))

lbl_cal   = label(pos=vector(0, -4.5, 0),
                  text="[C] or button below to recalibrate",
                  box=False, height=10, color=color.gray(0.45))

# ═══════════════════════════════════════════════════════════════════
#  GRAPHS
#  Create gcurve ONCE — update via gcurve.data each refresh cycle.
#  This keeps the legend fixed (no accumulation).
# ═══════════════════════════════════════════════════════════════════
g_raw = graph(
    title      = "Orientation — Time Domain  (last 4 s window)",
    xtitle     = "Time (s)",
    ytitle     = "Angle (deg)",
    width      = 580,
    height     = 220,
    background = color.black,
    foreground = color.white,
    align      = "left",
)
gcr_roll  = gcurve(graph=g_raw, color=color.orange,
                   label="Roll  (← Phyphox Pitch)", width=2)
gcr_pitch = gcurve(graph=g_raw, color=color.green,
                   label="Pitch (← Phyphox Yaw)",   width=2)
gcr_yaw   = gcurve(graph=g_raw, color=color.cyan,
                   label="Yaw   (← Phyphox Roll)",  width=2)

g_fft = graph(
    title      = "FFT Magnitude Spectrum — RPY  (128-pt  Hann window)",
    xtitle     = "Frequency (Hz)",
    ytitle     = "Magnitude",
    width      = 580,
    height     = 220,
    background = color.black,
    foreground = color.white,
    align      = "left",
)
gcf_roll  = gcurve(graph=g_fft, color=color.orange,
                   label="Roll FFT",  width=2)
gcf_pitch = gcurve(graph=g_fft, color=color.green,
                   label="Pitch FFT", width=2)
gcf_yaw   = gcurve(graph=g_fft, color=color.cyan,
                   label="Yaw FFT",   width=2)

# ═══════════════════════════════════════════════════════════════════
#  CALIBRATION
# ═══════════════════════════════════════════════════════════════════
q0 = None

def reset_cal(src=None):
    global q0
    q0 = None
    

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

# def start_recording():
#     global recording, rec_start_time, rec_data
#     if recording:
#         return          
#     rec_data       = []
#     rec_start_time = t_now
#     recording      = True
#     lbl_rec.text   = "⏺ Recording..."
#     lbl_rec.color  = color.red

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
    
    """
    Rotate 3-vector v = (vx, vy, vz) by unit quaternion q = (w, x, y, z).
    Uses the sandwich product:  p' = q  *  (0,v)  *  q*
    """

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
    arr   = np.asarray(buf, dtype=float) - np.mean(buf)
    win   = np.hanning(len(arr))
    mags  = np.abs(np.fft.rfft(arr * win))
    freqs = np.fft.rfftfreq(len(arr), DT)
    return freqs[1:], mags[1:]

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
roll_buf  = []
pitch_buf = []
yaw_buf   = []
t_now     = 0.0
frame_n   = 0

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

    lbl_info.color = color.white
    qw_s, qx_s, qy_s, qz_s, pitch_s, roll_s, yaw_s = raw

    # ── sensor-frame → VPython-frame remap ───────────────────────
    q_cur = (qw_s, -qx_s, qz_s, -qy_s)

    # ── first-frame calibration ───────────────────────────────────
    if q0 is None:
        q0 = q_cur
        

    q_rel = quat_mul(quat_conj(q0), q_cur)

   
    
    real_roll  = roll_s    # วางแบน = roll
    real_pitch = pitch_s   # ก้ม/เงย = pitch 
    real_yaw   = yaw_s     # หมุนรอบตัว = yaw

    # ── rolling buffers ───────────────────────────────────────────
    roll_buf.append(real_roll)
    pitch_buf.append(real_pitch)
    yaw_buf.append(real_yaw)
    if len(roll_buf) > BUFFER_SIZE:
        roll_buf.pop(0)
        pitch_buf.pop(0)
        yaw_buf.pop(0)

    # ── phone orientation ─────────────────────────────────────────
    phone.axis = vector(*qrot(q_rel, (1, 0, 0)))
    phone.up   = vector(*qrot(q_rel, (0, 1, 0)))

    # ── servo clamping ────────────────────────────────────────────
    pan  = max(-90.0, min(90.0,  real_yaw))
    tilt = max(-45.0, min(45.0,  real_pitch))
    s1   = pan  + 90.0
    s2   = tilt + 45.0

    # ── HUD ───────────────────────────────────────────────────────
    lbl_info.text = (f"Roll: {real_roll:+6.1f}°    "
                     f"Pitch: {real_pitch:+6.1f}°    "
                     f"Yaw: {real_yaw:+6.1f}°")
    lbl_raw.text  = (f"Phyphox raw →  P:{pitch_s:.1f}°  "
                     f"R:{roll_s:.1f}°  Y:{yaw_s:.1f}°")
    lbl_pan.text  = f"Pan  (Yaw):    {pan:+.1f}°  →  Servo 1 = {s1:.0f}°"
    lbl_tilt.text = f"Tilt (Pitch):  {tilt:+.1f}°  →  Servo 2 = {s2:.0f}°"
    lbl_roll.text = f"Roll:          {real_roll:+.1f}°"

    frame_n += 1
    t_now   += DT
    
    # print(f"pitch_s={pitch_s:+.1f}  roll_s={roll_s:+.1f}  yaw_s={yaw_s:+.1f}")
    
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
            round(real_roll,  2),
            round(real_pitch, 2),
            round(real_yaw,   2),
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
    
    # ──────────────────────────────────────────────────────────────
    #  GRAPH REFRESH  (~1 Hz, buffer full only)
    #  Use gcurve.data = [...] to replace points in-place.
    #  Legend entries are created once → they NEVER accumulate.
    # ──────────────────────────────────────────────────────────────
    if frame_n % GRAPH_EVERY == 0 and len(roll_buf) == BUFFER_SIZE:

        n  = BUFFER_SIZE
        t0 = t_now - n * DT

        # Time-domain: build list of [t, value] pairs and assign to .data
        pts_t = [t0 + i * DT for i in range(n)]
        gcr_roll.data  = [[pts_t[i], roll_buf[i]]  for i in range(n)]
        gcr_pitch.data = [[pts_t[i], pitch_buf[i]] for i in range(n)]
        gcr_yaw.data   = [[pts_t[i], yaw_buf[i]]   for i in range(n)]

        # FFT
        fr_R, mR = compute_fft(roll_buf)
        fr_P, mP = compute_fft(pitch_buf)
        fr_Y, mY = compute_fft(yaw_buf)

        gcf_roll.data  = [[fr_R[i], mR[i]] for i in range(len(fr_R))]
        gcf_pitch.data = [[fr_P[i], mP[i]] for i in range(len(fr_P))]
        gcf_yaw.data   = [[fr_Y[i], mY[i]] for i in range(len(fr_Y))]

        # Dominant frequency
        pk_R = fr_R[np.argmax(mR)]
        pk_P = fr_P[np.argmax(mP)]
        pk_Y = fr_Y[np.argmax(mY)]
        lbl_fft.text = (f"FFT Peak\n"
                        f"────────────\n"
                        f"Roll:  {pk_R:.2f} Hz\n"
                        f"Pitch: {pk_P:.2f} Hz\n"
                        f"Yaw:   {pk_Y:.2f} Hz")