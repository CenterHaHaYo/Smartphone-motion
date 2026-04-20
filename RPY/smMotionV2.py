from vpython import *
import requests
import numpy as np

URL = "http://172.20.10.1"

# ──────────────────────────────────────────────
#  SCENE SETUP
# ──────────────────────────────────────────────
scene = canvas(
    title="Smartphone Motion | Digital Twin + Kalman Filter",
    width=900, height=500,
    background=color.white
)
scene.camera.pos  = vector(5, 5, 10)
scene.camera.axis = vector(-5, -5, -10)

# ── Phone model ──
phone = box(size=vector(2.8, 0.3, 1.5), color=color.cyan)
top_marker = box(pos=vector(0, 0.2, -0.6),
                 size=vector(0.4, 0.15, 0.3),
                 color=color.red)

# ── Axes ──
axis_len   = 2
axis_thick = 0.02



label(pos=vector(0,0,0), text="O", box=False, height=12, color=color.black)
arrow(pos=vector(0,0,0), axis=vector(axis_len,0,0),color=color.orange,shaftwidth=axis_thick)
label(pos=vector(axis_len+0.2,0,0), text="X (Roll axis)",box=False, height=12, color=color.orange)
arrow(pos=vector(0,0,0), axis=vector(0,axis_len,0),color=color.green, shaftwidth=axis_thick)
label(pos=vector(0,axis_len+0.3,0), text="Y (Pitch axis)",box=False, height=12, color=color.green)
arrow(pos=vector(0,0,0), axis=vector(0,0,axis_len),color=color.blue,  shaftwidth=axis_thick)
label(pos=vector(0,0,axis_len+0.3), text="Z",box=False, height=14, color=color.blue)

# ──────────────────────────────────────────────
#  COMPARISON GRAPH  (Raw vs Kalman)
# ──────────────────────────────────────────────
graph_pitch = graph(
    title="Pitch — Raw vs Kalman",
    xtitle="Time (s)", ytitle="Degrees",
    width=580, height=200,
    background=color.white
)
gc_pitch_raw    = gcurve(graph=graph_pitch, color=color.red,   label="Raw Pitch")
gc_pitch_kalman = gcurve(graph=graph_pitch, color=color.blue,  label="Kalman Pitch")

graph_roll = graph(
    title="Roll — Raw vs Kalman",
    xtitle="Time (s)", ytitle="Degrees",
    width=580, height=200,
    background=color.white
)
gc_roll_raw    = gcurve(graph=graph_roll, color=color.orange, label="Raw Roll")
gc_roll_kalman = gcurve(graph=graph_roll, color=color.cyan,   label="Kalman Roll")

graph_yaw = graph(
    title="Yaw — Raw vs Kalman",
    xtitle="Time (s)", ytitle="Degrees",
    width=580, height=200,
    background=color.white
)
gc_yaw_raw    = gcurve(graph=graph_yaw, color=color.green,   label="Raw Yaw")
gc_yaw_kalman = gcurve(graph=graph_yaw, color=color.magenta, label="Kalman Yaw")

# ──────────────────────────────────────────────
#  LABELS
# ──────────────────────────────────────────────
info_lbl = label(pos=vector(0, 6,   0), text="Waiting...", box=False)
yaw_lbl  = label(pos=vector(0, 5,   0), text="", box=False, color=color.blue)
pitch_lbl = label(pos=vector(0, 4.5, 0), text="", box=False, color=color.green)
roll_lbl = label(pos=vector(0, 4,   0), text="", box=False, color=color.orange)
raw_lbl  = label(pos=vector(0, 5.5, 0), text="RAW: --", box=False,
                 color=color.gray(0.3), height=11)
fft_lbl  = label(pos=vector(-5, 1,  0), text="FFT: --", box=False,
                 color=color.black,   height=12)


# ──────────────────────────────────────────────
#  KALMAN FILTER CLASS
#  1D Kalman for each angle independently
# ──────────────────────────────────────────────
class KalmanAngle:
    """
    Simple 1-D Kalman filter for a single angle channel.

    State:   x  = [angle, angle_rate]   (2×1)
    Measure: z  = angle                 (scalar)

    Q  — process noise  (how much we trust the motion model)
    R  — measurement noise (how much we trust the sensor)

    Tune:
      ↑ Q  →  filter reacts faster but noisier
      ↑ R  →  filter smoother but slower to follow
    """
    def __init__(self, Q_angle=0.001, Q_rate=0.003, R_measure=0.03):
        self.Q_angle   = Q_angle
        self.Q_rate    = Q_rate
        self.R_measure = R_measure

        self.angle = 0.0        # filtered angle
        self.bias  = 0.0        # gyro bias estimate (we don't have gyro, so stays 0)
        self.P     = [[0.0, 0.0],
                      [0.0, 0.0]]   # error covariance matrix

    def update(self, new_angle, dt):
        """
        Prediction step
        (Since we have no gyro rate, rate = 0 → angle prediction = last angle)
        """
        rate = 0.0 - self.bias      # estimated rate (no gyro → 0)
        self.angle += dt * rate

        # Update error covariance
        self.P[0][0] += dt * (dt * self.P[1][1]
                              - self.P[0][1]
                              - self.P[1][0]
                              + self.Q_angle)
        self.P[0][1] -= dt * self.P[1][1]
        self.P[1][0] -= dt * self.P[1][1]
        self.P[1][1] += self.Q_rate * dt

        # Update step (Kalman gain)
        S = self.P[0][0] + self.R_measure      # innovation covariance
        K = [self.P[0][0] / S,
             self.P[1][0] / S]                 # Kalman gain

        y = new_angle - self.angle             # innovation
        self.angle += K[0] * y
        self.bias  += K[1] * y

        P00_temp = self.P[0][0]
        P01_temp = self.P[0][1]
        self.P[0][0] -= K[0] * P00_temp
        self.P[0][1] -= K[0] * P01_temp
        self.P[1][0] -= K[1] * P00_temp
        self.P[1][1] -= K[1] * P01_temp

        return self.angle


# Instantiate one filter per axis
kf_pitch = KalmanAngle(Q_angle=0.001, Q_rate=0.003, R_measure=0.03)
kf_roll  = KalmanAngle(Q_angle=0.001, Q_rate=0.003, R_measure=0.01)
kf_yaw   = KalmanAngle(Q_angle=0.001, Q_rate=0.003, R_measure=0.05)

# ──────────────────────────────────────────────
#  QUATERNION HELPERS
# ──────────────────────────────────────────────
def quat_conj(w, x, y, z):
    return (-w, -x, -y, -z)

def quat_mul(q1, q2):
    w1,x1,y1,z1 = q1
    w2,x2,y2,z2 = q2
    return (
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2
    )

def rotate_vec(qw, qx, qy, qz, vx, vy, vz):
    tw = -qx*vx - qy*vy - qz*vz
    tx =  qw*vx + qy*vz - qz*vy
    ty =  qw*vy - qx*vz + qz*vx
    tz =  qw*vz + qx*vy - qy*vx
    rx = tx*qw - tw*qx - ty*qz + tz*qy
    ry = ty*qw - tw*qy + tx*qz - tz*qx
    rz = tz*qw - tw*qz - tx*qy + ty*qx
    return rx, ry, rz

# ──────────────────────────────────────────────
#  EULER → QUATERNION
#  ใช้สำหรับสร้าง quaternion จาก filtered Euler angles
#  เพื่อ rotate phone model
# ──────────────────────────────────────────────
def euler_to_quat(pitch_deg, roll_deg, yaw_deg):
    """ZYX convention: Yaw → Pitch → Roll"""
    p = np.radians(pitch_deg) / 2
    r = np.radians(roll_deg)  / 2
    y = np.radians(yaw_deg)   / 2

    qw = (np.cos(r)*np.cos(p)*np.cos(y) + np.sin(r)*np.sin(p)*np.sin(y))
    qx = (np.sin(r)*np.cos(p)*np.cos(y) - np.cos(r)*np.sin(p)*np.sin(y))
    qy = (np.cos(r)*np.sin(p)*np.cos(y) + np.sin(r)*np.cos(p)*np.sin(y))
    qz = (np.cos(r)*np.cos(p)*np.sin(y) - np.sin(r)*np.sin(p)*np.cos(y))
    return float(qw), float(qx), float(qy), float(qz)

# ──────────────────────────────────────────────
#  READ DATA FROM PHYPHOX
# ──────────────────────────────────────────────
def read_data():
    try:
        r = requests.get(f"{URL}/get?w&x&y&z&pitch&roll&yaw", timeout=1)
        data = r.json()
        buf  = data["buffer"]
        qw    = buf["w"]["buffer"][-1]
        qx    = buf["x"]["buffer"][-1]
        qy    = buf["y"]["buffer"][-1]
        qz    = buf["z"]["buffer"][-1]
        pitch = buf["pitch"]["buffer"][-1]
        roll  = buf["roll"]["buffer"][-1]
        yaw   = buf["yaw"]["buffer"][-1]
        return qw, qx, qy, qz, pitch, roll, yaw
    except:
        return None

# ──────────────────────────────────────────────
#  FFT BUFFER
# ──────────────────────────────────────────────
BUFFER_SIZE  = 128
pitch_buffer = []
dt           = 1 / 30.0
t_elapsed    = 0.0

# ──────────────────────────────────────────────
#  CALIBRATION
# ──────────────────────────────────────────────
q0 = None

# ──────────────────────────────────────────────
#  MAIN LOOP
# ──────────────────────────────────────────────
while True:
    rate(30)
    t_elapsed += dt

    data = read_data()
    if data is None:
        continue

    qw, qx, qy, qz, pitch, roll, yaw = data

    # ── Calibration (capture first reading) ──
    if q0 is None:
        q0 = (qw, qx, qy, qz)
        kf_pitch.angle = pitch
        kf_roll.angle  = roll
        kf_yaw.angle   = yaw

    # ── Relative quaternion ──
    q_rel = quat_mul(quat_conj(*q0), (qw, qx, qy, qz))
    qw, qx, qy, qz = q_rel

    # ── Axis remapping (same as original) ──
    raw_roll  = pitch
    raw_pitch = yaw
    raw_yaw   = roll

    # ── Kalman filter ──
    k_pitch = kf_pitch.update(raw_pitch, dt)
    k_roll  = kf_roll.update(raw_roll,   dt)
    k_yaw   = kf_yaw.update(raw_yaw,     dt)

    # ── Rotate phone using FILTERED angles ──
    qw_f, qx_f, qy_f, qz_f = euler_to_quat(k_pitch, k_roll, k_yaw)
    ax, ay, az = rotate_vec(qw_f, qx_f, qy_f, qz_f, 1, 0, 0)
    ux, uy, uz = rotate_vec(qw_f, qx_f, qy_f, qz_f, 0, 1, 0)
    phone.axis = vector(ax, ay, az)
    phone.up   = vector(ux, uy, uz)

    # ── Sync top marker ──
    mx, my, mz = rotate_vec(qw_f, qx_f, qy_f, qz_f, 0, 0.2, -0.6)
    top_marker.pos  = vector(mx, my, mz)
    top_marker.axis = phone.axis
    top_marker.up   = phone.up

    # ── Update comparison graphs ──
    gc_pitch_raw.plot(t_elapsed,    raw_pitch)
    gc_pitch_kalman.plot(t_elapsed, k_pitch)
    gc_roll_raw.plot(t_elapsed,     raw_roll)
    gc_roll_kalman.plot(t_elapsed,  k_roll)
    gc_yaw_raw.plot(t_elapsed,      raw_yaw)
    gc_yaw_kalman.plot(t_elapsed,   k_yaw)

    # ── FFT on pitch ──
    pitch_buffer.append(raw_pitch)
    if len(pitch_buffer) > BUFFER_SIZE:
        pitch_buffer.pop(0)

    if len(pitch_buffer) == BUFFER_SIZE:
        data_fft   = np.array(pitch_buffer) - np.mean(pitch_buffer)
        fft_vals   = np.fft.fft(data_fft)
        freqs      = np.fft.fftfreq(len(data_fft), dt)
        idx        = np.where(freqs > 0)
        peak_freq  = freqs[idx][np.argmax(np.abs(fft_vals[idx]))]
        fft_lbl.text = f"FFT Peak: {peak_freq:.2f} Hz"

    # ── Servo mapping ──
    pan  = max(-90, min(90,  k_yaw))
    tilt = max(-45, min(45,  k_pitch))

    # ── HUD labels ──
    roll_lbl.text  = f"Roll: {k_roll:+.1f}°"
    pitch_lbl.text  = f"Pitch : {tilt:+.1f}°  → Servo2 (Tilt) = {tilt+45:.0f}°"
    yaw_lbl.text   = f"Yaw :  {pan:+.1f}°  → Servo1 (Pan) = {pan+90:.0f}°"
    info_lbl.text  = (f"Kalman  Pitch:{k_pitch:.1f}" f"Roll:{k_roll:.1f}  Yaw:{k_yaw:.1f}")
    raw_lbl.text   = (f"RAW ==>  Pitch:{raw_pitch:.1f}" f"Roll:{raw_roll:.1f}  Yaw:{raw_yaw:.1f}")