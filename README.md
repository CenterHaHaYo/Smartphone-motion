# Smartphone Motion — Real-Time RPY Visualizer

Turn a smartphone into a wireless orientation sensor.

These Python scripts read live attitude data from the **[Phyphox](https://phyphox.org/)**
app over Wi-Fi, calibrate a zero pose on the first frame, and animate a 3-D phone model
in **VPython** that follows the real device in real time. On top of that they compute
Roll / Pitch / Yaw, optionally smooth them with a Kalman filter, run an FFT to find the
dominant vibration frequency, log data to CSV, and preview the servo angles a 2-DOF
pan-tilt rig would need to mirror the motion.

```
 📱 Phone (Phyphox)  ──HTTP──>  Python  ──>  quaternion math  ──>  VPython 3-D scene
      Attitude sensor              │                                HUD + graphs
      w x y z, pitch roll yaw      └──>  Kalman / FFT  ──>  CSV + servo angles
```

### Attitude with Euler angles from Phyphox

| Symbol | Angle | Physical meaning |
|:--:|:--|:--|
| $\psi$ | Yaw | rotating the phone about the vertical axis |
| $\theta$ | Pitch | tilting the phone forward / backward |
| $\phi$ | Roll | tipping the phone left / right |

---

## Features

- **Real-time 3-D visualization** — a detailed phone model (body, screen, notch, camera bump) rotated by the live sensor quaternion.
- **One-key calibration** — press `C` at any time to make the current pose the new zero.
- **Kalman filtering** — an independent 1-D filter per axis smooths noisy readings, with raw-vs-filtered comparison graphs.
- **FFT analysis** — a rolling 128-sample window reports the dominant motion frequency on each axis.
- **CSV logging** — a one-click 10-second recording saved under `dataset/`.
- **Pan-tilt servo preview** — yaw/pitch mapped into servo ranges, plus a full 2-DOF gimbal simulator.

---

## Requirements

- Python 3.8 or newer
- A smartphone with the **Phyphox** app installed ([iOS](https://apps.apple.com/app/phyphox/id1127319693) / [Android](https://play.google.com/store/apps/details?id=de.rwth_aachen.phyphox))
- Phone and computer on the **same network**

```bash
pip install vpython requests numpy
```

That is the complete dependency list — everything else is Python's standard library.

---

## Setup

### 1. Start the sensor on the phone

1. Open **Phyphox** and select the **Attitude** experiment.
2. Tap the ⋮ menu → **Allow remote access**.
3. Phyphox displays an address such as `http://192.168.1.43:8080`. Write it down.
4. Press ▶ to start the measurement and leave the screen on.

### 2. Point the script at your phone

Every script has a `URL` constant near the top. **This is the one line you must edit.**

```python
URL = "http://192.168.1.43:8080"   # ← replace with the address Phyphox shows
```

The scripts currently ship with two different defaults, depending on how each one was
last used, so make sure you edit the file you are actually going to run:

| Script | Current default `URL` | Typical setup |
|:--|:--|:--|
| `RPY/2DOF_motion.py` | `http://172.20.10.1` | iPhone personal hotspot (port 80) |
| `RPY/FFT/smartphone_motionV1.py` | `http://172.20.10.1` | iPhone personal hotspot (port 80) |
| `RPY/FFT/smartphone_motionV2_fft.py` | `http://192.168.1.43:8080` | shared Wi-Fi LAN (port 8080) |
| `RPY/Kalman Filter/smMotionV2.py` | `http://192.168.1.43:8080` | shared Wi-Fi LAN (port 8080) |
| `RPY/Kalman Filter/smartphone_motionV2.py` | `http://172.20.10.1` | iPhone personal hotspot (port 80) |

One other constant is worth knowing about: `SAMPLE_RATE = 30` sets the polling rate in
Hz, and should roughly match the sensor rate configured in Phyphox. It also defines the
FFT frequency resolution (see [How it works](#how-it-works)).

### 3. Run

The `Kalman Filter` directory name contains a space, so quote the path:

```bash
python "RPY/Kalman Filter/smMotionV2.py"
```

VPython opens the scene in your default browser at `http://localhost:7000`. Hold the
phone flat and still for the first second — that pose becomes the reference orientation.

---

## Project structure

```
Smartphone-motion/
└── RPY/
    ├── 2DOF_motion.py                  Pan-tilt gimbal rig simulator
    ├── FFT/
    │   ├── smartphone_motionV1.py      Minimal baseline — simple box + pitch FFT
    │   └── smartphone_motionV2_fft.py  Detailed model + windowed FFT + CSV
    └── Kalman Filter/
        ├── smartphone_motionV2.py      Kalman added, model driven by filtered Euler
        └── smMotionV2.py               ★ Recommended — quaternion model, 3-axis Kalman + FFT + CSV
```

---

## Which script should I run?

All five share the same acquisition pipeline; they differ in what they do with the data.

| | 3-D model | Kalman | FFT | Graphs | CSV | `[C]` recalibrate |
|:--|:--:|:--:|:--:|:--:|:--:|:--:|
| `FFT/smartphone_motionV1.py` | plain box | — | pitch only | — | — | — |
| `FFT/smartphone_motionV2_fft.py` | detailed | — | pitch, windowed | RPY + spectrum | ✔ | ✔ |
| `Kalman Filter/smartphone_motionV2.py` | plain box | 3 axes | pitch only | raw vs filtered ×3 | — | — |
| **`Kalman Filter/smMotionV2.py`** | detailed | 3 axes | 3 axes, windowed | raw vs filtered ×3 | ✔ | ✔ |
| `2DOF_motion.py` | pan-tilt rig | — | — | pan & tilt | — | ✔ |

**Start with `RPY/Kalman Filter/smMotionV2.py`.** It is the most complete version and
the only one that combines a smooth quaternion-driven model with filtering, three-axis
spectral analysis and recording.

`2DOF_motion.py` is a different subject: instead of a phone it draws an articulated
pan-tilt camera gimbal (base, pole, pan body, tilt platform, camera and aim ray) and
drives its two joints from the phone's yaw and pitch — a preview of how a real
two-servo rig would behave.

```bash
python "RPY/Kalman Filter/smMotionV2.py"        # recommended
python "RPY/FFT/smartphone_motionV2_fft.py"     # FFT focus, no filtering
python RPY/2DOF_motion.py                       # pan-tilt rig simulator
```

---

## Controls

| Action | How |
|:--|:--|
| Recalibrate (set current pose as zero) | press `C`, or click **↺ Reset Calibration** |
| Record 10 seconds to CSV | click **⏺ Record 10s** |
| Orbit / zoom the 3-D scene | drag with the right mouse button / scroll |

Recalibration also resets the Kalman filters to the current raw angle, so the graphs
do not show a spike after re-zeroing.

---

## How it works

### 1. Data acquisition

Phyphox's remote-access server exposes measurement buffers over a single HTTP endpoint.
All five scripts issue exactly one request per frame:

```python
r = requests.get(f"{URL}/get?w&x&y&z&pitch&roll&yaw", timeout=1)
buf = r.json()["buffer"]
qw = buf["w"]["buffer"][-1]     # …and x, y, z, pitch, roll, yaw
```

Seven channels are requested: the orientation **quaternion** `w x y z` and the
**Euler angles** `pitch roll yaw`. Only the newest sample (`[-1]`) of each buffer is
used, so the loop always reflects the current pose rather than replaying history.
If the request fails the scene shows `⚠ No signal — check Wi-Fi / server` and keeps
retrying.

### 2. Quaternion pipeline

The phone's sensor frame is not VPython's frame, so the components are remapped, and
the orientation is then expressed *relative* to a calibration snapshot taken on the
first frame:

```python
q_cur = (qw_s, -qx_s, qz_s, -qy_s)       # sensor frame → VPython frame
if q0 is None: q0 = q_cur                # first frame = reference pose
q_rel = quat_mul(quat_conj(q0), q_cur)   # q_rel = q0⁻¹ ⊗ q_cur
```

Vectors are rotated with the sandwich product $q \cdot (0,\mathbf{v}) \cdot q^{*}$,
written out by hand in `qrot()`, and applied to the model's two orientation vectors:

```python
phone.axis = vector(*qrot(q_rel, (1, 0, 0)))
phone.up   = vector(*qrot(q_rel, (0, 1, 0)))
```

Driving the model **straight from the quaternion** — never through Euler angles — is
what keeps `smMotionV2.py` stable at any attitude, including pitch near ±90° where an
Euler-based rotation would gimbal-lock. The Kalman output there is used only for the
HUD, the graphs and the CSV. (The axis remap above was tuned against an iPhone 13 Pro
Max; a different device may need a different permutation.)

### 3. Kalman filter

Each axis gets its own 1-D filter with state $x = [\text{angle}, \text{bias}]^T$ and
the angle as the measurement — the classic formulation from the MPU-6050/Arduino world.
There is no gyro rate input here, so the predicted rate is `0.0 - bias` and the filter
behaves as an adaptive low-pass that tracks the sensor while rejecting jitter.

| Axis | `Q_angle` | `Q_rate` | `R_measure` |
|:--|:--:|:--:|:--:|
| Pitch | 0.001 | 0.003 | 0.03 |
| Roll | 0.001 | 0.003 | 0.01 |
| Yaw | 0.001 | 0.003 | 0.05 |

Tuning rule of thumb: **raise `Q`** for a faster but noisier response; **raise `R`** for
a smoother but laggier one. Yaw uses the largest `R` because it is the noisiest channel.

### 4. FFT

A rolling buffer of `BUFFER_SIZE = 128` samples at 30 Hz gives a 4.27 s window,
a resolution of Δf ≈ 0.23 Hz, and a Nyquist limit of 15 Hz. The newer scripts apply a
Hann window and drop the DC bin:

```python
def compute_fft(buf):
    arr   = np.asarray(buf, dtype=float) - np.mean(buf)
    win   = np.hanning(len(arr))
    mags  = np.abs(np.fft.rfft(arr * win))
    freqs = np.fft.rfftfreq(len(arr), DT)
    return freqs[1:], mags[1:]
```

The bin with the largest magnitude is reported as the dominant frequency in the
**FFT Peak** box — wave the phone at a steady rhythm and it reads back your pace in Hz.

### 5. Servo mapping

The angles are clamped to a hobby servo's travel and biased into an unsigned range:

```python
pan  = max(-90.0, min(90.0, yaw))      # → Servo 1 = pan  + 90   (0–180°)
tilt = max(-45.0, min(45.0, pitch))    # → Servo 2 = tilt + 45   (0–90°)
```

> **Note:** these values are **display only**. Nothing is transmitted to hardware —
> driving real servos is on the roadmap below.

---

## CSV output

Available in `smMotionV2.py` and `smartphone_motionV2_fft.py`. Clicking **⏺ Record 10s**
creates the `dataset/` folder if needed and writes:

```
dataset/motion_YYYYmmdd_HHMMSS.csv
```

Recording runs for `RECORD_DURATION = 10.0` seconds ≈ 300 rows at 30 Hz, flushed row by
row so the file survives an interrupted run.

| Column | Contents |
|:--|:--|
| `time_s` | elapsed time since the script started, seconds |
| `roll_deg`, `pitch_deg`, `yaw_deg` | processed angles (Kalman-filtered in `smMotionV2.py`) |
| `raw_pitch_s`, `raw_roll_s`, `raw_yaw_s` | unprocessed values straight from Phyphox |

---

## Troubleshooting

| Symptom | Likely cause and fix |
|:--|:--|
| `⚠ No signal — check Wi-Fi / server` | Wrong `URL`; remote access not enabled; phone and PC on different networks; the measurement is paused in Phyphox. |
| Connection works in a browser but not in Python | Check the port — hotspot setups often use port 80, LAN setups 8080. A firewall may be blocking outbound requests from Python. |
| Nothing opens in the browser | VPython serves the scene at `http://localhost:7000`; open it manually if the browser does not launch. |
| The model points the wrong way | Press `C` while the phone is flat and still. If a whole axis is inverted, your device needs a different remap in `q_cur`. |
| Motion is jerky or laggy | Lower `SAMPLE_RATE`, move closer to the router, or close other Phyphox-heavy apps. |
| Readings jitter too much | Raise `R_measure` in the Kalman constructors (see the table above). |

---

## Known Limitations

Issues found while reviewing the code. They are **documented here, not yet fixed** —
each one is a good first contribution.

- **CSV yaw column is wrong** — `RPY/Kalman Filter/smMotionV2.py` writes `k_roll` into
  both the `roll_deg` and `yaw_deg` columns, so recorded yaw duplicates roll and the
  filtered yaw (`k_yaw`) is never saved.
- **Incorrect quaternion conjugate** — `RPY/Kalman Filter/smartphone_motionV2.py`
  defines `quat_conj()` as `(-w, -x, -y, -z)`. The `w` component should not be negated;
  every other script correctly returns `(w, -x, -y, -z)`.
- **Gimbal lock in the older Kalman script** — the same file rotates the 3-D model from
  the *filtered Euler angles* re-converted to a quaternion, which misbehaves near
  pitch ±90°. `smMotionV2.py` avoids this by using `q_rel` directly.
- **Unbounded graph growth** — `RPY/Kalman Filter/smartphone_motionV2.py` calls
  `gcurve.plot()` every frame with no point cap, so memory use and redraw cost keep
  rising the longer it runs. The newer scripts update `gcurve.data` over a fixed window
  instead.
- **Stale header comment in `RPY/2DOF_motion.py`** — it describes an
  "Acceleration with g" sensor with pan following *roll*, but the code actually reads
  the quaternion and maps pan ← yaw, tilt ← pitch.
- **Unused import** — `import math` in `RPY/FFT/smartphone_motionV1.py` is never used.
- **Errors are silently swallowed** — `read_sensor()` catches every exception bare, so a
  typo in `URL`, a paused experiment and a single dropped packet all look identical.
- **Configuration is hard-coded** — the `URL` must be edited in the source; there are no
  command-line arguments or config file, and the five scripts duplicate most of their
  logic.

---

## Roadmap

- [ ] Fix the CSV yaw column and the `quat_conj` sign
- [ ] Drive real servos over serial / ESP32 instead of only displaying the angles
- [ ] Merge the five variants into one script with command-line flags
  (`--url`, `--kalman`, `--fft`, `--record`)
- [ ] Add a `requirements.txt` and auto-discover the phone on the network
- [ ] Replay mode: feed a recorded CSV back through the visualizer

---

## License

No license file is present in this repository yet, so all rights are reserved by
default. If you intend others to reuse the code, add a `LICENSE` file.
