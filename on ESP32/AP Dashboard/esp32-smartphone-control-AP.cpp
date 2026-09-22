// >>> ACTIVE FIRMWARE - this is the only source file PlatformIO builds
// (platformio.ini: src_dir = "on ESP32/AP Dashboard"). Edit directly here.
/*
  ESP32-S3 + Pan-Tilt Servo System — Access Point + Web Dashboard
  -----------------------------------------------------------------
  - ESP32-S3 broadcasts its OWN Wi-Fi (AP_SSID / AP_PASSWORD), no home router needed
  - Serves a mobile-friendly dashboard at http://192.168.4.1/ (or http://pantilt.local/)
  - Two control modes:
      AUTO   -> polls a phone's motion-sensor server app for pitch/roll/yaw (as before)
      MANUAL -> drag the on-page joystick to drive pan/tilt directly
  - Same LPF -> deadband -> rate-limiter pipeline as esp32-smartphone-control-2.cpp

  Required libraries (same as before, WebServer/ESPmDNS ship with the ESP32 core):
    - ArduinoJson
    - ESP32Servo
*/

#include <WiFi.h>
#include <WebServer.h>
#include <ESPmDNS.h>
#include <HTTPClient.h>
#include <ArduinoJson.h>
#include <ESP32Servo.h>

// ================= Access Point =================
const char* AP_SSID_C     = "ESP32-S3";
const char* AP_PASSWORD_C = "87654321";

// Phone must run the same motion-sensor server app as before, and be the ONLY
// device connected to this AP. ESP32's DHCP server normally hands the first
// client 192.168.4.2 — if your phone shows a different IP, edit this line.
const char* motionURL = "http://192.168.4.2:8080/get?pitch&roll&yaw";

WebServer server(80);

// ================= Servo objects =================
Servo servoPan;
Servo servoTilt;

// ================= Servo pins =================
const int PAN_SERVO_PIN  = 4;
const int TILT_SERVO_PIN = 6;

// ================= Servo limits =================
const int PAN_MIN   = 0;
const int PAN_MAX   = 180;
const int TILT_MIN  = 30;
const int TILT_MAX  = 150;

// ================= Servo center =================
const int PAN_CENTER  = 90;
const int TILT_CENTER = 90;

// ================= Motion ranges (AUTO mode) =================
const float YAW_MIN_INPUT   = -90.0;
const float YAW_MAX_INPUT   =  90.0;
const float PITCH_MIN_INPUT = -45.0;
const float PITCH_MAX_INPUT =  45.0;

// ================= Filtering =================
float filteredPanInput  = 0.0;
float filteredTiltInput = 0.0;
const float alpha = 0.10;

// ================= Deadband =================
const float PAN_DEADBAND_DEG  = 1.5;
const float TILT_DEADBAND_DEG = 1.5;
float heldPanInput  = 0.0;
float heldTiltInput = 0.0;

// ================= Rate limiter =================
const float PAN_MAX_RATE_DPS  = 90.0;
const float TILT_MAX_RATE_DPS = 60.0;
const unsigned long servoInterval = 20;
const float MAX_DT = 0.1;

float targetPanAngle    = PAN_CENTER;
float targetTiltAngle   = TILT_CENTER;
float currentPanAngle   = PAN_CENTER;
float currentTiltAngle  = TILT_CENTER;
int lastPanWritten  = PAN_CENTER;
int lastTiltWritten = TILT_CENTER;

// ================= Timing =================
unsigned long lastRead = 0;
const unsigned long readInterval = 100;
unsigned long lastServoUpdate = 0;

// ================= Mode + manual control =================
enum ControlMode { MODE_AUTO, MODE_MANUAL };
volatile ControlMode currentMode = MODE_AUTO;
float manualPanAngle  = PAN_CENTER;
float manualTiltAngle = TILT_CENTER;

// ================= Latest raw values (for dashboard) =================
float lastRawPitch = 0, lastRawRoll = 0, lastRawYaw = 0;

// ================= Utility =================
float lowPassFilter(float oldValue, float newValue, float a) {
  return oldValue + a * (newValue - oldValue);
}
float clampFloat(float x, float lo, float hi) {
  if (x < lo) return lo;
  if (x > hi) return hi;
  return x;
}
float mapFloat(float x, float in_min, float in_max, float out_min, float out_max) {
  float ratio = (x - in_min) / (in_max - in_min);
  return out_min + ratio * (out_max - out_min);
}
float applyDeadband(float input, float &held, float threshold) {
  if (fabsf(input - held) >= threshold) held = input;
  return held;
}
float rateLimit(float current, float target, float maxRate, float dt) {
  float maxStep = maxRate * dt;
  float diff = target - current;
  if (diff >  maxStep) diff =  maxStep;
  if (diff < -maxStep) diff = -maxStep;
  return current + diff;
}
void setPanTilt(int panAngle, int tiltAngle) {
  panAngle  = constrain(panAngle, PAN_MIN, PAN_MAX);
  tiltAngle = constrain(tiltAngle, TILT_MIN, TILT_MAX);
  servoPan.write(panAngle);
  servoTilt.write(tiltAngle);
}
void centerServos() { setPanTilt(PAN_CENTER, TILT_CENTER); }

// ================= Access Point =================
void startAccessPoint() {
  WiFi.mode(WIFI_AP);
  WiFi.softAP(AP_SSID_C, AP_PASSWORD_C);
  delay(100);

  Serial.println("Access Point started");
  Serial.print("SSID: "); Serial.println(AP_SSID_C);
  Serial.print("Dashboard: http://"); Serial.println(WiFi.softAPIP());

  if (MDNS.begin("pantilt")) {
    Serial.println("Dashboard (mDNS): http://pantilt.local/");
  }
}

// ================= AUTO mode: motion processing =================
void driveServosFromMotion(float pitch, float yaw) {
  filteredPanInput  = lowPassFilter(filteredPanInput, yaw, alpha);
  filteredTiltInput = lowPassFilter(filteredTiltInput, pitch, alpha);

  float panInput  = applyDeadband(filteredPanInput,  heldPanInput,  PAN_DEADBAND_DEG);
  float tiltInput = applyDeadband(filteredTiltInput, heldTiltInput, TILT_DEADBAND_DEG);

  targetPanAngle = clampFloat(
    mapFloat(panInput, YAW_MIN_INPUT, YAW_MAX_INPUT, PAN_MIN, PAN_MAX), PAN_MIN, PAN_MAX);
  targetTiltAngle = clampFloat(
    mapFloat(tiltInput, PITCH_MIN_INPUT, PITCH_MAX_INPUT, TILT_MIN, TILT_MAX), TILT_MIN, TILT_MAX);
}

void updateServos(float dt) {
  currentPanAngle  = rateLimit(currentPanAngle,  targetPanAngle,  PAN_MAX_RATE_DPS,  dt);
  currentTiltAngle = rateLimit(currentTiltAngle, targetTiltAngle, TILT_MAX_RATE_DPS, dt);

  int panServoAngle  = (int)lroundf(currentPanAngle);
  int tiltServoAngle = (int)lroundf(currentTiltAngle);

  if (panServoAngle != lastPanWritten || tiltServoAngle != lastTiltWritten) {
    setPanTilt(panServoAngle, tiltServoAngle);
    lastPanWritten  = panServoAngle;
    lastTiltWritten = tiltServoAngle;
  }
}

void readSmartphoneMotion() {
  // Skip entirely if no phone is connected yet - avoids blocking the
  // dashboard on an HTTP timeout with nobody there to answer.
  if (WiFi.softAPgetStationNum() == 0) return;

  HTTPClient http;
  http.begin(motionURL);
  http.setTimeout(500);

  int httpCode = http.GET();
  if (httpCode == HTTP_CODE_OK) {
    String payload = http.getString();
    StaticJsonDocument<512> doc;
    DeserializationError error = deserializeJson(doc, payload);
    if (error) { http.end(); return; }

    float pitch = doc["buffer"]["pitch"]["buffer"][0] | 0.0;
    float roll  = doc["buffer"]["roll"]["buffer"][0]  | 0.0;
    float yaw   = doc["buffer"]["yaw"]["buffer"][0]   | 0.0;

    lastRawPitch = pitch;
    lastRawRoll  = roll;
    lastRawYaw   = yaw;

    driveServosFromMotion(pitch, yaw);
  }
  http.end();
}

// ================= Web handlers =================
const char INDEX_HTML[] PROGMEM = R"HTMLPAGE(
<!DOCTYPE html>
<html lang="th">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1, user-scalable=no">
<title>Pan-Tilt Dashboard</title>
<style>
  * { box-sizing: border-box; -webkit-tap-highlight-color: transparent; }
  body { margin:0; padding:16px; background:#0f1115; color:#e8e8e8;
    font-family:-apple-system,Segoe UI,Roboto,sans-serif; text-align:center; }
  h1 { font-size:18px; margin:4px 0 12px; color:#7cc4ff; }
  .status { font-size:13px; color:#999; margin-bottom:14px; }
  .status span { color:#7cc4ff; }
  .cards { display:grid; grid-template-columns:repeat(3,1fr); gap:8px; margin-bottom:14px; }
  .card { background:#1a1d24; border-radius:10px; padding:10px 4px; }
  .card .label { font-size:11px; color:#999; }
  .card .value { font-size:20px; font-weight:600; margin-top:2px; }
  .servos { display:grid; grid-template-columns:1fr 1fr; gap:8px; margin-bottom:16px; }
  #joy { touch-action:none; background:#1a1d24; border-radius:50%; margin:8px auto; display:block; }
  .modebar { display:flex; gap:8px; justify-content:center; margin:14px 0; flex-wrap:wrap; }
  button { background:#22262f; color:#e8e8e8; border:1px solid #333a45; border-radius:8px;
    padding:10px 14px; font-size:14px; }
  button.active { background:#2f6fed; border-color:#2f6fed; }
  .mode-label { font-size:14px; margin-top:6px; color:#7cc4ff; }
</style>
</head>
<body>
  <h1>ESP32-S3 Pan-Tilt Dashboard</h1>
  <div class="status">เชื่อมต่อ: <span id="clients">-</span> เครื่อง | โหมด: <span id="modeText">-</span></div>

  <div class="cards">
    <div class="card"><div class="label">Pitch</div><div class="value" id="pitch">0.0</div></div>
    <div class="card"><div class="label">Roll</div><div class="value" id="roll">0.0</div></div>
    <div class="card"><div class="label">Yaw</div><div class="value" id="yaw">0.0</div></div>
  </div>

  <div class="servos">
    <div class="card"><div class="label">Pan (servo)</div><div class="value" id="panAngle">90</div></div>
    <div class="card"><div class="label">Tilt (servo)</div><div class="value" id="tiltAngle">90</div></div>
  </div>

  <canvas id="joy" width="240" height="240"></canvas>
  <div class="mode-label">ลากวงกลมด้านบนเพื่อควบคุมด้วยมือ</div>

  <div class="modebar">
    <button id="btnAuto">เอียงมือถือ (Auto)</button>
    <button id="btnCenter">กึ่งกลาง</button>
  </div>

<script>
  const canvas = document.getElementById('joy');
  const ctx = canvas.getContext('2d');
  const R = 100, CX = 120, CY = 120, KNOB_R = 26;
  let knob = { x: CX, y: CY };
  let dragging = false;
  let lastSend = 0;

  function draw() {
    ctx.clearRect(0, 0, 240, 240);
    ctx.beginPath(); ctx.arc(CX, CY, R, 0, Math.PI * 2);
    ctx.fillStyle = '#12141a'; ctx.fill();
    ctx.strokeStyle = '#333a45'; ctx.stroke();
    ctx.beginPath(); ctx.arc(knob.x, knob.y, KNOB_R, 0, Math.PI * 2);
    ctx.fillStyle = dragging ? '#2f6fed' : '#4a5568'; ctx.fill();
  }

  function pointFromEvent(e) {
    const rect = canvas.getBoundingClientRect();
    return { x: e.clientX - rect.left, y: e.clientY - rect.top };
  }

  function setKnob(x, y) {
    let dx = x - CX, dy = y - CY;
    const dist = Math.sqrt(dx * dx + dy * dy);
    if (dist > R) { dx = dx * R / dist; dy = dy * R / dist; }
    knob.x = CX + dx; knob.y = CY + dy;
    draw();

    const now = Date.now();
    if (now - lastSend > 100) {
      lastSend = now;
      const nx = dx / R, ny = dy / R;
      const pan = 90 + nx * 90;
      const tilt = 90 - ny * 60;
      fetch('/control?pan=' + pan.toFixed(1) + '&tilt=' + tilt.toFixed(1));
    }
  }

  canvas.addEventListener('pointerdown', e => { dragging = true; const p = pointFromEvent(e); setKnob(p.x, p.y); });
  canvas.addEventListener('pointermove', e => { if (dragging) { const p = pointFromEvent(e); setKnob(p.x, p.y); } });
  window.addEventListener('pointerup', () => { dragging = false; draw(); });

  document.getElementById('btnAuto').onclick = () => fetch('/auto');
  document.getElementById('btnCenter').onclick = () => { knob = { x: CX, y: CY }; draw(); fetch('/center'); };

  async function poll() {
    try {
      const r = await fetch('/data');
      const d = await r.json();
      document.getElementById('pitch').textContent = d.rawPitch.toFixed(1);
      document.getElementById('roll').textContent  = d.rawRoll.toFixed(1);
      document.getElementById('yaw').textContent   = d.rawYaw.toFixed(1);
      document.getElementById('panAngle').textContent  = d.panAngle.toFixed(0);
      document.getElementById('tiltAngle').textContent = d.tiltAngle.toFixed(0);
      document.getElementById('clients').textContent = d.clients;
      document.getElementById('modeText').textContent = d.mode === 'auto' ? 'เอียงมือถือ' : 'จอยสติ๊ก';
      document.getElementById('btnAuto').className = d.mode === 'auto' ? 'active' : '';
    } catch (e) {}
  }

  draw();
  setInterval(poll, 250);
</script>
</body>
</html>
)HTMLPAGE";

void handleRoot()    { server.send_P(200, "text/html", INDEX_HTML); }

void handleData() {
  StaticJsonDocument<256> doc;
  doc["mode"]       = (currentMode == MODE_AUTO) ? "auto" : "manual";
  doc["clients"]    = WiFi.softAPgetStationNum();
  doc["rawPitch"]   = lastRawPitch;
  doc["rawRoll"]    = lastRawRoll;
  doc["rawYaw"]     = lastRawYaw;
  doc["targetPan"]  = targetPanAngle;
  doc["targetTilt"] = targetTiltAngle;
  doc["panAngle"]   = currentPanAngle;
  doc["tiltAngle"]  = currentTiltAngle;

  String out;
  serializeJson(doc, out);
  server.send(200, "application/json", out);
}

void handleControl() {
  if (server.hasArg("pan"))  manualPanAngle  = constrain(server.arg("pan").toFloat(),  (float)PAN_MIN,  (float)PAN_MAX);
  if (server.hasArg("tilt")) manualTiltAngle = constrain(server.arg("tilt").toFloat(), (float)TILT_MIN, (float)TILT_MAX);
  currentMode = MODE_MANUAL;
  server.send(200, "application/json", "{\"ok\":true}");
}

void handleAuto()   { currentMode = MODE_AUTO; server.send(200, "application/json", "{\"ok\":true}"); }
void handleCenter() {
  manualPanAngle  = PAN_CENTER;
  manualTiltAngle = TILT_CENTER;
  currentMode = MODE_MANUAL;
  server.send(200, "application/json", "{\"ok\":true}");
}
void handleNotFound() { server.send(404, "text/plain", "Not found"); }

// ================= Setup =================
void setup() {
  Serial.begin(115200);
  delay(1000);

  servoPan.setPeriodHertz(50);
  servoTilt.setPeriodHertz(50);
  servoPan.attach(PAN_SERVO_PIN, 500, 2400);
  servoTilt.attach(TILT_SERVO_PIN, 500, 2400);
  centerServos();

  startAccessPoint();

  server.on("/", handleRoot);
  server.on("/data", handleData);
  server.on("/control", handleControl);
  server.on("/auto", handleAuto);
  server.on("/center", handleCenter);
  server.onNotFound(handleNotFound);
  server.begin();

  lastServoUpdate = millis();

  Serial.println("ESP32-S3 Pan-Tilt AP + Dashboard ready");
}

// ================= Loop =================
void loop() {
  server.handleClient();

  unsigned long now = millis();

  if (now - lastServoUpdate >= servoInterval) {
    float dt = (now - lastServoUpdate) / 1000.0;
    if (dt > MAX_DT) dt = MAX_DT;
    lastServoUpdate = now;

    if (currentMode == MODE_MANUAL) {
      targetPanAngle  = manualPanAngle;
      targetTiltAngle = manualTiltAngle;
    }
    updateServos(dt);
  }

  if (currentMode == MODE_AUTO && now - lastRead >= readInterval) {
    lastRead = now;
    readSmartphoneMotion();
  }
}