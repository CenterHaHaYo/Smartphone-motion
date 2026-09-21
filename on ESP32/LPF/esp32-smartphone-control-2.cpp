/*
  ESP32 + Smartphone Motion + 2-DOF Pan-Tilt Servo System
  --------------------------------------------------------
  Function:
    - Reads smartphone motion from HTTP
    - Uses:
        Pan  = Yaw
        Tilt = Pitch
    - Controls two servos smoothly

  Signal pipeline (per axis):
    raw angle -> low-pass filter -> deadband -> map to servo angle
              -> limit to servo range -> rate limiter -> servo.write()

  Smartphone endpoint:
    http://192.168.1.8:8080/get?pitch&roll&yaw

  Required libraries:
    - WiFi.h
    - HTTPClient.h
    - ArduinoJson
    - ESP32Servo

  Install from Arduino Library Manager:
    - ArduinoJson
    - ESP32Servo
*/

#include <WiFi.h>
#include <HTTPClient.h>
#include <ArduinoJson.h>
#include <ESP32Servo.h>

// ================= Wi-Fi =================
const char* ssid     = WIFI_SSID;
const char* password = WIFI_PASSWORD;

// ================= Smartphone motion URL =================
const char* motionURL = "http://172.20.10.1/get?pitch&roll&yaw";

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

// ================= Motion ranges =================
// Pan uses yaw
const float YAW_MIN_INPUT   = -90.0;
const float YAW_MAX_INPUT   =  90.0;

// Tilt uses pitch
const float PITCH_MIN_INPUT = -45.0;
const float PITCH_MAX_INPUT =  45.0;

// ================= Filtering =================
float filteredPanInput  = 0.0;   // from yaw
float filteredTiltInput = 0.0;   // from pitch
const float alpha = 0.15;        // lower = smoother

// ================= Deadband =================

const float PAN_DEADBAND_DEG  = 1.5;   // input degrees (yaw)
const float TILT_DEADBAND_DEG = 1.5;   // input degrees (pitch)

float heldPanInput  = 0.0;   // last accepted yaw value
float heldTiltInput = 0.0;   // last accepted pitch value

// ================= Rate limiter =================
const float PAN_MAX_RATE_DPS  = 90.0;
const float TILT_MAX_RATE_DPS = 60.0;

// Servo output is updated more often than the HTTP read so the ramp is smooth
const unsigned long servoInterval = 20;   // ms (50 Hz)
const float MAX_DT = 0.1;                 // s, cap after a blocking HTTP call

float targetPanAngle    = PAN_CENTER;    
float targetTiltAngle   = TILT_CENTER;
float currentPanAngle   = PAN_CENTER;    
float currentTiltAngle  = TILT_CENTER;

int lastPanWritten  = PAN_CENTER;
int lastTiltWritten = TILT_CENTER;

// ================= Timing =================
unsigned long lastRead = 0;
const unsigned long readInterval = 100; // ms
unsigned long lastServoUpdate = 0;

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

// Hold-last deadband: output changes only if input moved >= threshold from held value
float applyDeadband(float input, float &held, float threshold) {
  if (fabsf(input - held) >= threshold) {
    held = input;
  }
  return held;
}

// Move current toward target by at most maxRate * dt
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

void centerServos() {
  setPanTilt(PAN_CENTER, TILT_CENTER);
}

// ================= Wi-Fi =================
void connectWiFi() {
  WiFi.mode(WIFI_STA);
  WiFi.begin(ssid, password);

  Serial.print("Connecting to Wi-Fi");
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }

  Serial.println();
  Serial.println("Wi-Fi connected");
  Serial.print("ESP32 IP address: ");
  Serial.println(WiFi.localIP());
}

// ================= Motion processing =================
// Runs at HTTP rate (readInterval): filter -> deadband -> map -> new target
void driveServosFromMotion(float pitch, float yaw) {
  // 2-DOF definition
  // Pan  = Yaw
  // Tilt = Pitch

  filteredPanInput  = lowPassFilter(filteredPanInput, yaw, alpha);
  filteredTiltInput = lowPassFilter(filteredTiltInput, pitch, alpha);

  float panInput  = applyDeadband(filteredPanInput,  heldPanInput,  PAN_DEADBAND_DEG);
  float tiltInput = applyDeadband(filteredTiltInput, heldTiltInput, TILT_DEADBAND_DEG);

  targetPanAngle = clampFloat(
    mapFloat(panInput, YAW_MIN_INPUT, YAW_MAX_INPUT, PAN_MIN, PAN_MAX),
    PAN_MIN, PAN_MAX
  );

  targetTiltAngle = clampFloat(
    mapFloat(tiltInput, PITCH_MIN_INPUT, PITCH_MAX_INPUT, TILT_MIN, TILT_MAX),
    TILT_MIN, TILT_MAX
  );

  Serial.print("Filtered Pan(Yaw): ");
  Serial.print(filteredPanInput, 2);
  Serial.print(" deg");

  Serial.print(" | Filtered Tilt(Pitch): ");
  Serial.print(filteredTiltInput, 2);
  Serial.print(" deg");

  Serial.print(" | TargetPan: ");
  Serial.print(targetPanAngle, 1);

  Serial.print(" | TargetTilt: ");
  Serial.println(targetTiltAngle, 1);
}

// Runs at servoInterval: rate limiter -> servo.write()
void updateServos(float dt) {
  currentPanAngle  = rateLimit(currentPanAngle,  targetPanAngle,  PAN_MAX_RATE_DPS,  dt);
  currentTiltAngle = rateLimit(currentTiltAngle, targetTiltAngle, TILT_MAX_RATE_DPS, dt);

  int panServoAngle  = (int)lroundf(currentPanAngle);
  int tiltServoAngle = (int)lroundf(currentTiltAngle);

  // Write only when the integer angle actually changes
  if (panServoAngle != lastPanWritten || tiltServoAngle != lastTiltWritten) {
    setPanTilt(panServoAngle, tiltServoAngle);
    lastPanWritten  = panServoAngle;
    lastTiltWritten = tiltServoAngle;
  }
}

// ================= HTTP read =================
void readSmartphoneMotion() {
  if (WiFi.status() != WL_CONNECTED) {
    Serial.println("Wi-Fi disconnected. Reconnecting...");
    connectWiFi();
    return;
  }

  HTTPClient http;
  http.begin(motionURL);
  http.setTimeout(1000);

  int httpCode = http.GET();

  if (httpCode > 0) {
    if (httpCode == HTTP_CODE_OK) {
      String payload = http.getString();

      StaticJsonDocument<512> doc;
      DeserializationError error = deserializeJson(doc, payload);

      if (error) {
        Serial.print("JSON parse failed: ");
        Serial.println(error.c_str());
        http.end();
        return;
      }

      float pitch = doc["buffer"]["pitch"]["buffer"][0] | 0.0;
      float roll  = doc["buffer"]["roll"]["buffer"][0]  | 0.0;
      float yaw   = doc["buffer"]["yaw"]["buffer"][0]   | 0.0;

      Serial.print("Pitch: ");
      Serial.print(pitch, 2);
      Serial.print(" deg | Roll: ");
      Serial.print(roll, 2);
      Serial.print(" deg | Yaw: ");
      Serial.print(yaw, 2);
      Serial.println(" deg");

      // 2-DOF Pan-Tilt mapping
      driveServosFromMotion(pitch, yaw);

    } else {
      Serial.print("HTTP error code: ");
      Serial.println(httpCode);
    }
  } else {
    Serial.print("HTTP GET failed: ");
    Serial.println(http.errorToString(httpCode));
  }

  http.end();
}

// ================= Setup =================
void setup() {
  Serial.begin(115200);
  delay(1000);

  // Servo setup
  servoPan.setPeriodHertz(50);
  servoTilt.setPeriodHertz(50);

  servoPan.attach(PAN_SERVO_PIN, 500, 2400);
  servoTilt.attach(TILT_SERVO_PIN, 500, 2400);

  centerServos();

  connectWiFi();

  lastServoUpdate = millis();

  Serial.println("ESP32 Smartphone Pan-Tilt Servo System Ready");
  Serial.println("Pan = Yaw, Tilt = Pitch");
}

// ================= Loop =================
void loop() {
  unsigned long now = millis();

  // Fast tick: rate-limited servo motion
  if (now - lastServoUpdate >= servoInterval) {
    float dt = (now - lastServoUpdate) / 1000.0;
    if (dt > MAX_DT) dt = MAX_DT;   // avoid a big jump after a blocking HTTP call
    lastServoUpdate = now;
    updateServos(dt);
  }

  // Slow tick: read smartphone motion, update targets
  if (now - lastRead >= readInterval) {
    lastRead = now;
    readSmartphoneMotion();
  }
}