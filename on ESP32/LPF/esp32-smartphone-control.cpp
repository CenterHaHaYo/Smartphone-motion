/*
  ESP32 + Smartphone Motion + 2-DOF Pan-Tilt Servo System
  --------------------------------------------------------
  Function:
    - Reads smartphone motion from HTTP
    - Uses:
        Pan  = Yaw
        Tilt = Pitch
    - Controls two servos smoothly

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
#include "secrets.h"

// ================= Wi-Fi =================
// SSID/password come from include/secrets.h, which is gitignored.
// Copy include/secrets.h.example to include/secrets.h and fill in your own values.
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

// ================= Timing =================
unsigned long lastRead = 0;
const unsigned long readInterval = 100; // ms

// ================= Utility =================
float lowPassFilter(float oldValue, float newValue, float a) {
  return oldValue + a * (newValue - oldValue);
}

int mapFloatToInt(float x, float in_min, float in_max, int out_min, int out_max) {
  float ratio = (x - in_min) / (in_max - in_min);
  float value = out_min + ratio * (out_max - out_min);
  return (int)value;
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
void driveServosFromMotion(float pitch, float yaw) {
  // 2-DOF definition
  // Pan  = Yaw
  // Tilt = Pitch

  filteredPanInput  = lowPassFilter(filteredPanInput, yaw, alpha);
  filteredTiltInput = lowPassFilter(filteredTiltInput, pitch, alpha);

  int panServoAngle = mapFloatToInt(
    filteredPanInput,
    YAW_MIN_INPUT, YAW_MAX_INPUT,
    PAN_MIN, PAN_MAX
  );

  int tiltServoAngle = mapFloatToInt(
    filteredTiltInput,
    PITCH_MIN_INPUT, PITCH_MAX_INPUT,
    TILT_MIN, TILT_MAX
  );

  panServoAngle  = constrain(panServoAngle, PAN_MIN, PAN_MAX);
  tiltServoAngle = constrain(tiltServoAngle, TILT_MIN, TILT_MAX);

  setPanTilt(panServoAngle, tiltServoAngle);

  Serial.print("Filtered Pan(Yaw): ");
  Serial.print(filteredPanInput, 2);
  Serial.print(" deg");

  Serial.print(" | Filtered Tilt(Pitch): ");
  Serial.print(filteredTiltInput, 2);
  Serial.print(" deg");

  Serial.print(" | ServoPan: ");
  Serial.print(panServoAngle);

  Serial.print(" | ServoTilt: ");
  Serial.println(tiltServoAngle);
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

  Serial.println("ESP32 Smartphone Pan-Tilt Servo System Ready");
  Serial.println("Pan = Yaw, Tilt = Pitch");
}

// ================= Loop =================
void loop() {
  unsigned long now = millis();

  if (now - lastRead >= readInterval) {
    lastRead = now;
    readSmartphoneMotion();
  }
}