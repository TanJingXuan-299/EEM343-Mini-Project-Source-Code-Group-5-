/*
  EEM343 6-DOF robot arm Arduino controller

  Works with eem343_robot_arm_planner.py option 4.

  Hardware from the programming guide:
    - Arduino Nano
    - PCA9685 servo driver at I2C address 0x40
    - HC-05 Bluetooth on SoftwareSerial pins 2(RX), 7(TX)
    - Servo frequency: 50 Hz

  Emergency stop:
    - Connect a normally-open push button between D11 and GND.
    - D11 uses INPUT_PULLUP, so pressing the button pulls D11 LOW.
    - A software stop can also be sent with command S.
    - By default, stop holds the last servo command instead of cutting PWM.
      Use a real power switch to isolate servo power in a true emergency.
    - After an emergency stop, send R to reset the stopped state.

  Serial/Bluetooth commands:
    H
      Move to guide home position.

    F,us1,us2,us3,us4,us5,us6,dt_ms
      Apply one pre-calculated frame immediately.

    C
      Clear the queued trajectory buffer.

    Q,us1,us2,us3,us4,us5,us6,dt_ms
      Add one pre-calculated frame to the queued trajectory buffer.

    G
      Start playing the queued trajectory with Arduino-side timing.

    S
      Software emergency stop. Servo pulses are disabled.

    R
      Reset stopped state. The PC app streams smooth home motion separately.

    ?
      Print status.
*/

#include <Wire.h>
#include <Adafruit_PWMServoDriver.h>
#include <SoftwareSerial.h>

Adafruit_PWMServoDriver pwm = Adafruit_PWMServoDriver(0x40);
SoftwareSerial Bluetooth(2, 7);  // Arduino RX, TX for HC-05 TX, RX

const uint16_t SERVO_FREQ = 50;
const uint8_t SERVO_COUNT = 6;
const uint8_t SERVO_CHANNEL[SERVO_COUNT] = {0, 1, 2, 3, 4, 5};

const uint16_t SERVO_MIN_US[SERVO_COUNT] = {500, 500, 500, 500, 500, 500};
const uint16_t SERVO_MAX_US[SERVO_COUNT] = {2500, 2500, 2500, 2500, 2500, 2500};

// From the programming guide: pwm.setPWM(0..5, 0, 324/319/131/287/299/307).
// Converted to microseconds with count * 20000 / 4096 at 50 Hz.
const uint16_t HOME_US[SERVO_COUNT] = {1582, 1557, 639, 1401, 1459, 1499};

const uint8_t ESTOP_PIN = 11;
const uint8_t AUX_OUT_1 = 4;
const uint8_t AUX_OUT_2 = 5;
const uint8_t AUX_OUT_3 = 6;
const bool DISABLE_PWM_ON_STOP = false;

char serialBuffer[96];
char bluetoothBuffer[96];
uint8_t serialIndex = 0;
uint8_t bluetoothIndex = 0;

struct MotionFrame {
  uint16_t us[SERVO_COUNT];
  uint16_t dtMs;
};

const uint8_t QUEUE_CAPACITY = 32;
MotionFrame frameQueue[QUEUE_CAPACITY];
uint8_t queueHead = 0;
uint8_t queueTail = 0;
uint8_t queueCount = 0;
bool queuePlaying = false;
bool queuePlaybackArmed = false;
uint32_t nextFrameDueMs = 0;

bool stopped = false;
bool physicalStopLatched = false;
uint32_t frameCounter = 0;

uint16_t microsecondsToCount(uint16_t us) {
  return (uint32_t)us * 4096UL / 20000UL;
}

void disableServoPulses() {
  for (uint8_t i = 0; i < SERVO_COUNT; i++) {
    pwm.setPWM(SERVO_CHANNEL[i], 0, 0);
  }
}

void triggerStop() {
  stopped = true;
  queuePlaying = false;
  queuePlaybackArmed = false;
  if (DISABLE_PWM_ON_STOP) {
    disableServoPulses();
  }
}

void checkPhysicalStop() {
  if (digitalRead(ESTOP_PIN) == LOW) {
    physicalStopLatched = true;
    triggerStop();
  }
}

void setServoMicroseconds(uint8_t servoIndex, uint16_t us) {
  us = constrain(us, SERVO_MIN_US[servoIndex], SERVO_MAX_US[servoIndex]);
  pwm.setPWM(SERVO_CHANNEL[servoIndex], 0, microsecondsToCount(us));
}

void clearFrameQueue() {
  queueHead = 0;
  queueTail = 0;
  queueCount = 0;
  queuePlaying = false;
  queuePlaybackArmed = false;
}

bool enqueueFrame(const uint16_t us[], uint16_t dtMs) {
  if (queueCount >= QUEUE_CAPACITY) {
    return false;
  }

  for (uint8_t i = 0; i < SERVO_COUNT; i++) {
    frameQueue[queueTail].us[i] = constrain(us[i], SERVO_MIN_US[i], SERVO_MAX_US[i]);
  }
  frameQueue[queueTail].dtMs = max((uint16_t)1, dtMs);
  queueTail = (queueTail + 1) % QUEUE_CAPACITY;
  queueCount++;
  if (queuePlaybackArmed && !queuePlaying) {
    queuePlaying = true;
    nextFrameDueMs = millis();
  }
  return true;
}

bool popFrame(MotionFrame &frame) {
  if (queueCount == 0) {
    return false;
  }

  frame = frameQueue[queueHead];
  queueHead = (queueHead + 1) % QUEUE_CAPACITY;
  queueCount--;
  return true;
}

void applyFrame(const uint16_t us[]) {
  for (uint8_t i = 0; i < SERVO_COUNT; i++) {
    setServoMicroseconds(i, us[i]);
  }
  frameCounter++;
}

void updateQueuedMotion() {
  if (stopped || !queuePlaying) {
    return;
  }

  uint32_t nowMs = millis();
  if ((int32_t)(nowMs - nextFrameDueMs) < 0) {
    return;
  }

  MotionFrame frame;
  if (!popFrame(frame)) {
    queuePlaying = false;
    return;
  }

  applyFrame(frame.us);
  nextFrameDueMs = nowMs + frame.dtMs;
}

void moveHome() {
  if (stopped) {
    return;
  }

  clearFrameQueue();
  for (uint8_t i = 0; i < SERVO_COUNT; i++) {
    setServoMicroseconds(i, HOME_US[i]);
  }
}

void printStatus(Stream &reply) {
  reply.print(stopped ? "STOPPED" : "READY");
  reply.print(",frames=");
  reply.print(frameCounter);
  reply.print(",pin11=");
  reply.print(digitalRead(ESTOP_PIN) == LOW ? "LOW" : "HIGH");
  reply.print(",physical=");
  reply.print(physicalStopLatched ? "YES" : "NO");
  reply.print(",queue=");
  reply.print(queueCount);
  reply.print(",playing=");
  reply.print(queuePlaying ? "YES" : "NO");
  reply.print(",armed=");
  reply.println(queuePlaybackArmed ? "YES" : "NO");
}

void resetStop(Stream &reply) {
  stopped = false;
  physicalStopLatched = false;
  clearFrameQueue();
  frameCounter = 0;
  reply.println("RESET_OK");
}

bool parseFramePayload(char *payload, uint16_t us[], uint16_t &dtMs, Stream &reply) {
  char *token = strtok(payload, ",");
  for (uint8_t i = 0; i < SERVO_COUNT; i++) {
    if (token == NULL) {
      reply.println("ERR,SHORT_FRAME");
      return false;
    }
    us[i] = (uint16_t)atoi(token);
    token = strtok(NULL, ",");
  }

  if (token == NULL) {
    reply.println("ERR,NO_DT");
    return false;
  }
  dtMs = (uint16_t)atoi(token);
  return true;
}

void processFrame(char *line, Stream &reply) {
  checkPhysicalStop();

  if (line[0] == '?') {
    printStatus(reply);
    return;
  }

  if (line[0] == 'S') {
    triggerStop();
    reply.println("STOPPED,SOFTWARE");
    return;
  }

  if (line[0] == 'R') {
    if (digitalRead(ESTOP_PIN) == LOW) {
      stopped = true;
      physicalStopLatched = true;
      reply.println("ESTOP,PIN11_STILL_LOW");
      return;
    }
    resetStop(reply);
    return;
  }

  if (stopped) {
    reply.println(physicalStopLatched ? "ESTOP,PHYSICAL" : "STOPPED,SOFTWARE");
    return;
  }

  if (line[0] == 'C') {
    clearFrameQueue();
    reply.println("OK,CLEAR");
    return;
  }

  if (line[0] == 'G') {
    if (queueCount == 0) {
      reply.println("ERR,QUEUE_EMPTY");
      return;
    }
    queuePlaybackArmed = true;
    queuePlaying = true;
    nextFrameDueMs = millis();
    reply.print("OK,PLAY,queue=");
    reply.println(queueCount);
    return;
  }

  if (line[0] == 'H') {
    moveHome();
    reply.println("OK,HOME");
    return;
  }

  if ((line[0] != 'F' && line[0] != 'Q') || line[1] != ',') {
    reply.println("ERR,BAD_CMD");
    return;
  }

  uint16_t us[SERVO_COUNT];
  uint16_t dtMs = 0;
  if (!parseFramePayload(line + 2, us, dtMs, reply)) {
    return;
  }

  if (line[0] == 'Q') {
    if (!enqueueFrame(us, dtMs)) {
      reply.print("ERR,Q_FULL,queue=");
      reply.println(queueCount);
      return;
    }
    reply.print("OKQ,queue=");
    reply.println(queueCount);
    return;
  }

  clearFrameQueue();
  applyFrame(us);
  reply.print("OK,");
  reply.println(frameCounter);
}

void readStream(Stream &stream, char *buffer, uint8_t &index) {
  while (stream.available() > 0) {
    char c = (char)stream.read();
    if (c == '\r') {
      continue;
    }

    if (c == '\n') {
      buffer[index] = '\0';
      if (index > 0) {
        processFrame(buffer, stream);
      }
      index = 0;
      continue;
    }

    if (index < 95) {
      buffer[index++] = c;
    } else {
      index = 0;
      stream.println("ERR,LINE_TOO_LONG");
    }
  }
}

void setup() {
  Serial.begin(115200);
  Bluetooth.begin(9600);

  pwm.begin();
  pwm.setOscillatorFrequency(27000000);
  pwm.setPWMFreq(SERVO_FREQ);
  delay(10);

  pinMode(AUX_OUT_1, OUTPUT);
  pinMode(AUX_OUT_2, OUTPUT);
  pinMode(AUX_OUT_3, OUTPUT);
  pinMode(ESTOP_PIN, INPUT_PULLUP);

  moveHome();
  delay(1000);
  Serial.println("READY");
  Bluetooth.println("READY");
}

void loop() {
  checkPhysicalStop();
  updateQueuedMotion();
  readStream(Serial, serialBuffer, serialIndex);
  readStream(Bluetooth, bluetoothBuffer, bluetoothIndex);
}
