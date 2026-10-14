/*
  Arduino sketch: receive simple serial commands to drive vibration motor
  Protocol (115200 baud, newline terminated):
    V:<0-100>  -> set vibration strength percent
    X          -> stop vibration

  Wiring example:
    - Use a transistor (e.g., NPN 2N2222) + flyback diode to drive vibration motor from 5V
    - PWM pin -> base via resistor (e.g., 1k)
    - Motor between +5V and transistor collector, emitter to GND; diode across motor
*/

const int PWM_PIN = 5;        // choose a PWM capable pin
int targetDuty = 0;           // 0..255
unsigned long lastCmdMs = 0;
const unsigned long FAILSAFE_MS = 3000; // stop if no command for 3s

String rx;

void setup() {
  pinMode(PWM_PIN, OUTPUT);
  analogWrite(PWM_PIN, 0);
  Serial.begin(115200);
}

void loop() {
  while (Serial.available() > 0) {
    char c = (char)Serial.read();
    if (c == '\n' || c == '\r') {
      handleLine(rx);
      rx = "";
    } else {
      rx += c;
      if (rx.length() > 64) rx = ""; // simple overflow guard
    }
  }

  // failsafe: stop motor if connection lost
  if (millis() - lastCmdMs > FAILSAFE_MS) {
    targetDuty = 0;
  }
  analogWrite(PWM_PIN, targetDuty);
}

void handleLine(String line) {
  line.trim();
  if (line.length() == 0) return;
  lastCmdMs = millis();

  if (line.charAt(0) == 'V') {
    int colon = line.indexOf(':');
    if (colon > 0) {
      int pct = line.substring(colon + 1).toInt();
      if (pct < 0) pct = 0;
      if (pct > 100) pct = 100;
      targetDuty = map(pct, 0, 100, 0, 255);
    }
    return;
  }
  if (line.charAt(0) == 'X') {
    targetDuty = 0;
    return;
  }
}



