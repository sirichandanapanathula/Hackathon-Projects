#define FSR_PIN A0
#define PRESSURE_PIN A1
#define FLEX_PIN A2

int fsrValue;
int pressureValue;
int flexValue;

void setup() {
  Serial.begin(9600);
}

void loop() {

  // Read sensor values
  fsrValue = analogRead(FSR_PIN);
  pressureValue = analogRead(PRESSURE_PIN);
  flexValue = analogRead(FLEX_PIN);

  Serial.println("------ Sensor Readings ------");

  // FSR SENSOR
  Serial.print("FSR Value: ");
  Serial.print(fsrValue);

  if (fsrValue < 10) {
    Serial.println(" - No Pressure");
  }
  else if (fsrValue < 50) {
    Serial.println(" - Light Touch");
  }
  else if (fsrValue < 200) {
    Serial.println(" - Medium Pressure");
  }
  else {
    Serial.println(" - Strong Pressure");
  }

  // PRESSURE SENSOR
  Serial.print("Pressure Sensor Value: ");
  Serial.print(pressureValue);

  if (pressureValue < 100) {
    Serial.println(" - Low Pressure");
  }
  else if (pressureValue < 400) {
    Serial.println(" - Moderate Pressure");
  }
  else if (pressureValue < 700) {
    Serial.println(" - High Pressure");
  }
  else {
    Serial.println(" - Extreme Pressure");
  }

  // FLEX SENSOR
  Serial.print("Flex Sensor Value: ");
  Serial.print(flexValue);

  if (flexValue < 300) {
    Serial.println(" - Finger Straight");
  }
  else if (flexValue < 600) {
    Serial.println(" - Slight Bend");
  }
  else {
    Serial.println(" - Fully Bent");
  }

  Serial.println("------------------------------");

  delay(1000);
}