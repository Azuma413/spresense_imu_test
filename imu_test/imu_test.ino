// void setup() {
//     pinMode(LED0, OUTPUT);
//     pinMode(LED1, OUTPUT);
//     pinMode(LED2, OUTPUT);
//     pinMode(LED3, OUTPUT);
// }

// void loop() {
//     digitalWrite(LED0, HIGH);
//     delay(100);
//     digitalWrite(LED1, HIGH);
//     delay(100);
//     digitalWrite(LED2, HIGH);
//     delay(100);
//     digitalWrite(LED3, HIGH);
//     delay(1000);

//     digitalWrite(LED0, LOW);
//     delay(100);
//     digitalWrite(LED1, LOW);
//     delay(100);
//     digitalWrite(LED2, LOW);
//     delay(100);
//     digitalWrite(LED3, LOW);
//     delay(1000);
// }
#include <Wire.h>
#include "SparkFun_ISM330DHCX.h"
#include "udp_send.hpp"

SparkFun_ISM330DHCX myISM1;
SparkFun_ISM330DHCX myISM2;
const char* ssid = "NotFreeWiFi";
const char* password = "924865hirekatsu";
#define WINDOW_SIZE 10 // Number of readings to average
// 保存用の変数
float accelX1[WINDOW_SIZE];
float accelY1[WINDOW_SIZE];
float accelZ1[WINDOW_SIZE];
float gyroX1[WINDOW_SIZE];
float gyroY1[WINDOW_SIZE];
float gyroZ1[WINDOW_SIZE];
float accelX2[WINDOW_SIZE];
float accelY2[WINDOW_SIZE];
float accelZ2[WINDOW_SIZE];
float gyroX2[WINDOW_SIZE];
float gyroY2[WINDOW_SIZE];
float gyroZ2[WINDOW_SIZE];
// 送信用の変数
float send_list[12];

int idx_count = 0;

// Structs for X,Y,Z data
sfe_ism_data_t accelData;
sfe_ism_data_t gyroData;
UDPSend udp_send(ssid, password, 12);

void setup()
{
  pinMode(LED0, OUTPUT);
  pinMode(LED1, OUTPUT);
  pinMode(LED2, OUTPUT);
  pinMode(LED3, OUTPUT);
	Wire.begin();

  // Uncomment the next line to use 400kHz I2C. Essential when running the accel and gyro at 416Hz or faster.
	Wire.setClock(400000);
	Serial.begin(115200);
  udp_send.init();

  // 0x60から0x6Fまでのアドレスを探す
  // for (int i = 0x60; i <= 0x6F; i++)
  // {
  //   Wire.beginTransmission(i);
  //   if (Wire.endTransmission() == 0)
  //   {
  //     Serial.print("Found device at 0x");
  //     Serial.println(i, HEX);
  //   }
  // }
	while (!myISM1.begin(0x6A))
	{
		Serial.println("0x6A did not begin...");
		delay(1000);
	}

	while (!myISM2.begin(0x6B))
	{
		Serial.println("0x6B did not begin...");
		delay(1000);
	}

    // Reset and configure first device
    myISM1.deviceReset();
    while (!myISM1.getDeviceReset())
    {
        delay(1);
    }
    myISM1.setDeviceConfig();
    myISM1.setBlockDataUpdate();
    myISM1.setAccelDataRate(ISM_XL_ODR_208Hz);
    myISM1.setAccelFullScale(ISM_4g);
    myISM1.setGyroDataRate(ISM_GY_ODR_104Hz);
    myISM1.setGyroFullScale(ISM_250dps);
    myISM1.setAccelFilterLP2();
    myISM1.setAccelSlopeFilter(ISM_LP_ODR_DIV_100);
    myISM1.setGyroFilterLP1();
    myISM1.setGyroLP1Bandwidth(ISM_MEDIUM);

    // Reset and configure second device
    myISM2.deviceReset();
    while (!myISM2.getDeviceReset())
    {
        delay(1);
    }
    myISM2.setDeviceConfig();
    myISM2.setBlockDataUpdate();
    myISM2.setAccelDataRate(ISM_XL_ODR_208Hz);
    myISM2.setAccelFullScale(ISM_4g);
    myISM2.setGyroDataRate(ISM_GY_ODR_104Hz);
    myISM2.setGyroFullScale(ISM_250dps);
    myISM2.setAccelFilterLP2();
    myISM2.setAccelSlopeFilter(ISM_LP_ODR_DIV_100);
    myISM2.setGyroFilterLP1();
    myISM2.setGyroLP1Bandwidth(ISM_MEDIUM);
}

void loop()
{
  // データを更新
  if (myISM1.checkAccelStatus())
  {
    digitalWrite(LED0, HIGH);
    myISM1.getAccel(&accelData);
    accelX1[idx_count] = accelData.xData;
    accelY1[idx_count] = accelData.yData;
    accelZ1[idx_count] = accelData.zData;
  }else{
    digitalWrite(LED0, LOW);
  }
  if (myISM1.checkGyroStatus())
  {
    digitalWrite(LED1, HIGH);
    myISM1.getGyro(&gyroData);
    gyroX1[idx_count] = gyroData.xData;
    gyroY1[idx_count] = gyroData.yData;
    gyroZ1[idx_count] = gyroData.zData;
  }else{
    digitalWrite(LED1, LOW);
  }
  if (myISM2.checkAccelStatus())
  {
    digitalWrite(LED2, HIGH);
    myISM2.getAccel(&accelData);
    accelX2[idx_count] = accelData.xData;
    accelY2[idx_count] = accelData.yData;
    accelZ2[idx_count] = accelData.zData;
  }else{
    digitalWrite(LED2, LOW);
  }
  if (myISM2.checkGyroStatus())
  {
    digitalWrite(LED3, HIGH);
    myISM2.getGyro(&gyroData);
    gyroX2[idx_count] = gyroData.xData;
    gyroY2[idx_count] = gyroData.yData;
    gyroZ2[idx_count] = gyroData.zData;
  }else{
    digitalWrite(LED3, LOW);
  }
  idx_count = (idx_count + 1)%WINDOW_SIZE;
  // それぞれの平均を計算して表示
  // send_listを初期化
  for (int i = 0; i < 12; i++)
  {
    send_list[i] = 0;
  }
  for (int i = 0; i < WINDOW_SIZE; i++)
  {
    send_list[0] += accelX1[i];
    send_list[1] += accelY1[i];
    send_list[2] += accelZ1[i];
    send_list[3] += gyroX1[i];
    send_list[4] += gyroY1[i];
    send_list[5] += gyroZ1[i];
    send_list[6] += accelX2[i];
    send_list[7] += accelY2[i];
    send_list[8] += accelZ2[i];
    send_list[9] += gyroX2[i];
    send_list[10] += gyroY2[i];
    send_list[11] += gyroZ2[i];
  }
  for (int i = 0; i < 12; i++)
  {
    send_list[i] /= WINDOW_SIZE;
  }
  udp_send.send(send_list);
  // Serial.print("Accel1: ");
  // Serial.print("X: ");
  // Serial.print(accelX1_sum / WINDOW_SIZE);
  // Serial.print(" ");
  // Serial.print("Y: ");
  // Serial.print(accelY1_sum / WINDOW_SIZE);
  // Serial.print(" ");
  // Serial.print("Z: ");
  // Serial.print(accelZ1_sum / WINDOW_SIZE);
  // Serial.println(" ");
  // Serial.print("Gyro1: ");
  // Serial.print("X: ");
  // Serial.print(gyroX1_sum / WINDOW_SIZE);
  // Serial.print(" ");
  // Serial.print("Y: ");
  // Serial.print(gyroY1_sum / WINDOW_SIZE);
  // Serial.print(" ");
  // Serial.print("Z: ");
  // Serial.println(gyroZ1_sum / WINDOW_SIZE);
  // Serial.print("Accel2: ");
  // Serial.print("X: ");
  // Serial.print(accelX2_sum / WINDOW_SIZE);
  // Serial.print(" ");
  // Serial.print("Y: ");
  // Serial.print(accelY2_sum / WINDOW_SIZE);
  // Serial.print(" ");
  // Serial.print("Z: ");
  // Serial.print(accelZ2_sum / WINDOW_SIZE);
  // Serial.println(" ");
  // Serial.print("Gyro2: ");
  // Serial.print("X: ");
  // Serial.print(gyroX2_sum / WINDOW_SIZE);
  // Serial.print(" ");
  // Serial.print("Y: ");
  // Serial.print(gyroY2_sum / WINDOW_SIZE);
  // Serial.print(" ");
  // Serial.print("Z: ");
  // Serial.println(gyroZ2_sum / WINDOW_SIZE);
  // Serial.println();
  delay(10);
}