#include <Wire.h>
#include "SparkFun_ISM330DHCX.h"
#include "udp_send.hpp"

SparkFun_ISM330DHCX myISM;
const char* ssid = "NotFreeWiFi";
const char* password = "924865hirekatsu";

// 右手 0
// 左手 1
// 右足 2
// 左足 3
#define ID 3
#define ACCEL_SCALE 2000
#define GYRO_SCALE 300000

// 送信用の変数
float send_list[2]; // 2 values (accel norm, gyro norm)

// Structs for X,Y,Z data
sfe_ism_data_t accelData;
sfe_ism_data_t gyroData;
UDPSend udp_send(ssid, password, 3, ID); // サイズを3に変更 (ID + AccelNorm + GyroNorm)

void setup()
{
  pinMode(LED0, OUTPUT);
  pinMode(LED1, OUTPUT);
	Wire.begin();

  // Uncomment the next line to use 400kHz I2C. Essential when running the accel and gyro at 416Hz or faster.
	Wire.setClock(400000);
	Serial.begin(115200);
  udp_send.init();

  // 0x60から0x6Fまでのアドレスを探す
  for (int i = 0x60; i <= 0x6F; i++)
  {
    Wire.beginTransmission(i);
    if (Wire.endTransmission() == 0)
    {
      Serial.print("Found device at 0x");
      Serial.println(i, HEX);
      while(!myISM.begin(i))
      {
        Serial.println("ISM330DHCX did not begin...");
        delay(1000);
      }
      break;
    }
  }

    // Reset and configure IMU device
    myISM.deviceReset();
    while (!myISM.getDeviceReset())
    {
        delay(1);
    }
    myISM.setDeviceConfig();
    myISM.setBlockDataUpdate();
    myISM.setAccelDataRate(ISM_XL_ODR_208Hz);
    myISM.setAccelFullScale(ISM_4g);
    myISM.setGyroDataRate(ISM_GY_ODR_104Hz);
    myISM.setGyroFullScale(ISM_250dps);
    myISM.setAccelFilterLP2();
    myISM.setAccelSlopeFilter(ISM_LP_ODR_DIV_100);
    myISM.setGyroFilterLP1();
    myISM.setGyroLP1Bandwidth(ISM_MEDIUM);
}

void loop()
{
  // データを更新
  if (myISM.checkAccelStatus())
  {
    digitalWrite(LED0, HIGH);
    myISM.getAccel(&accelData);
  }else{
    digitalWrite(LED0, LOW);
  }
  
  if (myISM.checkGyroStatus())
  {
    digitalWrite(LED1, HIGH);
    myISM.getGyro(&gyroData);
  }else{
    digitalWrite(LED1, LOW);
  }
  
  // ノルム（大きさ）を計算して送信 (IDはUDPSendクラス内で追加される)
  float accelNorm = sqrt(accelData.xData * accelData.xData + 
                         accelData.yData * accelData.yData + 
                         accelData.zData * accelData.zData);
  
  float gyroNorm = sqrt(gyroData.xData * gyroData.xData + 
                        gyroData.yData * gyroData.yData + 
                        gyroData.zData * gyroData.zData);
  
  send_list[0] = accelNorm / ACCEL_SCALE;
  send_list[1] = gyroNorm / GYRO_SCALE;
  
  udp_send.send(send_list);
  
  // Serial.print("ID: ");
  // Serial.println(ID);
  // Serial.print("Accel: ");
  // Serial.print("X: ");
  // Serial.print(accelData.xData);
  // Serial.print(" Y: ");
  // Serial.print(accelData.yData);
  // Serial.print(" Z: ");
  // Serial.println(accelData.zData);
  // Serial.print("Gyro: ");
  // Serial.print("X: ");
  // Serial.print(gyroData.xData);
  // Serial.print(" Y: ");
  // Serial.print(gyroData.yData);
  // Serial.print(" Z: ");
  // Serial.println(gyroData.zData);
  // Serial.println();
  
  delay(10);
}
