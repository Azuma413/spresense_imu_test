#include "multi_imu.hpp"
#include "ism330dhcx.hpp"
#include "serial_send.hpp"
#include "madgwick_filter.hpp"

Madgwick MadgwickFilter;
MultiIMU imu;
float data[MultiIMU::DATA_LENGTH];
SerialSend serial_send(MultiIMU::DATA_LENGTH);
int16_t ax, ay, az;//加速度　int16_tは2バイトの符号付き整数
int16_t gx, gy, gz;//角速度　同上
float send_list[3]; // 3 values (ID, roll, pitch, yaw)

void setup() {
  if (imu.begin()) {
    imu.startSensing(960, 16, 4000);
  }
  serial_send.init();
  MadgwickFilter.begin(100);//フィルタのサンプリング 100Hz
}

void loop() {
  if (imu.getData(data)) {
    MadgwickFilter.updateIMU(data[0], data[1], data[2], data[3], data[4], data[5]);
    send_list[0] = MadgwickFilter.getRoll();
    send_list[1] = MadgwickFilter.getPitch();
    send_list[2]  = MadgwickFilter.getYaw();
    serial_send.send(data);
    // Serial.print("Roll: ");
    // Serial.print(send_list[0]);
    // Serial.print("Pitch: ");
    // Serial.print(send_list[1]);
    // Serial.print("Yaw: ");
    // Serial.println(send_list[2]);
  }
}
