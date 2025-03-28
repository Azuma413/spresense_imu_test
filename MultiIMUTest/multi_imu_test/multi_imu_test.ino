#include "multi_imu.hpp"
#include "ism330dhcx.hpp"
#include "serial_send.hpp"
#include "madgwick_filter.hpp"
#include <math.h>

Madgwick MadgwickFilter;
MultiIMU imu;
// ISM330DHCX imu;
float data[MultiIMU::DATA_LENGTH];
SerialSend serial_send(3);
float send_list[3]; // Roll, Pitch, Yaw

void setup() {
    Serial.begin(115200);
    if (imu.begin()) {
        imu.startSensing(960, 16, 4000);
    }
    serial_send.init();
    MadgwickFilter.begin(100);//フィルタのサンプリング 100Hz
}

void loop() {
    if (imu.getData(data)) {
        // 姿勢の更新 rad/s, m/s^2
        MadgwickFilter.updateIMU(data[3], data[4], data[5], data[0], data[1], data[2]);
        send_list[0] = MadgwickFilter.getRoll();
        send_list[1] = MadgwickFilter.getPitch();
        send_list[2] = MadgwickFilter.getYaw();
        serial_send.send(send_list);
    }
}
