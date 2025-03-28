#include "multi_imu.hpp"
#include "ism330dhcx.hpp"
#include "serial_send.hpp"
#include "madgwick_filter.hpp"
#include <math.h>

Madgwick MadgwickFilter;
MultiIMU imu;
// ISM330DHCX imu;
float data[MultiIMU::DATA_LENGTH];
SerialSend serial_send(MultiIMU::DATA_LENGTH);
float send_list[6] = {0,0,0,0,0,0}; // Roll, Pitch, Yaw, X, Y, Z
float velocity[3] = {0, 0, 0};  // X, Y, Z velocity
float last_update = 0;          // 前回の更新時刻
const float gravity = 9.80665;     // 重力加速度

// 加速度を世界座標系に変換する関数
void convertToWorldFrame(float ax, float ay, float az, float roll, float pitch, float yaw, float* world_acc) {
    // ラジアンに変換
    float phi = roll * M_PI / 180.0;
    float theta = pitch * M_PI / 180.0;
    float psi = yaw * M_PI / 180.0;
    
    // 回転行列を使って座標変換
    world_acc[0] = ax * (cos(theta) * cos(psi)) +
                   ay * (sin(phi) * sin(theta) * cos(psi) - cos(phi) * sin(psi)) +
                   az * (cos(phi) * sin(theta) * cos(psi) + sin(phi) * sin(psi));
    
    world_acc[1] = ax * (cos(theta) * sin(psi)) +
                   ay * (sin(phi) * sin(theta) * sin(psi) + cos(phi) * cos(psi)) +
                   az * (cos(phi) * sin(theta) * sin(psi) - sin(phi) * cos(psi));
    
    world_acc[2] = ax * (-sin(theta)) +
                   ay * (sin(phi) * cos(theta)) +
                   az * (cos(phi) * cos(theta));
    
    // 重力成分を除去
    world_acc[2] -= gravity;
}

void setup() {
    Serial.begin(115200);
    if (imu.begin()) {
        imu.startSensing(960, 16, 4000);
    }
    serial_send.init();
    MadgwickFilter.begin(100);//フィルタのサンプリング 100Hz
    last_update = millis() / 1000.0;
}

void loop() {
    if (imu.getData(data)) {
        float current_time = millis() / 1000.0;
        float dt = current_time - last_update;

        // 姿勢の更新 rad/s, m/s^2
        MadgwickFilter.updateIMU(data[3], data[4], data[5], data[0], data[1], data[2]);
        send_list[0] = MadgwickFilter.getRoll();
        send_list[1] = MadgwickFilter.getPitch();
        send_list[2] = MadgwickFilter.getYaw();

        // 加速度を世界座標系に変換
        float world_acc[3];
        convertToWorldFrame(data[0], data[1], data[2], 
                          send_list[0], send_list[1], send_list[2],
                          world_acc);

        // 速度と位置の更新（台形積分）
        for (int i = 0; i < 3; i++) {
            velocity[i] += world_acc[i] * dt;
            send_list[3+i] += velocity[i] * dt + 0.5 * world_acc[i] * dt * dt;
        }

        // 結果の出力
        Serial.print("Position (m) - X: ");
        Serial.print(world_acc[0]);
        // Serial.print(send_list[3]);
        Serial.print(" Y: ");
        Serial.print(world_acc[1]);
        // Serial.print(send_list[4]);
        Serial.print(" Z: ");
        Serial.println(world_acc[2]);
        // Serial.println(send_list[5]);

        last_update = current_time;
    }
}
