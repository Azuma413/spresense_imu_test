#include "multi_imu.hpp"
#include "ism330dhcx.hpp"
#include "serial_send.hpp"
#include "madgwick_filter.hpp"
#include "low_pass_filter.hpp"
#include "kalman_filter.hpp"
#include "zero_velocity_detector.hpp"
#include <math.h>

// センサーとフィルタのインスタンス
Madgwick MadgwickFilter;
MultiIMU imu;
LowPassFilter lpf(0.1f);            // ローパスフィルタ（カットオフ周波数を調整）
KalmanFilter kf(0.01f, 0.1f, 0.01f, 0.1f);  // カルマンフィルタ
ZeroVelocityDetector zvd(0.05f, 0.1f, 50);  // 静止状態検出器

// データ配列
float data[MultiIMU::DATA_LENGTH];           // センサーデータ
SerialSend serial_send(MultiIMU::DATA_LENGTH);
float send_list[6] = {0,0,0,0,0,0};         // Roll, Pitch, Yaw, X, Y, Z
float filtered_acc[3] = {0, 0, 0};          // フィルタ後の加速度
float world_acc[3] = {0, 0, 0};             // 世界座標系の加速度
float last_update = 0;                       // 前回の更新時刻
const float gravity = 9.80665;               // 重力加速度

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
    MadgwickFilter.begin(100);  // Madgwickフィルタのサンプリング 100Hz
    last_update = millis() / 1000.0;
}

void loop() {
    if (imu.getData(data)) {
        float current_time = millis() / 1000.0;
        float dt = current_time - last_update;

        // 1. 加速度データのローパスフィルタリング
        lpf.update(data, filtered_acc);

        // 2. 姿勢の更新（Madgwickフィルタ）
        MadgwickFilter.updateIMU(data[3], data[4], data[5], filtered_acc[0], filtered_acc[1], filtered_acc[2]);
        send_list[0] = MadgwickFilter.getRoll();
        send_list[1] = MadgwickFilter.getPitch();
        send_list[2] = MadgwickFilter.getYaw();

        // 3. 加速度を世界座標系に変換
        convertToWorldFrame(filtered_acc[0], filtered_acc[1], filtered_acc[2],
                          send_list[0], send_list[1], send_list[2],
                          world_acc);

        // 4. 静止状態の検出
        if (zvd.isStatic(&filtered_acc[0], &data[3])) {
            // 静止状態では速度をリセット
            float zero_velocity[3] = {0, 0, 0};
            float current_pos[3];
            kf.getState(current_pos, nullptr, nullptr);
            kf.reset();
            // 現在位置を保持
            for (int i = 0; i < 3; i++) {
                send_list[3+i] = current_pos[i];
            }
        } else {
            // 5. カルマンフィルタの更新
            kf.predict(dt);
            kf.update(world_acc);

            // 状態の取得
            float position[3], velocity[3];
            kf.getState(position, velocity, nullptr);

            // 位置の更新
            for (int i = 0; i < 3; i++) {
                send_list[3+i] = position[i];
            }
        }

        // 結果の出力
        Serial.print("Position (m) - X: ");
        Serial.print(send_list[3]);
        Serial.print(" Y: ");
        Serial.print(send_list[4]);
        Serial.print(" Z: ");
        Serial.println(send_list[5]);

        last_update = current_time;
    }
}
