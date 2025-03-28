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
LowPassFilter lpf(20.0f);           // ローパスフィルタ（カットオフ周波数20Hz）
KalmanFilter kf(0.001f, 0.01f, 0.001f, 0.05f);  // カルマンフィルタ
ZeroVelocityDetector zvd(0.1f, 0.2f, 20);  // 静止状態検出器

// データ配列
float data[MultiIMU::DATA_LENGTH];           // センサーデータ
SerialSend serial_send(MultiIMU::DATA_LENGTH);
float send_list[6] = {0,0,0,0,0,0};         // Roll, Pitch, Yaw, X, Y, Z
float filtered_acc[3] = {0, 0, 0};          // フィルタ後の加速度
float world_acc[3] = {0, 0, 0};             // 世界座標系の加速度
float last_update = 0;                       // 前回の更新時刻
const float gravity = 9.80665;               // 重力加速度

// 加速度を世界座標系に変換し、重力を補正する関数
void convertToWorldFrame(float ax, float ay, float az, float roll, float pitch, float yaw, float* world_acc) {
    // エラーチェック
    if (isnan(ax) || isnan(ay) || isnan(az) || 
        isnan(roll) || isnan(pitch) || isnan(yaw)) {
        world_acc[0] = world_acc[1] = world_acc[2] = 0.0f;
        return;
    }

    // ラジアンに変換
    float phi = roll * M_PI / 180.0;
    float theta = pitch * M_PI / 180.0;
    float psi = yaw * M_PI / 180.0;
    
    // DCM（Direction Cosine Matrix）の計算
    float R[3][3];
    
    // DCMの要素を計算
    R[0][0] = cos(theta) * cos(psi);
    R[0][1] = cos(theta) * sin(psi);
    R[0][2] = -sin(theta);
    
    R[1][0] = sin(phi) * sin(theta) * cos(psi) - cos(phi) * sin(psi);
    R[1][1] = sin(phi) * sin(theta) * sin(psi) + cos(phi) * cos(psi);
    R[1][2] = sin(phi) * cos(theta);
    
    R[2][0] = cos(phi) * sin(theta) * cos(psi) + sin(phi) * sin(psi);
    R[2][1] = cos(phi) * sin(theta) * sin(psi) - sin(phi) * cos(psi);
    R[2][2] = cos(phi) * cos(theta);
    
    // 重力ベクトルを計算
    float gravity_body[3];
    gravity_body[0] = -gravity * sin(theta);
    gravity_body[1] = gravity * sin(phi) * cos(theta);
    gravity_body[2] = gravity * cos(phi) * cos(theta);
    
    // 重力を補正した加速度を計算
    float acc_body[3] = {ax, ay, az};
    float acc_compensated[3];
    for (int i = 0; i < 3; i++) {
        acc_compensated[i] = acc_body[i] - gravity_body[i];
    }
    
    // 世界座標系に変換
    for (int i = 0; i < 3; i++) {
        world_acc[i] = 0;
        for (int j = 0; j < 3; j++) {
            world_acc[i] += R[i][j] * acc_compensated[j];
        }
    }
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

        // エラーチェック
        bool has_nan = false;
        for (int i = 0; i < MultiIMU::DATA_LENGTH; i++) {
            if (isnan(data[i])) {
                has_nan = true;
                break;
            }
        }
        if (has_nan) {
            return;
        }

        // 4. 静止状態の検出と処理
        if (zvd.isStatic(&filtered_acc[0], &data[3])) {
            // 静止状態では速度のみをリセット
            float current_pos[3], current_vel[3], current_bias[3];
            kf.getState(current_pos, current_vel, current_bias);
            
            // 位置とバイアスは保持したまま、速度のみゼロにしてリセット
            for (int i = 0; i < 3; i++) {
                current_vel[i] = 0.0f;
            }
            
            // カルマンフィルタを再初期化（位置とバイアスは保持）
            kf.reset();
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

        // エラーチェック
        bool valid_data = true;
        for (int i = 0; i < 6; i++) {
            if (isnan(send_list[i]) || isinf(send_list[i])) {
                valid_data = false;
                break;
            }
        }

        if (valid_data) {
            // 結果の出力
            Serial.print("Position (m) - X: ");
            Serial.print(send_list[3]);
            Serial.print(" Y: ");
            Serial.print(send_list[4]);
            Serial.print(" Z: ");
            Serial.println(send_list[5]);
        } else {
            Serial.println("Invalid data detected");
            // 状態をリセット
            kf.reset();
            zvd.reset();
            lpf.reset();
            for (int i = 0; i < 6; i++) {
                send_list[i] = 0.0f;
            }
        }

        last_update = current_time;
    }
}
