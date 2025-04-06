#include "multi_imu.hpp"
#include "ism330dhcx.hpp"
#include "serial_send.hpp"
#include "madgwick_filter.hpp"
#include "kalman_filter.hpp"
#include "zero_velocity_detector.hpp"
#include <math.h>

// センサーとフィルタのインスタンス
Madgwick MadgwickFilter;
MultiIMU imu;
// プロセスノイズパラメータを調整：より小さな値にして位置ドリフトを抑制
KalmanFilter kf(0.00005f, 0.0002f, 0.00005f, 0.01f);  // カルマンフィルタのノイズパラメータを微調整
// 静止状態検出の閾値を調整：より感度を上げて静止状態をしっかり検出
ZeroVelocityDetector zvd(0.12f, 0.10f, 10);  // 静止状態検出器の閾値を調整

// データ配列
float data[MultiIMU::DATA_LENGTH];           // センサーデータ
// SerialSend serial_send(MultiIMU::DATA_LENGTH);
float send_list[6] = {0,0,0,0,0,0};         // Roll, Pitch, Yaw, X(cm), Y(cm), Z(cm)
float world_acc[3] = {0, 0, 0};             // 世界座標系の加速度
float last_update = 0;                       // 前回の更新時刻
const float gravity = 9.80665;               // 重力加速度
const int INIT_SAMPLES = 200;               // 初期化用サンプル数を増やして安定性向上
int init_counter = 0;                       // 初期化カウンタ
float init_acc[3] = {0};                    // 初期化用加速度累積
float init_gyro[3] = {0};                   // 初期化用ジャイロ累積
float last_vel[3] = {0, 0, 0};              // 前回の速度（異常な加速を検出するため）

void setup() {
    Serial.begin(115200);
    if (imu.begin()) {
        imu.startSensing(960, 16, 4000);
    }
    // serial_send.init();
    MadgwickFilter.begin(960);  // Madgwickフィルタのサンプリング 960Hz
    last_update = millis() / 1000.0;

    // IMUの初期化待ち - より長い初期化時間
    delay(200);
}

void loop() {
    if (imu.getData(data)) {
        // 初期化フェーズ
        if (init_counter < INIT_SAMPLES) {
            // 初期加速度とジャイロの平均を計算
            for (int i = 0; i < 3; i++) {
                init_acc[i] += data[i] / INIT_SAMPLES;
                init_gyro[i] += data[i+3] / INIT_SAMPLES;
            }
            init_counter++;
            if (init_counter == INIT_SAMPLES) {
                // 初期バイアスを設定
                float initial_bias[3];
                for (int i = 0; i < 3; i++) {
                    initial_bias[i] = init_acc[i];
                }
                // バイアス設定前にフィルタをリセット
                kf.reset();
                kf.setState(nullptr, nullptr, initial_bias);
                
                Serial.println("Initialization complete");
            }
            return;  // 初期化中は位置計算をスキップ
        }

        float current_time = millis() / 1000.0;
        float dt = current_time - last_update;
        
        // 異常なタイムステップをチェック
        if (dt > 0.1f || dt <= 0.0f) {
            dt = 0.01f; // 異常な値の場合は標準値を使用
        }
        
        // 姿勢の更新（Madgwickフィルタ）
        MadgwickFilter.updateIMU(data[3], data[4], data[5], data[0], data[1], data[2]);
        send_list[0] = MadgwickFilter.getRoll();
        send_list[1] = MadgwickFilter.getPitch();
        send_list[2] = MadgwickFilter.getYaw();

        // 加速度を世界座標系に変換（クォータニオンベース）- 改良版メソッドが使われる
        MadgwickFilter.convertAccelToWorldFrame(data[0], data[1], data[2], world_acc);

        // エラーチェック
        bool has_nan = false;
        for (int i = 0; i < MultiIMU::DATA_LENGTH; i++) {
            if (isnan(data[i])) {
                has_nan = true;
                Serial.println("NaN detected in sensor data");
                break;
            }
        }
        
        for (int i = 0; i < 3; i++) {
            if (isnan(world_acc[i]) || isinf(world_acc[i])) {
                has_nan = true;
                Serial.println("NaN/Inf detected in world acceleration");
                break;
            }
        }
        
        if (has_nan) {
            return;
        }

        // 静止状態の検出と処理 - 改良版の検出ロジックが使われる
        if (zvd.isStatic(&data[0], &data[3])) {
            // 静止状態では速度のみをリセット
            float current_pos[3], current_vel[3], current_bias[3];
            kf.getState(current_pos, current_vel, current_bias);
            
            // 位置とバイアスは保持したまま、速度のみゼロにしてリセット
            for (int i = 0; i < 3; i++) {
                current_vel[i] = 0.0f;
                last_vel[i] = 0.0f;  // 前回速度の記録もリセット
            }
            
            for (int i = 0; i < 3; i++) {
                send_list[3+i] = current_pos[i];
            }
            
            // カルマンフィルタを再初期化せず、速度のみリセット
            kf.setState(current_pos, current_vel, current_bias);
        } else {
            // 極端な加速度をチェック（物理的に不自然な加速を除外）
            bool valid_accel = true;
            for (int i = 0; i < 3; i++) {
                if (fabsf(world_acc[i]) > 20.0f) {  // 20 m/s²を超える加速度は異常と判断
                    valid_accel = false;
                    break;
                }
            }
            
            if (valid_accel) {
                // カルマンフィルタの更新（predictのみ使用）
                kf.predict(dt, world_acc);
                
                // 状態の取得
                float position[3], velocity[3], bias[3];
                kf.getState(position, velocity, bias);
                
                // 急激な速度変化をチェック（物理的に不自然な変化を抑制）
                for (int i = 0; i < 3; i++) {
                    float vel_change = fabsf(velocity[i] - last_vel[i]);
                    if (vel_change > 0.5f) {  // 0.5 m/sを超える急激な速度変化を抑制
                        velocity[i] = last_vel[i] + (velocity[i] > last_vel[i] ? 0.5f : -0.5f);
                        kf.setState(position, velocity, bias);
                    }
                    last_vel[i] = velocity[i];
                }
                
                // 位置の更新
                for (int i = 0; i < 3; i++) {
                    send_list[3+i] = position[i];
                }
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
            // 位置データをcmに変換して出力（小数点以下4桁まで表示）
            Serial.print(send_list[3] * 100.0f, 4);
            Serial.print(",");
            Serial.print(send_list[4] * 100.0f, 4);
            Serial.print(",");
            Serial.println(send_list[5] * 100.0f, 4);
        } else {
            Serial.println("Invalid data detected");
            // 状態をリセット
            kf.reset();
            zvd.reset();
            for (int i = 0; i < 6; i++) {
                send_list[i] = 0.0f;
            }
            for (int i = 0; i < 3; i++) {
                last_vel[i] = 0.0f;
            }
        }
        last_update = current_time;
    }
}
