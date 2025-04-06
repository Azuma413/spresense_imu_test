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
KalmanFilter kf(0.0001f, 0.0005f, 0.0001f, 0.05f);  // カルマンフィルタ
// 静止状態検出の閾値を調整：より感度を上げて静止状態をしっかり検出
ZeroVelocityDetector zvd(0.15f, 0.15f, 15);  // 静止状態検出器

// データ配列
float data[MultiIMU::DATA_LENGTH];           // センサーデータ
// SerialSend serial_send(MultiIMU::DATA_LENGTH);
float send_list[6] = {0,0,0,0,0,0};         // Roll, Pitch, Yaw, X(cm), Y(cm), Z(cm)
float world_acc[3] = {0, 0, 0};             // 世界座標系の加速度
float last_update = 0;                       // 前回の更新時刻
const float gravity = 9.80665;               // 重力加速度
const int INIT_SAMPLES = 150;               // 初期化用サンプル数を増やして安定性向上
int init_counter = 0;                       // 初期化カウンタ
float init_acc[3] = {0};                    // 初期化用加速度累積
float init_gyro[3] = {0};                   // 初期化用ジャイロ累積

void setup() {
    Serial.begin(115200);
    if (imu.begin()) {
        imu.startSensing(960, 16, 4000);
    }
    // serial_send.init();
    MadgwickFilter.begin(960);  // Madgwickフィルタのサンプリング 960Hz
    last_update = millis() / 1000.0;

    // IMUの初期化待ち
    delay(100);
}

void loop() {
    if (imu.getData(data)) {
        // 初期化フェーズ
        if (init_counter < INIT_SAMPLES) {
            // 初期加速度の平均を計算
            for (int i = 0; i < 3; i++) {
                init_acc[i] += data[i] / INIT_SAMPLES;
            }
            init_counter++;
            if (init_counter == INIT_SAMPLES) {
                // 初期バイアスを設定
                float initial_bias[3];
                for (int i = 0; i < 3; i++) {
                    initial_bias[i] = init_acc[i];
                }
                kf.setState(nullptr, nullptr, initial_bias);
            }
            return;  // 初期化中は位置計算をスキップ
        }

        float current_time = millis() / 1000.0;
        float dt = current_time - last_update;
        // 姿勢の更新（Madgwickフィルタ）
        MadgwickFilter.updateIMU(data[3], data[4], data[5], data[0], data[1], data[2]);
        send_list[0] = MadgwickFilter.getRoll();
        send_list[1] = MadgwickFilter.getPitch();
        send_list[2] = MadgwickFilter.getYaw();

        // 加速度を世界座標系に変換（クォータニオンベース）
        MadgwickFilter.convertAccelToWorldFrame(data[0], data[1], data[2], world_acc);

        // Serial.print(world_acc[0], 4);
        // Serial.print(",");
        // Serial.print(world_acc[1], 4);
        // Serial.print(",");
        // Serial.println(world_acc[2], 4);

        // エラーチェック
        bool has_nan = false;
        for (int i = 0; i < MultiIMU::DATA_LENGTH; i++) {
            if (isnan(data[i])) {
                has_nan = true;
                Serial.println("NaN detected in sensor data");
                break;
            }
        }
        if (has_nan) {
            return;
        }

        // 静止状態の検出と処理
        if (zvd.isStatic(&data[0], &data[3])) {
            // Serial.println("Static state detected");
            // 静止状態では速度のみをリセット
            float current_pos[3], current_vel[3], current_bias[3];
            kf.getState(current_pos, current_vel, current_bias);
            // 位置とバイアスは保持したまま、速度のみゼロにしてリセット
            for (int i = 0; i < 3; i++) {
                current_vel[i] = 0.0f;
            }
            for (int i = 0; i < 3; i++) {
                send_list[3+i] = current_pos[i];
            }
            // カルマンフィルタを再初期化（位置とバイアスは保持）
            kf.reset();
            // 位置とバイアスを再設定
            kf.setState(current_pos, current_vel, current_bias);
        } else {
            // カルマンフィルタの更新（predictのみ使用）
            kf.predict(dt, world_acc);
            // kf.update(world_acc); // 観測更新ステップをコメントアウト
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
            // 位置データをcmに変換して出力（小数点以下4桁まで表示）
            Serial.print(send_list[3] * 100.0f, 4);
            Serial.print(",");
            Serial.print(send_list[4] * 100.0f, 4);
            Serial.print(",");
            Serial.println(send_list[5] * 100.0f, 4);

            // 結果の出力
            // Serial.print("Posture (deg)\nRoll: ");
            // Serial.print(send_list[0]);
            // Serial.print(" Pitch: ");
            // Serial.print(send_list[1]);
            // Serial.print(" Yaw: ");
            // Serial.println(send_list[2]);
            // Serial.print("Position (cm)\nX: ");
            // Serial.print(send_list[3]);
            // Serial.print(" Y: ");
            // Serial.print(send_list[4]);
            // Serial.print(" Z: ");
            // Serial.println(send_list[5]);
        } else {
            Serial.println("Invalid data detected");
            // 状態をリセット
            kf.reset();
            zvd.reset();
            for (int i = 0; i < 6; i++) {
                send_list[i] = 0.0f;
            }
        }
        last_update = current_time;
    }
}
