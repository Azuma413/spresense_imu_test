#include "multi_imu.hpp"
#include "ism330dhcx.hpp"

// IMUの選択（使用するものをコメントアウトで切り替え）
// MultiIMU imu;
ISM330DHCX imu;

// IMUデータ用バッファ
float data[MultiIMU::DATA_LENGTH];
float prev_acc[3] = {0, 0, 0};  // 前回の加速度データ
float acc_diff[3] = {0, 0, 0};  // 加速度の差分
bool is_first_data = true;      // 初回データフラグ

// 静止検出のパラメータ
const float ACC_DIFF_THRESHOLD = 0.05;  // 加速度差分閾値 [m/s^2]
const float GYRO_THRESHOLD = 0.1;      // 角速度閾値 [rad/s]
const unsigned long STILL_TIME = 2000;     // 静止確認時間 [ms]
const unsigned long INTEGRATION_TIME = 10000; // 積分時間 [ms]

// 状態管理
bool still_detected = false;
unsigned long still_start_time = 0;
bool integration_started = false;
unsigned long integration_start_time = 0;

// 積分値
float acc_integral[3] = {0, 0, 0};
float gyro_integral[3] = {0, 0, 0};
float acc_abs_integral[3] = {0, 0, 0};
float gyro_abs_integral[3] = {0, 0, 0};
unsigned long last_integration_time = 0;

void setup() {
    Serial.begin(115200);
    if (imu.begin()) {
        // サンプリングレートを104Hzに設定
        imu.startSensing(104, 16, 4000);
    }
    Serial.println("Start");
}

// 加速度の差分を計算
void updateAccDiff(const float* data) {
    for (int i = 0; i < 3; i++) {
        acc_diff[i] = data[i] - prev_acc[i];
        prev_acc[i] = data[i];
    }
}

// 静止状態の検出
bool isStill(const float* data) {
    if (is_first_data) {
        is_first_data = false;
        return false;
    }
    Serial.print("加速度差分: ");
    for (int i = 0; i < 3; i++) {
        Serial.print(acc_diff[i], 6);
        Serial.print(" ");
    }
    Serial.println();
    Serial.print(" 角速度: ");
    for (int i = 0; i < 3; i++) {
        Serial.print(data[i + 3], 6);
        Serial.print(" ");
    }
    Serial.println();

    // 加速度差分と角速度の大きさをチェック
    for (int i = 0; i < 3; i++) {
        if (abs(acc_diff[i]) > ACC_DIFF_THRESHOLD) return false;  // 加速度差分
        if (abs(data[i + 3]) > GYRO_THRESHOLD) return false;     // 角速度
    }
    return true;
}

// 積分値の計算（台形積分）
void updateIntegrals(const float* data, float dt) {
    for (int i = 0; i < 3; i++) {
        // 通常の積分
        acc_integral[i] += data[i] * dt;
        gyro_integral[i] += data[i + 3] * dt;
        
        // 絶対値の積分
        acc_abs_integral[i] += abs(data[i]) * dt;
        gyro_abs_integral[i] += abs(data[i + 3]) * dt;
    }
}

// 結果の出力
void printResults() {
    Serial.println("===== 積分結果 =====");
    
    Serial.println("加速度積分値 [m/s]:");
    for (int i = 0; i < 3; i++) {
        Serial.print(String("XYZ"[i]) + ": ");
        Serial.println(acc_integral[i], 6);
    }
    
    Serial.println("\n加速度絶対値積分 [m/s]:");
    for (int i = 0; i < 3; i++) {
        Serial.print(String("XYZ"[i]) + ": ");
        Serial.println(acc_abs_integral[i], 6);
    }
    
    Serial.println("\n角速度積分値 [rad]:");
    for (int i = 0; i < 3; i++) {
        Serial.print(String("XYZ"[i]) + ": ");
        Serial.println(gyro_integral[i], 6);
    }
    
    Serial.println("\n角速度絶対値積分 [rad]:");
    for (int i = 0; i < 3; i++) {
        Serial.print(String("XYZ"[i]) + ": ");
        Serial.println(gyro_abs_integral[i], 6);
    }
    
    Serial.println("==================\n");
}

void resetIntegrals() {
    for (int i = 0; i < 3; i++) {
        acc_integral[i] = 0;
        gyro_integral[i] = 0;
        acc_abs_integral[i] = 0;
        gyro_abs_integral[i] = 0;
    }
}

void loop() {
    if (imu.getData(data)) {
        unsigned long current_time = millis();
        
        // 加速度の差分を更新
        updateAccDiff(data);
        
        if (!still_detected) {
            // 静止状態の検出
            if (isStill(data)) {
                if (still_start_time == 0) {
                    still_start_time = current_time;
                } else if (current_time - still_start_time >= STILL_TIME) {
                    still_detected = true;
                    integration_started = true;
                    integration_start_time = current_time;
                    last_integration_time = current_time;
                    resetIntegrals();
                    Serial.println("静止検出完了、積分開始");
                }
            } else {
                still_start_time = 0;
            }
        } else if (integration_started) {
            // 積分処理
            float dt = (current_time - last_integration_time) / 1000.0f; // 秒単位に変換
            updateIntegrals(data, dt);
            last_integration_time = current_time;
            
            // 積分終了判定
            if (current_time - integration_start_time >= INTEGRATION_TIME) {
                integration_started = false;
                still_detected = false;
                still_start_time = 0;
                printResults();
            }
        }
    }
}
