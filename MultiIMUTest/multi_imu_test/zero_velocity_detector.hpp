#ifndef ZERO_VELOCITY_DETECTOR_HPP
#define ZERO_VELOCITY_DETECTOR_HPP

#include <math.h>

class ZeroVelocityDetector {
private:
    // パラメータ
    float acc_threshold;        // 加速度閾値
    float gyro_threshold;       // 角速度閾値
    int static_count_threshold; // 静止判定カウント閾値
    int static_count;          // 静止状態カウンタ
    float gravity;             // 重力加速度

public:
    ZeroVelocityDetector(float acc_th = 0.05f, float gyro_th = 0.1f, 
                        int count_th = 50, float g = 9.80665f)
        : acc_threshold(acc_th)
        , gyro_threshold(gyro_th)
        , static_count_threshold(count_th)
        , static_count(0)
        , gravity(g) {}

    // パラメータの設定
    void setThresholds(float acc_th, float gyro_th, int count_th) {
        if (acc_th > 0.0f) acc_threshold = acc_th;
        if (gyro_th > 0.0f) gyro_threshold = gyro_th;
        if (count_th > 0) static_count_threshold = count_th;
    }

    // 静止状態の検出
    bool isStatic(const float* acc, const float* gyro) {
        // 加速度の大きさを計算
        float acc_magnitude = sqrtf(acc[0]*acc[0] + acc[1]*acc[1] + acc[2]*acc[2]);
        
        // 重力加速度からの差を計算
        float acc_diff = fabsf(acc_magnitude - gravity);

        // 角速度の大きさを計算
        float gyro_magnitude = sqrtf(gyro[0]*gyro[0] + gyro[1]*gyro[1] + gyro[2]*gyro[2]);

        // 静止状態の判定
        if (acc_diff < acc_threshold && gyro_magnitude < gyro_threshold) {
            static_count++;
        } else {
            static_count = 0;
        }

        return static_count >= static_count_threshold;
    }

    // カウンタのリセット
    void reset() {
        static_count = 0;
    }
};

#endif // ZERO_VELOCITY_DETECTOR_HPP
