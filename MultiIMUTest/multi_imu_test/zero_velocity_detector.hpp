#ifndef ZERO_VELOCITY_DETECTOR_HPP
#define ZERO_VELOCITY_DETECTOR_HPP

#include <math.h>

class ZeroVelocityDetector {
private:
    // パラメータ
    float acc_threshold;        // 加速度閾値
    float gyro_threshold;       // 角速度閾値
    float var_threshold;        // 分散閾値
    int static_count_threshold; // 静止判定カウント閾値
    int static_count;          // 静止状態カウンタ
    float gravity;             // 重力加速度

    // 分散計算用バッファ
    static const int BUFFER_SIZE = 10;
    float acc_buffer[BUFFER_SIZE][3];
    float gyro_buffer[BUFFER_SIZE][3];
    int buffer_index;

public:
    ZeroVelocityDetector(float acc_th = 0.1f, float gyro_th = 0.2f, 
                        int count_th = 20, float g = 9.80665f)
        : acc_threshold(acc_th)
        , gyro_threshold(gyro_th)
        , var_threshold(0.01f)
        , static_count_threshold(count_th)
        , static_count(0)
        , gravity(g)
        , buffer_index(0) {
        // バッファの初期化
        for (int i = 0; i < BUFFER_SIZE; i++) {
            for (int j = 0; j < 3; j++) {
                acc_buffer[i][j] = 0.0f;
                gyro_buffer[i][j] = 0.0f;
            }
        }
    }

    // パラメータの設定
    void setThresholds(float acc_th, float gyro_th, int count_th) {
        if (acc_th > 0.0f) acc_threshold = acc_th;
        if (gyro_th > 0.0f) gyro_threshold = gyro_th;
        if (count_th > 0) static_count_threshold = count_th;
    }

    // データの分散を計算
    void calculateVariance(const float buffer[][3], float* variance) {
        float mean[3] = {0.0f, 0.0f, 0.0f};
        
        // 平均の計算
        for (int i = 0; i < BUFFER_SIZE; i++) {
            for (int j = 0; j < 3; j++) {
                mean[j] += buffer[i][j];
            }
        }
        for (int j = 0; j < 3; j++) {
            mean[j] /= BUFFER_SIZE;
        }
        
        // 分散の計算
        for (int j = 0; j < 3; j++) {
            variance[j] = 0.0f;
            for (int i = 0; i < BUFFER_SIZE; i++) {
                float diff = buffer[i][j] - mean[j];
                variance[j] += diff * diff;
            }
            variance[j] /= BUFFER_SIZE;
        }
    }

    // 静止状態の検出
    bool isStatic(const float* acc, const float* gyro) {
        // バッファにデータを追加
        for (int i = 0; i < 3; i++) {
            acc_buffer[buffer_index][i] = acc[i];
            gyro_buffer[buffer_index][i] = gyro[i];
        }
        buffer_index = (buffer_index + 1) % BUFFER_SIZE;

        // 分散の計算
        float acc_var[3], gyro_var[3];
        calculateVariance(acc_buffer, acc_var);
        calculateVariance(gyro_buffer, gyro_var);

        // 加速度の大きさを計算
        float acc_magnitude = sqrtf(acc[0]*acc[0] + acc[1]*acc[1] + acc[2]*acc[2]);
        
        // 重力加速度からの差を計算
        float acc_diff = fabsf(acc_magnitude - gravity);

        // 角速度の大きさを計算
        float gyro_magnitude = sqrtf(gyro[0]*gyro[0] + gyro[1]*gyro[1] + gyro[2]*gyro[2]);

        // 分散の合計を計算
        float total_var = 0.0f;
        for (int i = 0; i < 3; i++) {
            total_var += acc_var[i] + gyro_var[i];
        }

        // 静止状態の判定
        if (acc_diff < acc_threshold && 
            gyro_magnitude < gyro_threshold && 
            total_var < var_threshold) {
            static_count++;
        } else {
            static_count = 0;
        }

        return static_count >= static_count_threshold;
    }

    // カウンタとバッファのリセット
    void reset() {
        static_count = 0;
        buffer_index = 0;
        for (int i = 0; i < BUFFER_SIZE; i++) {
            for (int j = 0; j < 3; j++) {
                acc_buffer[i][j] = 0.0f;
                gyro_buffer[i][j] = 0.0f;
            }
        }
    }
};

#endif // ZERO_VELOCITY_DETECTOR_HPP
