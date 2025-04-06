#ifndef KALMAN_FILTER_HPP
#define KALMAN_FILTER_HPP

#include <math.h>
#include <stdlib.h>

class KalmanFilter {
private:
    // 状態ベクトル [x, y, z, vx, vy, vz, ax_bias, ay_bias, az_bias]
    static const int STATE_DIM = 9;
    float state[STATE_DIM] = {0};
    float P[STATE_DIM][STATE_DIM] = {0};  // 誤差共分散行列

    // システムノイズのパラメータ
    float process_noise_pos;   // 位置のプロセスノイズ
    float process_noise_vel;   // 速度のプロセスノイズ
    float process_noise_bias;  // バイアスのプロセスノイズ
    float measurement_noise;   // 測定ノイズ

public:
    KalmanFilter(float pos_noise = 0.0001f, float vel_noise = 0.001f, 
                 float bias_noise = 0.00001f, float meas_noise = 0.1f) 
        : process_noise_pos(pos_noise)
        , process_noise_vel(vel_noise)
        , process_noise_bias(bias_noise)
        , measurement_noise(meas_noise) {
        
        reset();  // 適切な初期化を実行
    }

    // 異常値のチェック
    bool isValidFloat(float value) {
        return !isnan(value) && !isinf(value);
    }

    // エラー状態
    bool error_state = false;

    // 状態の直接設定
    void setState(const float* position, const float* velocity, const float* acc_bias) {
        if (position) {
            for (int i = 0; i < 3; i++) {
                state[i] = position[i];
            }
        }
        if (velocity) {
            for (int i = 0; i < 3; i++) {
                state[i + 3] = velocity[i];
            }
        }
        if (acc_bias) {
            for (int i = 0; i < 3; i++) {
                state[i + 6] = acc_bias[i];  // 加速度バイアス
            }
        }
    }


    // 予測ステップ
    void predict(float dt, const float* acceleration = nullptr) {
        if (error_state || !isValidFloat(dt)) return;

        // dt値の制限（異常な大きな値を防ぐ）
        dt = fmin(dt, 0.1f);

        // 状態の更新
        for (int i = 0; i < 3; i++) {
            float velocity = state[i + 3];  // 現在の速度

            // 値の範囲チェック
            if (!isValidFloat(velocity)) {
                error_state = true;
                return;
            }

            // 運動方程式に基づく状態の更新
            float acc = (acceleration != nullptr) ? (acceleration[i] - state[i + 6]) : 0.0f;  // バイアス補正済みの加速度
            
            // 小さな加速度をフィルタリング（ノイズ除去）
            float acc_threshold = 0.05f;  // 小さな加速度を無視する閾値
            if (fabsf(acc) < acc_threshold) {
                acc = 0.0f;
            }
            
            state[i] += velocity * dt + 0.5f * acc * dt * dt;  // 位置の更新（加速度を考慮）
            state[i + 3] += acc * dt;  // 速度の更新

            // バイアスの緩やかな減衰（システムの安定性を保つ）
            float decay_rate = exp(-0.005f * dt);  // より緩やかな減衰率
            state[i + 6] *= decay_rate;  // 加速度バイアスの減衰
        }

        // 誤差共分散行列の更新
        float F[STATE_DIM][STATE_DIM] = {0};  // システム行列
        float decay_rate = exp(-0.005f * dt); // 状態更新で使った減衰率と同じ値を使う

        for (int i = 0; i < STATE_DIM; i++) {
            F[i][i] = 1.0f;  // 対角要素を1で初期化
        }
        // 運動方程式とバイアスモデルに基づく要素の設定
        for (int i = 0; i < 3; i++) {
            F[i][i+3] = dt;                 // 位置への速度の影響: dPos/dVel = dt
            F[i][i+6] = -0.5f * dt * dt;    // 位置へのバイアスの影響: dPos/dBias = -0.5*dt^2 (acc = a_meas - bias)
            F[i+3][i+6] = -dt;              // 速度へのバイアスの影響: dVel/dBias = -dt (acc = a_meas - bias)
            F[i+6][i+6] = decay_rate;       // バイアスの減衰を遷移行列に反映
        }

        // P = F*P*F^T + Q
        float temp[STATE_DIM][STATE_DIM] = {0};
        float new_P[STATE_DIM][STATE_DIM] = {0};

        // F*P
        for (int i = 0; i < STATE_DIM; i++) {
            for (int j = 0; j < STATE_DIM; j++) {
                for (int k = 0; k < STATE_DIM; k++) {
                    temp[i][j] += F[i][k] * P[k][j];
                }
            }
        }

        // (F*P)*F^T + Q
        for (int i = 0; i < STATE_DIM; i++) {
            for (int j = 0; j < STATE_DIM; j++) {
                for (int k = 0; k < STATE_DIM; k++) {
                    new_P[i][j] += temp[i][k] * F[j][k];  // 転置を考慮
                }
                // プロセスノイズの追加
                if (i == j) {
                    if (i < 3) new_P[i][j] += process_noise_pos;        // 位置
                    else if (i < 6) new_P[i][j] += process_noise_vel;   // 速度
                    else if (i < 9) new_P[i][j] += process_noise_bias;  // バイアス
                }
            }
        }

        // 更新された共分散行列の保存
        for (int i = 0; i < STATE_DIM; i++) {
            for (int j = 0; j < STATE_DIM; j++) {
                P[i][j] = new_P[i][j];
            }
        }
    }

    // 状態の取得
    void getState(float* position, float* velocity, float* acceleration_bias) {
        for (int i = 0; i < 3; i++) {
            if (position) position[i] = state[i];
            if (velocity) velocity[i] = state[i + 3];
            if (acceleration_bias) acceleration_bias[i] = state[i + 6];
        }
    }

    // 状態のリセット
    void reset() {
        error_state = false;

        // すべての状態をゼロにリセット
        for (int i = 0; i < STATE_DIM; i++) {
            state[i] = 0.0f;
        }

        // 誤差共分散行列の初期化（非対角要素はゼロ）
        for (int i = 0; i < STATE_DIM; i++) {
            for (int j = 0; j < STATE_DIM; j++) {
                P[i][j] = 0.0f;
            }
        }

        // 対角要素の設定
        for (int i = 0; i < 3; i++) {
            P[i][i] = 1e-6f;                   // 位置の初期不確かさ（非常に小さく）
            P[i+3][i+3] = 1e-4f;              // 速度の初期不確かさ（小さく）
            P[i+6][i+6] = 1.0f;               // バイアスの初期不確かさ（大きく）
        }

        // 起動時の安定化のために初期バイアスを仮設定
        for (int i = 0; i < 3; i++) {
            state[i+6] = 0.1f;  // 小さな初期バイアスを設定
        }
    }
};

#endif // KALMAN_FILTER_HPP
