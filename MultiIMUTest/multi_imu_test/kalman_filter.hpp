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
            state[i] += velocity * dt + 0.5f * acc * dt * dt;  // 位置の更新（加速度を考慮）
            state[i + 3] += acc * dt;  // 速度の更新

            // バイアスの緩やかな減衰（システムの安定性を保つ）
            float decay_rate = exp(-0.01f * dt);  // より緩やかな減衰率
            state[i + 6] *= decay_rate;  // 加速度バイアスの減衰
        }

        // 誤差共分散行列の更新
        float F[STATE_DIM][STATE_DIM] = {0};  // システム行列
        float decay_rate = exp(-0.01f * dt); // 状態更新で使った減衰率と同じ値を使う

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
            else if (i < 9) new_P[i][j] += process_noise_bias;  // 速度バイアス
            else new_P[i][j] += process_noise_bias;             // 加速度バイアス
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

    // // 更新ステップ (コメントアウト)
    // void update(const float* acc_measurement) {
    //     if (error_state) return;

    //     // 入力値のチェック
    //     for (int i = 0; i < 3; i++) {
    //         if (!isValidFloat(acc_measurement[i])) {
    //             error_state = true;
    //             return;
    //         }
    //     }

    //     // 観測行列の設定（加速度のみを観測）
    //     const int MEAS_DIM = 3;
    //     float H[MEAS_DIM][STATE_DIM] = {0};
    //     for (int i = 0; i < 3; i++) {
    //         H[i][i + 6] = 1.0f;  // 加速度バイアスの観測
    //     }

    //     // イノベーション（測定値と予測値の差）の計算
    //     float innovation[MEAS_DIM];
    //     for (int i = 0; i < MEAS_DIM; i++) {
    //         innovation[i] = acc_measurement[i] - state[i + 6];  // 測定値 - 推定バイアス
    //     }

    //     // S = H*P*H^T + R の計算
    //     float S[MEAS_DIM][MEAS_DIM] = {0};
    //     float temp[MEAS_DIM][STATE_DIM] = {0};

    //     // H*P の計算
    //     for (int i = 0; i < MEAS_DIM; i++) {
    //         for (int j = 0; j < STATE_DIM; j++) {
    //             for (int k = 0; k < STATE_DIM; k++) {
    //                 temp[i][j] += H[i][k] * P[k][j];
    //             }
    //         }
    //     }

    //     // (H*P)*H^T + R の計算
    //     for (int i = 0; i < MEAS_DIM; i++) {
    //         for (int j = 0; j < MEAS_DIM; j++) {
    //             for (int k = 0; k < STATE_DIM; k++) {
    //                 S[i][j] += temp[i][k] * H[j][k];
    //             }
    //             if (i == j) {
    //                 S[i][j] += measurement_noise;  // 測定ノイズの追加
    //             }
    //         }
    //     }

    //     // カルマンゲイン K = P*H^T*S^(-1) の計算
    //     float K[STATE_DIM][MEAS_DIM] = {0};
    //     float S_inv[MEAS_DIM][MEAS_DIM] = {0};

    //     // S の逆行列計算（簡略化のため対角成分のみ）
    //     for (int i = 0; i < MEAS_DIM; i++) {
    //         S_inv[i][i] = 1.0f / (S[i][i] + 1e-6f);  // 数値安定性のために小さな値を加算
    //     }

    //     // P*H^T の計算
    //     float PH[STATE_DIM][MEAS_DIM] = {0};
    //     for (int i = 0; i < STATE_DIM; i++) {
    //         for (int j = 0; j < MEAS_DIM; j++) {
    //             for (int k = 0; k < STATE_DIM; k++) {
    //                 PH[i][j] += P[i][k] * H[j][k];
    //             }
    //         }
    //     }

    //     // K = PH * S^(-1) の計算
    //     for (int i = 0; i < STATE_DIM; i++) {
    //         for (int j = 0; j < MEAS_DIM; j++) {
    //             K[i][j] = PH[i][j] * S_inv[j][j];
    //         }
    //     }

    //     // 状態の更新
    //     for (int i = 0; i < STATE_DIM; i++) {
    //         float correction = 0;
    //         for (int j = 0; j < MEAS_DIM; j++) {
    //             correction += K[i][j] * innovation[j];
    //         }
    //         state[i] += correction;
    //     }

    //     // 誤差共分散の更新
    //     float temp_P[STATE_DIM][STATE_DIM] = {0};
    //     for (int i = 0; i < STATE_DIM; i++) {
    //         for (int j = 0; j < STATE_DIM; j++) {
    //             temp_P[i][j] = P[i][j];
    //         }
    //     }

    //     // 誤差共分散の更新（P = (I - KH)P）
    //     float I_KH[STATE_DIM][STATE_DIM] = {0};
        
    //     // I - KH の計算
    //     for (int i = 0; i < STATE_DIM; i++) {
    //         I_KH[i][i] = 1.0f;  // 単位行列の設定
    //         for (int j = 0; j < STATE_DIM; j++) {
    //             float kh_sum = 0.0f;
    //             for (int k = 0; k < 6; k++) {  // 6次元の観測
    //                 kh_sum += K[i][k] * H[k][j];
    //             }
    //             I_KH[i][j] -= kh_sum;
    //         }
    //     }

    //     // P = (I-KH)P
    //     for (int i = 0; i < STATE_DIM; i++) {
    //         for (int j = 0; j < STATE_DIM; j++) {
    //             float sum = 0.0f;
    //             for (int k = 0; k < STATE_DIM; k++) {
    //                 sum += I_KH[i][k] * temp_P[k][j];
    //             }
    //             P[i][j] = sum;
    //         }
    //     }
    // }

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
