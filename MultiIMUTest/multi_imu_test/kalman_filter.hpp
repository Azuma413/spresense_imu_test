#ifndef KALMAN_FILTER_HPP
#define KALMAN_FILTER_HPP

#include <math.h>

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
    KalmanFilter(float pos_noise = 0.01f, float vel_noise = 0.1f, 
                 float bias_noise = 0.01f, float meas_noise = 0.1f) 
        : process_noise_pos(pos_noise)
        , process_noise_vel(vel_noise)
        , process_noise_bias(bias_noise)
        , measurement_noise(meas_noise) {
        
        // 初期誤差共分散行列の設定
        for (int i = 0; i < STATE_DIM; i++) {
            for (int j = 0; j < STATE_DIM; j++) {
                P[i][j] = (i == j) ? 1.0f : 0.0f;
            }
        }
    }

    // 予測ステップ
    void predict(float dt) {
        // 状態の更新
        // 位置の更新
        for (int i = 0; i < 3; i++) {
            state[i] += state[i + 3] * dt + 
                       0.5f * (state[i + 6]) * dt * dt;  // 位置 += 速度*dt + 0.5*加速度*dt^2
            state[i + 3] += state[i + 6] * dt;          // 速度 += 加速度*dt
        }

        // 誤差共分散行列の更新
        float F[STATE_DIM][STATE_DIM] = {0};  // システム行列
        for (int i = 0; i < STATE_DIM; i++) {
            F[i][i] = 1.0f;  // 単位行列として初期化
        }
        // 運動方程式に基づく要素の設定
        for (int i = 0; i < 3; i++) {
            F[i][i+3] = dt;                    // 位置-速度の関係
            F[i][i+6] = 0.5f * dt * dt;       // 位置-加速度の関係
            F[i+3][i+6] = dt;                 // 速度-加速度の関係
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
                    if (i < 3) new_P[i][j] += process_noise_pos;
                    else if (i < 6) new_P[i][j] += process_noise_vel;
                    else new_P[i][j] += process_noise_bias;
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

    // 更新ステップ
    void update(const float* acc_measurement) {
        // 測定値と予測値の差
        float innovation[3];
        for (int i = 0; i < 3; i++) {
            innovation[i] = acc_measurement[i] - state[i + 6];  // 測定加速度 - 予測バイアス
        }

        // カルマンゲインの計算
        float H[3][STATE_DIM] = {0};  // 観測行列
        for (int i = 0; i < 3; i++) {
            H[i][i + 6] = 1.0f;  // バイアスの観測
        }

        // S = H*P*H^T + R
        float S[3][3] = {0};
        float temp[3][STATE_DIM] = {0};

        // H*P
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < STATE_DIM; j++) {
                for (int k = 0; k < STATE_DIM; k++) {
                    temp[i][j] += H[i][k] * P[k][j];
                }
            }
        }

        // (H*P)*H^T + R
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                for (int k = 0; k < STATE_DIM; k++) {
                    S[i][j] += temp[i][k] * H[j][k];  // 転置を考慮
                }
                if (i == j) S[i][j] += measurement_noise;
            }
        }

        // K = P*H^T*S^(-1)
        float K[STATE_DIM][3] = {0};
        float S_inv[3][3] = {0};
        
        // 3x3行列の逆行列計算（簡略化のため、対角成分のみを考慮）
        for (int i = 0; i < 3; i++) {
            S_inv[i][i] = 1.0f / S[i][i];
        }

        // P*H^T
        float PH[STATE_DIM][3] = {0};
        for (int i = 0; i < STATE_DIM; i++) {
            for (int j = 0; j < 3; j++) {
                for (int k = 0; k < STATE_DIM; k++) {
                    PH[i][j] += P[i][k] * H[j][k];  // H^Tを考慮
                }
            }
        }

        // (P*H^T)*S^(-1)
        for (int i = 0; i < STATE_DIM; i++) {
            for (int j = 0; j < 3; j++) {
                K[i][j] = PH[i][j] * S_inv[j][j];
            }
        }

        // 状態の更新
        for (int i = 0; i < STATE_DIM; i++) {
            for (int j = 0; j < 3; j++) {
                state[i] += K[i][j] * innovation[j];
            }
        }

        // 誤差共分散の更新
        float temp_P[STATE_DIM][STATE_DIM] = {0};
        for (int i = 0; i < STATE_DIM; i++) {
            for (int j = 0; j < STATE_DIM; j++) {
                temp_P[i][j] = P[i][j];
            }
        }

        for (int i = 0; i < STATE_DIM; i++) {
            for (int j = 0; j < STATE_DIM; j++) {
                float sum = 0;
                for (int k = 0; k < 3; k++) {
                    sum += K[i][k] * H[k][j];
                }
                P[i][j] = temp_P[i][j] - sum;
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
        for (int i = 0; i < STATE_DIM; i++) {
            state[i] = 0.0f;
            for (int j = 0; j < STATE_DIM; j++) {
                P[i][j] = (i == j) ? 1.0f : 0.0f;
            }
        }
    }
};

#endif // KALMAN_FILTER_HPP
