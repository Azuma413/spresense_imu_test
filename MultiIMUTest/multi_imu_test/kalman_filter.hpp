#ifndef KALMAN_FILTER_HPP
#define KALMAN_FILTER_HPP

#include <math.h>
#include <stdlib.h>

class KalmanFilter {
private:
    // 状態ベクトル [x, y, z, vx, vy, vz, vx_bias, vy_bias, vz_bias, ax_bias, ay_bias, az_bias]
    static const int STATE_DIM = 12;
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
        
        // 初期誤差共分散行列の設定
        for (int i = 0; i < STATE_DIM; i++) {
            for (int j = 0; j < STATE_DIM; j++) {
                P[i][j] = (i == j) ? 1.0f : 0.0f;
            }
        }
    }

    // 異常値のチェック
    bool isValidFloat(float value) {
        return !isnan(value) && !isinf(value);
    }

    // エラー状態
    bool error_state = false;

    // 予測ステップ
    void predict(float dt) {
        if (error_state || !isValidFloat(dt)) return;

        // dt値の制限（異常な大きな値を防ぐ）
        dt = fmin(dt, 0.1f);

        // 状態の更新
        for (int i = 0; i < 3; i++) {
            float true_velocity = state[i + 3] - state[i + 6];  // バイアス補正済みの速度
            float true_acceleration = -state[i + 9];  // バイアス補正済みの加速度

            // 値の範囲チェック
            if (!isValidFloat(true_velocity) || !isValidFloat(true_acceleration)) {
                error_state = true;
                return;
            }

            // 状態の更新（運動方程式に基づく）
            state[i] += true_velocity * dt + 0.5f * true_acceleration * dt * dt;  // 位置の更新
            state[i + 3] += true_acceleration * dt;  // 速度の更新（生の速度）

            // バイアスの更新（指数減衰モデル）
            float decay_rate = exp(-0.1f * dt);  // 減衰率
            state[i + 6] *= decay_rate;  // 速度バイアスの減衰
            state[i + 9] *= decay_rate;  // 加速度バイアスの減衰
            
            // バイアスのプロセスノイズ追加
            float noise_std = sqrtf(process_noise_bias * dt);
            state[i + 6] += noise_std * (0.01f);  // 速度バイアスへの小さなノイズ
            state[i + 9] += noise_std * (0.1f);   // 加速度バイアスへの小さなノイズ
        }

        // 誤差共分散行列の更新
        float F[STATE_DIM][STATE_DIM] = {0};  // システム行列
        for (int i = 0; i < STATE_DIM; i++) {
            F[i][i] = 1.0f;  // 単位行列として初期化
        }
        // 運動方程式に基づく要素の設定
        for (int i = 0; i < 3; i++) {
            // 位置の更新（x = x + v*dt + 0.5*a*dt^2）
            F[i][i+3] = dt;                     // 位置への速度の影響
            F[i][i+9] = 0.5f * dt * dt;        // 位置への加速度の影響
            
            // 速度の更新（v = v + a*dt）
            F[i+3][i+3] = 1.0f;                // 速度の保持
            F[i+3][i+6] = -1.0f;               // 速度バイアスの影響
            F[i+3][i+9] = dt;                  // 速度への加速度の影響

            // バイアスの更新（ランダムウォークモデル）
            F[i+6][i+6] = 1.0f;                // 速度バイアス
            F[i+9][i+9] = 1.0f;                // 加速度バイアス
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

    // 更新ステップ
    void update(const float* acc_measurement) {
        if (error_state) return;

        // 入力値のチェック
        for (int i = 0; i < 3; i++) {
            if (!isValidFloat(acc_measurement[i])) {
                error_state = true;
                return;
            }
        }

        // イノベーション（測定値と予測値の差）の計算
        float acc_innovation[3];
        float vel_innovation[3];

        // 加速度のイノベーション計算
        // 測定値には加速度バイアスが含まれている（測定値 = 真の加速度 + バイアス）
        for (int i = 0; i < 3; i++) {
            acc_innovation[i] = acc_measurement[i] - state[i + 9];
        }

        // 速度のイノベーション計算（ZUPTが有効な場合）
        for (int i = 0; i < 3; i++) {
            // 真の速度の推定値を計算（測定速度 = 0）
            float true_velocity = state[i + 3] - state[i + 6];  // 速度 - 速度バイアス
            vel_innovation[i] = 0.0f - true_velocity;
        }

        // 観測行列の設定
        float H[6][STATE_DIM] = {0};  // 6次元の観測（3次元加速度 + 3次元速度）
        for (int i = 0; i < 3; i++) {
            // 加速度の観測（測定値 = 真の加速度 + バイアス）
            H[i][i + 9] = 1.0f;    // 加速度バイアスの観測
            
            // 速度の観測（ZUPT: 測定値 = 0 = 真の速度）
            H[i + 3][i + 3] = 1.0f;  // 速度の観測
            H[i + 3][i + 6] = -1.0f; // 速度バイアスの観測
        }

        // S = H*P*H^T + R
        float S[6][6] = {0};
        float temp[6][STATE_DIM] = {0};

        // H*P
        for (int i = 0; i < 6; i++) {
            for (int j = 0; j < STATE_DIM; j++) {
                for (int k = 0; k < STATE_DIM; k++) {
                    temp[i][j] += H[i][k] * P[k][j];
                }
            }
        }

        // (H*P)*H^T + R
        for (int i = 0; i < 6; i++) {  // 修正：3から6に変更
            for (int j = 0; j < 6; j++) {
                for (int k = 0; k < STATE_DIM; k++) {
                    S[i][j] += temp[i][k] * H[j][k];
                }
                // 測定ノイズの追加
                if (i == j) {
                    if (i < 3) {
                        S[i][j] += measurement_noise;  // 加速度の測定ノイズ
                    } else {
                        S[i][j] += 0.01f * measurement_noise;  // 速度の測定ノイズ（ZUPTの場合は小さく）
                    }
                }
            }
        }

        // K = P*H^T*S^(-1)
        float K[STATE_DIM][6] = {0};
        float S_inv[6][6] = {0};
        
        // 6x6行列の逆行列計算（簡略化のため、対角成分のみを考慮）
        for (int i = 0; i < 6; i++) {
            S_inv[i][i] = 1.0f / S[i][i];
        }

        // P*H^T
        float PH[STATE_DIM][6] = {0};  // 修正：[3]から[6]に変更
        for (int i = 0; i < STATE_DIM; i++) {
            for (int j = 0; j < 6; j++) {
                for (int k = 0; k < STATE_DIM; k++) {
                    PH[i][j] += P[i][k] * H[j][k];  // H^Tを考慮
                }
            }
        }

        // (P*H^T)*S^(-1)
        for (int i = 0; i < STATE_DIM; i++) {
            for (int j = 0; j < 6; j++) {
                K[i][j] = PH[i][j] * S_inv[j][j];
            }
        }

        // 状態の更新
        for (int i = 0; i < STATE_DIM; i++) {
            for (int j = 0; j < 3; j++) {
                state[i] += K[i][j] * acc_innovation[j];
                state[i] += K[i][j + 3] * vel_innovation[j];
            }
        }

        // 誤差共分散の更新
        float temp_P[STATE_DIM][STATE_DIM] = {0};
        for (int i = 0; i < STATE_DIM; i++) {
            for (int j = 0; j < STATE_DIM; j++) {
                temp_P[i][j] = P[i][j];
            }
        }

        // 誤差共分散の更新（P = (I - KH)P）
        float I_KH[STATE_DIM][STATE_DIM] = {0};
        
        // I - KH の計算
        for (int i = 0; i < STATE_DIM; i++) {
            I_KH[i][i] = 1.0f;  // 単位行列の設定
            for (int j = 0; j < STATE_DIM; j++) {
                float kh_sum = 0.0f;
                for (int k = 0; k < 6; k++) {  // 6次元の観測
                    kh_sum += K[i][k] * H[k][j];
                }
                I_KH[i][j] -= kh_sum;
            }
        }

        // P = (I-KH)P
        for (int i = 0; i < STATE_DIM; i++) {
            for (int j = 0; j < STATE_DIM; j++) {
                float sum = 0.0f;
                for (int k = 0; k < STATE_DIM; k++) {
                    sum += I_KH[i][k] * temp_P[k][j];
                }
                P[i][j] = sum;
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
        float saved_vbias[3], saved_abias[3];
        
        // バイアスの保存
        for (int i = 0; i < 3; i++) {
            saved_vbias[i] = state[i + 6];
            saved_abias[i] = state[i + 9];
        }
        
        // 位置と速度のリセット（バイアスは保持）
        for (int i = 0; i < 6; i++) {
            state[i] = 0.0f;
        }
        
        // バイアスの復元
        for (int i = 0; i < 3; i++) {
            state[i + 6] = saved_vbias[i];
            state[i + 9] = saved_abias[i];
        }
        
        // 誤差共分散行列の再初期化
        for (int i = 0; i < STATE_DIM; i++) {
            for (int j = 0; j < STATE_DIM; j++) {
                if (i == j) {
                    if (i < 3) P[i][j] = 0.001f;      // 位置の初期不確かさ
                    else if (i < 6) P[i][j] = 0.001f; // 速度の初期不確かさ
                    else if (i < 9) P[i][j] = 0.01f;  // 速度バイアスの初期不確かさ
                    else P[i][j] = 0.01f;             // 加速度バイアスの初期不確かさ
                } else {
                    P[i][j] = 0.0f;
                }
            }
        }
    }
};

#endif // KALMAN_FILTER_HPP
