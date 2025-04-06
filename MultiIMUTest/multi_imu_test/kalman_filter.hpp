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

    // 行列の正値定符号性を保証
    void ensureMatrixPositiveDefinite() {
        for (int i = 0; i < STATE_DIM; i++) {
            // 対角要素が正であることを確認
            if (P[i][i] <= 0.0f) {
                P[i][i] = 1e-6f;
            }
            
            // 対称性を保証
            for (int j = 0; j < i; j++) {
                P[i][j] = P[j][i] = (P[i][j] + P[j][i]) * 0.5f;
            }
        }
    }

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

    // エラー状態の回復
    void recoverFromError() {
        if (error_state) {
            // エラー状態をリセットし、状態共分散行列だけを再初期化
            error_state = false;
            
            // 現在の状態ベクトルは保持しつつ、共分散行列を初期化
            for (int i = 0; i < STATE_DIM; i++) {
                for (int j = 0; j < STATE_DIM; j++) {
                    P[i][j] = 0.0f;
                }
            }
            
            // 対角要素の設定 - リカバリー時は不確かさを大きく設定
            for (int i = 0; i < 3; i++) {
                P[i][i] = 1.0f;                // 位置の不確かさ
                P[i+3][i+3] = 1.0f;            // 速度の不確かさ
                P[i+6][i+6] = 10.0f;           // バイアスの不確かさ
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
            
            // 小さな加速度をフィルタリング（ノイズ除去）- 閾値を引き上げて感度を下げる
            float acc_threshold = 0.1f;  // 小さな加速度を無視する閾値を増加
            if (fabsf(acc) < acc_threshold) {
                acc = 0.0f;
            }
            
            state[i] += velocity * dt + 0.5f * acc * dt * dt;  // 位置の更新（加速度を考慮）
            state[i + 3] += acc * dt;  // 速度の更新

            // バイアスの緩やかな減衰（システムの安定性を保つ）
            float decay_rate = 0.9999f;  // 非常に緩やかな減衰率に変更（ほぼ減衰しない）
            state[i + 6] *= decay_rate;  // 加速度バイアスの減衰
        }

        // 誤差共分散行列の更新
        float F[STATE_DIM][STATE_DIM] = {0};  // システム行列
        float decay_rate = 0.9999f; // 状態更新で使った減衰率と同じ値を使う

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

        // 数値的安定性を保証
        ensureMatrixPositiveDefinite();
    }

    // 測定更新ステップ (新機能)
    void update(const float* position_measurement) {
        if (error_state || position_measurement == nullptr) return;
        
        // 測定が有効かチェック
        for (int i = 0; i < 3; i++) {
            if (!isValidFloat(position_measurement[i])) {
                error_state = true;
                return;
            }
        }

        // 観測行列H (位置のみ観測)
        float H[3][STATE_DIM] = {0};
        for (int i = 0; i < 3; i++) {
            H[i][i] = 1.0f;  // 位置の観測
        }
        
        // 測定残差計算
        float y[3];
        for (int i = 0; i < 3; i++) {
            y[i] = position_measurement[i] - state[i];
        }
        
        // 残差共分散 S = H*P*H^T + R
        float S[3][3] = {0};
        float HPH[3][3] = {0};
        
        // HP計算
        float HP[3][STATE_DIM] = {0};
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < STATE_DIM; j++) {
                for (int k = 0; k < STATE_DIM; k++) {
                    HP[i][j] += H[i][k] * P[k][j];
                }
            }
        }
        
        // HPH^T計算
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                for (int k = 0; k < STATE_DIM; k++) {
                    HPH[i][j] += HP[i][k] * H[j][k];  // 転置を考慮
                }
            }
        }
        
        // S = HPH^T + R
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                S[i][j] = HPH[i][j];
                if (i == j) {
                    S[i][j] += measurement_noise;
                }
            }
        }
        
        // カルマンゲイン K = P*H^T*S^(-1)
        float K[STATE_DIM][3] = {0};
        
        // まず P*H^T を計算
        float PHt[STATE_DIM][3] = {0};
        for (int i = 0; i < STATE_DIM; i++) {
            for (int j = 0; j < 3; j++) {
                for (int k = 0; k < STATE_DIM; k++) {
                    PHt[i][j] += P[i][k] * H[j][k];  // 転置を考慮
                }
            }
        }
        
        // S^-1の計算（3x3行列の逆行列）
        float detS = S[0][0] * (S[1][1] * S[2][2] - S[1][2] * S[2][1])
                   - S[0][1] * (S[1][0] * S[2][2] - S[1][2] * S[2][0])
                   + S[0][2] * (S[1][0] * S[2][1] - S[1][1] * S[2][0]);
                   
        // 逆行列の存在チェック
        if (fabsf(detS) < 1e-10f) {
            // 特異行列になっている場合、更新をスキップ
            return;
        }
        
        float invDetS = 1.0f / detS;
        
        float invS[3][3];
        invS[0][0] = (S[1][1] * S[2][2] - S[1][2] * S[2][1]) * invDetS;
        invS[0][1] = (S[0][2] * S[2][1] - S[0][1] * S[2][2]) * invDetS;
        invS[0][2] = (S[0][1] * S[1][2] - S[0][2] * S[1][1]) * invDetS;
        invS[1][0] = (S[1][2] * S[2][0] - S[1][0] * S[2][2]) * invDetS;
        invS[1][1] = (S[0][0] * S[2][2] - S[0][2] * S[2][0]) * invDetS;
        invS[1][2] = (S[0][2] * S[1][0] - S[0][0] * S[1][2]) * invDetS;
        invS[2][0] = (S[1][0] * S[2][1] - S[1][1] * S[2][0]) * invDetS;
        invS[2][1] = (S[0][1] * S[2][0] - S[0][0] * S[2][1]) * invDetS;
        invS[2][2] = (S[0][0] * S[1][1] - S[0][1] * S[1][0]) * invDetS;
        
        // カルマンゲイン K = PH^T * S^-1 の計算
        for (int i = 0; i < STATE_DIM; i++) {
            for (int j = 0; j < 3; j++) {
                for (int k = 0; k < 3; k++) {
                    K[i][j] += PHt[i][k] * invS[k][j];
                }
            }
        }
        
        // 状態更新 x = x + K*y
        for (int i = 0; i < STATE_DIM; i++) {
            for (int j = 0; j < 3; j++) {
                state[i] += K[i][j] * y[j];
            }
        }
        
        // 共分散更新 P = (I - K*H) * P
        float I_KH[STATE_DIM][STATE_DIM] = {0};
        
        // I - K*H の計算
        for (int i = 0; i < STATE_DIM; i++) {
            I_KH[i][i] = 1.0f;  // 単位行列Iの対角要素
            for (int j = 0; j < STATE_DIM; j++) {
                float KH_ij = 0.0f;
                for (int k = 0; k < 3; k++) {
                    KH_ij += K[i][k] * H[k][j];
                }
                I_KH[i][j] -= KH_ij;
            }
        }
        
        // (I - K*H) * P の計算
        float new_P[STATE_DIM][STATE_DIM] = {0};
        for (int i = 0; i < STATE_DIM; i++) {
            for (int j = 0; j < STATE_DIM; j++) {
                for (int k = 0; k < STATE_DIM; k++) {
                    new_P[i][j] += I_KH[i][k] * P[k][j];
                }
            }
        }
        
        // 更新された共分散行列の保存
        for (int i = 0; i < STATE_DIM; i++) {
            for (int j = 0; j < STATE_DIM; j++) {
                P[i][j] = new_P[i][j];
            }
        }
        
        // 数値的安定性の確保
        ensureMatrixPositiveDefinite();
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
