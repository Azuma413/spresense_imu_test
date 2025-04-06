//=============================================================================================
// madgwick_filter.hpp
//=============================================================================================
//
// Implementation of Madgwick's IMU and AHRS algorithms.
// See: http://www.x-io.co.uk/open-source-imu-and-ahrs-algorithms/
//
// From the x-io website "Open-source resources available on this website are
// provided under the GNU General Public Licence unless an alternative licence
// is provided in source."
//
// Date         Author          Notes
// 29/09/2011   SOH Madgwick    Initial release
// 02/10/2011   SOH Madgwick    Optimised for reduced CPU load
// 19/02/2012   SOH Madgwick    Magnetometer measurement is normalised
//
//=============================================================================================

#ifndef MADGWICK_FILTER_HPP
#define MADGWICK_FILTER_HPP

#include <math.h>

class Madgwick {
private:
    // アルゴリズムのゲイン
    const float betaDef = 1.0f;            // 2 * proportional gain
    const float sampleFreqDef = 512.0f;    // サンプリング周波数（Hz）

    float beta;                  // アルゴリズムゲイン
    float q0, q1, q2, q3;       // センサーフレームに対する補助フレームのクォータニオン
    float invSampleFreq;        // サンプリング周波数の逆数
    float roll, pitch, yaw;     // オイラー角
    char anglesComputed;        // 角度計算済みフラグ

    // 高速な逆平方根計算
    // 参照: http://en.wikipedia.org/wiki/Fast_inverse_square_root
    inline float invSqrt(float x) {
        float halfx = 0.5f * x;
        float y = x;
        long i = *(long*)&y;
        i = 0x5f3759df - (i>>1);
        y = *(float*)&i;
        y = y * (1.5f - (halfx * y * y));
        y = y * (1.5f - (halfx * y * y));
        return y;
    }

    // オイラー角の計算
    inline void computeAngles() {
        roll = atan2f(q0*q1 + q2*q3, 0.5f - q1*q1 - q2*q2);
        pitch = asinf(-2.0f * (q1*q3 - q0*q2));
        yaw = atan2f(q1*q2 + q0*q3, 0.5f - q2*q2 - q3*q3);
        anglesComputed = 1;
    }

public:
    // エラーチェック用の関数
    bool isValidFloat(float value) {
        return !isnan(value) && !isinf(value);
    }

    // エラー状態
    bool error_state = false;

    // コンストラクタ - 初期値の設定
    Madgwick() : beta(betaDef), q0(1.0f), q1(0.0f), q2(0.0f), q3(0.0f),
                 invSampleFreq(1.0f / sampleFreqDef), anglesComputed(0), error_state(false) {}

    // サンプリング周波数の設定
    void begin(float sampleFrequency) {
        if (sampleFrequency > 0.0f) {
            invSampleFreq = 1.0f / sampleFrequency;
        }
        error_state = false;
        // クォータニオンの再初期化
        q0 = 1.0f;
        q1 = q2 = q3 = 0.0f;
        anglesComputed = 0;
    }

    // 9DoFセンサーデータによるAHRSアルゴリズムの更新
    void update(float gx, float gy, float gz, float ax, float ay, float az, float mx, float my, float mz) {
        if (error_state) return;

        // 入力値のチェック
        if (!isValidFloat(gx) || !isValidFloat(gy) || !isValidFloat(gz) ||
            !isValidFloat(ax) || !isValidFloat(ay) || !isValidFloat(az) ||
            !isValidFloat(mx) || !isValidFloat(my) || !isValidFloat(mz)) {
            error_state = true;
            return;
        }
        float recipNorm;
        float s0, s1, s2, s3;
        float qDot1, qDot2, qDot3, qDot4;
        float hx, hy;
        float _2q0mx, _2q0my, _2q0mz, _2q1mx, _2bx, _2bz, _4bx, _4bz;
        float _2q0, _2q1, _2q2, _2q3, _2q0q2, _2q2q3;
        float q0q0, q0q1, q0q2, q0q3, q1q1, q1q2, q1q3, q2q2, q2q3, q3q3;

        // 地磁気測定が無効な場合はIMUアルゴリズムを使用
        if((mx == 0.0f) && (my == 0.0f) && (mz == 0.0f)) {
            updateIMU(gx, gy, gz, ax, ay, az);
            return;
        }

        // ジャイロスコープの度/秒をラジアン/秒に変換
        gx *= 0.0174533f;
        gy *= 0.0174533f;
        gz *= 0.0174533f;

        // ジャイロスコープからのクォータニオンの変化率
        qDot1 = 0.5f * (-q1 * gx - q2 * gy - q3 * gz);
        qDot2 = 0.5f * (q0 * gx + q2 * gz - q3 * gy);
        qDot3 = 0.5f * (q0 * gy - q1 * gz + q3 * gx);
        qDot4 = 0.5f * (q0 * gz + q1 * gy - q2 * gx);

        // 加速度計の測定が有効な場合のみフィードバック計算
        if(!((ax == 0.0f) && (ay == 0.0f) && (az == 0.0f))) {
            // 加速度計測定の正規化
            recipNorm = invSqrt(ax * ax + ay * ay + az * az);
            ax *= recipNorm;
            ay *= recipNorm;
            az *= recipNorm;

            // 地磁気測定の正規化
            recipNorm = invSqrt(mx * mx + my * my + mz * mz);
            mx *= recipNorm;
            my *= recipNorm;
            mz *= recipNorm;

            // 繰り返しの演算を避けるための補助変数
            _2q0mx = 2.0f * q0 * mx;
            _2q0my = 2.0f * q0 * my;
            _2q0mz = 2.0f * q0 * mz;
            _2q1mx = 2.0f * q1 * mx;
            _2q0 = 2.0f * q0;
            _2q1 = 2.0f * q1;
            _2q2 = 2.0f * q2;
            _2q3 = 2.0f * q3;
            _2q0q2 = 2.0f * q0 * q2;
            _2q2q3 = 2.0f * q2 * q3;
            q0q0 = q0 * q0;
            q0q1 = q0 * q1;
            q0q2 = q0 * q2;
            q0q3 = q0 * q3;
            q1q1 = q1 * q1;
            q1q2 = q1 * q2;
            q1q3 = q1 * q3;
            q2q2 = q2 * q2;
            q2q3 = q2 * q3;
            q3q3 = q3 * q3;

            // 地球の磁場の基準方向
            hx = mx * q0q0 - _2q0my * q3 + _2q0mz * q2 + mx * q1q1 + _2q1 * my * q2 + _2q1 * mz * q3 - mx * q2q2 - mx * q3q3;
            hy = _2q0mx * q3 + my * q0q0 - _2q0mz * q1 + _2q1mx * q2 - my * q1q1 + my * q2q2 + _2q2 * mz * q3 - my * q3q3;
            _2bx = sqrtf(hx * hx + hy * hy);
            _2bz = -_2q0mx * q2 + _2q0my * q1 + mz * q0q0 + _2q1mx * q3 - mz * q1q1 + _2q2 * my * q3 - mz * q2q2 + mz * q3q3;
            _4bx = 2.0f * _2bx;
            _4bz = 2.0f * _2bz;

            // 勾配降下アルゴリズムの修正ステップ
            s0 = -_2q2 * (2.0f * q1q3 - _2q0q2 - ax) + _2q1 * (2.0f * q0q1 + _2q2q3 - ay) - _2bz * q2 * (_2bx * (0.5f - q2q2 - q3q3) + _2bz * (q1q3 - q0q2) - mx) + (-_2bx * q3 + _2bz * q1) * (_2bx * (q1q2 - q0q3) + _2bz * (q0q1 + q2q3) - my) + _2bx * q2 * (_2bx * (q0q2 + q1q3) + _2bz * (0.5f - q1q1 - q2q2) - mz);
            s1 = _2q3 * (2.0f * q1q3 - _2q0q2 - ax) + _2q0 * (2.0f * q0q1 + _2q2q3 - ay) - 4.0f * q1 * (1 - 2.0f * q1q1 - 2.0f * q2q2 - az) + _2bz * q3 * (_2bx * (0.5f - q2q2 - q3q3) + _2bz * (q1q3 - q0q2) - mx) + (_2bx * q2 + _2bz * q0) * (_2bx * (q1q2 - q0q3) + _2bz * (q0q1 + q2q3) - my) + (_2bx * q3 - _4bz * q1) * (_2bx * (q0q2 + q1q3) + _2bz * (0.5f - q1q1 - q2q2) - mz);
            s2 = -_2q0 * (2.0f * q1q3 - _2q0q2 - ax) + _2q3 * (2.0f * q0q1 + _2q2q3 - ay) - 4.0f * q2 * (1 - 2.0f * q1q1 - 2.0f * q2q2 - az) + (-_4bx * q2 - _2bz * q0) * (_2bx * (0.5f - q2q2 - q3q3) + _2bz * (q1q3 - q0q2) - mx) + (_2bx * q1 + _2bz * q3) * (_2bx * (q1q2 - q0q3) + _2bz * (q0q1 + q2q3) - my) + (_2bx * q0 - _4bz * q2) * (_2bx * (q0q2 + q1q3) + _2bz * (0.5f - q1q1 - q2q2) - mz);
            s3 = _2q1 * (2.0f * q1q3 - _2q0q2 - ax) + _2q2 * (2.0f * q0q1 + _2q2q3 - ay) + (-_4bx * q3 + _2bz * q1) * (_2bx * (0.5f - q2q2 - q3q3) + _2bz * (q1q3 - q0q2) - mx) + (-_2bx * q0 + _2bz * q2) * (_2bx * (q1q2 - q0q3) + _2bz * (q0q1 + q2q3) - my) + _2bx * q1 * (_2bx * (q0q2 + q1q3) + _2bz * (0.5f - q1q1 - q2q2) - mz);
            recipNorm = invSqrt(s0 * s0 + s1 * s1 + s2 * s2 + s3 * s3); // ステップ量の正規化
            s0 *= recipNorm;
            s1 *= recipNorm;
            s2 *= recipNorm;
            s3 *= recipNorm;

            // フィードバックステップの適用
            qDot1 -= beta * s0;
            qDot2 -= beta * s1;
            qDot3 -= beta * s2;
            qDot4 -= beta * s3;
        }

        // クォータニオンの積分
        q0 += qDot1 * invSampleFreq;
        q1 += qDot2 * invSampleFreq;
        q2 += qDot3 * invSampleFreq;
        q3 += qDot4 * invSampleFreq;

        // クォータニオンの正規化
        recipNorm = invSqrt(q0 * q0 + q1 * q1 + q2 * q2 + q3 * q3);
        q0 *= recipNorm;
        q1 *= recipNorm;
        q2 *= recipNorm;
        q3 *= recipNorm;
        anglesComputed = 0;
    }

    // 6DoFセンサーデータによるIMUアルゴリズムの更新
    void updateIMU(float gx, float gy, float gz, float ax, float ay, float az) {
        if (error_state) return;

        // 入力値のチェック
        if (!isValidFloat(gx) || !isValidFloat(gy) || !isValidFloat(gz) ||
            !isValidFloat(ax) || !isValidFloat(ay) || !isValidFloat(az)) {
            error_state = true;
            return;
        }

        // 加速度の大きさをチェック
        float acc_magnitude = sqrtf(ax * ax + ay * ay + az * az);
        if (acc_magnitude < 0.01f || acc_magnitude > 20.0f) {
            // 異常な加速度値
            return;
        }
        float recipNorm;
        float s0, s1, s2, s3;
        float qDot1, qDot2, qDot3, qDot4;
        float _2q0, _2q1, _2q2, _2q3, _4q0, _4q1, _4q2 ,_8q1, _8q2, q0q0, q1q1, q2q2, q3q3;

        // ジャイロスコープからのクォータニオンの変化率
        qDot1 = 0.5f * (-q1 * gx - q2 * gy - q3 * gz);
        qDot2 = 0.5f * (q0 * gx + q2 * gz - q3 * gy);
        qDot3 = 0.5f * (q0 * gy - q1 * gz + q3 * gx);
        qDot4 = 0.5f * (q0 * gz + q1 * gy - q2 * gx);

        // 加速度計の測定が有効な場合のみフィードバック計算
        if(!((ax == 0.0f) && (ay == 0.0f) && (az == 0.0f))) {

            // 加速度計測定の正規化
            recipNorm = invSqrt(ax * ax + ay * ay + az * az);
            ax *= recipNorm;
            ay *= recipNorm;
            az *= recipNorm;

            // 繰り返しの演算を避けるための補助変数
            _2q0 = 2.0f * q0;
            _2q1 = 2.0f * q1;
            _2q2 = 2.0f * q2;
            _2q3 = 2.0f * q3;
            _4q0 = 4.0f * q0;
            _4q1 = 4.0f * q1;
            _4q2 = 4.0f * q2;
            _8q1 = 8.0f * q1;
            _8q2 = 8.0f * q2;
            q0q0 = q0 * q0;
            q1q1 = q1 * q1;
            q2q2 = q2 * q2;
            q3q3 = q3 * q3;

            // 勾配降下アルゴリズムの修正ステップ
            s0 = _4q0 * q2q2 + _2q2 * ax + _4q0 * q1q1 - _2q1 * ay;
            s1 = _4q1 * q3q3 - _2q3 * ax + 4.0f * q0q0 * q1 - _2q0 * ay - _4q1 + _8q1 * q1q1 + _8q1 * q2q2 + _4q1 * az;
            s2 = 4.0f * q0q0 * q2 + _2q0 * ax + _4q2 * q3q3 - _2q3 * ay - _4q2 + _8q2 * q1q1 + _8q2 * q2q2 + _4q2 * az;
            s3 = 4.0f * q1q1 * q3 - _2q1 * ax + 4.0f * q2q2 * q3 - _2q2 * ay;
            recipNorm = invSqrt(s0 * s0 + s1 * s1 + s2 * s2 + s3 * s3); // ステップ量の正規化
            s0 *= recipNorm;
            s1 *= recipNorm;
            s2 *= recipNorm;
            s3 *= recipNorm;

            // フィードバックステップの適用
            qDot1 -= beta * s0;
            qDot2 -= beta * s1;
            qDot3 -= beta * s2;
            qDot4 -= beta * s3;
        }

        // クォータニオンの積分
        q0 += qDot1 * invSampleFreq;
        q1 += qDot2 * invSampleFreq;
        q2 += qDot3 * invSampleFreq;
        q3 += qDot4 * invSampleFreq;

        // クォータニオンの正規化
        recipNorm = invSqrt(q0 * q0 + q1 * q1 + q2 * q2 + q3 * q3);
        q0 *= recipNorm;
        q1 *= recipNorm;
        q2 *= recipNorm;
        q3 *= recipNorm;
        anglesComputed = 0;
    }

    // オイラー角の取得（度単位）
    float getRoll() {
        if (error_state) return 0.0f;
        if (!anglesComputed) computeAngles();
        return roll * 57.29578f;
    }
    float getPitch() {
        if (error_state) return 0.0f;
        if (!anglesComputed) computeAngles();
        return pitch * 57.29578f;
    }
    float getYaw() {
        if (error_state) return 0.0f;
        if (!anglesComputed) computeAngles();
        return yaw * 57.29578f + 180.0f;
    }

    // オイラー角の取得（ラジアン単位）
    float getRollRadians() {
        if (error_state) return 0.0f;
        if (!anglesComputed) computeAngles();
        return roll;
    }
    float getPitchRadians() {
        if (error_state) return 0.0f;
        if (!anglesComputed) computeAngles();
        return pitch;
    }
    float getYawRadians() {
        if (error_state) return 0.0f;
        if (!anglesComputed) computeAngles();
        return yaw;
    }

    // エラー状態のリセット
    void reset() {
        error_state = false;
        q0 = 1.0f;
        q1 = q2 = q3 = 0.0f;
        anglesComputed = 0;
    }

    // クォータニオンを使用して加速度を世界座標系に変換
    void convertAccelToWorldFrame(float ax, float ay, float az, float* world_acc) {
        if (error_state) {
            world_acc[0] = world_acc[1] = world_acc[2] = 0.0f;
            return;
        }

        // クォータニオンを使用して回転を適用
        float qw = q0, qx = q1, qy = q2, qz = q3;
        
        // 重力ベクトルの計算（センサー座標系）
        float gx = 2 * (qx * qz - qw * qy) * 9.80665f;
        float gy = 2 * (qw * qx + qy * qz) * 9.80665f;
        float gz = (qw * qw - qx * qx - qy * qy + qz * qz) * 9.80665f;
        
        // 重力を補正
        float acc_no_gravity[3] = {ax - gx, ay - gy, az - gz};
        
        // 世界座標系に変換
        world_acc[0] = (1 - 2*qy*qy - 2*qz*qz) * acc_no_gravity[0] + 
                      (2*qx*qy - 2*qz*qw) * acc_no_gravity[1] + 
                      (2*qx*qz + 2*qy*qw) * acc_no_gravity[2];
        
        world_acc[1] = (2*qx*qy + 2*qz*qw) * acc_no_gravity[0] + 
                      (1 - 2*qx*qx - 2*qz*qz) * acc_no_gravity[1] + 
                      (2*qy*qz - 2*qx*qw) * acc_no_gravity[2];
        
        world_acc[2] = (2*qx*qz - 2*qy*qw) * acc_no_gravity[0] + 
                      (2*qy*qz + 2*qx*qw) * acc_no_gravity[1] + 
                      (1 - 2*qx*qx - 2*qy*qy) * acc_no_gravity[2];
    }
};

#endif // MADGWICK_FILTER_HPP
