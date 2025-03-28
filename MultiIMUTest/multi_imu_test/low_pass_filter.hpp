#ifndef LOW_PASS_FILTER_HPP
#define LOW_PASS_FILTER_HPP

#include <math.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

class LowPassFilter {
private:
    // フィルタパラメータ
    float alpha;  // カットオフ周波数を決定する係数 (0.0 < alpha < 1.0)
    float filtered_value[3];  // 3軸のフィルタリング後の値

public:
    // コンストラクタ
    // alpha = dt / (RC + dt), RC = 1/(2π * fc)
    // fc: カットオフ周波数(Hz), dt: サンプリング周期(s)
    LowPassFilter(float cutoff_freq = 20.0f, float sample_period = 1.0f/960.0f) {
        const float RC = 1.0f / (2.0f * M_PI * cutoff_freq);
        alpha = sample_period / (RC + sample_period);
        filtered_value[0] = 0.0f;
        filtered_value[1] = 0.0f;
        filtered_value[2] = 0.0f;
    }

    // カットオフ周波数の設定
    void setCutoffFreq(float cutoff_freq, float sample_period = 1.0f/960.0f) {
        if (cutoff_freq > 0.0f) {
            const float RC = 1.0f / (2.0f * M_PI * cutoff_freq);
            alpha = sample_period / (RC + sample_period);
        }
    }

    // 3軸データのフィルタリング
    void update(const float* input, float* output) {
        for (int i = 0; i < 3; i++) {
            filtered_value[i] = alpha * input[i] + (1.0f - alpha) * filtered_value[i];
            output[i] = filtered_value[i];
        }
    }

    // フィルタ値のリセット
    void reset() {
        for (int i = 0; i < 3; i++) {
            filtered_value[i] = 0.0f;
        }
    }
};

#endif // LOW_PASS_FILTER_HPP
