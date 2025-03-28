#ifndef LOW_PASS_FILTER_HPP
#define LOW_PASS_FILTER_HPP

class LowPassFilter {
private:
    // フィルタパラメータ
    float alpha;  // カットオフ周波数を決定する係数 (0.0 < alpha < 1.0)
    float filtered_value[3];  // 3軸のフィルタリング後の値

public:
    // コンストラクタ
    LowPassFilter(float alpha_param = 0.1f) : alpha(alpha_param) {
        filtered_value[0] = 0.0f;
        filtered_value[1] = 0.0f;
        filtered_value[2] = 0.0f;
    }

    // フィルタ係数の設定
    void setAlpha(float new_alpha) {
        if (new_alpha > 0.0f && new_alpha < 1.0f) {
            alpha = new_alpha;
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
