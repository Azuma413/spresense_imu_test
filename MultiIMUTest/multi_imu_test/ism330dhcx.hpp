#ifndef ISM330DHCX_HPP
#define ISM330DHCX_HPP

#include <Wire.h>
#include "SparkFun_ISM330DHCX.h"
#include "multi_imu.hpp"

class ISM330DHCX : public MultiIMU {
public:
    ISM330DHCX() {}

    // IMUの初期化
    bool begin() {
        Wire.begin();
        Wire.setClock(400000); // 400kHz I2C

        // デバイスのアドレススキャン（0x60-0x6F）
        for (int i = 0x60; i <= 0x6F; i++) {
            Wire.beginTransmission(i);
            if (Wire.endTransmission() == 0) {
                if (imu.begin(i)) {
                    // センサーの初期設定
                    imu.deviceReset();
                    while (!imu.getDeviceReset()) {
                        delay(1);
                    }
                    
                    imu.setDeviceConfig();
                    imu.setBlockDataUpdate();
                    
                    // 初期フィルタ設定
                    imu.setAccelFilterLP2();
                    imu.setAccelSlopeFilter(ISM_LP_ODR_DIV_100);
                    imu.setGyroFilterLP1();
                    imu.setGyroLP1Bandwidth(ISM_MEDIUM);
                    
                    return true;
                }
            }
        }
        return false;
    }

    // センシングの開始
    bool startSensing(int rate, int adrange, int gdrange) {
        // サンプルプログラムと同じ設定を使用
        imu.setAccelDataRate(ISM_XL_ODR_208Hz);
        imu.setAccelFullScale(ISM_4g);
        imu.setGyroDataRate(ISM_GY_ODR_104Hz);
        imu.setGyroFullScale(ISM_250dps);
        return true;
    }

    // データの取得
    bool getData(float* data) {
        if (data == nullptr) return false;

        // 加速度データの読み取り
        if (imu.checkAccelStatus()) {
            imu.getAccel(&accelData);
        }

        // ジャイロデータの読み取り
        if (imu.checkGyroStatus()) {
            imu.getGyro(&gyroData);
        }

        // データの格納（MultiIMUと同じ形式）
        // 加速度データ
        data[0] = accelData.xData;
        data[1] = accelData.yData;
        data[2] = accelData.zData;

        // 角速度データ
        data[3] = gyroData.xData;
        data[4] = gyroData.yData;
        data[5] = gyroData.zData;

        // 温度データ（ISM330DHCXの温度データ取得方法に応じて実装）
        // 注：この実装では仮の値として0を設定
        data[6] = 0;

        return true;
    }

private:
    SparkFun_ISM330DHCX imu;
    sfe_ism_data_t accelData;
    sfe_ism_data_t gyroData;
};

#endif // ISM330DHCX_HPP
