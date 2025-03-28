#ifndef MULTI_IMU_HPP
#define MULTI_IMU_HPP

#include <stdio.h>
#include <sys/ioctl.h>
#include <fcntl.h>
#include <nuttx/sensors/cxd5602pwbimu.h>
#include <arch/board/cxd56_cxd5602pwbimu.h>

class MultiIMU {
public:
    static const int ACC_DATA_LENGTH = 3;   // ax, ay, az
    static const int GYRO_DATA_LENGTH = 3;  // gx, gy, gz
    static const int TEMP_DATA_LENGTH = 1;  // temp
    static const int DATA_LENGTH = ACC_DATA_LENGTH + GYRO_DATA_LENGTH + TEMP_DATA_LENGTH;

    MultiIMU() : fd(-1) {}

    // IMUの初期化
    bool begin() {
        board_cxd5602pwbimu_initialize(5);
        fd = open(CXD5602PWBIMU_DEVPATH, O_RDONLY);
        return fd >= 0;
    }

    // センシングの開始 rate[1920, 960, 480, 240, 120, 60, 30, 15] adrange[16, 8, 4, 2], gdrange[4000, 2000, 1000, 500, 250, 125]
    bool startSensing(int rate, int adrange, int gdrange) {
        if (fd < 0) return false;

        cxd5602pwbimu_range_t range;
        ioctl(fd, SNIOC_SSAMPRATE, rate);
        range.accel = adrange;
        range.gyro = gdrange;
        ioctl(fd, SNIOC_SDRANGE, (unsigned long)(uintptr_t)&range);
        ioctl(fd, SNIOC_SFIFOTHRESH, MAX_NFIFO);
        ioctl(fd, SNIOC_ENABLE, 1);

        // 初期の50msのデータを破棄
        int cnt = rate / 20; // 50msのデータサイズ
        cnt = ((cnt + MAX_NFIFO - 1) / MAX_NFIFO) * MAX_NFIFO;
        if (cnt == 0) cnt = MAX_NFIFO;

        while (cnt) {
            read(fd, rawData, sizeof(rawData[0]) * MAX_NFIFO);
            cnt -= MAX_NFIFO;
        }

        return true;
    }

    // データの取得
    bool getData(float* data) {
        if (fd < 0 || data == nullptr) return false;

        if (!readRawData()) return false;
        convertRawData(data);
        return true;
    }

private:
    static const int MAX_NFIFO = 4;
    static constexpr const char* CXD5602PWBIMU_DEVPATH = "/dev/imu0";

    int fd;
    cxd5602pwbimu_data_t rawData[MAX_NFIFO];

    // 生データの読み込み
    bool readRawData() {
        int ret = read(fd, rawData, sizeof(rawData[0]) * MAX_NFIFO);
        return ret == sizeof(rawData[0]) * MAX_NFIFO;
    }

    // 生データの変換
    void convertRawData(float* data) {
        // 最新のデータ（配列の最後のデータ）を使用
        const auto& latest = rawData[MAX_NFIFO - 1];
        
        // 加速度データ m/s^2
        data[0] = latest.ax;
        data[1] = latest.ay;
        data[2] = latest.az;
        
        // 角速度データ rad/s
        data[3] = latest.gx;
        data[4] = latest.gy;
        data[5] = latest.gz;
        
        // 温度データ
        data[6] = latest.temp;
    }
};

#endif // MULTI_IMU_HPP
