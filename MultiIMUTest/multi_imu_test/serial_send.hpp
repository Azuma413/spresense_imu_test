#ifndef SERIAL_SEND_HPP
#define SERIAL_SEND_HPP

#include <Arduino.h>
#include <stdint.h>
#include <string.h>

class SerialSend {
public:
    // UDPSendと同様にサイズを指定するコンストラクタ
    SerialSend(size_t size) {
        this->size = size * sizeof(float);
    }

    // シリアル通信の初期化
    void init() {
        Serial.begin(115200);
        while (!Serial) {
            ; // シリアルポートの準備待ち
        }
    }

    // float配列データをシリアル送信
    void send(const float* data) {
        uint8_t buffer[size];
        memcpy(buffer, data, size);
        Serial.write(buffer, size);
    }

private:
    size_t size;    // データサイズ（float型配列の要素数 * sizeof(float)）
};

#endif // SERIAL_SEND_HPP
