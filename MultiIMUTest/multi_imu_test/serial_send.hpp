#ifndef SERIAL_SEND_HPP
#define SERIAL_SEND_HPP

#include <Arduino.h>
#include <stdint.h>
#include <string.h>

// パケットフォーマットの定義
#define PACKET_HEADER 0xAA
#define PACKET_FOOTER 0x55

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
        
        // ヘッダー送信
        Serial.write(PACKET_HEADER);
        
        // データ長送信
        Serial.write(static_cast<uint8_t>(size));
        
        // データ送信
        Serial.write(buffer, size);
        
        // チェックサム計算＆送信
        uint8_t checksum = calculateChecksum(buffer, size);
        Serial.write(checksum);
        
        // フッター送信
        Serial.write(PACKET_FOOTER);
    }

private:
    size_t size;    // データサイズ（float型配列の要素数 * sizeof(float)）
    
    // チェックサム計算
    uint8_t calculateChecksum(const uint8_t* data, size_t length) {
        uint8_t sum = 0;
        for (size_t i = 0; i < length; i++) {
            sum += data[i];
        }
        return sum;
    }
};

#endif // SERIAL_SEND_HPP
