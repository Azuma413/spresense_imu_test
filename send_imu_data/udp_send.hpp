#ifndef UDP_SEND_HPP
#define UDP_SEND_HPP

#include <TelitWiFi.h>

#define UDPSRVR_IP     "192.168.0.255"
#define UDPSRVR_PORT   "8888"
#define LOCAL_PORT     "10001"

class UDPSend {
public:
    UDPSend(const char* ssid, const char* password, size_t size, int id = 0): ssid(ssid), password(password), id(id) {
		this->size = size*sizeof(float);
	}
    void init();
    void send(const float* data);

private:
    TelitWiFi gs2200;
    TWIFI_Params gsparams;
	size_t size;
	const char* ssid;
	const char* password;
	int id;
	char server_cid = 0;
};

void UDPSend::init() {
	// AtCmd_Init();
	Init_GS2200_SPI_type(iS110B_TypeC);
	gsparams.mode = ATCMD_MODE_STATION;
	gsparams.psave = ATCMD_PSAVE_DEFAULT;
	if (gs2200.begin(gsparams)) {
		Serial.println("GS2200 initialization failed");
		return;
	}
	if (gs2200.activate_station(ssid, password)) {
		Serial.println("GS2200 station activation failed");
		return;
	}
	bool served_flag = false;
	while (!served_flag) {
		server_cid = gs2200.connectUDP(UDPSRVR_IP, UDPSRVR_PORT, LOCAL_PORT);
		Serial.println(server_cid);
		if (server_cid == ATCMD_INVALID_CID) {
			continue;
		}
		served_flag = true;
	}
}

void UDPSend::send(const float* data) {
    // Create a new buffer that includes the ID at the beginning
    float data_with_id[size/sizeof(float)];
    data_with_id[0] = id;
    
    // Copy the rest of the data
    for (size_t i = 1; i < size/sizeof(float); i++) {
        data_with_id[i] = data[i-1];
    }
    
    uint8_t buffer[size];
    memcpy(buffer, data_with_id, size);
    gs2200.write(server_cid, buffer, size);
}

#endif // UDP_SEND_HPP
