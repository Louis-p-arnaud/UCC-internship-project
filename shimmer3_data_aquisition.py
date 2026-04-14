# pip install pyshimmer
from pyshimmer import ShimmerBluetooth, DEFAULT_BAUDRATE, DataPacket, EChannelType
import serial


def stream_shimmer(port="/dev/rfcomm0"):
    serial_port = serial.Serial(port, DEFAULT_BAUDRATE)
    shim = ShimmerBluetooth(serial_port)
    shim.initialize()

    # Active GSR + PPG
    shim.set_sampling_rate(64.0)  # 64 Hz = optimal rate for NormWear
    shim.start_streaming()

    buffer = []
    while True:
        pkt: DataPacket = shim.read_data_packet_extended(convert=True)
        gsr = pkt[EChannelType.GSR_RAW]
        ppg = pkt[EChannelType.INTERNAL_ADC_A13]
        buffer.append([gsr, ppg])
        yield buffer