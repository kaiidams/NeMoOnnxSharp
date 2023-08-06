import pyaudio as pa
import socket
import time

SAMPLE_RATE = 16000
CHANNELS = 1
CHUNK_SIZE = 1024
PORT = 17843


def main():
    p = pa.PyAudio()

    input_devices = []
    for i in range(p.get_device_count()):
        dev = p.get_device_info_by_index(i)
        if dev.get('maxInputChannels', 0) >= 1:
            device_name = dev.get('name')
            input_devices.append(device_name)

    if not input_devices:
        print('No audio input device found.')

    if False:
        print('Available audio input devices:')
        for i, device_name in enumerate(input_devices):
            print(f'{i}: {device_name}')

    device_index = 0
    device_name = input_devices[device_index]
    print(f'Using audio input device: {device_index} {device_name}')

    serversocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    serversocket.bind(("0.0.0.0", PORT))
    serversocket.listen(1)
    print(f'Listening TCP port {PORT}')


    while True:
        (clientsocket, address) = serversocket.accept()

        empty_counter = 0

        def callback(in_data, frame_count, time_info, status):
            clientsocket.send(in_data)
            return (in_data, pa.paContinue)

        stream = p.open(format=pa.paInt16,
                        channels=CHANNELS,
                        rate=SAMPLE_RATE,
                        input=True,
                        input_device_index=device_index,
                        stream_callback=callback,
                        frames_per_buffer=CHUNK_SIZE)

        stream.start_stream()
        
        try:
            while stream.is_active():
                time.sleep(0.1)
        except:
            pass
        finally:
            stream.stop_stream()
            stream.close()
            p.terminate()

            print("Connection closed")

        clientsocket.close()


main()