import cv2
import socket
import numpy as np


def get_image(sock, buffer):
    while True:
        data, _ = sock.recvfrom(65000)  # Max UDP packet size
        buffer.extend(data)

        # Check for JPEG end marker
        if len(buffer) > 2 and buffer[-2:] == b'\xff\xd9':
            try:
                frame = cv2.imdecode(np.frombuffer(buffer, dtype=np.uint8), cv2.IMREAD_COLOR)
                if frame is not None:
                    return frame, bytearray()  # Return frame and reset buffer
                else:
                    print("Decoding failed, resetting buffer")
                    return None, bytearray()  # Reset buffer on failure
            except Exception as e:
                print(f"Error decoding frame: {e}")
                return None, bytearray()  # Reset buffer on error


def create_server(multicast_address='239.192.1.1'):
    # UDP setup
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1) 
    
    # Spezieller Fix für macOS
    if hasattr(socket, 'SO_REUSEPORT'):
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
        
    # An alle verfügbaren Interfaces binden (leerer String ist auf Mac oft besser als '0.0.0.0')
    sock.bind(('', 5555))
    
    mreq = socket.inet_aton(multicast_address) + socket.inet_aton('0.0.0.0')
    sock.setsockopt(socket.IPPROTO_IP, socket.IP_ADD_MEMBERSHIP, mreq)

    buffer = bytearray()
    return sock, buffer


def stop_everything(sock):
    cv2.destroyAllWindows()
    sock.close()


def main():
    print("Hallo")
    sock, buffer = create_server()
    while True:
        try:
            frame, buffer = get_image(sock, buffer)  # Update buffer
            print(frame)
            if frame is not None:
                cv2.imshow('Received Frame', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Error: {e}")
            buffer = bytearray()  # Reset buffer on error

    print("Stopping client...")
    stop_everything(sock)


if __name__ == '__main__':
    main()
