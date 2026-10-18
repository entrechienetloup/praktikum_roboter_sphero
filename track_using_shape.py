"""
This script provides a simple baseline to get you started for the sphero-racing project.
Usage:

To print some help:
    python get_started.py -h

Example application, showing images and connecting to sphero "SM-B9F7", letting it glow in red (255, 0, 0)
    python get_started.py --show --color=255,0,0 --name=SM-B9F7

"""
import numpy as np
import cv2
import socket
import time
import argparse

from spherov2.sphero_edu import SpheroEduAPI
from spherov2.types import Color
from spherov2 import scanner


def get_image(udp_socket, udp_buffer):
    """
    Get an image and the resulting udp_buffer given the udp socket and buffer pair.
    The returned image will be None if any error occurs when getting the image.
    :param udp_socket: socket to use
    :param udp_buffer: current buffer
    :return: (frame, buffer) where frame is a cv2 image of the current data. frame will be None in case there was
    any error
    """
    while True:
        data, _ = udp_socket.recvfrom(65000)  # Max UDP packet size
        udp_buffer.extend(data)

        # Check for JPEG end marker
        if len(udp_buffer) > 2 and udp_buffer[-2:] == b'\xff\xd9':
            try:
                frame = cv2.imdecode(np.frombuffer(udp_buffer, dtype=np.uint8), cv2.IMREAD_COLOR)
                if frame is not None:
                    return frame, bytearray()  # Return frame and reset buffer
                else:
                    print("Decoding failed, resetting buffer")
                    return None, bytearray()  # Reset buffer on failure
            except Exception as e:
                print(f"Error decoding frame: {e}")
                return None, bytearray()  # Reset buffer on error


def get_udp_socket_and_buffer(multicast_address: str = '239.192.1.1'):
    """
    Given a multicast address, return udp socket and buffer.
    In general, you probably don't want to change from the default value '239.192.1.1', which corresponds to the server
    hosted for the practical course.
    :param multicast_address: IPv4 multicast address
    :return: (socket, buffer) for the udp image receiver
    """
    # UDP setup
    udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    udp_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)  # Allow address reuse
    udp_socket.bind(('0.0.0.0', 5555))
    mreq = socket.inet_aton(multicast_address) + socket.inet_aton('0.0.0.0')
    udp_socket.setsockopt(socket.IPPROTO_IP, socket.IP_ADD_MEMBERSHIP, mreq)

    udp_buffer = bytearray()
    return udp_socket, udp_buffer


def set_color(color, droid_api):
    """
    Given a droid_api, set the color of the sphero to the given RGB value
    :param color: [R, G, B] value to set the sphero to. Values are in [0, 255], e.g., red is [255, 0, 0]
    :param droid_api: droid_api object used to set the color
    """
    droid_api.set_main_led(Color(*color))


def set_heading(heading, droid_api: SpheroEduAPI):
    """
    Given a droid_api, set the heading of the sphero
    :param heading: heading in [0, 360] to set
    :param droid_api: droid_api object used to set the color
    """
    heading = int(heading % 360)
    droid_api.set_heading(heading)


def set_speed(speed, droid_api: SpheroEduAPI):
    """
    Given a droid_api, set the forward speed of the sphero. Negative values indicate backward speeds.
    :param speed: speed in [-255, 255] to set.
    :param droid_api: droid_api object used to set the color
    """
    speed = int(np.clip(speed, -255, 255))
    droid_api.set_speed(speed)


def track_ball_shape(frame, debug=False):
    """
    Tracks a ball by shape (circle).
    Returns: (x, y, r) or None
    """

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (9, 9), 2)

    circles = cv2.HoughCircles(
        gray,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=25,
        param1=120,   # edge detector threshold
        param2=14,    # circle center threshold (lower => more detections)
        minRadius=4,
        maxRadius=35
    )

    if circles is None:
        if debug:
            cv2.imshow("shape-gray", gray)
        return None

    circles = np.round(circles[0]).astype(int)

    # If multiple circles: choose the best by "bright center" score
    best = None
    best_score = -1.0

    h, w = gray.shape

    for (x, y, r) in circles:
        if r <= 0:
            continue
        if x < 0 or y < 0 or x >= w or y >= h:
            continue

        # Score = mean brightness inside the circle (ball/LED usually brighter than background)
        mask = np.zeros_like(gray, dtype=np.uint8)
        cv2.circle(mask, (x, y), r, 255, -1)

        score = cv2.mean(gray, mask=mask)[0]

        if score > best_score:
            best_score = score
            best = (int(x), int(y), int(r))

    if debug:
        vis = frame.copy()
        for (x, y, r) in circles:
            cv2.circle(vis, (x, y), r, (0, 255, 255), 1)
        if best is not None:
            bx, by, br = best
            cv2.circle(vis, (bx, by), br, (0, 255, 0), 2)
            cv2.circle(vis, (bx, by), 2, (0, 255, 0), -1)
        cv2.imshow("shape-candidates", vis)

    return best



def example_main():
    # Create the parser
    parser = argparse.ArgumentParser(description='Sphero Racing basic node.')
    parser.add_argument('-n', '--name', help='Unique ID of the sphero to connect to, in format "SM-XXXX"',
                        default='SM-CFB1')
    parser.add_argument('-c', '--color', help='RGB Color to set in format 255,255,255', default='255,0,0')
    parser.add_argument('--show', action='store_true', help='Enable visual output')
    # ...   # add more arguments here
    args = parser.parse_args()

    # parse the color from a string with "," separators to RGB
    color = [int(x) for x in args.color.split(",")]
    assert len(color) == 3, f"Color string '{args.color}' was not parsed to three ints!"

    # prepare the udp socket for receiving images
    udp_socket, udp_buffer = get_udp_socket_and_buffer()
    # connect to the sphero
    toy = scanner.find_toy(toy_name=args.name)
    try:  # this try-except wraps around everything to catch KeyboardInterrupts and shutdown in the end!
        with SpheroEduAPI(toy) as droid_api:  # this is always needed, the droid_api object is what's used to connect

            # ============================ Example code / YOUR CODE GOES HERE
            """
            This example code shows a video of the received video data, and simultaneously lets the robot drive up and 
            down at a fixed speed. Every one second, the robot turns around.
            
            Note that if you want to use cv2.imshow(), using time.sleep(1) to let the robot drive will also freeze your
            imshow. This is one way to avoid that, but of course there are others
            """
            heading_index = 0
            headings = [0,90,180,270]

            orentation = None
            last_position = None
            position = None
            colors = [[255,0,0], [0,255,0], [0,0,255]]
            index_color = 0
            history = []

            frames = []

            old_time = time.time()
            set_speed(30,droid_api)

            while True:  # we simply keep going forever!
                frame, udp_buffer = get_image(udp_socket, udp_buffer)  # get the newest image and update the udp buffer
                if frame is not None:  # check args (and ensure frame is not None!)
                    if cv2.waitKey(1) & 0xFF == ord('q'):  # need cv2.waitKey to show image, and stop if q is pressed
                        break  # stop the overall while loop, leading to the shutdown process
                    # read image

                    if(time.time() - old_time) > 1:
                        heading_index = (heading_index +1) % len(headings)
                        set_heading(headings[heading_index],droid_api)




                    # show frame
                    cv2.imshow('Received Frame', frame)
                    pos = track_ball_shape(frame, debug=False)
                    vis = frame.copy()

                    if pos is not None:
                        x, y, r = pos
                        cv2.circle(vis, (x, y), r, (0, 255, 0), 2)
                        cv2.circle(vis, (x, y), 2, (0, 255, 0), -1)

                    cv2.imshow("Tracking", vis)
                    #cv2.imshow("R channel", r)
                    #cv2.imshow("Red mask", mask)




                    #


    except KeyboardInterrupt:
        pass
    finally:  # perform shutdown
        # stop sphero (speed 0)
        with SpheroEduAPI(toy) as droid_api:
            droid_api.set_speed(0)
        # close cv2 windows in case some are open
        cv2.destroyAllWindows()
        # close the udp socket
        udp_socket.close()


if __name__ == '__main__':
    example_main()
