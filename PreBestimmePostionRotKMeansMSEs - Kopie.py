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
from ultralytics import YOLO


def bestimme_position_aus_kmeans_und_weissmaske(
    mittelpunkte_k1,
    mittelpunkte_k2,
    mittelpunkte_k3,
    mse_k1,
    mse_k2,
    white_mask,
    radius_weiss=20
):
    if white_mask is None:
        return None

    # Fall 1: k=1 ist schon gut genug
    if mse_k1 < 400:
        if mittelpunkte_k1 is not None and len(mittelpunkte_k1) > 0:
            return mittelpunkte_k1[0]
        return None

    # Fall 2: k=2 ist gut genug -> nur diese beiden Zentren prüfen
    if mse_k2 < 300:
        kandidaten = mittelpunkte_k2
    else:
        # Sonst alle 3 Zentren prüfen
        kandidaten = mittelpunkte_k3

    if kandidaten is None or len(kandidaten) == 0:
        return None

    beste_position = None
    max_weiss = -1

    for mx, my in kandidaten:
        kreismaske = np.zeros_like(white_mask, dtype=np.uint8)
        cv2.circle(kreismaske, (int(mx), int(my)), radius_weiss, 255, -1)

        weisse_pixel = cv2.bitwise_and(white_mask, kreismaske)
        anzahl_weiss = cv2.countNonZero(weisse_pixel)

        print(f"Zentrum {(mx, my)} -> weiße Pixel im Radius {radius_weiss}: {anzahl_weiss}")

        if anzahl_weiss > max_weiss:
            max_weiss = anzahl_weiss
            beste_position = (int(mx), int(my))

    return beste_position

def rot_kmeans_zentren(frame_img, k=3):
    if frame_img is None:
        return None

    hsv = cv2.cvtColor(frame_img, cv2.COLOR_BGR2HSV)

    lower_red_1 = np.array([0, 100, 80])
    upper_red_1 = np.array([10, 255, 255])

    lower_red_2 = np.array([170, 100, 80])
    upper_red_2 = np.array([180, 255, 255])

    mask1 = cv2.inRange(hsv, lower_red_1, upper_red_1)
    mask2 = cv2.inRange(hsv, lower_red_2, upper_red_2)
    rotmaske = cv2.bitwise_or(mask1, mask2)

    ys, xs = np.where(rotmaske > 0)

    if len(xs) == 0:
        print("Keine roten Pixel gefunden")
        return None

    # Schwerpunkt = Mittelwert der roten Pixel
    cx = int(np.mean(xs))
    cy = int(np.mean(ys))

    # Kreis-Maske um den Schwerpunkt
    radius = 50
    hoehe, breite = rotmaske.shape
    kreis_maske = np.zeros_like(rotmaske)

    cv2.circle(kreis_maske, (cx, cy), radius, 255, -1)

    # Nur rote Pixel im Radius 50 behalten
    rotmaske = cv2.bitwise_and(rotmaske, kreis_maske)

    # Neue Punkte aus der beschnittenen Maske
    ys, xs = np.where(rotmaske > 0)

    if len(xs) < k:
        print(f"Zu wenige rote Pixel im Radius {radius} für k={k}")
        return None

    ys, xs = np.where(rotmaske > 0)

    if len(xs) < k:
        print(f"Zu wenige rote Pixel für k={k}")
        return None

    punkte = np.column_stack((xs, ys)).astype(np.float32)
    punkte = punkte.reshape(-1, 2)

    kriterium = (
        cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
        100,
        0.2
    )
    versuche = 10

    compactness, labels, centers = cv2.kmeans(
        punkte,
        k,
        None,
        kriterium,
        versuche,
        cv2.KMEANS_PP_CENTERS
    )

    labels = labels.flatten()
    centers = np.asarray(centers, dtype=np.float32)

    print("Form von punkte:", punkte.shape)
    print("Form von centers:", centers.shape)
    print("centers:", centers)

    if centers.ndim != 2 or centers.shape[1] != 2:
        print("Unerwartete Form von centers")
        return None

    mittelpunkte = []
    pixelanzahlen = []

    for i in range(k):
        anzahl = int(np.sum(labels == i))
        mittelpunkt = (int(centers[i, 0]), int(centers[i, 1]))

        mittelpunkte.append(mittelpunkt)
        pixelanzahlen.append(anzahl)

        print(f"Mittelpunkt {i+1}: {mittelpunkt} Pixel: {anzahl}")

    mse = float(compactness / len(punkte))

    return {
        "rotmaske": rotmaske,
        "mittelpunkte": mittelpunkte,
        "pixelanzahlen": pixelanzahlen,
        "fehler": float(compactness),
        "mse": mse
    }

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

def finde_umschlossene_gebiete(rotmaske, min_flaeche=20):
    if rotmaske is None:
        return []

    # Sicherheit: binär machen
    _, bin_mask = cv2.threshold(rotmaske, 127, 255, cv2.THRESH_BINARY)

    konturen, hierarchy = cv2.findContours(
        bin_mask,
        cv2.RETR_CCOMP,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if hierarchy is None:
        return []

    hierarchy = hierarchy[0]
    gebiete = []

    for i, h in enumerate(hierarchy):
        next_idx, prev_idx, child_idx, parent_idx = h

        # Außenkontur mit mindestens einem Loch
        if child_idx != -1:
            child = child_idx
            while child != -1:
                flaeche = cv2.contourArea(konturen[child])

                if flaeche >= min_flaeche:
                    M = cv2.moments(konturen[child])
                    if M["m00"] != 0:
                        cx = int(M["m10"] / M["m00"])
                        cy = int(M["m01"] / M["m00"])
                    else:
                        cx, cy = 0, 0

                    gebiete.append({
                        "kontur_index_aussen": i,
                        "kontur_index_innen": child,
                        "flaeche": flaeche,
                        "mittelpunkt": (cx, cy),
                        "kontur": konturen[child]
                    })

                child = hierarchy[child][0]  # nächstes Geschwister-Loch

    return gebiete




def example_main():


    # Create the parser
    parser = argparse.ArgumentParser(description='Sphero Racing basic node.')
    parser.add_argument('-n', '--name', help='Unique ID of the sphero to connect to, in format "SM-XXXX"',
                        default='SM-C080')
    parser.add_argument('-c', '--color', help='RGB Color to set in format 255,255,255', default='255,0,0')
    parser.add_argument('--show', action='store_true', help='Enable visual output')
    # ...   # add more arguments here
    args = parser.parse_args()

    # parse the color from a string with "," separators to RGB
    color = [int(x) for x in args.color.split(",")]
    assert len(color) == 3, f"Color string '{args.color}' was not parsed to three ints!"

    # prepare the udp socket for receiving images
    udp_socket, udp_buffer = get_udp_socket_and_buffer()

    last_frame = None


    old_time = time.time()

    last_frames = []

    while True:
        # get the newest image and update the udp buffer
        frame, udp_buffer = get_image(udp_socket, udp_buffer)
        # check args (and ensure frame is not None!)
        if frame is not None:
            # need cv2.waitKey to show image, and stop if q is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            last_frames.append(frame)
            if(len(last_frames) > 400):
                last_frames.pop(0)

            erg1 = rot_kmeans_zentren(frame, k=1)
            erg2 = rot_kmeans_zentren(frame, k=10)
            erg3 = rot_kmeans_zentren(frame, k=10)



            lower_white = np.array([0, 0, 180])
            upper_white = np.array([180, 60, 255])

            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            white_mask = cv2.inRange(hsv, lower_white, upper_white)

            position = bestimme_position_aus_kmeans_und_weissmaske(
                mittelpunkte_k1=erg1["mittelpunkte"] if erg1 else None,
                mittelpunkte_k2=erg2["mittelpunkte"] if erg2 else None,
                mittelpunkte_k3=erg3["mittelpunkte"] if erg3 else None,
                mse_k1=erg1["mse"] if erg1 else float("inf"),
                mse_k2=erg2["mse"] if erg2 else float("inf"),
                white_mask=white_mask,
                radius_weiss=5
            )

            lower_red_1 = np.array([0, 100, 80])
            upper_red_1 = np.array([10, 255, 255])

            lower_red_2 = np.array([170, 100, 80])
            upper_red_2 = np.array([180, 255, 255])

            mask1 = cv2.inRange(hsv, lower_red_1, upper_red_1)
            mask2 = cv2.inRange(hsv, lower_red_2, upper_red_2)
            rotmaske = cv2.bitwise_or(mask1, mask2)

            #cv2.imshow("rotmaske", rotmaske)
            #cv2.imshow("Weissmaske", white_mask)




            gebiete = finde_umschlossene_gebiete(rotmaske, min_flaeche=18)

            if len(gebiete) > 0:
                print("Es gibt von Rot umschlossene Gebiete.")
                for g in gebiete:
                    print("Lochfläche:", g["flaeche"], "Mittelpunkt:", g["mittelpunkt"])
            else:
                print("Kein umschlossenes Gebiet gefunden.")


            anzeige = cv2.cvtColor(rotmaske, cv2.COLOR_GRAY2BGR)

            gebiete = finde_umschlossene_gebiete(rotmaske, min_flaeche=10)

            for g in gebiete:
                cx, cy = g["mittelpunkt"]
                cv2.drawContours(anzeige, [g["kontur"]], -1, (0, 255, 0), 2)
                cv2.circle(anzeige, (cx, cy), 4, (0, 0, 255), -1)
                cv2.putText(
                    anzeige,
                    f"A={int(g['flaeche'])}",
                    (cx + 5, cy - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    1,
                    cv2.LINE_AA
                )

            cv2.imshow("Umschlossene Gebiete", anzeige)

            cv2.imshow("Stream", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break




            # ich habe ein frame "frame" das ich mit cv2.imshow("Tracking", frame) zeige. auf dem bild befindet sich ein sphero (sieht von oben aus wie golfball). kannst du
            # mir die vorhersage von yolo für die position des  golfball geben ?

            # kannst du jetzt eine methode schreiben, die den ball aufgrund seines shapes (rund) detektiert und analog die position und den radius zurückgibt ?
            # hierzu finde erst alles runde im bild und zeige mir dannach alles, was weiß und rund ist auf dem bildschirm an.














if __name__ == '__main__':
    example_main()
