"""
This script provides a simple baseline to get you started for the sphero-racing project.
Usage:
    python get_started.py --show --color=255,0,0 --name=SM-B9F7
"""
import numpy as np
import cv2
import socket
import time
import argparse
import math
import random
from collections import deque

from spherov2.sphero_edu import SpheroEduAPI
from spherov2.types import Color
from spherov2 import scanner

# --- KONFIGURATION ---
START_POS = (460, 190)
FINAL_TARGET = (430, 60)
GRID_SIZE = 20
DANGER_BUFFER = 5

def get_image(udp_socket, udp_buffer):
    while True:
        data, _ = udp_socket.recvfrom(65000)
        udp_buffer.extend(data)
        if len(udp_buffer) > 2 and udp_buffer[-2:] == b'\xff\xd9':
            try:
                frame = cv2.imdecode(np.frombuffer(udp_buffer, dtype=np.uint8), cv2.IMREAD_COLOR)
                if frame is not None:
                    return frame, bytearray()
                else:
                    return None, bytearray()
            except Exception:
                return None, bytearray()


def get_udp_socket_and_buffer(multicast_address='239.192.1.1'):
    udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    udp_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    udp_socket.bind(('0.0.0.0', 5555))
    mreq = socket.inet_aton(multicast_address) + socket.inet_aton('0.0.0.0')
    udp_socket.setsockopt(socket.IPPROTO_IP, socket.IP_ADD_MEMBERSHIP, mreq)
    return udp_socket, bytearray()


def set_color(color, droid_api):
    droid_api.set_main_led(Color(*color))


def set_heading(heading, droid_api: SpheroEduAPI):
    droid_api.set_heading(int(heading % 360))


def set_speed(speed, droid_api: SpheroEduAPI):
    droid_api.set_speed(int(np.clip(speed, -255, 255)))


def click_event(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        print(f"Klick: ({x}, {y})")


# ==========================================================
# BILDVERARBEITUNG & LOGIK
# ==========================================================

def get_danger_zone(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_yellow = np.array([22, 100, 100])
    upper_yellow = np.array([40, 255, 255])

    # 1. Reine gelbe Wände erkennen
    raw_walls = cv2.inRange(hsv, lower_yellow, upper_yellow)

    # 2. Lücken schließen
    kernel_close = np.ones((5, 5), np.uint8)
    raw_walls = cv2.morphologyEx(raw_walls, cv2.MORPH_CLOSE, kernel_close)

    # 3. Gefahrenzone dick aufblasen (Puffer für den Roboter)
    kernel_dilate = np.ones((DANGER_BUFFER, DANGER_BUFFER), np.uint8)
    danger_zone = cv2.dilate(raw_walls, kernel_dilate, iterations=1)

    return danger_zone


def find_sphero(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # NEUE WERTE: Für den GRÜNEN Ball
    # H: 45-85 fängt sattes Grün ein und grenzt sich vom Gelb (ca. 20-40) sauber ab
    # S: 100-255 ignoriert den grauen Boden und die weißen Ping-Pong-Bälle
    # V: 100-255 ignoriert tiefe Schatten
    lower_green = np.array([35, 40, 40])
    upper_green = np.array([90, 255, 255])
    green_mask = cv2.inRange(hsv, lower_green, upper_green)

    # Rauschen entfernen, damit die Maske sauber bleibt
    kernel = np.ones((5, 5), np.uint8)
    green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_OPEN, kernel)
    green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_CLOSE, kernel)

    # Debug-Fenster anpassen
    cv2.imshow('Debug: Gruene Maske', green_mask)

    # Konturensuche
    contours, _ = cv2.findContours(green_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    best_center = None
    max_area = 0

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 10:
            if area > max_area:
                max_area = area
                M = cv2.moments(cnt)
                if M["m00"] > 0:
                    best_center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

    return best_center

def calculate_a_star_path(danger_zone, start_pos, target_pos, grid_size=GRID_SIZE):
    height, width = danger_zone.shape
    start_node = (start_pos[0] // grid_size, start_pos[1] // grid_size)
    target_node = (target_pos[0] // grid_size, target_pos[1] // grid_size)

    open_set = {start_node}
    came_from = {}
    g_score = {start_node: 0}

    def heuristic(a, b):
        return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)

    f_score = {start_node: heuristic(start_node, target_node)}

    def is_passable(gx, gy):
        x, y = gx * grid_size, gy * grid_size
        if x < 0 or y < 0 or x + grid_size > width or y + grid_size > height:
            return False
        cell = danger_zone[y:y + grid_size, x:x + grid_size]
        return cv2.countNonZero(cell) == 0

    if not is_passable(*start_node):
        print("A* WARNUNG: Startposition liegt in der Wand! Versuche trotzdem zu planen...")

    # Optionaler Fallback, falls das Ziel in der Wand liegt
    if not is_passable(*target_node):
        print("A* FEHLER: Ziel liegt in der Wand.")
        return None

    while open_set:
        current = min(open_set, key=lambda node: f_score.get(node, float('inf')))

        if current == target_node:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.reverse()

            waypoints = []
            if len(path) > 1:
                # Lege alle 5 Rasterpunkte (ca. 100 Pixel) einen "Brotkrümel" ab
                for i in range(1, len(path), 5):
                    waypoints.append(path[i])

            # Das finale Ziel immer als letzten Wegpunkt garantieren
            if path and path[-1] not in waypoints:
                waypoints.append(path[-1])

            final_waypoints = []
            for wp in waypoints:
                cx = wp[0] * grid_size + grid_size // 2
                cy = wp[1] * grid_size + grid_size // 2
                final_waypoints.append({"x": cx, "y": cy, "r": 40})
            return final_waypoints

        open_set.remove(current)

        for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0), (1, 1), (-1, -1), (1, -1), (-1, 1)]:
            neighbor = (current[0] + dx, current[1] + dy)
            if not is_passable(*neighbor):
                continue

            # Diagonales Schneiden verhindern
            if dx != 0 and dy != 0:
                if not is_passable(current[0] + dx, current[1]) or not is_passable(current[0], current[1] + dy):
                    continue

            cost = 1.41 if dx != 0 and dy != 0 else 1.0
            tentative_g_score = g_score[current] + cost

            if tentative_g_score < g_score.get(neighbor, float('inf')):
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g_score
                f_score[neighbor] = tentative_g_score + heuristic(neighbor, target_node)
                open_set.add(neighbor)

    return None
def check_collision_course(sphero_pos, danger_zone_mask):
    x, y = sphero_pos
    if y < danger_zone_mask.shape[0] and x < danger_zone_mask.shape[1]:
        if danger_zone_mask[y, x] > 0:
            return True
    return False

# ==========================================================
last_command_time = 0
COMMAND_INTERVAL = 0.1  # Nur alle 100ms ein Befehl an den Roboter
last_sent_heading = None
last_sent_speed = None


def send_safe_command(heading, speed, droid_api):
    global last_command_time, last_sent_heading, last_sent_speed
    current_time = time.time()

    # Nur senden, wenn genug Zeit vergangen ist ODER sich der Befehl massiv geändert hat
    if (current_time - last_command_time > COMMAND_INTERVAL) or (heading != last_sent_heading):
        set_heading(heading, droid_api)
        set_speed(speed, droid_api)
        last_command_time = current_time
        last_sent_heading = heading
        last_sent_speed = speed


def example_main():
    parser = argparse.ArgumentParser(description='Sphero Racing basic node.')
    parser.add_argument('-n', '--name', default='SM-B9F7')
    parser.add_argument('-c', '--color', default='255,0,0')
    parser.add_argument('--show', action='store_true')
    args = parser.parse_args()

    color = [int(x) for x in args.color.split(",")]
    udp_socket, udp_buffer = get_udp_socket_and_buffer()
    toy = scanner.find_toy(toy_name=args.name)

    try:
        with SpheroEduAPI(toy) as droid_api:
            set_color(color, droid_api)

            waypoint_areas = []
            static_path_calculated = False  # <-- NEU
            current_wp_index = 0

            state = "INIT"
            calib_start_time = 0
            calib_start_pos = None
            heading_offset = 0
            evade_start_time = 0
            lost_start_time = 0

            pos_history = deque(maxlen=15)

            cv2.namedWindow('Received Frame')
            cv2.setMouseCallback('Received Frame', click_event)
            static_path_calculated = False
            last_calc_time = 0
            while True:
                frame, udp_buffer = get_image(udp_socket, udp_buffer)

                if frame is not None:
                    danger_zone = get_danger_zone(frame)

                    # FIX: Den Ball IMMER suchen, damit die Variable für den Stream existiert!
                    sphero_pos = find_sphero(frame)

                    # --- 1. ROUTE BERECHNEN (Mit 2-Sekunden-Pause bei Fehler) ---
                    if not static_path_calculated:
                        if time.time() - last_calc_time > 2.0:
                            print(f"Prüfe Route von {START_POS} nach {FINAL_TARGET}...")
                            waypoint_areas = calculate_a_star_path(danger_zone, START_POS, FINAL_TARGET)

                            if waypoint_areas:
                                print(f"Route festgelegt: {len(waypoint_areas)} Wegpunkte.")
                                static_path_calculated = True
                            else:
                                print("FEHLER: Wand blockiert! Klicke in das Videofenster für neue Koordinaten.")

                            last_calc_time = time.time()

                    # --- 2. ROBOTER-STEUERUNG (Nur wenn Route existiert) ---
                            # --- 2. ROBOTER-STEUERUNG (Nur wenn Route existiert) ---
                            # --- 2. ROBOTER-STEUERUNG (Nur wenn Route existiert) ---
                    if static_path_calculated:

                        # A) WENN DER BALL NICHT ZU SEHEN IST
                        # A) WENN DER BALL NICHT ZU SEHEN IST
                        # A) WENN DER BALL NICHT ZU SEHEN IST
                        if sphero_pos is None:
                            if state in ["RACING", "CALIBRATING_WAIT"]:
                                print("Sichtkontakt kurz verloren! Behalte Kurs bei...")
                                state = "LOST"
                                lost_start_time = time.time()
                                # FIX: Vollbremsung entfernt! Er rollt bei kurzen Aussetzern einfach weiter.

                            elif state == "LOST":
                                # Erst wenn er 1,5 Sekunden WIRKLICH weg ist, greifen wir ein
                                if time.time() - lost_start_time > 1.5:
                                    print("Suche blind weiter...")
                                    random_heading = random.randint(0, 359)
                                    send_safe_command(random_heading, 25, droid_api)
                                    lost_start_time = time.time()

                            elif state == "EVADING":
                                if time.time() - evade_start_time > 1.0:
                                    send_safe_command(last_sent_heading if last_sent_heading else 0, 0, droid_api)
                                    evade_start_time = time.time()
                                    state = "POST_EVADE"

                            elif state == "POST_EVADE":
                                if time.time() - evade_start_time > 3.0:
                                    print("Befreiung beendet. Prüfe, ob Weg frei ist...")
                                    racing_start_time = time.time()
                                    pos_history.clear()
                                    state = "RACING"

                            elif state == "CALIBRATING_DRIVE":
                                if time.time() - calib_start_time > 1.0:
                                    send_safe_command(0, 0, droid_api)
                                    calib_wait_start = time.time()
                                    state = "CALIBRATING_WAIT"

                        # B) WENN DER BALL SICHTBAR IST
                        else:
                            pos_history.append(sphero_pos)

                            if state == "LOST":
                                print("Sphero wieder da! Setze fort...")
                                racing_start_time = time.time()
                                pos_history.clear()
                                state = "RACING"

                            # --- STATE MACHINE ---
                            if state == "INIT":
                                print("\n--- STARTE KALIBRIERUNG ---")
                                calib_start_pos = sphero_pos
                                send_safe_command(0, 5, droid_api)
                                calib_start_time = time.time()
                                state = "CALIBRATING_DRIVE"

                            elif state == "CALIBRATING_DRIVE":
                                if time.time() - calib_start_time > 1.0:
                                    send_safe_command(0, 0, droid_api)
                                    print("Warte 3 Sekunden...")
                                    calib_wait_start = time.time()
                                    state = "CALIBRATING_WAIT"

                            elif state == "CALIBRATING_WAIT":
                                if time.time() - calib_wait_start > 3.0:
                                    dx = sphero_pos[0] - calib_start_pos[0]
                                    dy = sphero_pos[1] - calib_start_pos[1]

                                    if math.hypot(dx, dy) < 10:
                                        print("Kalibrierung fehlgeschlagen (nicht bewegt). Rückstoß...")
                                        send_safe_command(0, -45, droid_api)
                                        calib_start_time = time.time()
                                        state = "CALIB_RECOVERY"
                                    else:
                                        heading_offset = (math.degrees(math.atan2(dy, dx)) + 360) % 360
                                        print(f"Kalibrierung beendet! Offset: {int(heading_offset)} Grad")
                                        racing_start_time = time.time()
                                        pos_history.clear()
                                        state = "RACING"

                            elif state == "CALIB_RECOVERY":
                                if time.time() - calib_start_time > 1.0:
                                    send_safe_command(0, 0, droid_api)
                                    state = "INIT"

                            elif state == "RACING":
                                is_stuck = False

                                # FIX: 2 Sekunden Schonfrist und nur noch 3 Pixel Toleranz!
                                if time.time() - racing_start_time > 2.0:
                                    if len(pos_history) == pos_history.maxlen:
                                        start_p = pos_history[0]
                                        end_p = pos_history[-1]
                                        dist_moved = math.hypot(end_p[0] - start_p[0], end_p[1] - start_p[1])

                                        if dist_moved < 3:
                                            is_stuck = True

                                if is_stuck:
                                    print("Kamera meldet Stillstand! Befreie in EINE Richtung...")
                                    evade_heading = random.randint(0, 359)
                                    send_safe_command(evade_heading, 45, droid_api)
                                    evade_start_time = time.time()
                                    pos_history.clear()
                                    state = "EVADING"
                                else:
                                    # 1. LAG-AUSGLEICH & WEGPUNKT-PRINT
                                    if current_wp_index < len(waypoint_areas):
                                        max_check = min(current_wp_index + 2, len(waypoint_areas) - 1)
                                        for i in range(max_check, current_wp_index - 1, -1):
                                            area = waypoint_areas[i]
                                            if math.hypot(sphero_pos[0] - area["x"], sphero_pos[1] - area["y"]) <= area[
                                                "r"]:
                                                # FIX: Die reguläre Erfolgsmeldung!
                                                if i > current_wp_index:
                                                    print(
                                                        f"Lag-Ausgleich: Wegpunkt {i + 1} erreicht! Überspringe alte Punkte.")
                                                else:
                                                    print(f"✅ Wegpunkt {current_wp_index + 1} erfolgreich erreicht!")
                                                current_wp_index = i + 1
                                                break

                                    # 2. FAHREN ODER STOPPEN
                                    if current_wp_index < len(waypoint_areas):
                                        area = waypoint_areas[current_wp_index]
                                        angle = (math.degrees(math.atan2(area["y"] - sphero_pos[1],
                                                                         area["x"] - sphero_pos[0])) + 360) % 360
                                        sphero_heading = (angle - heading_offset) % 360
                                        send_safe_command(sphero_heading, 40, droid_api)
                                    else:
                                        print("🎉 ZIEL ERREICHT!")
                                        send_safe_command(0, 0, droid_api)
                                        state = "FINISHED"

                            elif state == "EVADING":
                                if time.time() - evade_start_time > 1.0:
                                    send_safe_command(last_sent_heading if last_sent_heading else 0, 0, droid_api)
                                    evade_start_time = time.time()
                                    state = "POST_EVADE"

                            elif state == "POST_EVADE":
                                if time.time() - evade_start_time > 3.0:
                                    print("Befreiung beendet. Prüfe, ob Weg frei ist...")
                                    racing_start_time = time.time()
                                    pos_history.clear()
                                    state = "RACING"
                    # ==========================================
                    # VISUALISIERUNG (Live-Zeichnungen)
                    # ==========================================
                    if args.show:
                        # 1. Gefahrenzone rot transparent drüberlegen
                        overlay = np.zeros_like(frame)
                        overlay[danger_zone > 0] = (0, 0, 255)  # Rot
                        cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)

                        # 2. Wegpunkte zeichnen (grün)
                        if state in ["RACING", "FINISHED"] and waypoint_areas:
                            for i, wp in enumerate(waypoint_areas):
                                color = (0, 255, 0) if i >= current_wp_index else (100, 100, 100)  # Erledigte grau
                                cv2.circle(frame, (wp["x"], wp["y"]), wp["r"], color, 2)
                                cv2.putText(frame, str(i + 1), (wp["x"] - 5, wp["y"] + 5), cv2.FONT_HERSHEY_SIMPLEX,
                                            0.5, color, 2)

                        # 3. Ball & Richtungspfeil zeichnen
                        if sphero_pos:
                            cv2.circle(frame, sphero_pos, 15, (255, 0, 0), -1)  # Blauer Punkt

                            if len(pos_history) == pos_history.maxlen:
                                start_p = pos_history[0]
                                end_p = pos_history[-1]
                                # Nur Pfeil zeichnen, wenn er sich auch bewegt hat
                                if math.hypot(end_p[0] - start_p[0], end_p[1] - start_p[1]) > 2:
                                    # Pfeil verlängern für bessere Sichtbarkeit
                                    vec_x = (end_p[0] - start_p[0]) * 3
                                    vec_y = (end_p[1] - start_p[1]) * 3
                                    arrow_end = (end_p[0] + vec_x, end_p[1] + vec_y)
                                    cv2.arrowedLine(frame, end_p, arrow_end, (0, 255, 255), 3, tipLength=0.3)

                        cv2.imshow('Received Frame', frame)
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            break

    except KeyboardInterrupt:
        pass
    finally:
        with SpheroEduAPI(toy) as droid_api:
            droid_api.set_speed(0)
        cv2.destroyAllWindows()
        udp_socket.close()


if __name__ == '__main__':
    example_main()