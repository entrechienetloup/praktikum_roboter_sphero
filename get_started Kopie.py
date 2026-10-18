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

from spherov2.sphero_edu import SpheroEduAPI
from spherov2.types import Color
from spherov2 import scanner

# --- KONFIGURATION ---
START_POS = (213, 185)
FINAL_TARGET = (419, 197)
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
    lower_yellow = np.array([15, 40, 30])
    upper_yellow = np.array([45, 255, 255])

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

    # NEUE WERTE: Für WEIßE Objekte
    # H: 0-180 (Der Farbwert ist bei echtem Weiß irrelevant)
    # S: 0-60 (Sehr geringe Farbsättigung)
    # V: 200-255 (Sehr hohe Helligkeit)
    lower_white = np.array([0, 0, 200])
    upper_white = np.array([180, 60, 255])
    white_mask = cv2.inRange(hsv, lower_white, upper_white)

    # Rauschen entfernen
    kernel = np.ones((5, 5), np.uint8)
    white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_OPEN, kernel)
    white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_CLOSE, kernel)

    # Debug-Fenster anpassen
    cv2.imshow('Debug: Weisse Maske', white_mask)

    # Konturensuche auf der neuen weißen Maske
    contours, _ = cv2.findContours(white_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

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

            # Liefere jeden Punkt der A*-Route als exakte (X, Y) Pixelkoordinate zurück
            pixel_path = []
            for p in path:
                pixel_path.append((p[0] * grid_size + grid_size // 2, p[1] * grid_size + grid_size // 2))
            return pixel_path

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

def get_pure_pursuit_target(path, current_pos, start_idx, lookahead=30):
    if not path:
        return None, 0

    # FIX: Er darf ab seinem Lesezeichen nur noch 5 Punkte nach vorne suchen.
    # So kann er nicht mehr auf Linien hinter Wänden überspringen!
    search_end = min(start_idx + 5, len(path))

    closest_idx = start_idx
    min_dist = float('inf')

    for i in range(start_idx, search_end):
        d = math.hypot(path[i][0] - current_pos[0], path[i][1] - current_pos[1])
        if d < min_dist:
            min_dist = d
            closest_idx = i

    target_idx = closest_idx
    for i in range(closest_idx, len(path)):
        d = math.hypot(path[i][0] - current_pos[0], path[i][1] - current_pos[1])
        if d >= lookahead:
            target_idx = i
            break
    else:
        target_idx = len(path) - 1

    return path[target_idx], closest_idx

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
            lost_start_time = 0
            last_telemetry_time = 0

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

                    # ====================================================
                    # KUGELSICHERE ARCHITEKTUR: NOTFALL ZUERST!
                    # ====================================================

                    # --- A) GLOBALE NOTFALL-ÜBERWACHUNG (Höchste Priorität) ---
                    if sphero_pos is None:
                        if not state.startswith("LOST"):
                            print("Sichtkontakt verloren! Starte Notfall-Suche...")
                            state = "LOST"
                            lost_start_time = time.time()

                        elif state == "LOST":
                            if time.time() - lost_start_time > 1.5:
                                print("Suche blind weiter (Fahre 1 Sekunde)...")
                                random_heading = random.randint(0, 359)
                                send_safe_command(random_heading, 35, droid_api)
                                lost_drive_start = time.time()
                                state = "LOST_DRIVE"

                        elif state == "LOST_DRIVE":
                            if time.time() - lost_drive_start > 1.0:
                                print("Stoppe blinde Suche und leere Video-Puffer (3s)...")
                                send_safe_command(last_sent_heading if last_sent_heading else 0, 0, droid_api)
                                lost_wait_start = time.time()
                                state = "LOST_WAIT"

                        elif state == "LOST_WAIT":
                            if time.time() - lost_wait_start > 2.0:
                                lost_start_time = time.time() - 1.5
                                state = "LOST"

                    # --- B) WENN DER BALL SICHTBAR IST ---
                    else:
                        # 1. Comeback aus dem Notfall behandeln
                        if state.startswith("LOST"):
                            print("Sphero wieder da!")
                            send_safe_command(0, 0, droid_api)
                            if heading_offset == 0:
                                print("Fehler: Kalibrierung wurde unterbrochen! Starte neu...")
                                state = "INIT"
                            else:
                                print("Setze Rennen fort...")
                                static_path_calculated = False
                                state = "RACING"

                        # 2. ROUTE BERECHNEN (Nur im Rennen und wenn nötig!)
                        if not static_path_calculated and state.startswith("RACING"):
                            if time.time() - last_calc_time > 2.0:
                                print(f"Prüfe Route von {sphero_pos} nach {FINAL_TARGET}...")
                                waypoint_areas = calculate_a_star_path(danger_zone, sphero_pos, FINAL_TARGET)
                                if waypoint_areas:
                                    print(f"Route festgelegt: {len(waypoint_areas)} Wegpunkte.")
                                    static_path_calculated = True
                                    current_path_idx = 0
                                else:
                                    print("FEHLER: Wand blockiert! Klicke ins Fenster für neue Koordinaten.")
                                last_calc_time = time.time()

                        # 3. ROBOTER-STEUERUNG (State Machine)
                        if state == "INIT":
                            print("\n--- STARTE KALIBRIERUNG ---")
                            calib_start_pos = sphero_pos
                            send_safe_command(0, 5, droid_api)
                            calib_start_time = time.time()
                            state = "CALIBRATING_DRIVE"

                        elif state == "CALIBRATING_DRIVE":
                            if time.time() - calib_start_time > 1.0:
                                send_safe_command(0, 0, droid_api)
                                print("Warte 2 Sekunden...")
                                calib_wait_start = time.time()
                                state = "CALIBRATING_WAIT"

                        elif state == "CALIBRATING_WAIT":
                            if time.time() - calib_wait_start > 2.0:
                                dx = sphero_pos[0] - calib_start_pos[0]
                                dy = sphero_pos[1] - calib_start_pos[1]

                                if math.hypot(dx, dy) < 10:
                                    print("Kalibrierung fehlgeschlagen! Befreie in zufällige Richtung...")
                                    random_heading = random.randint(0, 359)
                                    send_safe_command(random_heading, 35, droid_api)
                                    calib_recovery_start = time.time()
                                    state = "CALIB_RECOVERY_DRIVE"
                                else:
                                    heading_offset = (math.degrees(math.atan2(dy, dx)) + 360) % 360
                                    print(f"Kalibrierung beendet! Offset: {int(heading_offset)} Grad")
                                    static_path_calculated = False
                                    state = "RACING"

                        elif state == "CALIB_RECOVERY_DRIVE":
                            if time.time() - calib_recovery_start > 1.0:
                                print("Stoppe Befreiung und leere Video-Puffer (3s)...")
                                send_safe_command(last_sent_heading if last_sent_heading else 0, 0, droid_api)
                                calib_wait_start = time.time()
                                state = "CALIB_RECOVERY_WAIT"

                        elif state == "CALIB_RECOVERY_WAIT":
                            if time.time() - calib_wait_start > 2.0:
                                state = "INIT"

                        elif state == "RACING":
                            # Nur losfahren, wenn die Route auch erfolgreich berechnet wurde!
                            if static_path_calculated:
                                target_pos, current_path_idx = get_pure_pursuit_target(waypoint_areas, sphero_pos,
                                                                                       current_path_idx, lookahead=30)

                                final_wp = waypoint_areas[-1]
                                dist_to_final = math.hypot(sphero_pos[0] - final_wp[0], sphero_pos[1] - final_wp[1])

                                if dist_to_final < 40:
                                    print("ZIEL ERREICHT!")
                                    send_safe_command(0, 0, droid_api)
                                    state = "FINISHED"
                                    continue

                                if target_pos:
                                    angle = (math.degrees(math.atan2(target_pos[1] - sphero_pos[1],
                                                                     target_pos[0] - sphero_pos[0])) + 360) % 360
                                    sphero_heading = (angle - heading_offset) % 360
                                    print("Fahre für 1 Sekunde...")
                                    send_safe_command(sphero_heading, 35, droid_api)
                                    racing_start_time = time.time()
                                    drive_start_pos = sphero_pos
                                    state = "RACING_DRIVE"

                        elif state == "RACING_DRIVE":
                            if time.time() - racing_start_time > 1.0:
                                dist_moved = math.hypot(sphero_pos[0] - drive_start_pos[0],
                                                        sphero_pos[1] - drive_start_pos[1])
                                if dist_moved < 15:
                                    print("Festgefahren! Befreie blind in zufällige Richtung...")
                                    random_heading = random.randint(0, 359)
                                    send_safe_command(random_heading, 35, droid_api)
                                    evade_drive_start = time.time()
                                    state = "EVADE_DRIVE"
                                else:
                                    print("Stoppe und leere den Video-Puffer (3s)...")
                                    send_safe_command(last_sent_heading if last_sent_heading else 0, 0, droid_api)
                                    racing_wait_start = time.time()
                                    state = "RACING_WAIT"

                        elif state == "RACING_WAIT":
                            if time.time() - racing_wait_start > 2.0:
                                static_path_calculated = False
                                state = "RACING"

                        elif state == "EVADE_DRIVE":
                            if time.time() - evade_drive_start > 1.0:
                                print("Stoppe Befreiung und leere Video-Puffer (3s)...")
                                send_safe_command(last_sent_heading if last_sent_heading else 0, 0, droid_api)
                                evade_wait_start = time.time()
                                state = "EVADE_WAIT"

                        elif state == "EVADE_WAIT":
                            if time.time() - evade_wait_start > 2.0:
                                static_path_calculated = False
                                state = "RACING"
                    # --- TELEMETRIE-AUSGABE (Alle 0.5 Sekunden) ---
                    if time.time() - last_telemetry_time > 0.5:
                        if sphero_pos is not None:
                            # Vorher: if state == "RACING" and waypoint_areas:
                            if state.startswith("RACING") and waypoint_areas:
                                _, current_path_idx = get_pure_pursuit_target(waypoint_areas, sphero_pos,
                                                                              current_path_idx, lookahead=30)
                                points_left = len(waypoint_areas) - current_path_idx
                                print(
                                    f"[{state}] Kurs: {int(last_sent_heading if last_sent_heading else 0)}° | Speed: {last_sent_speed} | Verbleibende Pfad-Punkte: {points_left}")
                        last_telemetry_time = time.time()
                    # ==========================================
                    # VISUALISIERUNG (Live-Zeichnungen)
                    # ==========================================
                    if args.show:
                        # 1. Gefahrenzone rot transparent drüberlegen
                        overlay = np.zeros_like(frame)
                        overlay[danger_zone > 0] = (0, 0, 255)  # Rot
                        cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)

                        # 2. Wegpunkte zeichnen (grün)
                        # 2. Die Vektor-Route (Ideallinie) zeichnen
                        # Vorher: if state in ["RACING", "FINISHED"] and waypoint_areas:
                        if (state.startswith("RACING") or state == "FINISHED") and waypoint_areas:
                            for i in range(len(waypoint_areas) - 1):
                                cv2.line(frame, waypoint_areas[i], waypoint_areas[i + 1], (0, 150, 0), 4)

                            # Vorher: if sphero_pos and state == "RACING":
                            if sphero_pos and state.startswith("RACING"):
                                target_pos, current_path_idx = get_pure_pursuit_target(waypoint_areas, sphero_pos,
                                                                                       current_path_idx, lookahead=30)
                        # 3. Ball & Richtungspfeil zeichnen
                        if sphero_pos:
                            cv2.circle(frame, sphero_pos, 15, (255, 0, 0), -1)  # Blauer Punkt

                            # NEU: Pfeil basierend auf den gesendeten Programm-Befehlen
                            if last_sent_heading is not None and last_sent_speed is not None and last_sent_speed != 0:
                                # Winkel vom Roboter-System in das Bildschirm-System zurückrechnen
                                visual_angle = math.radians((last_sent_heading + heading_offset) % 360)

                                # Pfeillänge skaliert mit der gesendeten Motor-Geschwindigkeit
                                arrow_len = abs(last_sent_speed) * 1.5

                                vec_x = math.cos(visual_angle) * arrow_len
                                vec_y = math.sin(visual_angle) * arrow_len
                                arrow_end = (int(sphero_pos[0] + vec_x), int(sphero_pos[1] + vec_y))

                                # Orangefarbener Pfeil für die reinen Befehle
                                cv2.arrowedLine(frame, sphero_pos, arrow_end, (0, 165, 255), 3, tipLength=0.3)

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