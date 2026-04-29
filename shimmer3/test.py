import shimmer, util
import serial.tools.list_ports
import time

TYPE = util.SHIMMER_GSRplus


def list_available_ports():
    """Affiche tous les ports série/BT disponibles avant toute connexion."""
    candidates = list(serial.tools.list_ports.comports())

    print("\n=== Ports série/Bluetooth disponibles ===")
    if not candidates:
        print("  Aucun port trouvé.")
    for p in candidates:
        print(f"  [{p.device}] - {p.description} | hwid: {p.hwid}")
    print("=========================================\n")

    return candidates


def find_shimmer_port(candidates):
    """Tente de se connecter uniquement aux ports suspects d'être un Shimmer."""
    shimmer_keywords = ["shimmer", "rfcomm", "bluetooth", "serial", "com"]

    # Filtrage préliminaire par nom/description
    filtered = [
        p for p in candidates
        if any(kw in (p.description + p.hwid).lower() for kw in shimmer_keywords)
    ]

    # Si aucun candidat filtré, on tente tous les ports
    targets = filtered if filtered else candidates
    print(f"Ports candidats pour Shimmer : {[p.device for p in targets]}\n")

    for p in targets:
        port = p.device
        print(f"  Tentative sur {port} ({p.description})...")
        s = shimmer.Shimmer3(TYPE, debug=False)
        try:
            s.connect(com_port=port, write_rtc=False, update_all_properties=True, reset_sensors=False)
            name = ""
            try:
                name = s.get_device_name()
            except Exception:
                pass
            print(f"  → Connecté. Nom détecté : '{name}'")
            return s, port
        except Exception as e:
            print(f"  → Échec : {e}")
            try:
                s.disconnect(reset_obj_to_init=True)
            except Exception:
                pass

    return None, None


# ── Main ──────────────────────────────────────────────────────────────────────

candidates = list_available_ports()

shimmer_obj, PORT = find_shimmer_port(candidates)

if shimmer_obj is None:
    raise RuntimeError("Aucun Shimmer3 trouvé sur les ports série disponibles.")

print(f"✓ Shimmer connecté sur {PORT}\n")

try:
    shimmer_obj.set_sampling_rate(2.0)
    shimmer_obj.set_enabled_sensors(util.SENSOR_LOW_NOISE_ACCELEROMETER)
    shimmer_obj.print_object_properties()

    shimmer_obj.start_bt_streaming()

    while True:
        n_of_packets, packets = shimmer_obj.read_data_packet_extended(calibrated=True)
        if n_of_packets > 0:
            for packet in packets:
                print(packet)

except KeyboardInterrupt:
    print("\nArrêt demandé.")
    shimmer_obj.stop_bt_streaming()
    shimmer_obj.disconnect(reset_obj_to_init=True)