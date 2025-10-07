import requests
import pandas as pd
from datetime import datetime
import time
import os
import json

# ==== CONFIGURATION ====
CONFIG_FILE = "config.json"
DEFAULT_CONFIG = {
    "origin": "Tijuana,MX",
    "destination_a": "San Diego,CA",
    "destination_b": "Los Angeles,CA",
    "interval_minutes": 5,
    "csv_file": "traffic_data.csv"
}


def load_config():
    if os.path.isfile(CONFIG_FILE):
        with open(CONFIG_FILE, 'r') as f:
            return json.load(f)
    else:
        with open(CONFIG_FILE, 'w') as f:
            json.dump(DEFAULT_CONFIG, f, indent=4)
        return DEFAULT_CONFIG


config = load_config()
ORIGIN = config["origin"]
DESTINATION_A = config["destination_a"]
DESTINATION_B = config["destination_b"]
API_KEY = config["api_key"]
INTERVAL_MINUTES = config["interval_minutes"]
CSV_FILE = config["csv_file"]
# ========================


def get_travel_time(destination):
    url = (
        f"https://maps.googleapis.com/maps/api/directions/json?"
        f"origin={ORIGIN}&destination={destination}"
        f"&departure_time=now&alternatives=true&mode=driving&key={API_KEY}"
    )
    print(f"DEBUG: API URL: {url}")
    response = requests.get(url)
    print(f"DEBUG: Response status: {response.status_code}")
    data = response.json()
    print(f"DEBUG: Response data keys: {list(data.keys())}")

    routes = []
    try:
        print(f"DEBUG: Number of routes: {len(data['routes'])}")
        for i, route in enumerate(data["routes"]):
            print(f"DEBUG: Processing route {i}")
            duration_in_traffic = route["legs"][0]["duration_in_traffic"]["value"]
            duration_text = route["legs"][0]["duration_in_traffic"]["text"]
            summary = route.get("summary", f"Route {i}")
            routes.append((duration_in_traffic, duration_text, i, summary))
            print(
                f"DEBUG: Route {i}: {summary} - {duration_text} ({duration_in_traffic}s)")
        return routes
    except (IndexError, KeyError) as e:
        print(
            f"Warning: Could not extract travel time for {destination}. Error: {e}")
        print("Full response:")
        print(data)
        return []


def log_to_csv(timestamp, routes, destination):
    rows = []
    for duration_sec, duration_text, route_id, summary in routes:
        rows.append({
            "timestamp": timestamp,
            "weekday": datetime.now().strftime("%A"),
            "time": datetime.now().strftime("%H:%M"),
            "duration_sec": duration_sec,
            "duration_text": duration_text,
            "destination": destination,
            "route": route_id,
            "summary": summary
        })
    df = pd.DataFrame(rows)
    if not os.path.isfile(CSV_FILE):
        df.to_csv(CSV_FILE, index=False)
    else:
        df.to_csv(CSV_FILE, mode='a', header=False, index=False)


def log_debug(message):
    with open("debug.log", "a") as f:
        f.write(f"{datetime.now().isoformat()}: {message}\n")
    print(message)


def main():
    log_debug(
        f"Starting traffic logger: {ORIGIN} -> {DESTINATION_A} and {DESTINATION_B}")
    log_debug(f"Interval: {INTERVAL_MINUTES} minutes")
    log_debug("Press Ctrl+C to stop.\n")

    while True:
        log_debug("DEBUG: Checking logging_enabled.txt")
        if not os.path.isfile("logging_enabled.txt"):
            log_debug("DEBUG: Logging disabled. Waiting...")
            time.sleep(10)
            continue

        log_debug("DEBUG: Logging enabled.")
        manual_log = os.path.isfile("manual_log.txt")
        if manual_log:
            os.remove("manual_log.txt")
            log_debug("DEBUG: Manual log triggered and file removed.")
            log_debug("DEBUG: Manual logging started.")

        now = datetime.now().isoformat()
        log_debug(f"DEBUG: Current time: {now}")

        # Log for Destination A
        log_debug(f"DEBUG: Fetching routes for Destination A: {DESTINATION_A}")
        routes_a = get_travel_time(DESTINATION_A)
        log_debug(
            f"DEBUG: Routes A received: {len(routes_a) if routes_a else 0} routes")
        if routes_a:
            log_to_csv(now, routes_a, "A")
            for dur, text, rid, summary in routes_a:
                log_debug(
                    f"[{now}] Destination A Route {rid} ({summary}): {text} ({dur}s)")
        else:
            log_debug(f"[{now}] Failed to fetch data for Destination A.")

        # Log for Destination B
        log_debug(f"DEBUG: Fetching routes for Destination B: {DESTINATION_B}")
        routes_b = get_travel_time(DESTINATION_B)
        log_debug(
            f"DEBUG: Routes B received: {len(routes_b) if routes_b else 0} routes")
        if routes_b:
            log_to_csv(now, routes_b, "B")
            for dur, text, rid, summary in routes_b:
                log_debug(
                    f"[{now}] Destination B Route {rid} ({summary}): {text} ({dur}s)")
        else:
            log_debug(f"[{now}] Failed to fetch data for Destination B.")

        log_debug("DEBUG: Logging cycle complete.")
        if manual_log:
            log_debug("DEBUG: Manual logging finished.")
        if not manual_log:
            log_debug(f"DEBUG: Sleeping for {INTERVAL_MINUTES * 60} seconds")
            time.sleep(INTERVAL_MINUTES * 60)
        else:
            log_debug("DEBUG: Manual log, no sleep.")


if __name__ == "__main__":
    main()
