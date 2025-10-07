import os
import json
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import time
import numpy as np
import requests
from datetime import datetime

st.set_page_config(page_title="Traffic Dashboard", page_icon="🚗",
                   layout="wide", initial_sidebar_state="expanded")

# Load config first
CONFIG_FILE = "config.json"
if os.path.isfile(CONFIG_FILE):
    with open(CONFIG_FILE, 'r') as f:
        config = json.load(f)
else:
    config = {"origin": "Cto. Zinfandel 3970, 22564", "destination_a": "Blvd. Agua Caliente 4558 C-1, Aviacion, 22014 Tijuana, B.C.",
              "destination_b": "Avenida Universidad 13021, Parque Internacional Industrial Tijuana, 22424 Tijuana, B.C."}

# Load current config for refresh
if os.path.isfile(CONFIG_FILE):
    with open(CONFIG_FILE, 'r') as f:
        current_config = json.load(f)
else:
    current_config = config

refresh_interval = max(10, current_config.get("interval_minutes", 10) * 30)

st.title(
    f"Traffic Data Dashboard: {config['origin']} → A: {config['destination_a']} | B: {config['destination_b']}")

# Sidebar filters
st.sidebar.header("Filters")
date_range = st.sidebar.date_input("Date Range", [], key="date_range")
weekday_filter = st.sidebar.multiselect("Weekdays", options=["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"], default=[
                                        "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"], key="weekday_filter")
destination_filter = st.sidebar.selectbox(
    "Destination", options=["All", "A", "B"], index=0, key="destination_filter")
route_filter = st.sidebar.multiselect("Routes", options=[0, 1, 2, 3, 4], default=[
                                      0, 1, 2, 3, 4], key="route_filter")

# Logger Control
st.sidebar.header("Logger Control")
logging_enabled = os.path.isfile("logging_enabled.txt")
if logging_enabled:
    st.sidebar.success("Logger is running")
else:
    st.sidebar.warning("Logger is stopped")

current_interval = st.sidebar.number_input("Interval (minutes)", value=config.get(
    "interval_minutes", 10), min_value=1, key="interval_input", disabled=logging_enabled)

if st.sidebar.button("Start Logging", key="start_logging", disabled=logging_enabled):
    with open("logging_enabled.txt", 'w') as f:
        f.write("enabled")
    st.sidebar.success("Logging started!")
    st.rerun()

if st.sidebar.button("Stop Logging", key="stop_logging", disabled=not logging_enabled):
    if os.path.isfile("logging_enabled.txt"):
        os.remove("logging_enabled.txt")
    st.sidebar.success("Logging stopped!")
    st.rerun()

if st.sidebar.button("Log Data Now", key="manual_log", disabled=not logging_enabled):
    with open("manual_log.txt", 'w') as f:
        f.write("manual")
    st.sidebar.success("Manual log triggered! Data will be logged shortly.")

st.sidebar.markdown("---")
st.sidebar.write(f"Auto-refresh every {refresh_interval} seconds")

# Configuration
st.sidebar.header("Logger Configuration")

current_origin = st.sidebar.text_input(
    "Origin", value=config["origin"], key="origin_input", disabled=logging_enabled)
current_destination_a = st.sidebar.text_input(
    "Destination A", value=config["destination_a"], key="destination_a_input", disabled=logging_enabled)
current_destination_b = st.sidebar.text_input(
    "Destination B", value=config["destination_b"], key="destination_b_input", disabled=logging_enabled)

if st.sidebar.button("Update Configuration", key="update_config", disabled=logging_enabled):
    new_config = config.copy()
    new_config["origin"] = current_origin
    new_config["destination_a"] = current_destination_a
    new_config["destination_b"] = current_destination_b
    new_config["interval_minutes"] = current_interval
    # Preserve other keys like api_key
    with open(CONFIG_FILE, 'w') as f:
        json.dump(new_config, f, indent=4)
    st.sidebar.success(
        "Configuration updated!")
    st.rerun()

if st.sidebar.button("Clear All Data", key="clear_data"):
    # Clear the CSV file
    with open('traffic_data.csv', 'w') as f:
        # Write header only
        f.write(
            'timestamp,weekday,time,duration_sec,duration_text,destination,route,summary\n')
    st.sidebar.success("All data cleared!")
    st.rerun()

# Function to load and process data


@st.cache_data(ttl=10)  # Cache for 10 seconds
def load_data():
    try:
        df = pd.read_csv('traffic_data.csv')
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['hour'] = df['timestamp'].dt.hour
        return df
    except pd.errors.ParserError:
        # If parsing error due to mismatched columns, try with specific columns
        st.warning("CSV format issue detected. Attempting to fix...")
        # Assume the new format
        df = pd.read_csv('traffic_data.csv', names=[
                         'timestamp', 'weekday', 'time', 'duration_sec', 'duration_text', 'destination', 'route', 'summary'], header=0)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['hour'] = df['timestamp'].dt.hour
        return df


# Logging functions from traffic_logger.py
def get_travel_time(destination, api_key):
    url = (
        f"https://maps.googleapis.com/maps/api/directions/json?"
        f"origin={config['origin']}&destination={destination}"
        f"&departure_time=now&alternatives=true&mode=driving&key={api_key}"
    )
    response = requests.get(url)
    data = response.json()

    routes = []
    try:
        for i, route in enumerate(data["routes"]):
            duration_in_traffic = route["legs"][0]["duration_in_traffic"]["value"]
            duration_text = route["legs"][0]["duration_in_traffic"]["text"]
            summary = route.get("summary", f"Route {i}")
            routes.append((duration_in_traffic, duration_text, i, summary))
        return routes
    except (IndexError, KeyError) as e:
        log_debug(
            f"Warning: Could not extract travel time for {destination}. Error: {e}")
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
    if not os.path.isfile('traffic_data.csv'):
        df.to_csv('traffic_data.csv', index=False)
    else:
        df.to_csv('traffic_data.csv', mode='a', header=False, index=False)


def log_debug(message):
    with open("debug.log", "a") as f:
        f.write(f"{datetime.now().isoformat()}: {message}\n")
    print(message)


def perform_logging():
    api_key = st.secrets.get("api_key", config.get("api_key"))
    if not api_key:
        log_debug("No API key found in secrets or config.")
        return

    now = datetime.now().isoformat()
    log_debug(f"Performing logging at {now}")

    # Log for Destination A
    routes_a = get_travel_time(config["destination_a"], api_key)
    if routes_a:
        log_to_csv(now, routes_a, "A")
        for dur, text, rid, summary in routes_a:
            log_debug(
                f"[{now}] Destination A Route {rid} ({summary}): {text} ({dur}s)")
    else:
        log_debug(f"[{now}] Failed to fetch data for Destination A.")

    # Log for Destination B
    routes_b = get_travel_time(config["destination_b"], api_key)
    if routes_b:
        log_to_csv(now, routes_b, "B")
        for dur, text, rid, summary in routes_b:
            log_debug(
                f"[{now}] Destination B Route {rid} ({summary}): {text} ({dur}s)")
    else:
        log_debug(f"[{now}] Failed to fetch data for Destination B.")

    log_debug("Logging cycle complete.")


# Auto-refresh based on interval
placeholder = st.empty()

# Initialize session state for logging
if 'last_log_time' not in st.session_state:
    st.session_state.last_log_time = None

while True:
    with placeholder.container():
        # Handle logging
        logging_enabled = os.path.isfile("logging_enabled.txt")
        manual_log_triggered = os.path.isfile("manual_log.txt")
        current_time = time.time()
        interval_seconds = config.get("interval_minutes", 10) * 60

        if logging_enabled:
            if manual_log_triggered:
                # Perform manual log
                perform_logging()
                os.remove("manual_log.txt")
                st.session_state.last_log_time = current_time
            elif st.session_state.last_log_time is None or (current_time - st.session_state.last_log_time) >= interval_seconds:
                # Perform automatic log
                perform_logging()
                st.session_state.last_log_time = current_time

        df = load_data()

        if df.empty:
            st.write("No data available yet.")
        else:
            # Apply filters
            filtered_df = df.copy()
            if date_range:
                if len(date_range) == 2:
                    start_date, end_date = date_range
                    filtered_df = filtered_df[(filtered_df['timestamp'].dt.date >= start_date) & (
                        filtered_df['timestamp'].dt.date <= end_date)]
            if weekday_filter:
                filtered_df = filtered_df[filtered_df['weekday'].isin(
                    weekday_filter)]
            if destination_filter != "All":
                filtered_df = filtered_df[filtered_df['destination']
                                          == destination_filter]
            if 'route' in df.columns and route_filter:
                filtered_df = filtered_df[filtered_df['route'].isin(
                    route_filter)]

            # Data summary card
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Readings", len(filtered_df))
            with col2:
                if not filtered_df.empty:
                    time_span = (filtered_df['timestamp'].max() -
                                 filtered_df['timestamp'].min()).total_seconds() / 3600
                    st.metric("Data Time Span", f"{time_span:.1f} hours")
                else:
                    st.metric("Data Time Span", "0 hours")
            with col3:
                last_update = df['timestamp'].max().strftime(
                    "%Y-%m-%d %H:%M:%S")
                st.metric("Last Update", last_update)

            # Basic stats and optimal
            col4, col5 = st.columns(2)
            with col4:
                st.subheader("Detailed Statistics")
                if not filtered_df.empty:
                    desc = filtered_df['duration_sec'].describe()
                    st.write(f"Count: {desc['count']:.0f}")
                    st.write(
                        f"Mean: {desc['mean']:.0f} sec ({desc['mean']/60:.1f} min)")
                    st.write(f"Median: {desc['50%']:.0f} sec")
                    st.write(f"Std Dev: {desc['std']:.0f} sec")
                    st.write(f"Min: {desc['min']:.0f} sec")
                    st.write(f"Max: {desc['max']:.0f} sec")
                    st.write(
                        f"Variance: {filtered_df['duration_sec'].var():.0f}")
                else:
                    st.write("No data matches filters.")
            with col5:
                if not filtered_df.empty and not filtered_df.groupby('hour')['duration_sec'].mean().empty:
                    hour_avg = filtered_df.groupby(
                        'hour')['duration_sec'].mean()
                    min_hour = hour_avg.idxmin()
                    min_dur = hour_avg.min()
                    st.subheader("Optimal Departure Window")
                    st.write(f"Best time: {min_hour}:00")
                    st.metric("Average Duration", f"{min_dur/60:.1f} min")

                    # Alert for high traffic
                    latest_dur = filtered_df['duration_sec'].iloc[-1]
                    if latest_dur > desc['mean'] + desc['std']:
                        st.warning(
                            f"🚨 High traffic alert! Latest duration: {latest_dur} sec ({latest_dur/60:.1f} min)")
                    elif latest_dur > desc['75%']:
                        st.info(
                            f"⚠️ Elevated traffic. Latest: {latest_dur} sec")
                else:
                    st.subheader("Optimal Departure Window")
                    st.write("Insufficient data.")

            # Visualizations
            st.subheader("Visualizations")
            viz_col1, viz_col2 = st.columns(2)
            with viz_col1:
                st.write("Average Duration by Weekday")
                if not filtered_df.empty:
                    weekday_avg = filtered_df.groupby(
                        'weekday')['duration_sec'].mean()
                    st.bar_chart(weekday_avg)
            with viz_col2:
                st.write("Average Duration by Hour")
                if not filtered_df.empty:
                    hour_avg = filtered_df.groupby(
                        'hour')['duration_sec'].mean()
                    st.bar_chart(hour_avg)

            # Heatmap
            st.subheader("Traffic Heatmap (Hour vs Weekday)")
            if not filtered_df.empty:
                pivot = filtered_df.pivot_table(
                    values='duration_sec', index='weekday', columns='hour', aggfunc='mean')
                fig, ax = plt.subplots(figsize=(10, 6))
                cax = ax.imshow(pivot, cmap='RdYlGn_r', aspect='auto')
                ax.set_xticks(range(len(pivot.columns)))
                ax.set_xticklabels(pivot.columns)
                ax.set_yticks(range(len(pivot.index)))
                ax.set_yticklabels(pivot.index)
                plt.colorbar(cax, ax=ax, label='Duration (seconds)')
                st.pyplot(fig)

            # Route Comparison
            st.subheader("Route Comparison")
            if not filtered_df.empty and 'route' in df.columns:
                route_avg = filtered_df.groupby(
                    'route')['duration_sec'].mean().sort_values()
                st.bar_chart(route_avg)
                best_route = route_avg.idxmin()
                st.write(
                    f"Recommended Route: {best_route} (average {route_avg.min()/60:.1f} min)")

                # Route details
                with st.expander("View Route Details and Maps"):
                    for dest in ['A', 'B']:
                        if dest in filtered_df['destination'].values:
                            st.write(f"**Destination {dest}**")
                            dest_data = filtered_df[filtered_df['destination'] == dest]
                            for route_id in sorted(dest_data['route'].unique()):
                                route_data = dest_data[dest_data['route']
                                                       == route_id]
                                avg_dur = route_data['duration_sec'].mean()
                                summary = route_data['summary'].iloc[0] if 'summary' in route_data.columns and not route_data[
                                    'summary'].empty else f"Route {route_id}"
                                st.write(
                                    f"  Route {route_id} ({summary}): Average {avg_dur/60:.1f} min")
                                # Link to Google Maps
                                origin_enc = config['origin'].replace(' ', '+')
                                dest_name = config['destination_a'] if dest == 'A' else config['destination_b']
                                dest_enc = dest_name.replace(' ', '+')
                                url = f"https://www.google.com/maps/dir/{origin_enc}/{dest_enc}/?dirflg=h"
                                st.markdown(
                                    f"  [View Route Options on Google Maps]({url})")

            # Time series
            st.subheader("Duration Over Time")
            if not filtered_df.empty:
                fig, ax = plt.subplots()
                ax.plot(filtered_df['timestamp'], filtered_df['duration_sec'])
                ax.set_xlabel('Timestamp')
                ax.set_ylabel('Duration (seconds)')
                ax.tick_params(axis='x', rotation=45)
                st.pyplot(fig)

                # Simple predictive analytics
                if len(filtered_df) > 1:
                    times = (
                        filtered_df['timestamp'] - filtered_df['timestamp'].min()).dt.total_seconds().values
                    durations = filtered_df['duration_sec'].values
                    if len(times) > 1:
                        coeffs = np.polyfit(times, durations, 1)
                        next_time = times[-1] + 600  # Assume next in 10 min
                        predicted = np.polyval(coeffs, next_time)
                        st.subheader("Predictive Analytics")
                        st.write(
                            f"Predicted next duration: {predicted:.0f} sec ({predicted/60:.1f} min) based on linear trend.")
            else:
                st.write("No data to display.")

            # Export data
            st.subheader("Export Data")
            if not filtered_df.empty:
                csv = filtered_df.to_csv(index=False)
                st.download_button(
                    label="Download Filtered CSV",
                    data=csv,
                    file_name='filtered_traffic_data.csv',
                    mime='text/csv',
                    key="download_csv"
                )
            else:
                st.write("No data to export.")

            # Debug log
            st.subheader("Debug Log")
            if os.path.isfile("debug.log"):
                try:
                    with open("debug.log", "r", encoding='utf-8', errors='ignore') as f:
                        lines = f.readlines()
                        last_lines = lines[-20:]  # Last 20 lines
                        debug_text = "".join(last_lines)
                        st.text_area("Recent Debug Messages",
                                     debug_text, height=200)
                except Exception as e:
                    st.write(f"Error reading debug log: {e}")
            else:
                st.write("No debug log available.")

    time.sleep(refresh_interval)
