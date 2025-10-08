import os
import json
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import time
import numpy as np
import requests
from datetime import datetime
import pytz
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="Traffic Dashboard", page_icon="🚗",
                   layout="wide", initial_sidebar_state="expanded")

# Load config first
CONFIG_FILE = "config.json"
if os.path.isfile(CONFIG_FILE):
    with open(CONFIG_FILE, 'r') as f:
        config = json.load(f)
else:
    config = {"origin": "Cto. Zinfandel 3970, 22564", "destination_a": "Blvd. Agua Caliente 4558 C-1, Aviacion, 22014 Tijuana, B.C.",
              "destination_b": "Avenida Universidad 13021, Parque Internacional Industrial Tijuana, 22424 Tijuana, B.C.",
              "destination_c": "Cto. Zinfandel 3970, 22564", "destination_d": "Cto. Zinfandel 3970, 22564"}

# Load current config for refresh
if os.path.isfile(CONFIG_FILE):
    with open(CONFIG_FILE, 'r') as f:
        current_config = json.load(f)
else:
    current_config = config

refresh_interval = max(10, current_config.get("interval_minutes", 10) * 30)

# Check for API key
api_key = st.secrets.get("api_key", config.get("api_key"))
if not api_key or api_key == "PUT_YOUR_API_KEY_HERE":
    st.error("🚨 API Key Missing! Please set your Google Maps API key in Streamlit secrets or replace 'PUT_YOUR_API_KEY_HERE' in config.json.")
    st.stop()

st.title(
    f"Traffic Data Dashboard: {config['origin']} → A: {config['destination_a']} | B: {config['destination_b']} | C: {config['destination_c']} | D: {config['destination_d']}")

# Sidebar filters
st.sidebar.header("Filters")
date_range = st.sidebar.date_input("Date Range", [], key="date_range")
weekday_filter = st.sidebar.multiselect("Weekdays", options=["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"], default=[
                                        "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"], key="weekday_filter")
destination_filter = st.sidebar.selectbox(
    "Destination", options=["All", "A", "B", "C", "D"], index=0, key="destination_filter")
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

# Notes for logging
current_notes = st.sidebar.text_area("Notes (for manual logs)", value="", height=100, key="notes_input",
                                     help="Add notes for unusual traffic events, crashes, etc. These will be included in manual log entries.")

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
current_destination_c = st.sidebar.text_input(
    "Destination C", value=config["destination_c"], key="destination_c_input", disabled=logging_enabled)
current_destination_d = st.sidebar.text_input(
    "Destination D", value=config["destination_d"], key="destination_d_input", disabled=logging_enabled)

if st.sidebar.button("Update Configuration", key="update_config", disabled=logging_enabled):
    new_config = config.copy()
    new_config["origin"] = current_origin
    new_config["destination_a"] = current_destination_a
    new_config["destination_b"] = current_destination_b
    new_config["destination_c"] = current_destination_c
    new_config["destination_d"] = current_destination_d
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
            'timestamp,weekday,time,duration_sec,duration_text,destination,route,summary,notes\n')
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
                         'timestamp', 'weekday', 'time', 'duration_sec', 'duration_text', 'destination', 'route', 'summary', 'notes'], header=0)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['hour'] = df['timestamp'].dt.hour
        return df


# Logging functions from traffic_logger.py
def get_travel_time(origin, destination, api_key, dest_label):
    url = (
        f"https://maps.googleapis.com/maps/api/directions/json?"
        f"origin={origin}&destination={destination}"
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
            warnings = route.get("warnings", [])
            warnings_text = "; ".join(warnings) if warnings else ""
            routes.append((duration_in_traffic, duration_text,
                          i, summary, warnings_text))

        # Filter to allowed routes if specified
        allowed = config.get("allowed_routes", {}).get(dest_label, [])
        if allowed:
            routes = [r for r in routes if r[3] in allowed]

        return routes
    except (IndexError, KeyError) as e:
        log_debug(
            f"Warning: Could not extract travel time from {origin} to {destination}. Error: {e}")
        return []


def log_to_csv(timestamp, routes, destination, notes=""):
    # Use Tijuana timezone (Pacific Daylight Time, GMT-7)
    tijuana_tz = pytz.timezone('America/Tijuana')
    now = datetime.now(tijuana_tz)
    rows = []
    for duration_sec, duration_text, route_id, summary, warnings in routes:
        combined_notes = f"{notes}; {warnings}".strip("; ").strip()
        rows.append({
            "timestamp": timestamp,
            "weekday": now.strftime("%A"),
            "time": now.strftime("%H:%M"),
            "duration_sec": duration_sec,
            "duration_text": duration_text,
            "destination": destination,
            "route": route_id,
            "summary": summary,
            "notes": combined_notes
        })
    df = pd.DataFrame(rows)
    if not os.path.isfile('traffic_data.csv'):
        df.to_csv('traffic_data.csv', index=False)
    else:
        df.to_csv('traffic_data.csv', mode='a', header=False, index=False)


def log_debug(message):
    tijuana_tz = pytz.timezone('America/Tijuana')
    with open("debug.log", "a") as f:
        f.write(f"{datetime.now(tijuana_tz).isoformat()}: {message}\n")
    print(message)


def perform_logging(notes=""):
    api_key = st.secrets.get("api_key", config.get("api_key"))
    if not api_key:
        log_debug("No API key found in secrets or config.")
        return

    now = datetime.now().isoformat()
    log_debug(f"Performing logging at {now}")

    # Log for Destination A
    routes_a = get_travel_time(
        config["origin"], config["destination_a"], api_key, "A")
    if routes_a:
        log_to_csv(now, routes_a, "A", notes)
        for dur, text, rid, summary, warnings in routes_a:
            log_debug(
                f"[{now}] Destination A Route {rid} ({summary}): {text} ({dur}s)")
    else:
        log_debug(f"[{now}] Failed to fetch data for Destination A.")

    # Log for Destination B
    routes_b = get_travel_time(
        config["origin"], config["destination_b"], api_key, "B")
    if routes_b:
        log_to_csv(now, routes_b, "B", notes)
        for dur, text, rid, summary, warnings in routes_b:
            log_debug(
                f"[{now}] Destination B Route {rid} ({summary}): {text} ({dur}s)")
    else:
        log_debug(f"[{now}] Failed to fetch data for Destination B.")

    # Log for Destination C (return from A)
    routes_c = get_travel_time(
        config["destination_a"], config["destination_c"], api_key, "C")
    if routes_c:
        log_to_csv(now, routes_c, "C", notes)
        for dur, text, rid, summary, warnings in routes_c:
            log_debug(
                f"[{now}] Destination C Route {rid} ({summary}): {text} ({dur}s)")
    else:
        log_debug(f"[{now}] Failed to fetch data for Destination C.")

    # Log for Destination D (return from B)
    routes_d = get_travel_time(
        config["destination_b"], config["destination_d"], api_key, "D")
    if routes_d:
        log_to_csv(now, routes_d, "D", notes)
        for dur, text, rid, summary, warnings in routes_d:
            log_debug(
                f"[{now}] Destination D Route {rid} ({summary}): {text} ({dur}s)")
    else:
        log_debug(f"[{now}] Failed to fetch data for Destination D.")

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
                perform_logging(current_notes)
                os.remove("manual_log.txt")
                st.session_state.last_log_time = current_time
            elif st.session_state.last_log_time is None or (current_time - st.session_state.last_log_time) >= interval_seconds:
                # Perform automatic log
                perform_logging()
                st.session_state.last_log_time = current_time

        df = load_data()

        # Apply filters even if df is empty for consistency
        filtered_df = df.copy() if not df.empty else pd.DataFrame()
        if not df.empty and date_range:
            if len(date_range) == 2:
                start_date, end_date = date_range
                filtered_df = filtered_df[(filtered_df['timestamp'].dt.date >= start_date) & (
                    filtered_df['timestamp'].dt.date <= end_date)]
        if not df.empty and weekday_filter:
            filtered_df = filtered_df[filtered_df['weekday'].isin(
                weekday_filter)]
        if not df.empty and destination_filter != "All":
            filtered_df = filtered_df[filtered_df['destination']
                                      == destination_filter]
        if not df.empty and 'route' in df.columns and route_filter:
            filtered_df = filtered_df[filtered_df['route'].isin(
                route_filter)]

        if df.empty:
            st.write("No data available yet.")
        else:
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
                st.subheader("Optimal Departure Windows")

                # Best for outbound (A/B) in morning 4AM-9AM
                outbound_df = filtered_df[filtered_df['destination'].isin(
                    ['A', 'B'])] if not filtered_df.empty else pd.DataFrame()
                if not outbound_df.empty:
                    morning_df = outbound_df[(outbound_df['hour'] >= 4) & (
                        outbound_df['hour'] <= 9)]
                    if not morning_df.empty and not morning_df.groupby('hour')['duration_sec'].mean().empty:
                        morning_avg = morning_df.groupby(
                            'hour')['duration_sec'].mean()
                        best_morning = morning_avg.idxmin()
                        best_dur = morning_avg.min()
                        st.write(
                            f"**Outbound (A/B) Morning:** {best_morning}:00 ({best_dur/60:.1f} min)")
                    else:
                        st.write(
                            "**Outbound (A/B) Morning:** Insufficient data in 4AM-9AM range")
                else:
                    st.write(
                        "**Outbound (A/B) Morning:** No outbound data available")

                # Best for return (C/D) in evening 4PM-8PM
                return_df = filtered_df[filtered_df['destination'].isin(
                    ['C', 'D'])] if not filtered_df.empty else pd.DataFrame()
                if not return_df.empty:
                    evening_df = return_df[(return_df['hour'] >= 16) & (
                        return_df['hour'] <= 20)]
                    if not evening_df.empty and not evening_df.groupby('hour')['duration_sec'].mean().empty:
                        evening_avg = evening_df.groupby(
                            'hour')['duration_sec'].mean()
                        best_evening = evening_avg.idxmin()
                        best_dur = evening_avg.min()
                        st.write(
                            f"**Return (C/D) Evening:** {best_evening}:00 ({best_dur/60:.1f} min)")
                    else:
                        st.write(
                            "**Return (C/D) Evening:** Insufficient data in 4PM-8PM range")
                else:
                    st.write("**Return (C/D) Evening:** No return data available")

                # Alert for high traffic
                if not filtered_df.empty:
                    desc = filtered_df['duration_sec'].describe()
                    latest_dur = filtered_df['duration_sec'].iloc[-1]
                    if latest_dur > desc['mean'] + desc['std']:
                        st.warning(
                            f"🚨 High traffic alert! Latest duration: {latest_dur} sec ({latest_dur/60:.1f} min)")
                    elif latest_dur > desc['75%']:
                        st.info(
                            f"⚠️ Elevated traffic. Latest: {latest_dur} sec")

            # Enhanced Visualizations with Plotly
            st.subheader("📊 Enhanced Traffic Visualizations")

            if not filtered_df.empty:
                # Convert duration to minutes for better readability
                filtered_df['duration_min'] = filtered_df['duration_sec'] / 60

                # 1. Interactive Bar Chart: Average Duration by Weekday
                st.subheader("📅 Average Travel Time by Weekday")
                weekday_avg = filtered_df.groupby(
                    'weekday')['duration_min'].mean().reset_index()
                weekday_avg['weekday'] = pd.Categorical(weekday_avg['weekday'],
                                                        categories=[
                                                            'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'],
                                                        ordered=True)
                weekday_avg = weekday_avg.sort_values('weekday')

                fig_weekday = px.bar(weekday_avg, x='weekday', y='duration_min',
                                     title='Average Travel Time by Day of Week',
                                     labels={
                                         'duration_min': 'Duration (minutes)', 'weekday': 'Day'},
                                     color='duration_min',
                                     color_continuous_scale='RdYlGn_r')
                fig_weekday.update_layout(showlegend=False)
                st.plotly_chart(
                    fig_weekday, use_container_width=True, key='weekday_chart')

                # 2. Interactive Line Chart: Average Duration by Hour
                st.subheader("🕐 Average Travel Time by Hour")
                hour_avg = filtered_df.groupby(
                    'hour')['duration_min'].mean().reset_index()

                fig_hour = px.line(hour_avg, x='hour', y='duration_min',
                                   title='Average Travel Time Throughout the Day',
                                   labels={
                                       'duration_min': 'Duration (minutes)', 'hour': 'Hour of Day'},
                                   markers=True, line_shape='spline')
                fig_hour.update_xaxes(tickmode='linear', tick0=0, dtick=1)
                fig_hour.update_traces(
                    mode='lines+markers', hovertemplate='Hour: %{x}<br>Duration: %{y:.1f} min')
                st.plotly_chart(
                    fig_hour, use_container_width=True, key='hour_chart')

                # 3. Interactive Heatmap: Hour vs Weekday
                st.subheader("🔥 Traffic Intensity Heatmap")
                pivot = filtered_df.pivot_table(
                    values='duration_min', index='weekday', columns='hour', aggfunc='mean')
                # Reorder weekdays
                weekday_order = ['Monday', 'Tuesday', 'Wednesday',
                                 'Thursday', 'Friday', 'Saturday', 'Sunday']
                pivot = pivot.reindex(weekday_order)

                fig_heatmap = go.Figure(data=go.Heatmap(
                    z=pivot.values,
                    x=pivot.columns,
                    y=pivot.index,
                    colorscale='RdYlGn_r',
                    hoverongaps=False,
                    hovertemplate='Day: %{y}<br>Hour: %{x}<br>Duration: %{z:.1f} min<extra></extra>'
                ))
                fig_heatmap.update_layout(
                    title='Traffic Duration Heatmap (Hour vs Weekday)',
                    xaxis_title='Hour of Day',
                    yaxis_title='Day of Week'
                )
                st.plotly_chart(
                    fig_heatmap, use_container_width=True, key='heatmap_chart')

                # 4. New: Duration Distribution Histogram
                st.subheader("📈 Travel Time Distribution")
                fig_hist = px.histogram(filtered_df, x='duration_min',
                                        title='Distribution of Travel Times',
                                        labels={
                                            'duration_min': 'Duration (minutes)', 'count': 'Frequency'},
                                        nbins=30,
                                        color_discrete_sequence=['#636EFA'])
                fig_hist.update_layout(showlegend=False)
                st.plotly_chart(
                    fig_hist, use_container_width=True, key='hist_chart')

                # 5. New: Route Performance Comparison
                if 'route' in filtered_df.columns:
                    st.subheader("🛣️ Route Performance Comparison")
                    route_avg = filtered_df.groupby(['destination', 'route'])[
                        'duration_min'].mean().reset_index()
                    route_avg['route_label'] = route_avg['destination'] + \
                        '-' + route_avg['route'].astype(str)

                    fig_routes = px.bar(route_avg, x='route_label', y='duration_min',
                                        title='Average Travel Time by Route',
                                        labels={
                                            'duration_min': 'Duration (minutes)', 'route_label': 'Route'},
                                        color='destination',
                                        color_discrete_map={'A': '#FF6B6B', 'B': '#4ECDC4', 'C': '#45B7D1', 'D': '#96CEB4'})
                    fig_routes.update_layout(xaxis_tickangle=-45)
                    st.plotly_chart(
                        fig_routes, use_container_width=True, key='routes_chart')

                # 6. Enhanced Time Series with Trend
                st.subheader("📈 Travel Time Trends Over Time")
                # Sort by timestamp
                time_df = filtered_df.sort_values('timestamp')

                fig_trend = px.scatter(time_df, x='timestamp', y='duration_min',
                                       title='Travel Time Trends with Moving Average',
                                       labels={
                                           'duration_min': 'Duration (minutes)', 'timestamp': 'Time'},
                                       trendline='rolling', trendline_options=dict(window=10),
                                       color='destination',
                                       color_discrete_map={'A': '#FF6B6B', 'B': '#4ECDC4', 'C': '#45B7D1', 'D': '#96CEB4'})
                st.plotly_chart(
                    fig_trend, use_container_width=True, key='trend_chart')

            else:
                st.write("No data available for visualizations.")
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
                        for dest in ['A', 'B', 'C', 'D']:
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
                                    if dest == 'A':
                                        origin_enc = config['origin'].replace(
                                            ' ', '+')
                                        dest_name = config['destination_a']
                                    elif dest == 'B':
                                        origin_enc = config['origin'].replace(
                                            ' ', '+')
                                        dest_name = config['destination_b']
                                    elif dest == 'C':
                                        origin_enc = config['destination_a'].replace(
                                            ' ', '+')
                                        dest_name = config['destination_c']
                                    elif dest == 'D':
                                        origin_enc = config['destination_b'].replace(
                                            ' ', '+')
                                        dest_name = config['destination_d']
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
