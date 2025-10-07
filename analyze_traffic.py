import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV data
df = pd.read_csv('traffic_data.csv')

# Convert timestamp to datetime
df['timestamp'] = pd.to_datetime(df['timestamp'])

# Basic statistics
print("Basic Statistics for Duration (seconds):")
print(df['duration_sec'].describe())
print("\n")

# Average duration by weekday
print("Average Duration by Weekday:")
weekday_avg = df.groupby('weekday')['duration_sec'].mean()
print(weekday_avg)
print("\n")

# Average duration by hour
df['hour'] = df['timestamp'].dt.hour
print("Average Duration by Hour:")
hour_avg = df.groupby('hour')['duration_sec'].mean()
print(hour_avg)
print("\n")

# Plot duration over time
plt.figure(figsize=(10, 5))
plt.plot(df['timestamp'], df['duration_sec'])
plt.title('Traffic Duration Over Time')
plt.xlabel('Timestamp')
plt.ylabel('Duration (seconds)')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('traffic_analysis.png')
print("Time series plot saved as traffic_analysis.png")

# Visualize average travel time by weekday
plt.figure(figsize=(8, 5))
weekday_avg.plot(kind='bar')
plt.title('Average Travel Time by Weekday')
plt.xlabel('Weekday')
plt.ylabel('Average Duration (seconds)')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('weekday_avg.png')
print("Weekday average plot saved as weekday_avg.png")

# Visualize average travel time by hour
plt.figure(figsize=(8, 5))
hour_avg.plot(kind='bar')
plt.title('Average Travel Time by Hour')
plt.xlabel('Hour')
plt.ylabel('Average Duration (seconds)')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('hour_avg.png')
print("Hour average plot saved as hour_avg.png")

# Identify optimal departure windows
min_duration_hour = hour_avg.idxmin()
min_duration = hour_avg.min()
print(
    f"\nOptimal departure window: {min_duration_hour}:00 with average duration of {min_duration} seconds ({min_duration/60:.1f} minutes)")
