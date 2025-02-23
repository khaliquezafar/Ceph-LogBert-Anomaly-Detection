import random
import time
from datetime import datetime

# Define log levels
LOG_LEVELS = ["INFO", "DEBUG", "WARNING", "ERROR", "CRITICAL"]

# Define Ceph components
COMPONENTS = [
    "mon", "osd", "mgr", "mds", "rados", "rbd", "rgw", "crush", "auth", "journal", "pgmap"
]

# Define normal log messages
NORMAL_MESSAGES = [
    "Cluster health check passed",
    "OSD heartbeat received",
    "PGs are active+clean",
    "Monitor quorum established",
    "OSD map update completed",
    "Client request completed successfully",
    "Scrub process completed for OSD {id}",
    "Client authenticated successfully for pool {pool}",
    "PG {pg_id} successfully recovered",
    "Data rebalancing in progress...",
    "Bucket index updated in RGW",
    "RADOS object read successful",
    "Cluster network is stable",
    "New monitor leader elected",
    "CephFS metadata synchronized",
    "Snapshot creation completed",
    "Background deep scrub initiated",
    "Garbage collection completed",
]

# Define anomalies
ANOMALIES = [
    ("OSD Failure", "ERROR", "OSD {id} marked down due to missed heartbeats"),
    ("Slow Requests", "WARNING", "OSD {id} experiencing slow requests"),
    ("High Latency", "WARNING", "High latency detected on OSD {id}"),
    ("Disk Failure", "CRITICAL", "Disk failure detected on host {host}, OSD {id}"),
    ("Network Issue", "ERROR", "Lost connection to OSD {id}, retrying..."),
    ("PG Stuck", "CRITICAL", "PG {pg_id} stuck in inconsistent state"),
    ("High CPU Usage", "WARNING", "High CPU usage detected on {host}"),
    ("Memory Leak", "ERROR", "Memory leak detected in process {component}"),
    ("Corrupt Data", "CRITICAL", "Corrupt data found in PG {pg_id}, recovery needed"),
    ("Misplaced Objects", "WARNING", "{count} objects misplaced in PG {pg_id}"),
    ("Failed Authentication", "ERROR", "Failed client authentication attempt for pool {pool}"),
    ("Snapshot Corruption", "CRITICAL", "Snapshot corrupted in pool {pool}, manual intervention needed"),
    ("High Disk I/O", "WARNING", "High disk I/O detected on OSD {id}"),
    ("Object Store Full", "CRITICAL", "Ceph object store near full capacity, threshold exceeded"),
    ("Slow Scrubbing", "WARNING", "Deep scrub on OSD {id} taking longer than expected"),
    ("Stuck Requests", "ERROR", "Several client requests are stuck, PG {pg_id} unresponsive"),
    ("Cache Eviction Failure", "ERROR", "Failed to evict objects from cache tier"),
    ("Filesystem Error", "CRITICAL", "CephFS mount failure detected on client node"),
]

# Function to generate a timestamp
def get_timestamp():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# Function to generate a normal log entry
def generate_normal_log():
    timestamp = get_timestamp()
    level = random.choice(LOG_LEVELS[:3])  # Mostly INFO, DEBUG, or WARNING
    component = random.choice(COMPONENTS)
    message = random.choice(NORMAL_MESSAGES).format(
        id=random.randint(0, 50),
        pg_id=random.randint(100, 999),
        pool=random.choice(["pool1", "pool2", "pool3"]),
    )
    return f"{timestamp} {level} {component}: {message}"

# Function to generate an anomaly log entry
def generate_anomaly_log():
    timestamp = get_timestamp()
    anomaly, level, message_template = random.choice(ANOMALIES)

    # Replace placeholders dynamically
    message = message_template.format(
        id=random.randint(0, 50),
        host=f"ceph-node{random.randint(1, 5)}",
        pg_id=random.randint(100, 999),
        pool=random.choice(["pool1", "pool2", "pool3"]),
        count=random.randint(1000, 5000),
        component=random.choice(COMPONENTS),
    )
    return f"{timestamp} {level} {anomaly}: {message}"

# Function to generate logs with anomalies
def generate_logs(output_file="../data/synthetic_logs.log", num_log=100000, anomaly_ratio=0.20):
    with open(output_file, "w") as f:
        for i in range(num_log):
            # Inject anomalies based on the defined ratio
            if random.random() < anomaly_ratio:
                log_entry = generate_anomaly_log()
            else:
                log_entry = generate_normal_log()

            f.write(log_entry + "\n")

            # Simulate time gaps between log entries
            time.sleep(random.uniform(0.01, 0.1))  # Simulating real-time logs

    print(f"Log file '{output_file}' generated with {num_log} entries.")

# Run the log generator
if __name__ == "__main__":
    generate_logs()
    print("Synthetic logs generated in data/synthetic_logs.log")