# generate_test_logs.py
import random
import os
from datetime import datetime, timedelta

NORMAL_TEMPLATES = [
    "log_channel(cluster) log [DBG] : osdmap {e}: {total} total, {up} up",
    "log_channel(cluster) log [DBG] : pgmap v{version}: {pgs} pgs: {peering} peering, {active_clean} active+clean; {data} KiB data, {used} GiB used, {avail} GiB avail",
    "log_channel(audit) log [INF] : from='{entity}' entity='{entity_name}' cmd=[{command}] : dispatch",
    "log_channel(cluster) log [DBG] : scrub starts",
    "log_channel(cluster) log [DBG] : scrub ok",
    "log_channel(client) log [INF] : client {client_id} executed command {command} successfully",
    "log_channel(network) log [INF] : network connection established: {network_id}",
    "log_channel(pg) log [INF] : PG {pg_id} active and clean",
    "log_channel(cluster) log [INF] : osd {osd_id} is healthy"
]

ANOMALY_TEMPLATES = [
    "log_channel(cluster) log [WRN] : OSD {osd_id} is down: {error_message}",
    "log_channel(cluster) log [ERR] : OSD {osd_id} failure: unable to recover data from {pg_id}",
    "log_channel(cluster) log [ERR] : OSD {osd_id} failure: disk full, unable to write data",
    "log_channel(cluster) log [ERR] : MON {mon_id} crashed due to {crash_reason}",
    "log_channel(cluster) log [ERR] : OSD {osd_id} marked as faulty: {error_details}",
    "log_channel(cluster) log [ERR] : OSD {osd_id} failed to recover from {pg_id} after {time_period}",
    "log_channel(cluster) log [ERR] : OSD {osd_id} disk corruption detected, data may be lost",
    "log_channel(cluster) log [ERR] : OSD {osd_id} cannot find disk, going into error state"
] 

def generate_log_line(template, current_time):
    values = {
        "e": random.randint(100, 200),
        "total": random.randint(10, 20),
        "up": random.randint(5, 10),
        "version": random.randint(1000, 9999),
        "pgs": random.randint(100, 500),
        "peering": random.randint(1, 50),
        "active_clean": random.randint(80, 100),
        "data": random.randint(50000, 100000),
        "used": random.randint(1, 50),
        "avail": random.randint(50, 100),
        "entity": f"client.{random.randint(1, 100)}",
        "entity_name": f"client-name-{random.randint(1, 100)}",
        "command": random.choice(["status", "osd tree", "df"]),
        "client_id": random.randint(1000, 9999),
        "network_id": f"net-{random.randint(100, 999)}",
        "pg_id": random.randint(1, 500),
        "osd_id": random.randint(0, 10),
        "error_message": random.choice(["no heartbeat", "I/O error", "timeout"]),
        "crash_reason": random.choice(["out of memory", "assert failed", "segfault"]),
        "error_details": random.choice(["read error", "unmounted", "disk offline"]),
        "time_period": f"{random.randint(10, 120)}s",
        "mon_id": random.randint(0, 5)
    }
    timestamp = current_time.strftime("%Y-%m-%d %H:%M:%S")
    return f"{timestamp} {template.format(**{k: v for k, v in values.items() if f'{{{k}}}' in template})}"

def generate_logs(n_total=500, anomaly_ratio=0.2):
    n_anomalies = int(n_total * anomaly_ratio)
    n_normal = n_total - n_anomalies

    start_time = datetime.now()
    logs = [
        generate_log_line(random.choice(NORMAL_TEMPLATES), start_time + timedelta(seconds=i)) for i in range(n_normal)
    ] + [
        generate_log_line(random.choice(ANOMALY_TEMPLATES), start_time + timedelta(seconds=n_normal + i)) for i in range(n_anomalies)
    ]
    random.shuffle(logs)
    return logs

if __name__ == "__main__":
    output_path = "../data/logs/ceph_real.log"
    logs = generate_logs(n_total=100)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for line in logs:
            f.write(line + "\n")

    print(f"✅ Generated 100 test logs (with 20% anomalies) at {output_path}")