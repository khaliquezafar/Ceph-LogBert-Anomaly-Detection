# gen_training_logs.py
import random
from datetime import datetime, timedelta

# === Normal Log Templates ===
NORMAL_TEMPLATES = [
    # Cluster logs
    "log_channel(cluster) log [DBG] : osdmap {e}: {total} total, {up} up",
    "log_channel(cluster) log [DBG] : pgmap v{version}: {pgs} pgs: {peering} peering, {active_clean} active+clean; {data} KiB data, {used} GiB used, {avail} GiB avail",
    "log_channel(cluster) log [DBG] : scrub starts",
    "log_channel(cluster) log [DBG] : scrub ok",
    "log_channel(cluster) log [INF] : osd {osd_id} is healthy",
    "log_channel(cluster) log [DBG] : recovery completed on osd.{osd_id}",
    "log_channel(cluster) log [INF] : mon.{mon_id} quorum formed with mons: {mon_list}",

    # Audit logs
    "log_channel(audit) log [INF] : from='{entity}' entity='{entity_name}' cmd=[{command}] : dispatch",

    # Client logs
    "log_channel(client) log [INF] : client {client_id} executed command {command} successfully",

    # Network logs
    "log_channel(network) log [INF] : network connection established: {network_id}",
    "log_channel(network) log [DBG] : heartbeat received from mon.{mon_id}",

    # PG logs
    "log_channel(pg) log [INF] : PG {pg_id} active and clean",
    "log_channel(pg) log [DBG] : PG {pg_id} scrub completed",

    # Metadata Server (MDS) logs
    "log_channel(mds) log [DBG] : MDS.{mds_id} is exporting namespace to rank {rank}",
    "log_channel(mds) log [INF] : MDS.{mds_id} session with client.{client_id} established",
    "log_channel(mds) log [DBG] : MDS.{mds_id} updated inode cache size: {inode_count}",

    # RADOS Gateway (RGW) logs
    "log_channel(rgw) log [INF] : RGW op success: {method} /{bucket}/{object_name}",
    "log_channel(rgw) log [DBG] : RGW bucket index updated for bucket '{bucket}'",
    "log_channel(rgw) log [INF] : RGW user '{user}' authenticated successfully",

    # RBD (RADOS Block Device) logs
    "log_channel(rbd) log [INF] : RBD image '{rbd_image}' mapped to /dev/rbd{rbd_id}",
    "log_channel(rbd) log [DBG] : RBD snapshot '{snapshot}' created for image '{rbd_image}'",
    "log_channel(rbd) log [INF] : RBD unmap operation for image '{rbd_image}' completed successfully"
]

# === Function to Generate a Single Normal Log Line ===
def generate_log_line(template, timestamp):
    return f"{timestamp} " + template.format(
        e=random.randint(100, 200),
        total=random.randint(10, 20),
        up=random.randint(5, 20),
        version=random.randint(1000, 9999),
        pgs=random.randint(100, 500),
        peering=random.randint(0, 50),
        active_clean=random.randint(50, 500),
        data=random.randint(50000, 100000),
        used=random.randint(1, 100),
        avail=random.randint(1, 100),
        entity=f"client.{random.randint(1, 100)}",
        entity_name=f"client-name-{random.randint(1, 100)}",
        command=random.choice(["status", "df", "osd tree", "mon_status", "auth list"]),
        client_id=random.randint(1000, 9999),
        network_id=f"net-{random.randint(100, 999)}",
        pg_id=random.randint(100, 999),
        osd_id=random.randint(0, 10),
        mon_id=random.randint(0, 5),
        mon_list=','.join([f"mon.{i}" for i in random.sample(range(0, 5), 3)]),
        mds_id=random.randint(0, 3),
        rank=random.randint(0, 3),
        inode_count=random.randint(1000, 100000),
        method=random.choice(["GET", "PUT", "DELETE"]),
        bucket=f"bucket-{random.randint(1, 20)}",
        object_name=f"object-{random.randint(100, 999)}.bin",
        user=f"user-{random.randint(1, 50)}",
        rbd_image=f"rbd-image-{random.randint(1, 10)}",
        rbd_id=random.randint(0, 10),
        snapshot=f"snap-{random.randint(100, 999)}",
    )

# === Generate Normal Logs ===
def generate_logs(n_logs=100000):
    logs = []
    base_time = datetime.now()
    for i in range(n_logs):
        ts = (base_time + timedelta(seconds=i * 10)).strftime("%Y-%m-%d %H:%M:%S")
        template = random.choice(NORMAL_TEMPLATES)
        logs.append(generate_log_line(template, ts))
    return logs

# === Save Logs to File ===
def save_logs_to_file(logs, path="../data/logs/ceph.log"):
    with open(path, "w", encoding="utf-8") as f:
        for line in logs:
            f.write(line + "\n")

if __name__ == "__main__":
    logs = generate_logs(100000)
    save_logs_to_file(logs)
    print("✅ Normal training logs saved to ../data/logs/ceph.log")
