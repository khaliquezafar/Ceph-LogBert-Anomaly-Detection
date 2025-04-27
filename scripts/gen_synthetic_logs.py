# === Ceph storage synthetic logs script ===

import random
from datetime import datetime, timedelta
import os

# === Log Templates  ===
NORMAL_TEMPLATES_UNLABELED = [
    "log_channel(cluster) log [DBG] : osdmap {e}: {total} total, {up} up",
    "log_channel(cluster) log [DBG] : pgmap v{version}: {pgs} pgs: {peering} peering, {active_clean} active+clean; {data} KiB data, {used} GiB used, {avail} GiB avail",
    "log_channel(cluster) log [DBG] : scrub starts",
    "log_channel(cluster) log [DBG] : scrub ok",
    "log_channel(cluster) log [INF] : osd {osd_id} is healthy",
    "log_channel(cluster) log [DBG] : heartbeat from OSD {osd_id}",
    "log_channel(cluster) log [DBG] : recovery completed on osd.{osd_id}",
    "log_channel(cluster) log [INF] : mon.{mon_id} quorum formed with mons: {mon_list}",
    "log_channel(mon) log [INF] : MON {mon_id} joined quorum",
    "log_channel(client) log [INF] : client {client_id} executed command {command} successfully",
    "log_channel(network) log [INF] : network connection established: {network_id}",
    "log_channel(network) log [DBG] : heartbeat received from mon.{mon_id}",
    "log_channel(pg) log [INF] : PG {pg_id} active and clean",
    "log_channel(pg) log [DBG] : PG {pg_id} scrub completed",
    "log_channel(cluster) log [DBG] : recovery completed for PG {pg_id}",
    "log_channel(cluster) log [INF] : snapshot {pg_id} created",
    "log_channel(mds) log [DBG] : MDS.{mds_id} is exporting namespace to rank {rank}",
    "log_channel(mds) log [INF] : MDS.{mds_id} session with client.{client_id} established",
    "log_channel(mds) log [DBG] : MDS.{mds_id} updated inode cache size: {inode_count}",
    "log_channel(rgw) log [INF] : RGW op success: {method} /{bucket}/{object_name}",
    "log_channel(rgw) log [DBG] : RGW bucket index updated for bucket '{bucket}'",
    "log_channel(rgw) log [INF] : RGW user '{user}' authenticated successfully",
    "log_channel(rbd) log [INF] : RBD image '{rbd_image}' mapped to /dev/rbd{rbd_id}",
    "log_channel(rbd) log [DBG] : RBD snapshot '{snapshot}' created for image '{rbd_image}'",
    "log_channel(rbd) log [INF] : RBD unmap operation for image '{rbd_image}' completed successfully",
    "log_channel(client) log [OK] : client {client_id} mount verified",
    "log_channel(cluster) log [INF] : RADOS operation completed for pool {pg_id}",
    "log_channel(cluster) log [DBG] : OSD {osd_id} backfill completed",
    "log_channel(cluster) log [DBG] : auto-scaler adjusted PG count for pool {pg_id}",
    "log_channel(cluster) log [INF] : reweight-by-utilization applied",
    "log_channel(mon) log [DBG] : monitor election finished",
    "log_channel(cluster) log [OK] : health check passed",
    "log_channel(mon) log [INF] : MON {mon_id} beacon received",
    "log_channel(mgr) log [INF] : mgr module reload succeeded",
    "log_channel(cluster) log [INF] : client reconnect complete",
    "log_channel(cluster) log [DBG] : MDS startup completed",
    "log_channel(mon) log [INF] : config update committed",
    "log_channel(audit) log [INF] : auth key accepted for client {client_id}",
    "log_channel(pg) log [DBG] : PG {pg_id} is clean",
    "log_channel(cluster) log [OK] : all services responding",
    "log_channel(mon) log [INF] : no pending health issues",
    "log_channel(cluster) log [INF] : disk utilization within thresholds",
    "log_channel(client) log [INF] : object read completed by client {client_id}",
    "log_channel(cluster) log [DBG] : logs rotated for OSD {osd_id}"
]

NORMAL_TEMPLATES_LABELED = [
    "log_channel(cluster) log [DBG] : osdmap {e}: {total} total, {up} up",
    "log_channel(cluster) log [DBG] : scrub starts",
    "log_channel(cluster) log [DBG] : scrub ok",
    "log_channel(client) log [INF] : client {client_id} executed command {command} successfully",
    "log_channel(pg) log [INF] : PG {pg_id} active and clean",
    "log_channel(cluster) log [INF] : osd {osd_id} is healthy"
    "log_channel(cluster) log [DBG] : osdmap {e}: {total} total, {up} up",
    "log_channel(cluster) log [DBG] : pgmap v{version}: {pgs} pgs: {peering} peering, {active_clean} active+clean; {data} KiB data, {used} GiB used, {avail} GiB avail",
    "log_channel(cluster) log [DBG] : scrub starts",
    "log_channel(cluster) log [DBG] : scrub ok",
    "log_channel(cluster) log [INF] : osd {osd_id} is healthy",
    "log_channel(cluster) log [DBG] : heartbeat from OSD {osd_id}",
    "log_channel(cluster) log [DBG] : recovery completed on osd.{osd_id}",
    "log_channel(cluster) log [INF] : mon.{mon_id} quorum formed with mons: {mon_list}",
    "log_channel(mon) log [INF] : MON {mon_id} joined quorum",
    "log_channel(client) log [INF] : client {client_id} executed command {command} successfully",
    "log_channel(network) log [INF] : network connection established: {network_id}",
    "log_channel(network) log [DBG] : heartbeat received from mon.{mon_id}",
    "log_channel(pg) log [INF] : PG {pg_id} active and clean",
    "log_channel(pg) log [DBG] : PG {pg_id} scrub completed",
    "log_channel(cluster) log [DBG] : recovery completed for PG {pg_id}",
    "log_channel(cluster) log [INF] : snapshot {pg_id} created",
    "log_channel(mds) log [DBG] : MDS.{mds_id} is exporting namespace to rank {rank}",
    "log_channel(mds) log [INF] : MDS.{mds_id} session with client.{client_id} established",
    "log_channel(mds) log [DBG] : MDS.{mds_id} updated inode cache size: {inode_count}",
    "log_channel(rgw) log [INF] : RGW op success: {method} /{bucket}/{object_name}",
    "log_channel(rgw) log [DBG] : RGW bucket index updated for bucket '{bucket}'",
    "log_channel(rgw) log [INF] : RGW user '{user}' authenticated successfully",
    "log_channel(rbd) log [INF] : RBD image '{rbd_image}' mapped to /dev/rbd{rbd_id}",
    "log_channel(rbd) log [DBG] : RBD snapshot '{snapshot}' created for image '{rbd_image}'",
    "log_channel(rbd) log [INF] : RBD unmap operation for image '{rbd_image}' completed successfully",
    "log_channel(client) log [OK] : client {client_id} mount verified",
    "log_channel(cluster) log [INF] : RADOS operation completed for pool {pg_id}",
    "log_channel(cluster) log [DBG] : OSD {osd_id} backfill completed",
    "log_channel(cluster) log [DBG] : auto-scaler adjusted PG count for pool {pg_id}",
    "log_channel(cluster) log [INF] : reweight-by-utilization applied",
    "log_channel(mon) log [DBG] : monitor election finished",
    "log_channel(cluster) log [OK] : health check passed",
    "log_channel(mon) log [INF] : MON {mon_id} beacon received",
    "log_channel(mgr) log [INF] : mgr module reload succeeded",
    "log_channel(cluster) log [INF] : client reconnect complete",
    "log_channel(cluster) log [DBG] : MDS startup completed",
    "log_channel(mon) log [INF] : config update committed",
    "log_channel(audit) log [INF] : auth key accepted for client {client_id}",
    "log_channel(pg) log [DBG] : PG {pg_id} is clean",
    "log_channel(cluster) log [OK] : all services responding",
    "log_channel(mon) log [INF] : no pending health issues",
    "log_channel(cluster) log [INF] : disk utilization within thresholds",
    "log_channel(client) log [INF] : object read completed by client {client_id}",
    "log_channel(cluster) log [DBG] : logs rotated for OSD {osd_id}"
]

ANOMALY_TEMPLATES_LABELED = [
    "log_channel(cluster) log [WRN] : OSD {osd_id} is down: {error_message}",
    "log_channel(cluster) log [ERR] : OSD {osd_id} failure: unable to recover data from {pg_id}",
    "log_channel(cluster) log [ERR] : OSD {osd_id} failure: disk full, unable to write data",
    "log_channel(cluster) log [ERR] : MON {mon_id} crashed due to {crash_reason}",
    "log_channel(cluster) log [ERR] : OSD {osd_id} marked as faulty: {error_details}",
    "log_channel(cluster) log [ERR] : OSD {osd_id} failed to recover from {pg_id} after {time_period}",
    "log_channel(cluster) log [ERR] : OSD {osd_id} disk corruption detected, data may be lost",
    "log_channel(cluster) log [ERR] : OSD {osd_id} cannot find disk, going into error state"
    "log_channel(osd) log [WRN] : inconsistent PG {pg_id} detected [error={error}]",
    "log_channel(mon) log [ERR] : monitor {mon_id} failed to respond [status=DOWN]",
    "log_channel(client) log [ERR] : failed to write object [client={client_id}, reason={reason}]",
    "log_channel(cluster) log [CRIT] : cluster full - no space left to create object",
    "log_channel(mon) log [ERR] : election timeout for monitor {mon_id}",
    "log_channel(osd) log [WRN] : degraded PG {pg_id} [objects_missing={count}]",
    "log_channel(client) log [ERR] : client {client_id} disconnected unexpectedly",
    "log_channel(cluster) log [CRIT] : possible data corruption detected in PG {pg_id}",
    "log_channel(mon) log [ERR] : authentication failure for monitor {mon_id}",
    "log_channel(osd) log [WRN] : OSD {osd_id} heartbeat missed",
    "log_channel(client) log [CRIT] : snapshot write failed [client={client_id}]",
    "log_channel(osd) log [ERR] : read error from OSD {osd_id} [pg={pg_id}]",
    "log_channel(mon) log [CRIT] : monitor {mon_id} clock skew detected",
    "log_channel(cluster) log [ERR] : slow request detected [pg={pg_id}]",
    "log_channel(osd) log [CRIT] : full OSD {osd_id} [no_space_left]",
    "log_channel(mon) log [ERR] : missing monmap entry for {mon_id}"
]

NORMAL_TEMPLATES_TEST = [
"log_channel(cluster) log [DBG] : osdmap {e}: {total} total, {up} up",
    "log_channel(cluster) log [DBG] : pgmap v{version}: {pgs} pgs: {peering} peering, {active_clean} active+clean; {data} KiB data, {used} GiB used, {avail} GiB avail",
    "log_channel(cluster) log [DBG] : scrub starts",
    "log_channel(cluster) log [DBG] : scrub ok",
    "log_channel(cluster) log [INF] : osd {osd_id} is healthy",
    "log_channel(cluster) log [DBG] : heartbeat from OSD {osd_id}",
    "log_channel(cluster) log [DBG] : recovery completed on osd.{osd_id}",
    "log_channel(cluster) log [INF] : mon.{mon_id} quorum formed with mons: {mon_list}",
    "log_channel(client) log [INF] : client {client_id} executed command {command} successfully",
    "log_channel(network) log [DBG] : heartbeat received from mon.{mon_id}",
    "log_channel(pg) log [INF] : PG {pg_id} active and clean",
    "log_channel(pg) log [DBG] : PG {pg_id} scrub completed",
    "log_channel(cluster) log [DBG] : recovery completed for PG {pg_id}",
    "log_channel(cluster) log [INF] : snapshot {pg_id} created",
    "log_channel(mds) log [DBG] : MDS.{mds_id} is exporting namespace to rank {rank}",
    "log_channel(mds) log [INF] : MDS.{mds_id} session with client.{client_id} established",
    "log_channel(mds) log [DBG] : MDS.{mds_id} updated inode cache size: {inode_count}",
    "log_channel(rgw) log [INF] : RGW op success: {method} /{bucket}/{object_name}",
    "log_channel(rgw) log [DBG] : RGW bucket index updated for bucket '{bucket}'",
    "log_channel(rgw) log [INF] : RGW user '{user}' authenticated successfully",
    "log_channel(rbd) log [INF] : RBD image '{rbd_image}' mapped to /dev/rbd{rbd_id}",
    "log_channel(rbd) log [DBG] : RBD snapshot '{snapshot}' created for image '{rbd_image}'",
    "log_channel(rbd) log [INF] : RBD unmap operation for image '{rbd_image}' completed successfully",
    "log_channel(cluster) log [INF] : RADOS operation completed for pool {pg_id}",
    "log_channel(cluster) log [DBG] : OSD {osd_id} backfill completed",
    "log_channel(cluster) log [DBG] : auto-scaler adjusted PG count for pool {pg_id}",
    "log_channel(cluster) log [INF] : reweight-by-utilization applied",
    "log_channel(mon) log [INF] : MON {mon_id} beacon received",
    "log_channel(mgr) log [INF] : mgr module reload succeeded",
    "log_channel(cluster) log [INF] : client reconnect complete",
    "log_channel(cluster) log [DBG] : MDS startup completed",
    "log_channel(mon) log [INF] : config update committed",
    "log_channel(audit) log [INF] : auth key accepted for client {client_id}",
    "log_channel(pg) log [DBG] : PG {pg_id} is clean",
    "log_channel(mon) log [INF] : no pending health issues",
    "log_channel(client) log [INF] : object read completed by client {client_id}"
]

ANOMALY_TEMPLATES_TEST = [
    "log_channel(cluster) log [WRN] : OSD {osd_id} is down: {error_message}",
    "log_channel(cluster) log [ERR] : OSD {osd_id} failure: unable to recover data from {pg_id}",
    "log_channel(cluster) log [ERR] : OSD {osd_id} failure: disk full, unable to write data",
    "log_channel(cluster) log [ERR] : MON {mon_id} crashed due to {crash_reason}",
    "log_channel(cluster) log [ERR] : OSD {osd_id} marked as faulty: {error_details}",
    "log_channel(cluster) log [ERR] : OSD {osd_id} failed to recover from {pg_id} after {time_period}",
    "log_channel(cluster) log [ERR] : OSD {osd_id} disk corruption detected, data may be lost",
    "log_channel(cluster) log [ERR] : OSD {osd_id} cannot find disk, going into error state",
    "log_channel(cluster) log [WRN] : OSD {osd_id} is down: {error_message}",
    "log_channel(cluster) log [ERR] : OSD {osd_id} failure: unable to recover data from {pg_id}",
    "log_channel(cluster) log [ERR] : OSD {osd_id} failure: disk full, unable to write data",
    "log_channel(cluster) log [ERR] : MON {mon_id} crashed due to {crash_reason}",
    "log_channel(cluster) log [ERR] : OSD {osd_id} marked as faulty: {error_details}",
    "log_channel(cluster) log [ERR] : OSD {osd_id} failed to recover from {pg_id} after {time_period}",
    "log_channel(cluster) log [ERR] : OSD {osd_id} disk corruption detected, data may be lost",
    "log_channel(cluster) log [ERR] : OSD {osd_id} cannot find disk, going into error state"
    "log_channel(osd) log [WRN] : inconsistent PG {pg_id} detected [error={error}]",
    "log_channel(mon) log [ERR] : monitor {mon_id} failed to respond [status=DOWN]",
    "log_channel(client) log [ERR] : failed to write object [client={client_id}, reason={reason}]",
    "log_channel(cluster) log [CRIT] : cluster full - no space left to create object",
    "log_channel(mon) log [ERR] : election timeout for monitor {mon_id}",
    "log_channel(osd) log [WRN] : degraded PG {pg_id} [objects_missing={count}]",
    "log_channel(client) log [ERR] : client {client_id} disconnected unexpectedly",
    "log_channel(cluster) log [CRIT] : possible data corruption detected in PG {pg_id}",
    "log_channel(mon) log [ERR] : authentication failure for monitor {mon_id}",
    "log_channel(osd) log [WRN] : OSD {osd_id} heartbeat missed",
    "log_channel(client) log [CRIT] : snapshot write failed [client={client_id}]",
    "log_channel(osd) log [ERR] : read error from OSD {osd_id} [pg={pg_id}]",
    "log_channel(mon) log [CRIT] : monitor {mon_id} clock skew detected",
    "log_channel(cluster) log [ERR] : slow request detected [pg={pg_id}]",
    "log_channel(osd) log [CRIT] : full OSD {osd_id} [no_space_left]",
    "log_channel(mon) log [ERR] : missing monmap entry for {mon_id}"
]

# === Helper Function ===
def generate_random_log(template, timestamp, extra_labels=None):
    values = {
        "e": random.randint(100, 200),
        "total": random.randint(10, 20),
        "osd_total": random.randint(10, 20),
        "up": random.randint(5, 10),
        "osd_up": random.randint(5, 10),
        "version": random.randint(1000, 9999),
        "pgs": random.randint(100, 500),
        "peering": random.randint(0, 50),
        "active_clean": random.randint(50, 500),
        "data": random.randint(50000, 100000),
        "used": random.randint(1, 100),
        "avail": random.randint(1, 100),
        "entity": f"client.{random.randint(1, 100)}",
        "entity_name": f"client-name-{random.randint(1, 100)}",
        "command": random.choice(["status", "df", "osd tree", "mon_status", "auth list"]),
        "client_id": random.randint(1000, 9999),
        "node": random.choice(["node-a", "node-b", "node-c"]),
        "network_id": f"net-{random.randint(100, 999)}",
        "pg_id": random.randint(100, 999),
        "osdmap_id": random.randint(100, 999),
        "pool_id": random.randint(1, 10),
        "pool": random.randint(1, 10),
        "osd_id": random.randint(0, 10),
        "mon_id": random.randint(0, 5),
        "mon_list": ','.join([f"mon.{i}" for i in random.sample(range(0, 5), 3)]),
        "mons": ",".join(random.sample(["a", "b", "c", "d"], 3)),
        "mds_id": random.randint(0, 3),
        "rank": random.randint(0, 3),
        "inode_count": random.randint(1000, 100000),
        "method": random.choice(["GET", "PUT", "DELETE"]),
        "bucket": f"bucket-{random.randint(1, 20)}",
        "object_name": f"object-{random.randint(100, 999)}.bin",
        "user": f"user-{random.randint(1, 50)}",
        "rbd_image": f"rbd-image-{random.randint(1, 10)}",
        "rbd_id": random.randint(0, 10),
        "snapshot": f"snap-{random.randint(100, 999)}",
        "snap_id": random.randint(1000, 2000),
        "tag": random.choice(["auto", "manual"]),
        "region": random.choice(["us-east", "eu-west"]),
        "error": random.choice(["crc_mismatch", "unknown", "bitflip"]),
        "reason": random.choice(["permission_denied", "timeout", "disk_error", "stale_session"]),
        "count": random.randint(1, 10),
        "error_message": random.choice(["no heartbeat", "I/O error", "timeout"]),
        "crash_reason": random.choice(["out of memory", "assert failed", "segfault"]),
        "error_details": random.choice(["read error", "unmounted", "disk offline"]),
        "time_period": f"{random.randint(10, 120)}s"
    }
    log_line = f"{timestamp} {template.format(**values)}"
    if extra_labels:
        log_line += f" ##{extra_labels}"
    return log_line

# === Main Logic ===
def generate_combined_logs(log_type, num_logs):
    os.makedirs("../data/raw_logs/", exist_ok=True)

    if log_type == "raw":
        logs = []
        base_time = datetime.now()
        for i in range(num_logs):
            ts = (base_time + timedelta(seconds=i * 10)).strftime("%Y-%m-%d %H:%M:%S")
            template = random.choice(NORMAL_TEMPLATES_UNLABELED)
            logs.append(generate_random_log(template, ts))
        path = "../data/raw_logs/syn_unlab_ceph_logs.log"

    elif log_type == "finetuned":
        logs = []
        base_time = datetime.now()
        for i in range(num_logs):
            ts = (base_time + timedelta(seconds=random.randint(1, 3) * i)).strftime("%Y-%m-%d %H:%M:%S")
            if i % 20 == 0:
                template = random.choice(ANOMALY_TEMPLATES_LABELED)
                logs.append(generate_random_log(template, ts, extra_labels="ANOMALY"))
            else:
                template = random.choice(NORMAL_TEMPLATES_LABELED)
                logs.append(generate_random_log(template, ts, extra_labels="NORMAL"))
        path = "../data/raw_logs/syn_lab_ceph_logs.log"

    elif log_type == "test":
        logs = []
        base_time = datetime.now()
        n_normal = int(num_logs * 0.9)
        n_anomalies = num_logs - n_normal
        for i in range(n_normal):
            ts = (base_time + timedelta(seconds=i)).strftime("%Y-%m-%d %H:%M:%S")
            template = random.choice(NORMAL_TEMPLATES_TEST)
            logs.append(generate_random_log(template, ts))
        for i in range(n_anomalies):
            ts = (base_time + timedelta(seconds=90 + i)).strftime("%Y-%m-%d %H:%M:%S")
            template = random.choice(ANOMALY_TEMPLATES_TEST)
            logs.append(generate_random_log(template, ts))
        random.shuffle(logs)
        path = "../data/raw_logs/syn_test_ceph.log"

    else:
        print("Invalid log type specified. Use 'raw', 'finetuned', or 'test'.")
        return

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(logs))

    print(f"✅ Generated {len(logs)} logs at {path}")

# === User Prompt ===
if __name__ == "__main__":
    log_type = input("Enter log type (raw / finetuned / test): ").strip().lower()
    num_logs = int(input("Enter number of logs to generate: ").strip())
    generate_combined_logs(log_type, num_logs)
