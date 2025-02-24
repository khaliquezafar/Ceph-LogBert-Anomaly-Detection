import random
import time
import uuid

def generate_timestamp():
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time() - random.randint(0, 60)))

def generate_host():
    return f'ceph-node-{random.randint(1, 5)}'

def generate_pid():
    return random.randint(1000, 9999)

def generate_normal_log():
    timestamp = generate_timestamp()
    host = generate_host()
    pid = generate_pid()
    service = random.choice(['mon', 'osd', 'mgr', 'radosgw', 'mds', 'client'])
    severity = random.choice(['INFO', 'DEBUG', 'NOTICE'])
    message = random.choice([
        'OSD 1.2 heartbeat sent to OSD 1.1',
        'Client 10.0.0.1 connected to monitor',
        'PG 1.2 replicated successfully',
        'Scrubbing completed for PG 1.5',
        'Monitor quorum established: mon1, mon2, mon3',
        'Data replication completed for object pool',
        'Backfilling completed for PG 1.9',
        'Metadata operation completed successfully on MDS',
        'RADOS Gateway bucket created successfully',
        'File write operation completed by client 10.0.0.2',
        'Cluster status: HEALTH_OK',
        'Monitor map has been updated',
        'OSD disk space utilization: 70%',
        'OSD rebalancing completed',
        'MGR module loaded successfully',
    ])
    return f'{timestamp} {host} {service}[{pid}]: {severity}: {message}'

def generate_anomaly_log():
    timestamp = generate_timestamp()
    host = generate_host()
    pid = generate_pid()
    service = random.choice(['mon', 'osd', 'mgr', 'radosgw', 'mds', 'client'])
    severity = random.choice(['ERROR', 'WARNING', 'CRITICAL'])
    message = random.choice([
        'OSD 3.1 marked down due to heartbeat failure',
        'Monitor mon1 failed to reach quorum',
        'CRITICAL: PG 2.1 stuck in inconsistent state',
        'Slow request detected on OSD 4.3',
        'Data corruption detected in object pool',
        'Disk failure reported on ceph-node-4',
        'High I/O latency detected in radosgw service',
        'MDS failure detected: metadata inconsistency',
        'Monitor down, unable to connect to quorum',
        'Client connection timeout detected',
        'OSD disk space utilization exceeded 90%',
        'CRITICAL: Data loss detected on pool 1',
        'Network partition detected between OSD nodes',
        'Authentication failure in RADOS Gateway',
        'Cluster status: HEALTH_WARN',
        'Journal write failure detected on OSD node',
        'Uncorrectable ECC error detected in MDS cache',
    ])
    return f'{timestamp} {host} {service}[{pid}]: {severity}: {message}'

def generate_synthetic_logs(num_logs=10000):
    logs = []
    anomaly_count = int(num_logs * 0.1)  # 20% anomalies

    for _ in range(num_logs - anomaly_count):
        logs.append(generate_normal_log())

    for _ in range(anomaly_count):
        logs.append(generate_anomaly_log())

    random.shuffle(logs)
    return logs

def save_logs_to_file(filename='../data/synthetic_logs.log', num_logs=10000):
    logs = generate_synthetic_logs(num_logs)
    with open(filename, 'w') as file:
        for log in logs:
            file.write(log + '\n')
    print(f'Successfully generated {num_logs} logs with 10% anomalies in {filename}')

# Generate and save logs
save_logs_to_file('../data/synthetic_logs.log', 10000)
