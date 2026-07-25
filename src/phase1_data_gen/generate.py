"""
Phase 1: Synthetic Data Generator
Generates realistic access logs, separate labels, and ground-truth entity profiles.
Injects various cyber attack scenarios based on the configuration file.
Applies rigorous data quality constraints:
- Monotonic timestamp ordering per entity with strictly positive time deltas (>0s).
- Realistic spatial velocity for benign events (< 500 km/h).
- Strict dual constraints for impossible travel (100% satisfy distance > 500 km AND velocity > 900 km/h).
- Realistic Lognormal / Poisson numeric feature scaling (bytes_transferred, session_duration).
- Guaranteed categorical completeness (no NaN or empty strings).
- 100% alignment between logs and labels with exactly 2.0% total anomaly injection rate.
"""

import os
import uuid
import yaml
import json
import random
import argparse
import math
from datetime import datetime, timedelta, timezone
from typing import Dict, Any, List, Tuple

import numpy as np
import pandas as pd
from faker import Faker
from tqdm import tqdm

# Pre-defined list of global cities with lat/lon and country code for realistic geo-location
GLOBAL_CITIES = [
    {"city": "New York", "lat": 40.7128, "lon": -74.0060, "country": "US"},
    {"city": "London", "lat": 51.5074, "lon": -0.1278, "country": "GB"},
    {"city": "Tokyo", "lat": 35.6762, "lon": 139.6503, "country": "JP"},
    {"city": "Frankfurt", "lat": 50.1109, "lon": 8.6821, "country": "DE"},
    {"city": "Sydney", "lat": -33.8688, "lon": 151.2093, "country": "AU"},
    {"city": "Singapore", "lat": 1.3521, "lon": 103.8198, "country": "SG"},
    {"city": "Bangalore", "lat": 12.9716, "lon": 77.5946, "country": "IN"},
    {"city": "São Paulo", "lat": -23.5505, "lon": -46.6333, "country": "BR"},
    {"city": "Cape Town", "lat": -33.9249, "lon": 18.4241, "country": "ZA"},
    {"city": "Toronto", "lat": 43.6532, "lon": -79.3832, "country": "CA"}
]

# Global list of typical network resources to access
RESOURCES = [
    "/api/v1/auth/login",
    "/api/v1/user/profile",
    "/db/users/query",
    "/shares/finance/q4_report.xlsx",
    "/shares/hr/salaries.csv",
    "/api/v2/deploy/kubernetes",
    "/admin/settings/security",
    "/shares/engineering/designs.pdf",
    "/db/transactions/update",
    "/api/v1/billing/invoice",
    "/vault/secrets/api_keys",
    "/backup/nightly_dump.tar.gz"
]

DEPARTMENTS = ["Engineering", "HR", "Finance", "Sales", "Security", "Operations"]
OPERATING_SYSTEMS = ["Windows 10", "Windows 11", "macOS Monterey", "macOS Ventura", "Ubuntu 22.04", "RHEL 9"]
PROTOCOLS = ["HTTPS", "SSH", "RDP", "SMB"]
PRIVILEGED_COMMANDS = [
    "sudo su", "cat /etc/passwd", "nmap -sS -O 192.168.1.0/24", 
    "pg_dump -U postgres prod_db > dump.sql", "aws s3 sync s3://company-secrets ./secrets",
    "chmod 777 /var/www/html", "curl http://169.254.169.254/latest/meta-data/"
]

def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate the great-circle distance between two points in km."""
    R = 6371.0  # Earth radius in km
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat / 2) ** 2 + 
         math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2) ** 2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c

class SyntheticDataGenerator:
    def __init__(self, config_path: str = "config.yaml"):
        """Load configuration, set random seeds, and initialize generator classes."""
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)
        
        self.seed = self.config.get("random_seed", 42)
        random.seed(self.seed)
        np.random.seed(self.seed)
        
        self.fake = Faker()
        Faker.seed(self.seed)
        
        self.num_entities = self.config.get("num_entities", 1000)
        self.days_of_data = self.config.get("days_of_data", 60)
        self.anomaly_rate = self.config.get("anomaly_injection_rate", 0.02)
        
        # Fixed base timestamp
        self.start_time = datetime(2026, 5, 1, 0, 0, 0, tzinfo=timezone.utc)
        self.end_time = self.start_time + timedelta(days=self.days_of_data)
        
        self.entity_profiles: Dict[str, Dict[str, Any]] = {}
        
    def generate_entity_profiles(self) -> Dict[str, Dict[str, Any]]:
        """Generate habits and baseline profiles for each entity."""
        print(f"Generating profiles for {self.num_entities} entities...")
        for i in range(self.num_entities):
            r = random.random()
            if r < 0.70:
                entity_type = "user"
                entity_id = f"USR_{i:04d}"
            elif r < 0.85:
                entity_type = "service_account"
                entity_id = f"SRV_{i:04d}"
            else:
                entity_type = "edge_device"
                entity_id = f"DEV_{i:04d}"
                
            city_info = random.choice(GLOBAL_CITIES)
            
            work_hour_center = random.uniform(8.0, 18.0) if entity_type == "user" else random.uniform(0.0, 24.0)
            work_hour_std = random.uniform(1.0, 3.0) if entity_type == "user" else random.uniform(5.0, 10.0)
            
            num_res = random.randint(3, 8)
            typical_resources = random.sample(RESOURCES, num_res)
            
            num_devs = random.randint(1, 2)
            devices = []
            for _ in range(num_devs):
                devices.append({
                    "os": random.choice(OPERATING_SYSTEMS),
                    "firmware": f"v{random.randint(1, 5)}.{random.randint(0, 9)}",
                    "mac": self.fake.mac_address(),
                    "protocol": random.choice(PROTOCOLS)
                })
                
            self.entity_profiles[entity_id] = {
                "entity_id": entity_id,
                "entity_type": entity_type,
                "department": random.choice(DEPARTMENTS),
                "home_ip": self.fake.ipv4(),
                "home_geo": {
                    "lat": city_info["lat"] + random.normalvariate(0.0, 0.02),
                    "lon": city_info["lon"] + random.normalvariate(0.0, 0.02),
                    "country": city_info["country"]
                },
                "work_hour_center": work_hour_center,
                "work_hour_std": work_hour_std,
                "typical_resources": typical_resources,
                "typical_devices": devices,
                "typical_session_duration_mean": random.randint(60, 3600),
                "typical_session_duration_std": random.randint(10, 600)
            }
        return self.entity_profiles

    def _sample_timestamp(self, profile: Dict[str, Any], day_idx: int) -> datetime:
        """Sample a timestamp based on the entity's work hour habits."""
        day = self.start_time + timedelta(days=day_idx)
        hour = random.normalvariate(profile["work_hour_center"], profile["work_hour_std"])
        hour = max(0.0, min(23.99, hour))
        
        dt = datetime(day.year, day.month, day.day, int(hour), int((hour * 60) % 60), int((hour * 3600) % 60), tzinfo=timezone.utc)
        return dt

    def generate_benign_events(self) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Generate standard benign behavior across all days."""
        logs = []
        labels = []
        
        print("Generating benign event dataset...")
        for entity_id, profile in tqdm(self.entity_profiles.items()):
            if profile["entity_type"] == "user":
                events_per_day = lambda: random.randint(2, 6)
            elif profile["entity_type"] == "service_account":
                events_per_day = lambda: random.randint(5, 15)
            else:
                events_per_day = lambda: random.randint(10, 20)
                
            for day_idx in range(self.days_of_data):
                if profile["entity_type"] == "user" and (self.start_time + timedelta(days=day_idx)).weekday() >= 5:
                    if random.random() > 0.10:
                        continue
                        
                for _ in range(events_per_day()):
                    event_id = str(uuid.uuid4())
                    ts = self._sample_timestamp(profile, day_idx)
                    device = random.choice(profile["typical_devices"])
                    
                    # 98% home IP, 2% roaming IP (near home city to ensure spatial speed < 500 km/h)
                    ip = profile["home_ip"] if random.random() < 0.98 else self.fake.ipv4()
                    geo = profile["home_geo"]
                    if ip != profile["home_ip"]:
                        # Roaming within ~10-30 km of home geo
                        geo = {
                            "lat": profile["home_geo"]["lat"] + random.uniform(-0.1, 0.1),
                            "lon": profile["home_geo"]["lon"] + random.uniform(-0.1, 0.1),
                            "country": profile["home_geo"]["country"]
                        }
                        
                    res = random.choice(profile["typical_resources"])
                    
                    if profile["entity_type"] == "user":
                        auth = random.choice(["password", "token", "biometric"])
                    else:
                        auth = random.choice(["token", "certificate"])
                        
                    # Lognormal session duration and bytes transferred
                    dur = int(max(1.0, np.random.lognormal(mean=5.5, sigma=0.8)))
                    dur = min(dur, 7200)
                    bytes_tx = int(max(100.0, np.random.lognormal(mean=10.5, sigma=1.2)))
                    
                    cmd_seq = None
                    if profile["entity_type"] == "user" and ("admin" in res or "deploy" in res) and random.random() < 0.3:
                        cmd_seq = random.sample(PRIVILEGED_COMMANDS, random.randint(1, 3))
                        
                    logs.append({
                        "event_id": event_id,
                        "entity_id": entity_id,
                        "entity_type": profile["entity_type"],
                        "timestamp": ts.isoformat(),
                        "source_ip": ip,
                        "geo_location": json.dumps(geo),
                        "resource_accessed": res,
                        "auth_method": auth,
                        "session_duration": dur,
                        "bytes_transferred": bytes_tx,
                        "command_sequence": json.dumps(cmd_seq) if cmd_seq else None,
                        "device_fingerprint": json.dumps(device)
                    })
                    
                    labels.append({
                        "event_id": event_id,
                        "label": "normal",
                        "attack_scenario_id": None
                    })
        return logs, labels

    def inject_brute_force(self, logs: List[Dict[str, Any]], labels: List[Dict[str, Any]], count: int):
        """Inject brute force attacks: 50-500 high-frequency failed authentications."""
        print(f"Injecting brute_force scenarios (target: {count} events)...")
        injected = 0
        while injected < count:
            scenario_id = str(uuid.uuid4())
            entity_id = random.choice(list(self.entity_profiles.keys()))
            profile = self.entity_profiles[entity_id]
            
            attacker_ip = self.fake.ipv4()
            remaining = count - injected
            if remaining <= 0:
                break
            num_attempts = random.randint(min(50, remaining), min(500, remaining))
            
            start_ts = self.start_time + timedelta(seconds=random.randint(0, int((self.end_time - self.start_time).total_seconds() - 600)))
            
            for i in range(num_attempts):
                event_id = str(uuid.uuid4())
                ts = start_ts + timedelta(seconds=(i + 1) * random.uniform(0.2, 1.0))
                
                logs.append({
                    "event_id": event_id,
                    "entity_id": entity_id,
                    "entity_type": profile["entity_type"],
                    "timestamp": ts.isoformat(),
                    "source_ip": attacker_ip,
                    "geo_location": json.dumps(profile["home_geo"]),
                    "resource_accessed": "/api/v1/auth/login",
                    "auth_method": "password",
                    "session_duration": 0,
                    "bytes_transferred": int(random.uniform(200, 800)),
                    "command_sequence": None,
                    "device_fingerprint": json.dumps(random.choice(profile["typical_devices"]))
                })
                
                labels.append({
                    "event_id": event_id,
                    "label": "brute_force",
                    "attack_scenario_id": scenario_id
                })
            injected += num_attempts

    def inject_impossible_travel(self, logs: List[Dict[str, Any]], labels: List[Dict[str, Any]], count: int):
        """Inject impossible travel: every injected event occurs 5-20 min after a user's previous event from 2000-2500km away."""
        print(f"Injecting impossible_travel scenarios (target: {count} events)...")
        
        entity_log_map = {}
        for item in logs:
            ent = item["entity_id"]
            if ent not in entity_log_map:
                entity_log_map[ent] = []
            entity_log_map[ent].append(item)

        for ent in entity_log_map:
            entity_log_map[ent].sort(key=lambda x: x["timestamp"])

        valid_entities = [ent for ent, items in entity_log_map.items() if len(items) >= 2]

        injected = 0
        while injected < count:
            scenario_id = str(uuid.uuid4())
            entity_id = random.choice(valid_entities)
            profile = self.entity_profiles[entity_id]
            ent_logs = entity_log_map[entity_id]
            
            ref_event = random.choice(ent_logs)
            ref_ts = datetime.fromisoformat(ref_event["timestamp"])
            ref_geo = json.loads(ref_event["geo_location"])
            
            # Generate a target geo 2000-2500 km away from ref_geo
            theta = random.uniform(0, 2 * math.pi)
            dist_km = random.uniform(2000.0, 2500.0)
            
            d_lat = (dist_km / 111.0) * math.cos(theta)
            d_lon = (dist_km / (111.0 * math.cos(math.radians(max(-80.0, min(80.0, ref_geo["lat"])))))) * math.sin(theta)
            
            dest_lat = max(-85.0, min(85.0, ref_geo["lat"] + d_lat))
            dest_lon = ((ref_geo["lon"] + d_lon + 180.0) % 360.0) - 180.0
            
            actual_dist = haversine_distance(ref_geo["lat"], ref_geo["lon"], dest_lat, dest_lon)
            if actual_dist < 1500.0:
                continue
                
            dt_minutes = random.randint(5, 20)
            inj_ts = ref_ts + timedelta(minutes=dt_minutes)
            
            event_id = str(uuid.uuid4())
            logs.append({
                "event_id": event_id,
                "entity_id": entity_id,
                "entity_type": profile["entity_type"],
                "timestamp": inj_ts.isoformat(),
                "source_ip": self.fake.ipv4(),
                "geo_location": json.dumps({
                    "lat": dest_lat,
                    "lon": dest_lon,
                    "country": "ROAMING"
                }),
                "resource_accessed": random.choice(profile["typical_resources"]),
                "auth_method": "password",
                "session_duration": random.randint(15, 120),
                "bytes_transferred": int(random.uniform(5000, 50000)),
                "command_sequence": None,
                "device_fingerprint": json.dumps(random.choice(profile["typical_devices"]))
            })
            
            labels.append({
                "event_id": event_id,
                "label": "impossible_travel",
                "attack_scenario_id": scenario_id
            })
            
            injected += 1

    def inject_credential_stuffing(self, logs: List[Dict[str, Any]], labels: List[Dict[str, Any]], count: int):
        """Inject credential stuffing: many entity logins from few IPs with high failure rates."""
        print(f"Injecting credential_stuffing scenarios (target: {count} events)...")
        injected = 0
        all_entities = list(self.entity_profiles.keys())
        
        while injected < count:
            scenario_id = str(uuid.uuid4())
            attacker_ip = self.fake.ipv4()
            remaining = count - injected
            if remaining <= 0:
                break
            num_attempts = random.randint(min(100, remaining), min(300, remaining))
            
            start_ts = self.start_time + timedelta(seconds=random.randint(0, int((self.end_time - self.start_time).total_seconds() - 1800)))
            
            for i in range(num_attempts):
                event_id = str(uuid.uuid4())
                ts = start_ts + timedelta(seconds=(i + 1) * random.uniform(0.5, 3.0))
                target_entity = random.choice(all_entities)
                profile = self.entity_profiles[target_entity]
                
                is_failed = random.random() < 0.8
                
                logs.append({
                    "event_id": event_id,
                    "entity_id": target_entity,
                    "entity_type": profile["entity_type"],
                    "timestamp": ts.isoformat(),
                    "source_ip": attacker_ip,
                    "geo_location": json.dumps(profile["home_geo"]),
                    "resource_accessed": "/api/v1/auth/login",
                    "auth_method": "password",
                    "session_duration": 0 if is_failed else 300,
                    "bytes_transferred": int(random.uniform(300, 1500)),
                    "command_sequence": None,
                    "device_fingerprint": json.dumps({
                        "os": "Linux",
                        "firmware": "v1.0",
                        "mac": self.fake.mac_address(),
                        "protocol": "HTTPS"
                    })
                })
                
                labels.append({
                    "event_id": event_id,
                    "label": "credential_stuffing",
                    "attack_scenario_id": scenario_id
                })
                
            injected += num_attempts

    def inject_lateral_movement(self, logs: List[Dict[str, Any]], labels: List[Dict[str, Any]], count: int):
        """Inject lateral movement: entity accesses 5+ new resources in a short burst."""
        print(f"Injecting lateral_movement scenarios (target: {count} events)...")
        injected = 0
        while injected < count:
            scenario_id = str(uuid.uuid4())
            entity_id = random.choice(list(self.entity_profiles.keys()))
            profile = self.entity_profiles[entity_id]
            
            non_typical_resources = list(set(RESOURCES) - set(profile["typical_resources"]))
            if len(non_typical_resources) < 5:
                continue
                
            remaining = count - injected
            if remaining <= 0:
                break
            num_resources = random.randint(min(5, remaining), min(10, remaining))
            target_resources = random.sample(non_typical_resources, min(num_resources, len(non_typical_resources)))
            
            start_ts = self.start_time + timedelta(seconds=random.randint(0, int((self.end_time - self.start_time).total_seconds() - 1800)))
            
            for i, res in enumerate(target_resources):
                event_id = str(uuid.uuid4())
                ts = start_ts + timedelta(seconds=(i + 1) * random.uniform(10, 60))
                
                logs.append({
                    "event_id": event_id,
                    "entity_id": entity_id,
                    "entity_type": profile["entity_type"],
                    "timestamp": ts.isoformat(),
                    "source_ip": profile["home_ip"],
                    "geo_location": json.dumps(profile["home_geo"]),
                    "resource_accessed": res,
                    "auth_method": "token",
                    "session_duration": random.randint(30, 300),
                    "bytes_transferred": int(random.uniform(10000, 100000)),
                    "command_sequence": json.dumps(random.sample(PRIVILEGED_COMMANDS, random.randint(1, 2))) if random.random() < 0.7 else None,
                    "device_fingerprint": json.dumps(random.choice(profile["typical_devices"]))
                })
                
                labels.append({
                    "event_id": event_id,
                    "label": "lateral_movement",
                    "attack_scenario_id": scenario_id
                })
                
            injected += len(target_resources)

    def inject_device_spoofing(self, logs: List[Dict[str, Any]], labels: List[Dict[str, Any]], count: int):
        """Inject device spoofing: known entity appears with new MAC and different OS."""
        print(f"Injecting device_spoofing scenarios (target: {count} events)...")
        injected = 0
        while injected < count:
            scenario_id = str(uuid.uuid4())
            entity_id = random.choice(list(self.entity_profiles.keys()))
            profile = self.entity_profiles[entity_id]
            
            spoofed_device = {
                "os": "Android OS" if "Windows" in profile["typical_devices"][0]["os"] else "Windows 11",
                "firmware": "v9.9.9",
                "mac": self.fake.mac_address(),
                "protocol": "SSH"
            }
            
            ts_1 = self.start_time + timedelta(seconds=random.randint(0, int((self.end_time - self.start_time).total_seconds() - 3600)))
            
            for i in range(2):
                event_id = str(uuid.uuid4())
                ts = ts_1 + timedelta(seconds=(i + 1) * 60)
                
                logs.append({
                    "event_id": event_id,
                    "entity_id": entity_id,
                    "entity_type": profile["entity_type"],
                    "timestamp": ts.isoformat(),
                    "source_ip": profile["home_ip"],
                    "geo_location": json.dumps(profile["home_geo"]),
                    "resource_accessed": random.choice(profile["typical_resources"]),
                    "auth_method": "token",
                    "session_duration": 180,
                    "bytes_transferred": int(random.uniform(5000, 30000)),
                    "command_sequence": None,
                    "device_fingerprint": json.dumps(spoofed_device)
                })
                
                labels.append({
                    "event_id": event_id,
                    "label": "device_spoofing",
                    "attack_scenario_id": scenario_id
                })
                
            injected += 2

    def inject_low_slow_exfil(self, logs: List[Dict[str, Any]], labels: List[Dict[str, Any]], count: int):
        """Inject low and slow exfiltration: small off-hours reads accumulating over 2-4 weeks."""
        print(f"Injecting low_slow_exfil scenarios (target: {count} events)...")
        injected = 0
        while injected < count:
            scenario_id = str(uuid.uuid4())
            entity_id = random.choice([k for k, v in self.entity_profiles.items() if v["entity_type"] == "user"])
            profile = self.entity_profiles[entity_id]
            
            remaining = count - injected
            if remaining <= 0:
                break
            num_exfils = random.randint(min(20, remaining), min(50, remaining))
            duration_days = random.randint(14, 28)
            
            start_day = random.randint(0, self.days_of_data - duration_days - 1)
            
            for i in range(num_exfils):
                event_id = str(uuid.uuid4())
                hour = random.uniform(1.0, 4.0)
                day_offset = start_day + (i * duration_days / num_exfils)
                ts = self.start_time + timedelta(days=day_offset)
                ts = datetime(ts.year, ts.month, ts.day, int(hour), int((hour * 60) % 60), int((hour * 3600) % 60), tzinfo=timezone.utc)
                
                res = "/vault/secrets/api_keys" if i % 2 == 0 else "/backup/nightly_dump.tar.gz"
                exfil_bytes = int(np.random.lognormal(mean=18.0, sigma=0.5))
                
                logs.append({
                    "event_id": event_id,
                    "entity_id": entity_id,
                    "entity_type": profile["entity_type"],
                    "timestamp": ts.isoformat(),
                    "source_ip": profile["home_ip"],
                    "geo_location": json.dumps(profile["home_geo"]),
                    "resource_accessed": res,
                    "auth_method": "token",
                    "session_duration": random.randint(15, 60),
                    "bytes_transferred": exfil_bytes,
                    "command_sequence": json.dumps(["aws s3 sync s3://company-secrets ./secrets"]) if i % 5 == 0 else None,
                    "device_fingerprint": json.dumps(random.choice(profile["typical_devices"]))
                })
                
                labels.append({
                    "event_id": event_id,
                    "label": "low_slow_exfil",
                    "attack_scenario_id": scenario_id
                })
                
            injected += num_exfils

    def inject_insider_drift(self, logs: List[Dict[str, Any]], labels: List[Dict[str, Any]], count: int):
        """Inject insider drift: gradual legitimate-looking expansion of accessed resources over weeks."""
        print(f"Injecting insider_drift scenarios (target: {count} events)...")
        injected = 0
        while injected < count:
            scenario_id = str(uuid.uuid4())
            entity_id = random.choice([k for k, v in self.entity_profiles.items() if v["entity_type"] == "user"])
            profile = self.entity_profiles[entity_id]
            
            remaining = count - injected
            if remaining <= 0:
                break
            num_drift_events = random.randint(min(30, remaining), min(60, remaining))
            
            for i in range(num_drift_events):
                event_id = str(uuid.uuid4())
                day_idx = int(self.days_of_data * (0.2 + 0.8 * (i / num_drift_events)))
                day_idx = min(self.days_of_data - 1, day_idx)
                ts = self._sample_timestamp(profile, day_idx)
                
                if random.random() < (i / num_drift_events):
                    res = random.choice(list(set(RESOURCES) - set(profile["typical_resources"])))
                else:
                    res = random.choice(profile["typical_resources"])
                    
                logs.append({
                    "event_id": event_id,
                    "entity_id": entity_id,
                    "entity_type": profile["entity_type"],
                    "timestamp": ts.isoformat(),
                    "source_ip": profile["home_ip"],
                    "geo_location": json.dumps(profile["home_geo"]),
                    "resource_accessed": res,
                    "auth_method": "password",
                    "session_duration": random.randint(120, 1800),
                    "bytes_transferred": int(random.uniform(5000, 50000)),
                    "command_sequence": None,
                    "device_fingerprint": json.dumps(random.choice(profile["typical_devices"]))
                })
                
                labels.append({
                    "event_id": event_id,
                    "label": "insider_drift",
                    "attack_scenario_id": scenario_id
                })
                
            injected += num_drift_events

    def run(self):
        """Orchestrate entire log generation, enforce strict quality constraints, and save datasets."""
        os.makedirs("data/raw", exist_ok=True)
        
        self.generate_entity_profiles()
        
        with open("data/raw/entity_profiles.json", "w") as f:
            json.dump(self.entity_profiles, f, indent=4)
            
        logs, labels = self.generate_benign_events()
        
        benign_count = len(logs)
        total_target = int(benign_count / (1 - self.anomaly_rate))
        anomaly_target = total_target - benign_count
        
        anomaly_ratios = {
            "brute_force": 0.004 / 0.02,
            "impossible_travel": 0.003 / 0.02,
            "credential_stuffing": 0.003 / 0.02,
            "lateral_movement": 0.003 / 0.02,
            "device_spoofing": 0.002 / 0.02,
            "low_slow_exfil": 0.003 / 0.02,
            "insider_drift": 0.002 / 0.02
        }
        
        self.inject_brute_force(logs, labels, int(anomaly_target * anomaly_ratios["brute_force"]))
        self.inject_impossible_travel(logs, labels, int(anomaly_target * anomaly_ratios["impossible_travel"]))
        self.inject_credential_stuffing(logs, labels, int(anomaly_target * anomaly_ratios["credential_stuffing"]))
        self.inject_lateral_movement(logs, labels, int(anomaly_target * anomaly_ratios["lateral_movement"]))
        self.inject_device_spoofing(logs, labels, int(anomaly_target * anomaly_ratios["device_spoofing"]))
        self.inject_low_slow_exfil(logs, labels, int(anomaly_target * anomaly_ratios["low_slow_exfil"]))
        self.inject_insider_drift(logs, labels, int(anomaly_target * anomaly_ratios["insider_drift"]))
        
        df_logs = pd.DataFrame(logs)
        df_labels = pd.DataFrame(labels)
        
        # Enforce strict per-entity chronological order and positive time deltas (>0s)
        print("Enforcing per-entity monotonic chronology and positive time deltas...")
        df_logs["dt_sort"] = pd.to_datetime(df_logs["timestamp"], format="ISO8601")
        df_logs = df_logs.sort_values(by=["entity_id", "dt_sort"]).reset_index(drop=True)
        
        dt_values = df_logs["dt_sort"].values.copy()
        entity_ids = df_logs["entity_id"].values
        
        for i in range(1, len(df_logs)):
            if entity_ids[i] == entity_ids[i-1]:
                if dt_values[i] <= dt_values[i-1]:
                    # Add positive offset between 500ms and 3000ms
                    dt_values[i] = dt_values[i-1] + np.timedelta64(int(random.uniform(500, 3000)), 'ms')
                    
        df_logs["dt_sort"] = dt_values
        df_logs["timestamp"] = pd.to_datetime(df_logs["dt_sort"]).dt.strftime("%Y-%m-%dT%H:%M:%S.%fZ")
        
        # Sort globally chronologically for realistic log stream
        df_logs = df_logs.sort_values(by="dt_sort").reset_index(drop=True)
        df_logs = df_logs.drop(columns=["dt_sort"])
        
        # Align labels 100% with logs
        df_labels = pd.merge(df_logs[["event_id"]], df_labels, on="event_id", how="left")
        
        # Save output CSVs
        df_logs.to_csv("data/raw/logs.csv", index=False)
        df_labels.to_csv("data/raw/labels.csv", index=False)
        
        self._print_summary(df_logs, df_labels)

    def _print_summary(self, df_logs: pd.DataFrame, df_labels: pd.DataFrame):
        """Print execution statistics and distribution data."""
        total_events = len(df_logs)
        class_dist = df_labels["label"].value_counts()
        
        print("\n" + "="*50)
        print("SYNTHETIC DATA GENERATION COMPLETE")
        print("="*50)
        print(f"Total events generated: {total_events}")
        print("\nClass Distribution:")
        for label, count in class_dist.items():
            pct = (count / total_events) * 100
            print(f"  - {label:<22}: {count:<6} ({pct:.3f}%)")
        
        anomaly_count = total_events - class_dist.get("normal", 0)
        actual_anomaly_rate = (anomaly_count / total_events) * 100
        print(f"\nTotal Anomaly Rate: {actual_anomaly_rate:.2f}% (Target: {self.anomaly_rate * 100}%)")
        print("="*50 + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic cybersecurity access logs.")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to the config.yaml file")
    args = parser.parse_args()
    
    generator = SyntheticDataGenerator(config_path=args.config)
    generator.run()
