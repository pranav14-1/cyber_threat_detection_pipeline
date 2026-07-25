# Phase 1: Synthetic Data Generation & Threat Injection

This module is responsible for simulating realistic enterprise environment event logs and injecting synthetic cybersecurity threat campaigns (anomalies). 

## 🎯 Objectives
- **Simulate Normal Behavior:** Generate baseline event logs across various data sources representing routine business activities (e.g., users logging in during business hours, standard web browsing, routine database queries).
- **Inject Threat Scenarios:** Programmatically inject adversarial patterns mapped to the MITRE ATT&CK framework (e.g., Brute-force attacks, Lateral movement, Credential dumping, and Data exfiltration).
- **Multiple Data Sources:** Support formats representing system logs (Syslog), authentication logs (Windows Event ID 4624/4625), network traffic flows (NetFlow/PCAP metadata), and process creation logs (Sysmon Event ID 1).

## 🚀 Getting Started
Run the generator script to create raw logs:
```bash
python generate_logs.py --config config.yaml --output ../../data/raw/
```
