-- ==============================================================================
-- Optimized Postgres Schema for Phase 1 - Phase 6 Pipeline
-- ==============================================================================

-- 1. Entities Table (Supports Users, Service Accounts, and Edge Devices)
CREATE TABLE IF NOT EXISTS entities (
    entity_id VARCHAR(100) PRIMARY KEY,
    entity_type VARCHAR(50) NOT NULL,     -- 'user', 'service_account', 'edge_device'
    department VARCHAR(100),
    home_ip INET,
    home_geo JSONB,                        -- {"lat": 40.71, "lon": -74.00, "country": "US"}
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- 2. Access Logs Table (Matches Phase 1 Output Schema Exactly)
CREATE TABLE IF NOT EXISTS access_logs (
    event_id UUID PRIMARY KEY,
    timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
    entity_id VARCHAR(100) REFERENCES entities(entity_id) ON DELETE CASCADE,
    source_ip INET NOT NULL,
    geo_location JSONB NOT NULL,          -- Coordinates + Country
    resource_accessed TEXT NOT NULL,
    auth_method VARCHAR(50) NOT NULL,      -- 'password', 'token', 'certificate', 'biometric'
    session_duration INT NOT NULL,         -- in seconds
    command_sequence JSONB,                -- Order of actions taken
    device_fingerprint JSONB NOT NULL,    -- OS, MAC, Firmware, Protocol
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- 3. Temporal Activity Sequences (Array-Optimized for LSTM/Transformer)
CREATE TABLE IF NOT EXISTS sequences (
    id BIGSERIAL PRIMARY KEY,
    entity_id VARCHAR(100) REFERENCES entities(entity_id) ON DELETE CASCADE,
    session_identifier VARCHAR(255) NOT NULL,
    log_ids UUID[] NOT NULL,              -- Ordered list of event_ids in sequence window
    start_time TIMESTAMP WITH TIME ZONE NOT NULL,
    end_time TIMESTAMP WITH TIME ZONE NOT NULL,
    event_count INT NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- 4. Anomaly Scores (Outputs of Phase 2 Baseline & Phase 3 Sequence Models)
CREATE TABLE IF NOT EXISTS anomaly_scores (
    id BIGSERIAL PRIMARY KEY,
    event_id UUID REFERENCES access_logs(event_id) ON DELETE CASCADE,
    model_name VARCHAR(100) NOT NULL,        -- 'Isolation Forest', 'BiLSTM Autoencoder'
    score DOUBLE PRECISION NOT NULL,         -- Anomaly risk score [0.0 to 1.0]
    threshold DOUBLE PRECISION NOT NULL,     -- Flagging threshold
    is_anomaly BOOLEAN NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- 5. SOC Analyst Alerts (Phase 6 Dashboard Primary Queue)
CREATE TABLE IF NOT EXISTS alerts (
    id BIGSERIAL PRIMARY KEY,
    event_id UUID REFERENCES access_logs(event_id) ON DELETE CASCADE,
    entity_id VARCHAR(100) REFERENCES entities(entity_id) ON DELETE CASCADE,
    risk_score DOUBLE PRECISION NOT NULL,
    threat_level VARCHAR(50) NOT NULL,      -- 'LOW', 'MEDIUM', 'HIGH', 'CRITICAL'
    status VARCHAR(50) DEFAULT 'UNTRIAGED',  -- 'UNTRIAGED', 'CONFIRMED_ATTACK', 'FALSE_POSITIVE'
    assigned_to VARCHAR(100),
    comments TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- 6. Threat Classification Metadata (Phase 4 Model Outputs)
CREATE TABLE IF NOT EXISTS threat_classifications (
    id BIGSERIAL PRIMARY KEY,
    alert_id BIGINT REFERENCES alerts(id) ON DELETE CASCADE,
    attack_type VARCHAR(100) NOT NULL,       -- e.g., 'brute_force', 'impossible_travel'
    mitre_tactic VARCHAR(100) NOT NULL,      -- e.g., 'Credential Access'
    mitre_technique VARCHAR(100) NOT NULL,   -- e.g., 'T1110: Brute Force'
    confidence DOUBLE PRECISION NOT NULL,    -- Probability (0.0 to 1.0)
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- 7. Explainability Attributions (Phase 5 SHAP Output)
CREATE TABLE IF NOT EXISTS explanations (
    id BIGSERIAL PRIMARY KEY,
    alert_id BIGINT REFERENCES alerts(id) ON DELETE CASCADE,
    feature_name VARCHAR(150) NOT NULL,      -- e.g., 'geo_velocity_kmh'
    attribution_value DOUBLE PRECISION NOT NULL,
    narrative TEXT NOT NULL,                 -- Human-readable explanation sentence
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Indexes for Fast Dashboard Loading
CREATE INDEX IF NOT EXISTS idx_access_logs_timestamp ON access_logs(timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_access_logs_entity_id ON access_logs(entity_id);
CREATE INDEX IF NOT EXISTS idx_alerts_status_score ON alerts(status, risk_score DESC);
CREATE INDEX IF NOT EXISTS idx_explanations_alert ON explanations(alert_id);