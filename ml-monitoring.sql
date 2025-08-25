-- =========================================================
-- ml-monitoring.sql
-- ---------------------------------------------------------
-- Schema and sample data for a lightweight, self-contained
-- monitoring / alerting system for machine-learning models.
-- Compatible with PostgreSQL ≥ 12.
-- =========================================================

-- ------------------------------------------------------------------
-- 1. ENUM TYPES
-- ------------------------------------------------------------------
CREATE TYPE model_status AS ENUM ('training', 'online', 'offline', 'deprecated');
CREATE TYPE alert_severity AS ENUM ('low', 'medium', 'high', 'critical');
CREATE TYPE alert_status AS ENUM ('open', 'acknowledged', 'resolved');

-- ------------------------------------------------------------------
-- 2. CORE TABLES
-- ------------------------------------------------------------------

-- 2.1  Registered models
CREATE TABLE IF NOT EXISTS ml_model (
    id            SERIAL PRIMARY KEY,
    name          TEXT NOT NULL UNIQUE,
    version       TEXT NOT NULL,
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    deployed_at   TIMESTAMPTZ,
    status        model_status NOT NULL DEFAULT 'training',
    metadata      JSONB
);

-- 2.2  Model metrics collected at inference time
CREATE TABLE IF NOT EXISTS model_metric (
    id            BIGSERIAL PRIMARY KEY,
    model_id      INT NOT NULL REFERENCES ml_model(id) ON DELETE CASCADE,
    metric_name   TEXT NOT NULL,           -- e.g. 'accuracy', 'latency_ms'
    metric_value  DOUBLE PRECISION NOT NULL,
    recorded_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
    metadata      JSONB
);
CREATE INDEX idx_model_metric_model_time
    ON model_metric (model_id, recorded_at DESC);

-- 2.3  Alert definitions (threshold rules)
CREATE TABLE IF NOT EXISTS alert_rule (
    id            SERIAL PRIMARY KEY,
    model_id      INT NOT NULL REFERENCES ml_model(id) ON DELETE CASCADE,
    metric_name   TEXT NOT NULL,
    condition     TEXT NOT NULL CHECK (condition IN ('<', '<=', '>', '>=', '=')),
    threshold     DOUBLE PRECISION NOT NULL,
    severity      alert_severity NOT NULL,
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    enabled       BOOLEAN NOT NULL DEFAULT TRUE
);

-- 2.4  Alerts / incidents
CREATE TABLE IF NOT EXISTS alert (
    id            BIGSERIAL PRIMARY KEY,
    rule_id       INT NOT NULL REFERENCES alert_rule(id) ON DELETE CASCADE,
    triggered_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    resolved_at   TIMESTAMPTZ,
    status        alert_status NOT NULL DEFAULT 'open',
    details       JSONB
);
CREATE INDEX idx_alert_rule_status ON alert (rule_id, status);

-- ------------------------------------------------------------------
-- 3. VIEWS
-- ------------------------------------------------------------------

-- 3.1  Latest metric snapshot per (model, metric_name)
CREATE OR REPLACE VIEW v_latest_metric AS
SELECT DISTINCT ON (model_id, metric_name)
       model_id,
       metric_name,
       metric_value,
       recorded_at
FROM model_metric
ORDER BY model_id, metric_name, recorded_at DESC;

-- 3.2  Open alerts with rule context
CREATE OR REPLACE VIEW v_open_alerts AS
SELECT a.id          AS alert_id,
       m.name        AS model_name,
       m.version     AS model_version,
       r.metric_name,
       r.condition,
       r.threshold,
       r.severity,
       a.triggered_at,
       a.details
FROM alert a
JOIN alert_rule r ON r.id = a.rule_id
JOIN ml_model  m  ON m.id = r.model_id
WHERE a.status = 'open'
ORDER BY a.triggered_at DESC;

-- ------------------------------------------------------------------
-- 4. TRIGGERS
-- ------------------------------------------------------------------

-- 4.1  Keep alert_rule.updated_at in sync
CREATE OR REPLACE FUNCTION trg_set_timestamp()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at := NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER set_timestamp
BEFORE UPDATE ON alert_rule
FOR EACH ROW EXECUTE FUNCTION trg_set_timestamp();

-- 4.2  Auto-fire alerts when metrics breach thresholds
CREATE OR REPLACE FUNCTION trg_check_metric_alert()
RETURNS TRIGGER AS $$
DECLARE
    rule_rec RECORD;
    breach   BOOLEAN;
BEGIN
    FOR rule_rec IN
        SELECT * FROM alert_rule
        WHERE model_id = NEW.model_id
          AND metric_name = NEW.metric_name
          AND enabled = TRUE
    LOOP
        breach := CASE rule_rec.condition
                    WHEN '<'  THEN NEW.metric_value <  rule_rec.threshold
                    WHEN '<=' THEN NEW.metric_value <= rule_rec.threshold
                    WHEN '>'  THEN NEW.metric_value >  rule_rec.threshold
                    WHEN '>=' THEN NEW.metric_value >= rule_rec.threshold
                    WHEN '='  THEN NEW.metric_value =  rule_rec.threshold
                 END;
        IF breach THEN
            INSERT INTO alert (rule_id, details)
            VALUES (rule_rec.id,
                    jsonb_build_object(
                        'metric_value', NEW.metric_value,
                        'threshold',    rule_rec.threshold,
                        'condition',    rule_rec.condition
                    ));
        END IF;
    END LOOP;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER check_metric_alert
AFTER INSERT ON model_metric
FOR EACH ROW EXECUTE FUNCTION trg_check_metric_alert();

-- ------------------------------------------------------------------
-- 5. SAMPLE DATA
-- ------------------------------------------------------------------

-- 5.1  Two dummy models
INSERT INTO ml_model (name, version, deployed_at, status)
VALUES
    ('churn_model', 'v1.2.0', now() - INTERVAL '3 days', 'online'),
    ('fraud_model', 'v2.0.1', now() - INTERVAL '1 hour',  'online');

-- 5.2  Define alert rules
INSERT INTO alert_rule (model_id, metric_name, condition, threshold, severity)
VALUES
    (1, 'accuracy',      '<', 0.85, 'high'),
    (1, 'latency_ms',    '>', 200,  'medium'),
    (2, 'fraud_recall',  '<', 0.9,  'critical');

-- 5.3  Inject recent metrics
INSERT INTO model_metric (model_id, metric_name, metric_value)
VALUES
    (1, 'accuracy',      0.82),
    (1, 'latency_ms',    250),
    (2, 'fraud_recall',  0.87);

-- ------------------------------------------------------------------
-- 6. QUICK TEST
-- ------------------------------------------------------------------
-- SELECT * FROM v_latest_metric;
-- SELECT * FROM v_open_alerts;