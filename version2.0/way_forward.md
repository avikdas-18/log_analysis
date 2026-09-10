Build this as a hybrid system: physics/rule-based analytics for trustworthy energy-flow answers, ML for anomaly detection and health trends, and an LLM only to translate questions into approved analyses and explain the evidence.
You cannot reliably answer vehicle questions from raw CAN frames alone. Each vehicle needs its DBC/ARXML (or an equivalent signal-definition sheet) that defines CAN IDs, bit layouts, scaling, units, and sign conventions.
1. Target architecture
CAN logs (.asc/.blf/.mf4/.csv)
        ↓
Parse + decode with DBC / ARXML
        ↓
Time-aligned signal table + data-quality checks
        ↓
Derived electrical / mechanical features
        ↓
Rules + anomaly models + health-trend models
        ↓
Query API / LLM assistant / detailed HTML-PDF report
Store both the immutable raw log and decoded data with metadata: vehicle variant, DBC version, logging session, timezone, software version, and signal-quality results.
Python libraries are a good starting point:
- python-can for reading ASC, BLF, CSV, LOG, TRC, and some MF4 files. Documentation
- cantools for decoding CAN frames with DBC files.
- asammdf for MDF/MF4 measurement files and large time-series processing. Documentation
- pandas or polars, plus Parquet and DuckDB, for scalable data processing.
- plotly for interactive graphs; matplotlib for report figures.
- FastAPI for the backend and React/Streamlit for the interface.
- scikit-learn for first anomaly models, including Isolation Forest. Documentation
- An LLM API for natural-language querying and report prose, with strict tool/function calling.
2. Signals you should obtain
Create a canonical signal map so each vehicle-specific name maps to a common name.
Area	Minimum useful signals
Battery	Pack voltage/current, SOC, pack temperature, cell voltages, cell temperatures, contactor state, charge/discharge limits
Generator / engine	Generator speed, torque/current/voltage, generator-enable state, engine speed, fuel rate if available
Motor / inverter	Motor speed, torque, DC bus voltage/current, inverter temperature, motor temperature, drive state
Vehicle	Vehicle speed, accelerator/brake, gear, ambient temperature, odometer
Diagnostics	DTCs, fault states, CAN error frames, missing-message counters


Define sign conventions once and validate them with known drive events. For example:
- battery_current > 0: battery discharging
- battery_power = pack_voltage × battery_current
- positive motor power: propulsion
- negative motor power: regenerative braking
- positive generator power: generator delivering energy to the DC bus
Do not assume this convention—verify it against the OEM definitions.
3. Implement energy-flow analysis first
This delivers the most value and creates the foundation for ML.
At a fixed analysis rate, such as 10 Hz:
battery_power_kw = battery_voltage_v * battery_current_a / 1000
generator_power_kw = generator_voltage_v * generator_current_a / 1000
motor_mechanical_kw = motor_torque_nm * motor_speed_rpm * 2 * pi / 60 / 1000
Use actual DC-bus power when available; mechanical motor power is only an estimate because inverter and motor losses matter.
Example event logic:
GENERATOR_CHARGING_BATTERY =
    generator_enabled
    and generator_power_kw > 2
    and battery_power_kw < -1
    and not regen_active

GENERATOR_AND_BATTERY_DRIVING_MOTOR =
    generator_power_kw > 2
    and battery_power_kw > 1
    and motor_power_kw > 3
    and vehicle_speed_kph > 2
Use duration hysteresis, such as “true for at least 2 seconds,” to prevent noisy state changes.
For power-loss analysis, calculate an energy balance only when all participating signals are valid and measured on compatible voltage domains:
estimated_loss =
    generator_output
  + battery_discharge
  - motor_input
  - auxiliary_loads
  - battery_charge_power
Treat this as an estimate with a confidence interval. Sensor error, sampling offsets, unmeasured auxiliary loads, regenerative operation, and unknown inverter efficiency can otherwise look like “losses.”
4. Add a data-quality layer
This is non-negotiable. Bad decoding produces convincing but incorrect AI conclusions.
Detect and report:
- Missing messages and CAN bus errors
- Signal values outside physical ranges
- Counter/checksum failures where available
- Stale values and frozen sensors
- Timestamp resets or gaps
- Unit mismatch, scaling errors, and invalid states
- Signals sampled at mismatched rates
Resample signals to a common timeline, but preserve the original samples. Use suitable aggregation:
- mean for temperatures and voltages
- last-known value for states
- min/max for safety-relevant limits
- integration for energy
5. Battery-health analytics
Begin with transparent engineering features before deep learning.
Useful features per trip, hour, or equivalent full cycle:
- Minimum/maximum pack voltage and current
- Voltage dip under a comparable load
- Cell-voltage spread: max(cell_voltage) - min(cell_voltage)
- Cell temperature spread
- SOC drift versus estimated coulomb count
- Charging and discharging energy
- Charge/discharge time at high temperature
- Internal-resistance proxy during current pulses:
  R ≈ ΔV / ΔI, normalized by SOC and temperature
- Number/duration of contactor or BMS faults
- Trend slope and variance for all of the above
Important limitation: cell-voltage spread and voltage dip can flag risk, but they do not by themselves provide a trustworthy capacity/SOH value. Accurate SOH needs controlled charge/discharge information, good current measurement, temperature, and preferably repeated comparable operating conditions.
6. Failure and anomaly detection strategy
Use three layers.
1. Known-fault rules
   Detect conditions such as over-temperature, high cell imbalance, unexpected SOC behavior, generator-output shortfall, persistent energy-balance residual, and inverter derating.
2. Unsupervised anomaly detection
   Train on confirmed healthy trips. Start with robust z-scores and Isolation Forest over windowed features. Isolation Forest identifies observations that are unusually easy to isolate relative to normal data. Documentation
3. Supervised failure prediction
   When you have historical logs linked to service outcomes, train a classifier on windows preceding known failures. Gradient-boosted trees are usually the best first model because they work well with structured telemetry and can explain feature importance.
Only introduce LSTM/TCN/Transformer models after you have substantial, representative labeled data. A strong later design is a forecasting model that predicts the next signal window; sustained high prediction residuals become anomaly scores. Avoid claiming a specific future failure unless you have verified labels.
Each alert should include:
- Severity and confidence
- Time interval
- Signals and thresholds that triggered it
- Comparable baseline
- Plot and raw values
- Recommended inspection action
- Clear wording such as “early indicator,” not “failure confirmed”
7. AI question-answering design
Do not send a complete raw log to an LLM and ask it to reason directly. It will be expensive, untraceable, and unreliable for numerical analysis.
Instead:
User question
  → LLM produces validated analysis request
  → trusted analytics functions run on decoded data
  → charts, metrics, and evidence returned
  → LLM writes a concise explanation with cited time ranges
Example structured request:
{
  "intent": "generator_battery_motor_overlap",
  "time_range": "entire_log",
  "minimum_duration_seconds": 2,
  "required_signals": [
    "generator_power_kw",
    "battery_power_kw",
    "motor_power_kw"
  ]
}
The LLM should only select from registered analyses and known signal names. It must not generate formulas, thresholds, or findings without the analytics layer validating them.
Use retrieval-augmented generation for vehicle documents, DBC descriptions, fault manuals, and engineering rules. Keep the log analytics itself deterministic.
8. Detailed report format
A report for one uploaded log should include:
1. File and vehicle metadata, DBC version, duration, data coverage.
2. Data-quality findings and limitations.
3. Drive-mode timeline: idle, EV drive, hybrid drive, generator charging, regenerative braking, faults.
4. Energy-flow summary: generator energy, battery charge/discharge energy, motor demand, estimated residual/loss.
5. Battery health: SOC trend, voltage/current, cell imbalance, temperature distribution, resistance proxy.
6. Generator, inverter, and motor behavior.
7. Detected anomalies and early-warning indicators.
8. DTCs and correlation with operating conditions.
9. Evidence plots with exact timestamps.
10. Conclusions, confidence, caveats, and inspection recommendations.
Generate HTML first, then export it to PDF. HTML makes charts, tables, and audit links much easier to review.
9. Recommended implementation phases
Phase 1 — Parser and visual explorer
- Upload ASC/BLF/MF4/CSV logs.
- Decode through DBC.
- Display signal search, plots, data-quality checks, and exports.
- Save decoded signals to partitioned Parquet.
Phase 2 — Hybrid energy analysis
- Implement signal ontology and vehicle configuration.
- Add derived powers, drive modes, charging events, regeneration, and energy balance.
- Add deterministic answers to your three energy-flow questions.
Phase 3 — Health monitoring
- Add battery features, baselines, thresholds, and trend analysis.
- Implement rule alerts and Isolation Forest anomaly scoring.
- Capture expert review feedback for every alert.
Phase 4 — AI assistant and reporting
- LLM converts questions into safe analytics calls.
- Generate evidence-backed responses and detailed reports.
- Add document retrieval for OEM engineering documentation.
Phase 5 — Validated predictive models
- Join historical logs with repair/failure records.
- Train and evaluate failure-risk models.
- Deploy only after vehicle-level, time-based validation.
10. Validation metrics
Do not use random row-based train/test splits; that leaks near-identical periods of the same trip into both sets. Split by vehicle and time.
Track:
- Energy-balance error versus calibrated reference tests
- Event detection precision/recall for charging, propulsion, and regeneration
- False alerts per driving hour
- Missed known-failure rate
- Lead time before confirmed failure
- Calibration: whether “80% risk” events fail about 80% of the time
- Report correctness reviewed by a domain engineer
For a production vehicle tool, alerts should assist maintenance decisions, not autonomously control the vehicle. Preserve raw evidence and require qualified engineering review for safety-relevant conclusions.
The most practical first deliverable is: upload a CAN log + DBC, decode it, calculate power flow, detect generator/battery/motor operating modes, and generate an evidence-backed report. Once that is verified against real vehicle behavior, add health and failure models.
