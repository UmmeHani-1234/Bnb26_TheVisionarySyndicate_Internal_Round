import React from 'react';
import { Settings as SettingsIcon, Database, Server, Cpu, ShieldCheck } from 'lucide-react';
import { SystemStatus } from '../types';

interface SettingsViewProps {
  systemStatus: SystemStatus | null;
}

export const SettingsView: React.FC<SettingsViewProps> = ({ systemStatus }) => {
  return (
    <div style={styles.container}>
      <div style={styles.header}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
          <SettingsIcon size={18} color="#2563EB" />
          <h1 style={styles.title}>System Settings & Configuration</h1>
        </div>
        <p style={styles.subtitle}>
          Technical environment status, observability pipelines, and API parameters.
        </p>
      </div>

      <div style={styles.grid}>
        {/* Backend & Environment Card */}
        <div style={styles.card}>
          <div style={styles.cardHeader}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
              <Server size={15} color="#2563EB" />
              <h3 style={styles.cardTitle}>FastAPI Backend Service</h3>
            </div>
            <span className="badge badge-success">Online</span>
          </div>

          <div style={styles.settingRow}>
            <span style={styles.settingLabel}>API Endpoint</span>
            <span className="code-pill">http://localhost:8000</span>
          </div>

          <div style={styles.settingRow}>
            <span style={styles.settingLabel}>CORS Access Control</span>
            <span className="badge badge-neutral">Enabled (*)</span>
          </div>

          <div style={styles.settingRow}>
            <span style={styles.settingLabel}>System Version</span>
            <span style={styles.settingVal}>{systemStatus?.version || '1.0.0'}</span>
          </div>
        </div>

        {/* Database & Persistent Trace Storage */}
        <div style={styles.card}>
          <div style={styles.cardHeader}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
              <Database size={15} color="#16A34A" />
              <h3 style={styles.cardTitle}>Persistent Trace Storage</h3>
            </div>
            <span className="badge badge-success">Active</span>
          </div>

          <div style={styles.settingRow}>
            <span style={styles.settingLabel}>Repository Engine</span>
            <span style={styles.settingVal}>SQLite / PostgreSQL TraceRepository</span>
          </div>

          <div style={styles.settingRow}>
            <span style={styles.settingLabel}>Trace Event Sinks</span>
            <span className="badge badge-blue">Synchronous ExecutionRecorder</span>
          </div>

          <div style={styles.settingRow}>
            <span style={styles.settingLabel}>Checkpoints Storage</span>
            <span className="badge badge-neutral">Atomic Serialization</span>
          </div>
        </div>

        {/* AI Agent Configuration */}
        <div style={styles.card}>
          <div style={styles.cardHeader}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
              <Cpu size={15} color="#7C3AED" />
              <h3 style={styles.cardTitle}>Laptop Recommendation Agent</h3>
            </div>
            <span className="badge badge-replay">Ready</span>
          </div>

          <div style={styles.settingRow}>
            <span style={styles.settingLabel}>Pipeline Architecture</span>
            <span style={styles.settingVal}>8 Standard Observable Stages</span>
          </div>

          <div style={styles.settingRow}>
            <span style={styles.settingLabel}>Default Tools</span>
            <span className="code-pill">search_products, calculate_budget, check_specifications</span>
          </div>

          <div style={styles.settingRow}>
            <span style={styles.settingLabel}>Failure Injection Modes</span>
            <span className="badge badge-warning">wrong_tool, budget_overflow, invalid_output, timeout</span>
          </div>
        </div>

        {/* Independent Verifier */}
        <div style={styles.card}>
          <div style={styles.cardHeader}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
              <ShieldCheck size={15} color="#16A34A" />
              <h3 style={styles.cardTitle}>Independent Verifier</h3>
            </div>
            <span className="badge badge-success">Enforced</span>
          </div>

          <div style={styles.settingRow}>
            <span style={styles.settingLabel}>Verification Logic</span>
            <span style={styles.settingVal}>Deterministic Rule-Based + Spec Match</span>
          </div>

          <div style={styles.settingRow}>
            <span style={styles.settingLabel}>Replay Comparison</span>
            <span className="badge badge-neutral">3-Way Trace Alignment</span>
          </div>
        </div>
      </div>
    </div>
  );
};

const styles: Record<string, React.CSSProperties> = {
  container: {
    padding: '2rem',
    height: '100vh',
    overflowY: 'auto',
    backgroundColor: 'var(--bg-primary)',
    display: 'flex',
    flexDirection: 'column',
    gap: '1.25rem',
  },
  header: {
    display: 'flex',
    flexDirection: 'column',
    gap: '0.2rem',
  },
  title: {
    fontSize: '1.4rem',
    fontWeight: 700,
    color: 'var(--text-primary)',
    letterSpacing: '-0.01em',
  },
  subtitle: {
    fontSize: '0.85rem',
    color: 'var(--text-secondary)',
  },
  grid: {
    display: 'grid',
    gridTemplateColumns: 'repeat(2, 1fr)',
    gap: '1.25rem',
  },
  card: {
    backgroundColor: 'var(--bg-primary)',
    border: '1px solid var(--border-color)',
    borderRadius: '8px',
    padding: '1.25rem',
    display: 'flex',
    flexDirection: 'column',
    gap: '0.85rem',
    boxShadow: '0 1px 2px rgba(0,0,0,0.02)',
  },
  cardHeader: {
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'space-between',
    paddingBottom: '0.5rem',
    borderBottom: '1px solid var(--border-color)',
  },
  cardTitle: {
    fontSize: '0.88rem',
    fontWeight: 600,
    color: 'var(--text-primary)',
  },
  settingRow: {
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'space-between',
    fontSize: '0.82rem',
    padding: '0.25rem 0',
  },
  settingLabel: {
    color: 'var(--text-secondary)',
  },
  settingVal: {
    fontWeight: 500,
    color: 'var(--text-primary)',
  },
};
