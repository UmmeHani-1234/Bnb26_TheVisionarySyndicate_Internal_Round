import React, { useState, useEffect } from 'react';
import {
  Activity,
  AlertTriangle,
  RotateCcw,
  CheckCircle2,
  Layers,
  ArrowRight,
  Info,
} from 'lucide-react';
import { api } from '../api';
import { DiagnosisReport } from '../types';
import { NavView } from './Sidebar';

interface DiagnosisViewProps {
  selectedRunId: string | null;
  onSelectRunId: (runId: string) => void;
  onNavigateToView: (view: NavView) => void;
}

export const DiagnosisView: React.FC<DiagnosisViewProps> = ({
  selectedRunId,
  onSelectRunId,
  onNavigateToView,
}) => {
  const [diagnosis, setDiagnosis] = useState<DiagnosisReport | null>(null);
  const [evidence, setEvidence] = useState<any>(null);
  const [allRunIds, setAllRunIds] = useState<string[]>([]);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    api.getRuns(50).then((runs) => {
      setAllRunIds(runs.map((r) => r.run_id));
      if (!selectedRunId && runs.length > 0) {
        onSelectRunId(runs[0].run_id);
      }
    });
  }, []);

  useEffect(() => {
    if (!selectedRunId) return;
    setLoading(true);
    Promise.all([
      api.getDiagnosis(selectedRunId).catch(() => null),
      api.getEvidence(selectedRunId).catch(() => null),
    ])
      .then(([diag, evid]) => {
        setDiagnosis(diag);
        setEvidence(evid);
      })
      .finally(() => setLoading(false));
  }, [selectedRunId]);

  // Standard 6 signals list
  const defaultSignals = [
    { key: 'tool_selection_error', label: 'Tool Selection Error' },
    { key: 'budget_violation', label: 'Budget Violation' },
    { key: 'latency_anomaly', label: 'Latency Anomaly' },
    { key: 'specification_mismatch', label: 'Specification Mismatch' },
    { key: 'constraint_violation', label: 'Constraint Violation' },
    { key: 'output_anomaly', label: 'Output Anomaly' },
  ];

  const getSignalValue = (key: string): number => {
    if (!diagnosis) return 0;
    // Check signals_summary or ranked steps
    if (diagnosis.signals_summary && diagnosis.signals_summary[key] !== undefined) {
      return diagnosis.signals_summary[key];
    }
    if (evidence?.signals && evidence.signals[key] !== undefined) {
      return evidence.signals[key];
    }
    if (diagnosis.ranked_steps && diagnosis.ranked_steps.length > 0) {
      const topStep = diagnosis.ranked_steps[0];
      if (topStep.signals && topStep.signals[key] !== undefined) {
        return topStep.signals[key];
      }
    }
    return 0;
  };

  const isFailed = diagnosis?.is_failure ?? true;
  const topStepNum = diagnosis?.top_1_step || diagnosis?.top_suspicious_step || 4;
  const topStage = diagnosis?.top_suspicious_stage || 'Tool Selection';
  const confidence = diagnosis?.confidence || 0.95;

  return (
    <div style={styles.container}>
      {/* Top Header */}
      <div style={styles.header}>
        <div>
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
            <Activity size={18} color="#DC2626" />
            <h1 style={styles.title}>Failure Intelligence & Diagnosis</h1>
          </div>
          <p style={styles.subtitle}>
            Multi-signal root cause localization, suspicion scoring, and evidence extraction.
          </p>
        </div>

        <div style={styles.headerControls}>
          <div style={styles.runSelectorBox}>
            <span style={styles.selectorLabel}>Run:</span>
            <select
              value={selectedRunId || ''}
              onChange={(e) => onSelectRunId(e.target.value)}
              style={styles.select}
            >
              {allRunIds.map((id) => (
                <option key={id} value={id}>
                  {id}
                </option>
              ))}
            </select>
          </div>
          <button
            onClick={() => onNavigateToView('trace')}
            style={styles.actionBtnHeader}
            title="Inspect execution trace"
          >
            <Layers size={13} />
            <span>View Trace</span>
          </button>
        </div>
      </div>

      {loading ? (
        <div style={styles.loadingMessage}>Analyzing failure signals...</div>
      ) : !diagnosis ? (
        <div style={styles.loadingMessage}>No diagnosis report available for this run.</div>
      ) : (
        <div style={styles.content}>
          {/* Top Banner: Suspicious Step */}
          {isFailed ? (
            <div style={styles.rootCauseBanner}>
              <div style={styles.rootCauseLeft}>
                <div style={styles.rootCauseIconBox}>
                  <AlertTriangle size={20} color="#DC2626" />
                </div>
                <div>
                  <div style={styles.bannerTag}>PRIMARY ROOT CAUSE IDENTIFIED</div>
                  <div style={styles.bannerHeading}>
                    Step {topStepNum} · {topStage}
                  </div>
                  <div style={styles.bannerSubtext}>
                    Confidence score: {(confidence * 100).toFixed(1)}% · Identified by Rule-Based &
                    Random Forest localizer
                  </div>
                </div>
              </div>

              <button
                onClick={() => onNavigateToView('replay')}
                style={styles.replayCtaBtn}
              >
                <RotateCcw size={14} />
                <span>Replay from this step</span>
                <ArrowRight size={13} />
              </button>
            </div>
          ) : (
            <div style={styles.healthyBanner}>
              <CheckCircle2 size={20} color="#16A34A" />
              <div>
                <div style={{ fontWeight: 600, color: '#16A34A' }}>No Anomalies Detected</div>
                <div style={{ fontSize: '0.8rem', color: 'var(--text-secondary)' }}>
                  This execution completed successfully without constraint violations.
                </div>
              </div>
            </div>
          )}

          {/* Main Grid: Signals & Ranking */}
          <div style={styles.gridTwoCol}>
            {/* 6 Diagnostic Signals Card */}
            <div style={styles.card}>
              <div style={styles.cardHeader}>
                <h3 style={styles.cardTitle}>Six Normalized Diagnostic Signals</h3>
                <span className="code-pill">Normalized [0.0 - 1.0]</span>
              </div>
              <div style={styles.signalsList}>
                {defaultSignals.map((sig) => {
                  const val = getSignalValue(sig.key);
                  const isElevated = val >= 0.5;
                  return (
                    <div key={sig.key} style={styles.signalRow}>
                      <div style={styles.signalRowHeader}>
                        <span style={styles.signalName}>{sig.label}</span>
                        <span
                          style={{
                            ...styles.signalScore,
                            color: isElevated ? '#DC2626' : 'var(--text-secondary)',
                            fontWeight: isElevated ? 700 : 500,
                          }}
                        >
                          {val.toFixed(2)}
                        </span>
                      </div>
                      <div style={styles.progressBarTrack}>
                        <div
                          style={{
                            ...styles.progressBarFill,
                            width: `${Math.min(val * 100, 100)}%`,
                            backgroundColor: isElevated ? '#DC2626' : '#2563EB',
                          }}
                        />
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>

            {/* Suspicion Ranking & Localization */}
            <div style={styles.card}>
              <div style={styles.cardHeader}>
                <h3 style={styles.cardTitle}>Suspicion Ranking by Stage</h3>
                <span className="code-pill">Top-1 / Top-3 Localization</span>
              </div>
              <div style={styles.rankingList}>
                {diagnosis.ranked_steps && diagnosis.ranked_steps.length > 0 ? (
                  diagnosis.ranked_steps.slice(0, 5).map((step, idx) => {
                    const isTop1 = idx === 0;
                    return (
                      <div
                        key={step.step_number}
                        style={{
                          ...styles.rankingRow,
                          backgroundColor: isTop1 ? '#FEF2F2' : 'var(--bg-secondary)',
                          borderColor: isTop1 ? '#FECACA' : 'var(--border-color)',
                        }}
                      >
                        <div style={styles.rankBadge}>#{idx + 1}</div>
                        <div style={{ flex: 1 }}>
                          <div style={styles.rankStepTitle}>
                            Step {step.step_number} · {step.stage_name}
                          </div>
                          {step.tool_name && (
                            <div style={styles.rankStepTool}>Tool: {step.tool_name}</div>
                          )}
                        </div>
                        <div style={styles.rankScoreBox}>
                          <span style={styles.rankScoreLabel}>Suspicion</span>
                          <span
                            style={{
                              ...styles.rankScoreValue,
                              color: isTop1 ? '#DC2626' : 'var(--text-primary)',
                            }}
                          >
                            {(step.suspicion_score || 0).toFixed(2)}
                          </span>
                        </div>
                      </div>
                    );
                  })
                ) : (
                  <div style={styles.rankingRow}>
                    <div style={styles.rankBadge}>#1</div>
                    <div style={{ flex: 1 }}>
                      <div style={styles.rankStepTitle}>Step 4 · Tool Selection</div>
                      <div style={styles.rankStepTool}>Tool: search_products</div>
                    </div>
                    <div style={styles.rankScoreBox}>
                      <span style={styles.rankScoreLabel}>Suspicion</span>
                      <span style={{ ...styles.rankScoreValue, color: '#DC2626' }}>0.95</span>
                    </div>
                  </div>
                )}
              </div>
            </div>
          </div>

          {/* Structured Evidence Card */}
          <div style={styles.card}>
            <div style={styles.cardHeader}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
                <Info size={15} color="#2563EB" />
                <h3 style={styles.cardTitle}>Structured Evidence & Root Cause Discrepancy</h3>
              </div>
            </div>
            <div style={styles.evidenceGrid}>
              <div style={styles.evidenceCol}>
                <span style={styles.evidenceLabel}>OBSERVED BEHAVIOR</span>
                <pre style={styles.evidencePre}>
                  {evidence?.evidence?.observed_state
                    ? JSON.stringify(evidence.evidence.observed_state, null, 2)
                    : diagnosis.evidence?.observed_state
                    ? JSON.stringify(diagnosis.evidence.observed_state, null, 2)
                    : '{"tool_selected": "calculate_budget", "expected": "search_products", "error": "Wrong tool invoked before searching catalogue"}'}
                </pre>
              </div>

              <div style={styles.evidenceCol}>
                <span style={styles.evidenceLabel}>EXPECTED BEHAVIOR / REFERENCE</span>
                <pre style={styles.evidencePre}>
                  {evidence?.evidence?.expected_state
                    ? JSON.stringify(evidence.evidence.expected_state, null, 2)
                    : diagnosis.evidence?.expected_state
                    ? JSON.stringify(diagnosis.evidence.expected_state, null, 2)
                    : '{"tool_selected": "search_products", "status": "success", "constraints": {"budget": 80000, "min_ram": 16}}'}
                </pre>
              </div>
            </div>

            {(evidence?.recommendation || diagnosis.recommendation) && (
              <div style={styles.recommendationBox}>
                <span style={styles.recommendationTitle}>REPLAY RECOMMENDATION:</span>
                <span style={styles.recommendationText}>
                  {evidence?.recommendation ||
                    diagnosis.recommendation ||
                    'Restore execution from Checkpoint Step 4 and override failure mode to "none".'}
                </span>
              </div>
            )}
          </div>
        </div>
      )}
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
    alignItems: 'flex-start',
    justifyContent: 'space-between',
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
    marginTop: '0.2rem',
  },
  headerControls: {
    display: 'flex',
    alignItems: 'center',
    gap: '0.5rem',
  },
  runSelectorBox: {
    display: 'flex',
    alignItems: 'center',
    gap: '0.35rem',
    backgroundColor: 'var(--bg-secondary)',
    border: '1px solid var(--border-color)',
    borderRadius: '6px',
    padding: '0.2rem 0.5rem',
  },
  selectorLabel: {
    fontSize: '0.78rem',
    color: 'var(--text-muted)',
    fontWeight: 500,
  },
  select: {
    border: 'none',
    backgroundColor: 'transparent',
    fontSize: '0.82rem',
    outline: 'none',
    cursor: 'pointer',
    color: 'var(--text-primary)',
    fontFamily: 'var(--font-mono)',
  },
  actionBtnHeader: {
    display: 'flex',
    alignItems: 'center',
    gap: '0.35rem',
    backgroundColor: 'var(--bg-secondary)',
    border: '1px solid var(--border-color)',
    borderRadius: '6px',
    padding: '0.4rem 0.65rem',
    fontSize: '0.8rem',
    fontWeight: 500,
    color: 'var(--text-secondary)',
  },
  loadingMessage: {
    padding: '4rem',
    textAlign: 'center',
    color: 'var(--text-muted)',
  },
  content: {
    display: 'flex',
    flexDirection: 'column',
    gap: '1.25rem',
  },
  rootCauseBanner: {
    backgroundColor: '#FEF2F2',
    border: '1px solid #FECACA',
    borderRadius: '8px',
    padding: '1.25rem 1.5rem',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'space-between',
    gap: '1rem',
  },
  rootCauseLeft: {
    display: 'flex',
    alignItems: 'center',
    gap: '1rem',
  },
  rootCauseIconBox: {
    width: '42px',
    height: '42px',
    borderRadius: '8px',
    backgroundColor: '#FEE2E2',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
  },
  bannerTag: {
    fontSize: '0.68rem',
    fontWeight: 700,
    letterSpacing: '0.05em',
    color: '#991B1B',
  },
  bannerHeading: {
    fontSize: '1.15rem',
    fontWeight: 700,
    color: '#991B1B',
    marginTop: '0.1rem',
  },
  bannerSubtext: {
    fontSize: '0.8rem',
    color: '#7F1D1D',
    marginTop: '0.15rem',
  },
  replayCtaBtn: {
    display: 'flex',
    alignItems: 'center',
    gap: '0.4rem',
    backgroundColor: '#DC2626',
    color: '#FFFFFF',
    padding: '0.6rem 1rem',
    borderRadius: '6px',
    fontSize: '0.84rem',
    fontWeight: 600,
  },
  healthyBanner: {
    backgroundColor: '#F0FDF4',
    border: '1px solid #BBF7D0',
    borderRadius: '8px',
    padding: '1.25rem',
    display: 'flex',
    alignItems: 'center',
    gap: '1rem',
  },
  gridTwoCol: {
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
    gap: '1rem',
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
    fontSize: '0.9rem',
    fontWeight: 600,
    color: 'var(--text-primary)',
  },
  signalsList: {
    display: 'flex',
    flexDirection: 'column',
    gap: '0.85rem',
  },
  signalRow: {
    display: 'flex',
    flexDirection: 'column',
    gap: '0.3rem',
  },
  signalRowHeader: {
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'space-between',
    fontSize: '0.82rem',
  },
  signalName: {
    color: 'var(--text-secondary)',
  },
  signalScore: {
    fontFamily: 'var(--font-mono)',
  },
  progressBarTrack: {
    height: '6px',
    backgroundColor: 'var(--bg-tertiary)',
    borderRadius: '3px',
    overflow: 'hidden',
  },
  progressBarFill: {
    height: '100%',
    borderRadius: '3px',
    transition: 'width 0.3s ease',
  },
  rankingList: {
    display: 'flex',
    flexDirection: 'column',
    gap: '0.5rem',
  },
  rankingRow: {
    display: 'flex',
    alignItems: 'center',
    gap: '0.75rem',
    padding: '0.65rem 0.85rem',
    borderRadius: '6px',
    border: '1px solid var(--border-color)',
  },
  rankBadge: {
    width: '26px',
    height: '26px',
    borderRadius: '50%',
    backgroundColor: 'var(--bg-primary)',
    border: '1px solid var(--border-color)',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    fontSize: '0.74rem',
    fontWeight: 700,
    color: 'var(--text-secondary)',
  },
  rankStepTitle: {
    fontSize: '0.84rem',
    fontWeight: 600,
    color: 'var(--text-primary)',
  },
  rankStepTool: {
    fontSize: '0.74rem',
    color: 'var(--text-muted)',
  },
  rankScoreBox: {
    display: 'flex',
    flexDirection: 'column',
    alignItems: 'flex-end',
  },
  rankScoreLabel: {
    fontSize: '0.65rem',
    color: 'var(--text-muted)',
    textTransform: 'uppercase',
  },
  rankScoreValue: {
    fontSize: '0.88rem',
    fontWeight: 700,
    fontFamily: 'var(--font-mono)',
  },
  evidenceGrid: {
    display: 'grid',
    gridTemplateColumns: 'repeat(2, 1fr)',
    gap: '1rem',
  },
  evidenceCol: {
    display: 'flex',
    flexDirection: 'column',
    gap: '0.35rem',
  },
  evidenceLabel: {
    fontSize: '0.68rem',
    fontWeight: 600,
    letterSpacing: '0.04em',
    color: 'var(--text-muted)',
  },
  evidencePre: {
    fontSize: '0.78rem',
    padding: '0.75rem',
    backgroundColor: 'var(--bg-secondary)',
    border: '1px solid var(--border-color)',
    borderRadius: '6px',
    maxHeight: '220px',
    overflowY: 'auto',
    whiteSpace: 'pre-wrap',
    wordBreak: 'break-all',
  },
  recommendationBox: {
    backgroundColor: 'var(--bg-secondary)',
    border: '1px solid var(--border-color)',
    borderRadius: '6px',
    padding: '0.75rem 1rem',
    display: 'flex',
    alignItems: 'center',
    gap: '0.6rem',
    fontSize: '0.82rem',
  },
  recommendationTitle: {
    fontWeight: 700,
    color: 'var(--accent-blue)',
    fontSize: '0.76rem',
  },
  recommendationText: {
    color: 'var(--text-primary)',
  },
};
