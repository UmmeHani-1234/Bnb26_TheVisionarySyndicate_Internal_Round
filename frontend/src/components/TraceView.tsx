import React, { useState, useEffect } from 'react';
import {
  CheckCircle2,
  XCircle,
  Clock,
  Wrench,
  ChevronDown,
  ChevronRight,
  Activity,
  RotateCcw,
  Layers,
  Code2,
} from 'lucide-react';
import { api } from '../api';
import { RunTrace, AgentStep } from '../types';
import { NavView } from './Sidebar';

interface TraceViewProps {
  selectedRunId: string | null;
  onSelectRunId: (runId: string) => void;
  onNavigateToView: (view: NavView) => void;
}

export const TraceView: React.FC<TraceViewProps> = ({
  selectedRunId,
  onSelectRunId,
  onNavigateToView,
}) => {
  const [trace, setTrace] = useState<RunTrace | null>(null);
  const [allRunIds, setAllRunIds] = useState<string[]>([]);
  const [expandedSteps, setExpandedSteps] = useState<Record<number, boolean>>({});
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    // Fetch list of run IDs for the selector
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
    api
      .getRun(selectedRunId)
      .then((data) => {
        setTrace(data);
        // Expand first and failed steps by default
        const initExpanded: Record<number, boolean> = {};
        data.steps.forEach((s, idx) => {
          if (idx === 0 || s.status === 'failed' || s.status === 'error') {
            initExpanded[idx] = true;
          }
        });
        setExpandedSteps(initExpanded);
      })
      .catch((err) => console.error('Failed to load trace:', err))
      .finally(() => setLoading(false));
  }, [selectedRunId]);

  const toggleStep = (idx: number) => {
    setExpandedSteps((prev) => ({ ...prev, [idx]: !prev[idx] }));
  };

  const expandAll = () => {
    if (!trace) return;
    const allExp: Record<number, boolean> = {};
    trace.steps.forEach((_, idx) => (allExp[idx] = true));
    setExpandedSteps(allExp);
  };

  const collapseAll = () => {
    setExpandedSteps({});
  };

  return (
    <div style={styles.container}>
      {/* Top Header / Run Selector */}
      <div style={styles.header}>
        <div>
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
            <Layers size={18} color="#2563EB" />
            <h1 style={styles.title}>Execution Trace</h1>
          </div>
          <p style={styles.subtitle}>
            Step-by-step observable execution path across 8 pipeline stages.
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
            onClick={() => onNavigateToView('diagnosis')}
            style={styles.actionBtnHeader}
            title="Diagnose this run"
          >
            <Activity size={13} color="#DC2626" />
            <span>Diagnosis</span>
          </button>
          <button
            onClick={() => onNavigateToView('replay')}
            style={styles.actionBtnHeader}
            title="Replay from checkpoint"
          >
            <RotateCcw size={13} color="#7C3AED" />
            <span>Replay</span>
          </button>
        </div>
      </div>

      {loading ? (
        <div style={styles.loadingMessage}>Loading execution trace...</div>
      ) : !trace ? (
        <div style={styles.loadingMessage}>No trace selected. Select a run above.</div>
      ) : (
        <div style={styles.content}>
          {/* Metadata Card */}
          <div style={styles.metadataCard}>
            <div style={styles.metaRow}>
              <div>
                <span style={styles.metaLabel}>USER QUERY</span>
                <div style={styles.metaQuery}>{trace.user_request}</div>
              </div>
              <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
                {trace.status === 'success' ? (
                  <span className="badge badge-success">
                    <CheckCircle2 size={12} />
                    SUCCESS
                  </span>
                ) : (
                  <span className="badge badge-error">
                    <XCircle size={12} />
                    FAILED
                  </span>
                )}
                {trace.failure_mode && (
                  <span className="badge badge-warning">Mode: {trace.failure_mode}</span>
                )}
              </div>
            </div>

            <div style={styles.metaStats}>
              <div style={styles.statItem}>
                <span style={styles.statLabel}>Total Steps</span>
                <span style={styles.statValue}>{trace.steps?.length || 0}</span>
              </div>
              <div style={styles.statItem}>
                <span style={styles.statLabel}>Total Latency</span>
                <span style={styles.statValue}>
                  {trace.total_duration ? `${trace.total_duration.toFixed(2)}s` : '1.8s'}
                </span>
              </div>
              <div style={styles.statItem}>
                <span style={styles.statLabel}>Run ID</span>
                <span className="code-pill">{trace.run_id}</span>
              </div>
            </div>
          </div>

          {/* Timeline Controls */}
          <div style={styles.timelineBar}>
            <span style={styles.timelineBarTitle}>Standard 8-Stage Timeline</span>
            <div style={{ display: 'flex', gap: '0.4rem' }}>
              <button onClick={expandAll} style={styles.smallToggleBtn}>
                Expand All
              </button>
              <button onClick={collapseAll} style={styles.smallToggleBtn}>
                Collapse All
              </button>
            </div>
          </div>

          {/* Vertical 8-Stage Timeline */}
          <div style={styles.timelineContainer}>
            {trace.steps.map((step, idx) => {
              const isExpanded = !!expandedSteps[idx];
              const isFail = step.status === 'failed' || step.status === 'error';
              const stepNum = step.step_number || idx + 1;
              const stageName = step.stage || step.step_type || `Stage ${stepNum}`;

              return (
                <div key={idx} style={styles.stepWrapper}>
                  {/* Left Timeline Spine */}
                  <div style={styles.spine}>
                    <div
                      style={{
                        ...styles.spineNode,
                        backgroundColor: isFail ? '#DC2626' : '#16A34A',
                      }}
                    >
                      {stepNum}
                    </div>
                    {idx < trace.steps.length - 1 && <div style={styles.spineLine} />}
                  </div>

                  {/* Step Card */}
                  <div
                    style={{
                      ...styles.stepCard,
                      borderColor: isFail ? '#FCA5A5' : 'var(--border-color)',
                      backgroundColor: isFail ? '#FFFDFD' : 'var(--bg-primary)',
                    }}
                  >
                    {/* Header */}
                    <div style={styles.stepCardHeader} onClick={() => toggleStep(idx)}>
                      <div style={styles.stepTitleGroup}>
                        <button style={styles.expandChevron}>
                          {isExpanded ? <ChevronDown size={15} /> : <ChevronRight size={15} />}
                        </button>
                        <span style={styles.stepName}>{stageName}</span>
                        {step.tool_name && (
                          <span className="badge badge-neutral">
                            <Wrench size={11} />
                            {step.tool_name}
                          </span>
                        )}
                      </div>

                      <div style={styles.stepHeaderRight}>
                        {step.latency && (
                          <span style={styles.stepLatency}>
                            <Clock size={12} />
                            {step.latency.toFixed(2)}s
                          </span>
                        )}
                        {isFail ? (
                          <span className="badge badge-error">Failed</span>
                        ) : (
                          <span className="badge badge-success">Success</span>
                        )}
                      </div>
                    </div>

                    {/* Expanded Drawer */}
                    {isExpanded && (
                      <div style={styles.stepDetailsDrawer}>
                        {/* Input */}
                        {step.input !== undefined && (
                          <div style={styles.jsonSection}>
                            <div style={styles.sectionHeading}>
                              <Code2 size={12} />
                              <span>STEP INPUT</span>
                            </div>
                            <pre style={styles.codeBlock}>
                              {typeof step.input === 'string'
                                ? step.input
                                : JSON.stringify(step.input, null, 2)}
                            </pre>
                          </div>
                        )}

                        {/* Output */}
                        {step.output !== undefined && (
                          <div style={styles.jsonSection}>
                            <div style={styles.sectionHeading}>
                              <Code2 size={12} />
                              <span>STEP OUTPUT</span>
                            </div>
                            <pre
                              style={{
                                ...styles.codeBlock,
                                backgroundColor: isFail ? '#FEF2F2' : '#F8FAFC',
                              }}
                            >
                              {typeof step.output === 'string'
                                ? step.output
                                : JSON.stringify(step.output, null, 2)}
                            </pre>
                          </div>
                        )}

                        {/* Observable State Snapshot */}
                        {step.state_snapshot && (
                          <div style={styles.jsonSection}>
                            <div style={styles.sectionHeading}>
                              <Layers size={12} />
                              <span>OBSERVABLE STATE SNAPSHOT</span>
                            </div>
                            <pre style={styles.codeBlock}>
                              {JSON.stringify(step.state_snapshot, null, 2)}
                            </pre>
                          </div>
                        )}
                      </div>
                    )}
                  </div>
                </div>
              );
            })}
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
  metadataCard: {
    backgroundColor: 'var(--bg-secondary)',
    border: '1px solid var(--border-color)',
    borderRadius: '8px',
    padding: '1.25rem',
    display: 'flex',
    flexDirection: 'column',
    gap: '1rem',
  },
  metaRow: {
    display: 'flex',
    alignItems: 'flex-start',
    justifyContent: 'space-between',
    gap: '1rem',
  },
  metaLabel: {
    fontSize: '0.68rem',
    fontWeight: 600,
    letterSpacing: '0.05em',
    color: 'var(--text-muted)',
  },
  metaQuery: {
    fontSize: '0.95rem',
    fontWeight: 600,
    color: 'var(--text-primary)',
    marginTop: '0.2rem',
  },
  metaStats: {
    display: 'flex',
    gap: '2.5rem',
    paddingTop: '0.75rem',
    borderTop: '1px solid var(--border-color)',
  },
  statItem: {
    display: 'flex',
    flexDirection: 'column',
    gap: '0.15rem',
  },
  statLabel: {
    fontSize: '0.7rem',
    color: 'var(--text-muted)',
  },
  statValue: {
    fontSize: '0.88rem',
    fontWeight: 600,
    color: 'var(--text-primary)',
  },
  timelineBar: {
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'space-between',
    paddingBottom: '0.5rem',
    borderBottom: '1px solid var(--border-color)',
  },
  timelineBarTitle: {
    fontSize: '0.82rem',
    fontWeight: 600,
    color: 'var(--text-secondary)',
    textTransform: 'uppercase',
    letterSpacing: '0.04em',
  },
  smallToggleBtn: {
    fontSize: '0.74rem',
    padding: '0.2rem 0.5rem',
    backgroundColor: 'var(--bg-secondary)',
    border: '1px solid var(--border-color)',
    borderRadius: '4px',
    color: 'var(--text-secondary)',
  },
  timelineContainer: {
    display: 'flex',
    flexDirection: 'column',
    gap: '0.65rem',
  },
  stepWrapper: {
    display: 'flex',
    gap: '1rem',
  },
  spine: {
    width: '26px',
    display: 'flex',
    flexDirection: 'column',
    alignItems: 'center',
  },
  spineNode: {
    width: '22px',
    height: '22px',
    borderRadius: '50%',
    color: '#FFFFFF',
    fontSize: '0.7rem',
    fontWeight: 700,
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    zIndex: 2,
  },
  spineLine: {
    flex: 1,
    width: '2px',
    backgroundColor: 'var(--border-color)',
    margin: '4px 0',
  },
  stepCard: {
    flex: 1,
    border: '1px solid var(--border-color)',
    borderRadius: '8px',
    overflow: 'hidden',
    boxShadow: '0 1px 2px rgba(0,0,0,0.02)',
  },
  stepCardHeader: {
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'space-between',
    padding: '0.75rem 1rem',
    cursor: 'pointer',
    userSelect: 'none',
  },
  stepTitleGroup: {
    display: 'flex',
    alignItems: 'center',
    gap: '0.6rem',
  },
  expandChevron: {
    display: 'flex',
    alignItems: 'center',
    color: 'var(--text-muted)',
  },
  stepName: {
    fontSize: '0.88rem',
    fontWeight: 600,
    color: 'var(--text-primary)',
  },
  stepHeaderRight: {
    display: 'flex',
    alignItems: 'center',
    gap: '0.75rem',
  },
  stepLatency: {
    display: 'flex',
    alignItems: 'center',
    gap: '0.25rem',
    fontSize: '0.75rem',
    color: 'var(--text-muted)',
  },
  stepDetailsDrawer: {
    padding: '0.75rem 1rem 1rem 1rem',
    backgroundColor: 'var(--bg-secondary)',
    borderTop: '1px solid var(--border-color)',
    display: 'flex',
    flexDirection: 'column',
    gap: '0.75rem',
  },
  jsonSection: {
    display: 'flex',
    flexDirection: 'column',
    gap: '0.25rem',
  },
  sectionHeading: {
    display: 'flex',
    alignItems: 'center',
    gap: '0.35rem',
    fontSize: '0.68rem',
    fontWeight: 600,
    color: 'var(--text-muted)',
    letterSpacing: '0.04em',
  },
  codeBlock: {
    fontSize: '0.78rem',
    padding: '0.6rem 0.75rem',
    borderRadius: '6px',
    backgroundColor: '#FFFFFF',
    border: '1px solid var(--border-color)',
    overflowX: 'auto',
    whiteSpace: 'pre-wrap',
    wordBreak: 'break-all',
    color: 'var(--text-primary)',
    maxHeight: '260px',
  },
};
