import React, { useState, useEffect } from 'react';
import {
  RotateCcw,
  GitBranch,
  Play,
  Layers,
  ArrowRight,
  Sliders,
  CheckCircle2,
  XCircle,
  Clock,
  Sparkles,
  GitCompare,
} from 'lucide-react';
import { api } from '../api';
import { CheckpointItem } from '../types';
import { NavView } from './Sidebar';

interface ReplayViewProps {
  selectedRunId: string | null;
  onSelectRunId: (runId: string) => void;
  onNavigateToView: (view: NavView) => void;
  onSelectReplayRun: (origRunId: string, altRunId: string) => void;
}

export const ReplayView: React.FC<ReplayViewProps> = ({
  selectedRunId,
  onSelectRunId,
  onNavigateToView,
  onSelectReplayRun,
}) => {
  const [allRunIds, setAllRunIds] = useState<string[]>([]);
  const [checkpoints, setCheckpoints] = useState<CheckpointItem[]>([]);
  const [selectedCheckpointId, setSelectedCheckpointId] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [isReplaying, setIsReplaying] = useState(false);

  // Controlled Overrides state
  const [overrideBudget, setOverrideBudget] = useState<string>('');
  const [overrideFailureMode, setOverrideFailureMode] = useState<string>('none');
  const [overrideTool, setOverrideTool] = useState<string>('');

  // Result of replay execution
  const [replayResult, setReplayResult] = useState<any>(null);

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
    setReplayResult(null);
    api
      .getCheckpoints(selectedRunId)
      .then((res) => {
        setCheckpoints(res.checkpoints || []);
        if (res.checkpoints && res.checkpoints.length > 0) {
          // Select checkpoint around step 3 or 4 if available
          const preferred =
            res.checkpoints.find((c) => c.step_id === 4 || c.step_id === 3) || res.checkpoints[0];
          setSelectedCheckpointId(preferred.checkpoint_id);
        } else {
          setSelectedCheckpointId(null);
        }
      })
      .catch((err) => console.error('Failed to load checkpoints:', err))
      .finally(() => setLoading(false));
  }, [selectedRunId]);

  const selectedCheckpoint = checkpoints.find((c) => c.checkpoint_id === selectedCheckpointId);

  const handleExecuteReplay = async () => {
    if (!selectedRunId || !selectedCheckpointId) return;
    setIsReplaying(true);
    try {
      const overrideObj: any = {};
      if (overrideBudget.trim()) {
        overrideObj.max_budget = parseInt(overrideBudget, 10);
      }
      if (overrideFailureMode) {
        overrideObj.failure_mode = overrideFailureMode;
      }
      if (overrideTool.trim()) {
        overrideObj.tool_name = overrideTool.trim();
      }

      const res = await api.replayRun(selectedRunId, selectedCheckpointId, overrideObj);
      setReplayResult(res);
    } catch (err: any) {
      alert(`Replay failed: ${err.message || 'Unknown error'}`);
    } finally {
      setIsReplaying(false);
    }
  };

  return (
    <div style={styles.container}>
      {/* Top Header */}
      <div style={styles.header}>
        <div>
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
            <RotateCcw size={18} color="#7C3AED" />
            <h1 style={styles.title}>Replay & Alternative Execution</h1>
          </div>
          <p style={styles.subtitle}>
            Branch execution non-destructively from any verified checkpoint with controlled
            parameter overrides.
          </p>
        </div>

        <div style={styles.headerControls}>
          <div style={styles.runSelectorBox}>
            <span style={styles.selectorLabel}>Original Run:</span>
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
            <span>Trace</span>
          </button>
        </div>
      </div>

      {loading ? (
        <div style={styles.loadingMessage}>Loading checkpoints...</div>
      ) : checkpoints.length === 0 ? (
        <div style={styles.loadingMessage}>
          No checkpoints found for this run. Run the agent to generate auto-checkpoints.
        </div>
      ) : (
        <div style={styles.content}>
          {/* Main 2-column layout */}
          <div style={styles.gridTwoCol}>
            {/* Left Col: Checkpoint Selector */}
            <div style={styles.card}>
              <div style={styles.cardHeader}>
                <div style={{ display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
                  <GitBranch size={15} color="#7C3AED" />
                  <h3 style={styles.cardTitle}>Available Checkpoints</h3>
                </div>
                <span className="code-pill">{checkpoints.length} saved states</span>
              </div>

              <div style={styles.checkpointsList}>
                {checkpoints.map((cp) => {
                  const isSelected = cp.checkpoint_id === selectedCheckpointId;
                  return (
                    <div
                      key={cp.checkpoint_id}
                      onClick={() => setSelectedCheckpointId(cp.checkpoint_id)}
                      style={{
                        ...styles.checkpointItem,
                        backgroundColor: isSelected ? 'var(--bg-active)' : 'var(--bg-secondary)',
                        borderColor: isSelected ? 'var(--status-replay)' : 'var(--border-color)',
                      }}
                    >
                      <div style={styles.cpItemHeader}>
                        <span style={styles.cpStepBadge}>Step {cp.step_id}</span>
                        <span style={styles.cpStageName}>{cp.stage_name}</span>
                      </div>
                      <div style={styles.cpIdText}>{cp.checkpoint_id}</div>
                    </div>
                  );
                })}
              </div>

              {/* State Snapshot Preview */}
              {selectedCheckpoint && (
                <div style={styles.snapshotBox}>
                  <div style={styles.snapshotHeading}>
                    <Clock size={12} />
                    <span>RESTORED OBSERVABLE STATE SNAPSHOT</span>
                  </div>
                  <pre style={styles.snapshotPre}>
                    {JSON.stringify(selectedCheckpoint.state_snapshot, null, 2)}
                  </pre>
                </div>
              )}
            </div>

            {/* Right Col: Controlled Overrides & Replay Launcher */}
            <div style={styles.card}>
              <div style={styles.cardHeader}>
                <div style={{ display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
                  <Sliders size={15} color="#2563EB" />
                  <h3 style={styles.cardTitle}>Controlled Condition Overrides</h3>
                </div>
                <span className="badge badge-replay">Non-destructive</span>
              </div>

              <div style={styles.formContainer}>
                <p style={styles.formNotice}>
                  Modify exactly one condition to test counterfactual execution and verify
                  recovery. The original trace remains completely unchanged.
                </p>

                {/* Failure Mode Override */}
                <div style={styles.formGroup}>
                  <label style={styles.label}>Controlled Failure Mode</label>
                  <select
                    value={overrideFailureMode}
                    onChange={(e) => setOverrideFailureMode(e.target.value)}
                    style={styles.inputField}
                  >
                    <option value="none">none (Clear failure / Normal execution)</option>
                    <option value="wrong_tool">wrong_tool (Inject wrong tool)</option>
                    <option value="budget_overflow">budget_overflow (Inject overbudget)</option>
                    <option value="invalid_output">invalid_output (Inject spec mismatch)</option>
                  </select>
                  <span style={styles.helpText}>
                    Select "none" to repair the failure and verify recovery.
                  </span>
                </div>

                {/* Budget Constraint Override */}
                <div style={styles.formGroup}>
                  <label style={styles.label}>Override Max Budget (₹)</label>
                  <input
                    type="number"
                    placeholder="e.g. 90000"
                    value={overrideBudget}
                    onChange={(e) => setOverrideBudget(e.target.value)}
                    style={styles.inputField}
                  />
                  <span style={styles.helpText}>
                    Optional. Re-evaluates products under a relaxed or tighter budget.
                  </span>
                </div>

                {/* Force Tool Override */}
                <div style={styles.formGroup}>
                  <label style={styles.label}>Force Specific Tool Call</label>
                  <input
                    type="text"
                    placeholder="e.g. search_products"
                    value={overrideTool}
                    onChange={(e) => setOverrideTool(e.target.value)}
                    style={styles.inputField}
                  />
                  <span style={styles.helpText}>
                    Overrides the agent's first tool dispatch decision.
                  </span>
                </div>

                {/* Execute Button */}
                <button
                  onClick={handleExecuteReplay}
                  disabled={isReplaying}
                  style={styles.executeButton}
                >
                  <Play size={15} fill="#FFFFFF" />
                  <span>
                    {isReplaying ? 'Executing Replay Branch...' : 'Launch Alternative Execution'}
                  </span>
                </button>
              </div>

              {/* Replay Result Banner */}
              {replayResult && (
                <div style={styles.resultBanner}>
                  <div style={styles.resultHeader}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
                      {replayResult.status === 'success' ? (
                        <CheckCircle2 size={16} color="#16A34A" />
                      ) : (
                        <XCircle size={16} color="#DC2626" />
                      )}
                      <span style={{ fontWeight: 600 }}>
                        Replay Outcome: {replayResult.status.toUpperCase()}
                      </span>
                    </div>
                    <span className="badge badge-replay">
                      Spawned: {replayResult.replay_run_id}
                    </span>
                  </div>

                  <p style={styles.resultText}>
                    {replayResult.final_response || 'Alternative execution finished successfully.'}
                  </p>

                  <button
                    onClick={() => {
                      if (selectedRunId) {
                        onSelectReplayRun(selectedRunId, replayResult.replay_run_id);
                        onNavigateToView('comparison');
                      }
                    }}
                    style={styles.compareCtaBtn}
                  >
                    <GitCompare size={14} />
                    <span>Compare Original vs Replay Execution</span>
                    <ArrowRight size={13} />
                  </button>
                </div>
              )}
            </div>
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
  checkpointsList: {
    display: 'flex',
    flexDirection: 'column',
    gap: '0.5rem',
    maxHeight: '260px',
    overflowY: 'auto',
  },
  checkpointItem: {
    padding: '0.65rem 0.85rem',
    borderRadius: '6px',
    border: '1px solid var(--border-color)',
    cursor: 'pointer',
    display: 'flex',
    flexDirection: 'column',
    gap: '0.2rem',
    transition: 'all 0.1s ease',
  },
  cpItemHeader: {
    display: 'flex',
    alignItems: 'center',
    gap: '0.5rem',
  },
  cpStepBadge: {
    fontSize: '0.72rem',
    fontWeight: 700,
    padding: '0.1rem 0.4rem',
    borderRadius: '4px',
    backgroundColor: '#EDE9FE',
    color: '#7C3AED',
  },
  cpStageName: {
    fontSize: '0.82rem',
    fontWeight: 600,
    color: 'var(--text-primary)',
  },
  cpIdText: {
    fontSize: '0.72rem',
    color: 'var(--text-muted)',
    fontFamily: 'var(--font-mono)',
  },
  snapshotBox: {
    marginTop: '0.5rem',
    paddingTop: '0.75rem',
    borderTop: '1px solid var(--border-color)',
    display: 'flex',
    flexDirection: 'column',
    gap: '0.35rem',
  },
  snapshotHeading: {
    display: 'flex',
    alignItems: 'center',
    gap: '0.35rem',
    fontSize: '0.68rem',
    fontWeight: 600,
    color: 'var(--text-muted)',
    letterSpacing: '0.04em',
  },
  snapshotPre: {
    fontSize: '0.75rem',
    padding: '0.6rem 0.75rem',
    backgroundColor: 'var(--bg-secondary)',
    border: '1px solid var(--border-color)',
    borderRadius: '6px',
    maxHeight: '180px',
    overflowY: 'auto',
    whiteSpace: 'pre-wrap',
    wordBreak: 'break-all',
  },
  formContainer: {
    display: 'flex',
    flexDirection: 'column',
    gap: '1rem',
  },
  formNotice: {
    fontSize: '0.8rem',
    color: 'var(--text-secondary)',
    lineHeight: 1.4,
  },
  formGroup: {
    display: 'flex',
    flexDirection: 'column',
    gap: '0.25rem',
  },
  label: {
    fontSize: '0.78rem',
    fontWeight: 600,
    color: 'var(--text-primary)',
  },
  inputField: {
    padding: '0.5rem 0.75rem',
    backgroundColor: 'var(--bg-secondary)',
    border: '1px solid var(--border-color)',
    borderRadius: '6px',
    outline: 'none',
    fontSize: '0.84rem',
  },
  helpText: {
    fontSize: '0.72rem',
    color: 'var(--text-muted)',
  },
  executeButton: {
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    gap: '0.5rem',
    backgroundColor: '#7C3AED',
    color: '#FFFFFF',
    padding: '0.65rem 1rem',
    borderRadius: '6px',
    fontWeight: 600,
    fontSize: '0.86rem',
    marginTop: '0.5rem',
  },
  resultBanner: {
    marginTop: '1rem',
    padding: '1rem',
    backgroundColor: 'var(--bg-secondary)',
    border: '1px solid var(--border-color)',
    borderRadius: '8px',
    display: 'flex',
    flexDirection: 'column',
    gap: '0.75rem',
  },
  resultHeader: {
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'space-between',
    fontSize: '0.84rem',
  },
  resultText: {
    fontSize: '0.84rem',
    lineHeight: 1.45,
    color: 'var(--text-primary)',
  },
  compareCtaBtn: {
    display: 'flex',
    alignItems: 'center',
    gap: '0.4rem',
    backgroundColor: '#2563EB',
    color: '#FFFFFF',
    padding: '0.5rem 0.85rem',
    borderRadius: '6px',
    fontSize: '0.82rem',
    fontWeight: 600,
    alignSelf: 'flex-start',
  },
};
