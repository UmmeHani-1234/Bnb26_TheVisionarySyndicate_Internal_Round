import React, { useState, useEffect } from 'react';
import {
  Search,
  RotateCw,
  Layers,
  Activity,
  RotateCcw,
  CheckCircle2,
  XCircle,
  GitBranch,
} from 'lucide-react';
import { api } from '../api';
import { RunSummary } from '../types';
import { NavView } from './Sidebar';

interface RunsViewProps {
  onSelectRun: (runId: string, view: NavView) => void;
}

export const RunsView: React.FC<RunsViewProps> = ({ onSelectRun }) => {
  const [runs, setRuns] = useState<RunSummary[]>([]);
  const [filteredRuns, setFilteredRuns] = useState<RunSummary[]>([]);
  const [filterType, setFilterType] = useState<'all' | 'success' | 'failed' | 'replay'>('all');
  const [searchQuery, setSearchQuery] = useState('');
  const [loading, setLoading] = useState(false);

  const fetchRuns = async () => {
    setLoading(true);
    try {
      const data = await api.getRuns(100);
      setRuns(data);
    } catch (err) {
      console.error('Failed to load runs:', err);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchRuns();
  }, []);

  useEffect(() => {
    let result = [...runs];

    if (filterType === 'success') {
      result = result.filter((r) => r.status === 'success');
    } else if (filterType === 'failed') {
      result = result.filter((r) => r.status === 'failed' || r.status === 'error');
    } else if (filterType === 'replay') {
      result = result.filter((r) => r.is_replay || r.run_id.includes('replay'));
    }

    if (searchQuery.trim()) {
      const q = searchQuery.toLowerCase();
      result = result.filter(
        (r) =>
          r.run_id.toLowerCase().includes(q) ||
          (r.user_request && r.user_request.toLowerCase().includes(q))
      );
    }

    setFilteredRuns(result);
  }, [runs, filterType, searchQuery]);

  return (
    <div style={styles.container}>
      {/* Header */}
      <div style={styles.header}>
        <div>
          <h1 style={styles.title}>Agent Execution Runs</h1>
          <p style={styles.subtitle}>
            Inspect all recorded agent executions, failure logs, and alternative replay branches.
          </p>
        </div>
        <button onClick={fetchRuns} style={styles.refreshButton} title="Refresh runs">
          <RotateCw size={14} className={loading ? 'spin-animation' : ''} />
          <span>Refresh</span>
        </button>
      </div>

      {/* Filter and Search Bar */}
      <div style={styles.filterBar}>
        <div style={styles.filterChips}>
          {(['all', 'success', 'failed', 'replay'] as const).map((ft) => (
            <button
              key={ft}
              onClick={() => setFilterType(ft)}
              style={{
                ...styles.chip,
                backgroundColor: filterType === ft ? 'var(--bg-active)' : 'transparent',
                color: filterType === ft ? 'var(--text-primary)' : 'var(--text-muted)',
                fontWeight: filterType === ft ? 600 : 400,
              }}
            >
              {ft === 'all' && `All (${runs.length})`}
              {ft === 'success' && `Successful (${runs.filter((r) => r.status === 'success').length})`}
              {ft === 'failed' && `Failed (${runs.filter((r) => r.status === 'failed' || r.status === 'error').length})`}
              {ft === 'replay' && `Replays (${runs.filter((r) => r.is_replay || r.run_id.includes('replay')).length})`}
            </button>
          ))}
        </div>

        <div style={styles.searchBox}>
          <Search size={14} color="var(--text-muted)" />
          <input
            type="text"
            placeholder="Search by query or Run ID..."
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            style={styles.searchInput}
          />
        </div>
      </div>

      {/* Runs Table */}
      <div style={styles.tableCard}>
        {loading ? (
          <div style={styles.centerMessage}>Loading runs...</div>
        ) : filteredRuns.length === 0 ? (
          <div style={styles.centerMessage}>No matching runs found.</div>
        ) : (
          <table style={styles.table}>
            <thead>
              <tr style={styles.tableHeadRow}>
                <th style={{ ...styles.th, width: '170px' }}>Run ID</th>
                <th style={styles.th}>User Query</th>
                <th style={{ ...styles.th, width: '110px' }}>Status</th>
                <th style={{ ...styles.th, width: '130px' }}>Type / Mode</th>
                <th style={{ ...styles.th, width: '150px' }}>Timestamp</th>
                <th style={{ ...styles.th, width: '180px', textAlign: 'right' }}>Actions</th>
              </tr>
            </thead>
            <tbody>
              {filteredRuns.map((run) => {
                const isFail = run.status === 'failed' || run.status === 'error';
                const isReplay = run.is_replay || run.run_id.includes('replay');
                return (
                  <tr key={run.run_id} style={styles.tableRow}>
                    <td style={styles.td}>
                      <span className="code-pill">{run.run_id}</span>
                    </td>
                    <td style={styles.td}>
                      <span style={styles.queryText} title={run.user_request}>
                        {run.user_request || '(Empty query)'}
                      </span>
                    </td>
                    <td style={styles.td}>
                      {isFail ? (
                        <span className="badge badge-error">
                          <XCircle size={12} />
                          Failed
                        </span>
                      ) : (
                        <span className="badge badge-success">
                          <CheckCircle2 size={12} />
                          Success
                        </span>
                      )}
                    </td>
                    <td style={styles.td}>
                      {isReplay ? (
                        <span className="badge badge-replay">
                          <GitBranch size={11} />
                          Replay Branch
                        </span>
                      ) : run.failure_mode ? (
                        <span className="badge badge-warning">{run.failure_mode}</span>
                      ) : (
                        <span style={{ fontSize: '0.78rem', color: 'var(--text-muted)' }}>
                          Standard
                        </span>
                      )}
                    </td>
                    <td style={styles.td}>
                      <span style={styles.timestampText}>
                        {run.created_at ? new Date(run.created_at).toLocaleString() : 'Just now'}
                      </span>
                    </td>
                    <td style={{ ...styles.td, textAlign: 'right' }}>
                      <div style={styles.actionButtonGroup}>
                        <button
                          onClick={() => onSelectRun(run.run_id, 'trace')}
                          style={styles.actionBtn}
                          title="Inspect Execution Trace"
                        >
                          <Layers size={13} />
                          <span>Trace</span>
                        </button>
                        {isFail && (
                          <button
                            onClick={() => onSelectRun(run.run_id, 'diagnosis')}
                            style={{ ...styles.actionBtn, color: '#DC2626' }}
                            title="Diagnose Failure"
                          >
                            <Activity size={13} />
                            <span>Diagnose</span>
                          </button>
                        )}
                        <button
                          onClick={() => onSelectRun(run.run_id, 'replay')}
                          style={{ ...styles.actionBtn, color: '#7C3AED' }}
                          title="Replay from Checkpoint"
                        >
                          <RotateCcw size={13} />
                          <span>Replay</span>
                        </button>
                      </div>
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        )}
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
  refreshButton: {
    display: 'flex',
    alignItems: 'center',
    gap: '0.4rem',
    backgroundColor: 'var(--bg-secondary)',
    border: '1px solid var(--border-color)',
    borderRadius: '6px',
    padding: '0.45rem 0.75rem',
    fontSize: '0.82rem',
    fontWeight: 500,
    color: 'var(--text-primary)',
  },
  filterBar: {
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'space-between',
    gap: '1rem',
  },
  filterChips: {
    display: 'flex',
    gap: '0.35rem',
  },
  chip: {
    padding: '0.35rem 0.75rem',
    borderRadius: '6px',
    fontSize: '0.8rem',
    border: '1px solid var(--border-color)',
  },
  searchBox: {
    display: 'flex',
    alignItems: 'center',
    gap: '0.45rem',
    backgroundColor: 'var(--bg-secondary)',
    border: '1px solid var(--border-color)',
    borderRadius: '6px',
    padding: '0.35rem 0.65rem',
    width: '280px',
  },
  searchInput: {
    border: 'none',
    backgroundColor: 'transparent',
    outline: 'none',
    width: '100%',
    fontSize: '0.82rem',
  },
  tableCard: {
    backgroundColor: 'var(--bg-primary)',
    border: '1px solid var(--border-color)',
    borderRadius: '8px',
    overflow: 'hidden',
    boxShadow: '0 1px 2px rgba(0,0,0,0.02)',
  },
  table: {
    width: '100%',
    borderCollapse: 'collapse',
    textAlign: 'left',
  },
  tableHeadRow: {
    backgroundColor: 'var(--bg-secondary)',
    borderBottom: '1px solid var(--border-color)',
  },
  th: {
    padding: '0.7rem 1rem',
    fontSize: '0.74rem',
    fontWeight: 600,
    textTransform: 'uppercase',
    letterSpacing: '0.04em',
    color: 'var(--text-muted)',
  },
  tableRow: {
    borderBottom: '1px solid var(--border-color)',
  },
  td: {
    padding: '0.75rem 1rem',
    fontSize: '0.84rem',
    verticalAlign: 'middle',
  },
  queryText: {
    display: 'inline-block',
    maxWidth: '340px',
    whiteSpace: 'nowrap',
    overflow: 'hidden',
    textOverflow: 'ellipsis',
    color: 'var(--text-primary)',
    fontWeight: 500,
  },
  timestampText: {
    fontSize: '0.78rem',
    color: 'var(--text-muted)',
  },
  actionButtonGroup: {
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'flex-end',
    gap: '0.35rem',
  },
  actionBtn: {
    display: 'inline-flex',
    alignItems: 'center',
    gap: '0.25rem',
    padding: '0.25rem 0.5rem',
    borderRadius: '4px',
    backgroundColor: 'var(--bg-secondary)',
    border: '1px solid var(--border-color)',
    fontSize: '0.74rem',
    fontWeight: 500,
    color: 'var(--text-secondary)',
  },
  centerMessage: {
    padding: '3rem',
    textAlign: 'center',
    color: 'var(--text-muted)',
    fontSize: '0.9rem',
  },
};
