import {
  RunSummary,
  RunTrace,
  DiagnosisReport,
  CheckpointItem,
  TraceComparison,
  EvaluationSummary,
  Product,
  SystemStatus,
  AgentRunResponse,
} from './types';

const API_BASE = (import.meta.env.VITE_API_BASE_URL as string) || 'http://localhost:8000';

async function request<T>(endpoint: string, options: RequestInit = {}): Promise<T> {
  const url = `${API_BASE}${endpoint}`;
  const headers = {
    'Content-Type': 'application/json',
    ...(options.headers || {}),
  };

  const response = await fetch(url, { ...options, headers });
  if (!response.ok) {
    let errorDetail = `API error ${response.status}: ${response.statusText}`;
    try {
      const errJson = await response.json();
      if (errJson.detail) errorDetail = errJson.detail;
    } catch {
      // ignore
    }
    throw new Error(errorDetail);
  }
  return response.json() as Promise<T>;
}

export const api = {
  getSystemStatus: () => request<SystemStatus>('/system/status'),

  getRuns: (limit = 50, offset = 0) =>
    request<RunSummary[]>(`/runs?limit=${limit}&offset=${offset}`),

  getRun: (runId: string) => request<RunTrace>(`/runs/${runId}`),

  runAgent: (req: {
    request: string;
    run_id?: string;
    failure_mode?: string;
    history?: Array<{ role: string; content: string }>;
    conversation_id?: string;
  }) =>
    request<AgentRunResponse>('/agent/run', {
      method: 'POST',
      body: JSON.stringify(req),
    }),

  getDiagnosis: (runId: string) =>
    request<DiagnosisReport>(`/runs/${runId}/diagnosis`),

  getEvidence: (runId: string) =>
    request<any>(`/runs/${runId}/evidence`),

  getCheckpoints: (runId: string) =>
    request<{ run_id: string; count: number; checkpoints: CheckpointItem[] }>(
      `/runs/${runId}/checkpoints`
    ),

  replayRun: (
    runId: string,
    checkpointId: string,
    override?: { max_budget?: number; failure_mode?: string; tool_name?: string }
  ) =>
    request<{
      original_run_id: string;
      replay_run_id: string;
      checkpoint_id: string;
      checkpoint_step_id: number;
      replay_type: string;
      status: string;
      final_response: string;
      override_applied: any;
    }>(`/runs/${runId}/replay`, {
      method: 'POST',
      body: JSON.stringify({ checkpoint_id: checkpointId, override }),
    }),

  compareTraces: (runId: string, alternativeRunId: string, referenceRunId?: string) => {
    let url = `/runs/${runId}/compare/${alternativeRunId}`;
    if (referenceRunId) url += `?reference_run_id=${encodeURIComponent(referenceRunId)}`;
    return request<TraceComparison>(url);
  },

  getEvaluationSummary: () => request<EvaluationSummary>('/evaluation/summary'),
  getEvaluationLocalization: () => request<any>('/evaluation/localization'),
  getEvaluationByCategory: () => request<any>('/evaluation/by-category'),
  getEvaluationReplay: () => request<any>('/evaluation/replay'),

  runBenchmark: () =>
    request<any>('/evaluation/run-benchmark', {
      method: 'POST',
      body: JSON.stringify({ target_success: 15, target_failure: 30, force_regenerate: false }),
    }),

  getCatalogue: () => request<Product[]>('/catalogue'),

  getConversationMessages: (conversationId: string) =>
    request<{ conversation_id: string; count: number; messages: Array<{ id: number; role: string; content: string; timestamp: string }> }>(
      `/conversations/${conversationId}/messages`
    ),

  clearConversation: (conversationId: string) =>
    request<{ status: string; message: string }>(`/conversations/${conversationId}`, {
      method: 'DELETE',
    }),
};
