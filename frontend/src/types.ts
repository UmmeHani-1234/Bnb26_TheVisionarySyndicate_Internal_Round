export interface AgentStep {
  step_id?: number | string;
  step_number?: number;
  event_id?: string;
  step_type?: string;
  stage?: string;
  tool_name?: string | null;
  input?: any;
  output?: any;
  status: 'success' | 'failed' | 'error' | 'warning' | string;
  latency?: number;
  duration?: number;
  timestamp?: string;
  state_snapshot?: Record<string, any>;
  metadata?: Record<string, any>;
}

export interface RunTrace {
  run_id: string;
  user_request: string;
  status: 'success' | 'failed' | 'running' | string;
  created_at?: string;
  completed_at?: string;
  total_duration?: number;
  steps: AgentStep[];
  failure_mode?: string | null;
  is_replay?: boolean;
  original_run_id?: string | null;
  checkpoint_id?: string | null;
}

export interface RunSummary {
  run_id: string;
  user_request: string;
  status: string;
  created_at: string;
  step_count?: number;
  duration?: number;
  is_replay?: boolean;
  failure_mode?: string | null;
}

export interface DiagnosticSignal {
  name: string;
  score: number;
  weight?: number;
  details?: string;
}

export interface StepSuspicion {
  step_number: number;
  stage_name: string;
  suspicion_score: number;
  tool_name?: string | null;
  signals?: Record<string, number>;
  primary_reasons?: string[];
}

export interface DiagnosisReport {
  run_id: string;
  status: string;
  is_failure: boolean;
  top_suspicious_step?: number;
  top_suspicious_stage?: string;
  confidence?: number;
  top_1_step?: number;
  top_3_steps?: number[];
  ranked_steps?: StepSuspicion[];
  signals_summary?: Record<string, number>;
  evidence?: {
    observed_state?: any;
    expected_state?: any;
    discrepancy?: string;
    root_cause_explanation?: string;
  };
  recommendation?: string;
}

export interface CheckpointItem {
  checkpoint_id: string;
  run_id: string;
  step_id: number;
  stage_name: string;
  created_at: string;
  state_snapshot: Record<string, any>;
}

export interface TraceComparison {
  status: string;
  original_run_id: string;
  alternative_run_id: string;
  reference_run_id?: string;
  verification?: {
    status: 'RECOVERED' | 'NOT_RECOVERED' | 'UNKNOWN';
    is_recovered: boolean;
    confidence: number;
    verdict_reason: string;
  };
  alignment?: Array<{
    step_number: number;
    stage_name: string;
    original_step?: AgentStep;
    alternative_step?: AgentStep;
    reference_step?: AgentStep;
    has_diverged: boolean;
    difference_summary?: string;
  }>;
}

export interface EvaluationSummary {
  status: string;
  dataset_size?: number;
  summary?: {
    random_baseline?: { top_1: number; top_3: number; mrr: number };
    rule_based?: { top_1: number; top_3: number; mrr: number };
    random_forest?: { top_1: number; top_3: number; mrr: number };
  };
  category_breakdown?: Record<string, { top_1: number; top_3: number; count: number }>;
  replay_stats?: {
    branches_attempted: number;
    recovered: number;
    not_recovered: number;
    recovery_rate: number;
  };
}

export interface Product {
  id?: string | number;
  name: string;
  brand?: string;
  price: number;
  processor: string;
  ram: string | number;
  storage: string;
  gpu?: string;
  display?: string;
  rating?: number;
  category?: string;
  image_url?: string;
  description?: string;
}

export interface SystemStatus {
  status: string;
  backend: string;
  database: string;
  agent: string;
  version: string;
}
