import React, { useState, useRef, useEffect } from 'react';
import {
  Send,
  Sparkles,
  CheckCircle2,
  AlertTriangle,
  ArrowRight,
  Loader2,
  Bot,
  User,
  Sliders,
  Laptop as LaptopIcon,
} from 'lucide-react';
import { api } from '../api';
import { RunTrace, Product } from '../types';
import { NavView } from './Sidebar';

interface ChatMessage {
  id: string;
  sender: 'user' | 'agent';
  text: string;
  timestamp: string;
  runId?: string;
  trace?: RunTrace;
  status?: string;
  duration?: number;
  products?: Product[];
  failedStep?: {
    step_number: number;
    stage_name: string;
  };
}

interface ChatViewProps {
  onNavigateToRun: (runId: string, view: NavView) => void;
  catalogue: Product[];
}

export const ChatView: React.FC<ChatViewProps> = ({ onNavigateToRun, catalogue }) => {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [inputText, setInputText] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [failureMode, setFailureMode] = useState<string>('none');
  const [liveStep, setLiveStep] = useState<string | null>(null);

  const messagesEndRef = useRef<HTMLDivElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages, liveStep]);

  // Extract products mentioned in agent text
  const matchProductsFromText = (text: string): Product[] => {
    if (!text || catalogue.length === 0) return [];
    const matched: Product[] = [];
    const lowerText = text.toLowerCase();

    for (const prod of catalogue) {
      if (lowerText.includes(prod.name.toLowerCase())) {
        matched.push(prod);
      }
    }
    // Limit to top 4 matched cards
    return matched.slice(0, 4);
  };

  const handleSend = async (customText?: string) => {
    const textToSend = customText || inputText;
    if (!textToSend.trim() || isLoading) return;

    const userMsgId = `msg-${Date.now()}`;
    const userMsg: ChatMessage = {
      id: userMsgId,
      sender: 'user',
      text: textToSend.trim(),
      timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
    };

    setMessages((prev) => [...prev, userMsg]);
    setInputText('');
    setIsLoading(true);
    setLiveStep('Step 1: Input Received');

    // Simulate observable step progression before API responds
    const stepTimer1 = setTimeout(() => setLiveStep('Step 2: Parsing Intent & Budget'), 300);
    const stepTimer2 = setTimeout(() => setLiveStep('Step 3: Extracting Specifications'), 700);
    const stepTimer3 = setTimeout(() => setLiveStep('Step 4: Tool Execution & Verification'), 1100);

    try {
      const modeParam = failureMode === 'none' ? undefined : failureMode;
      const res = await api.runAgent({
        request: textToSend.trim(),
        failure_mode: modeParam,
      });

      clearTimeout(stepTimer1);
      clearTimeout(stepTimer2);
      clearTimeout(stepTimer3);
      setLiveStep(null);

      // Find failed step if failed
      let failedStepInfo: { step_number: number; stage_name: string } | undefined;
      if (res.status === 'failed' && res.trace?.steps) {
        const failed = res.trace.steps.find((s) => s.status === 'failed' || s.status === 'error');
        if (failed) {
          failedStepInfo = {
            step_number: failed.step_number || 4,
            stage_name: failed.stage || failed.step_type || 'Tool Execution',
          };
        }
      }

      // Check for products in final response or extracted state
      let foundProducts: Product[] = matchProductsFromText(res.final_response);

      // If text didn't match exact name, check trace steps
      if (foundProducts.length === 0 && res.trace?.steps) {
        for (const step of res.trace.steps) {
          if (step.output && typeof step.output === 'object') {
            if (Array.isArray(step.output.products)) {
              foundProducts = step.output.products.slice(0, 4);
              break;
            }
          }
        }
      }

      const agentMsg: ChatMessage = {
        id: `agent-${Date.now()}`,
        sender: 'agent',
        text: res.final_response || '(No response received)',
        timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
        runId: res.run_id,
        trace: res.trace,
        status: res.status,
        duration: res.trace?.total_duration || 1.8,
        products: foundProducts,
        failedStep: failedStepInfo,
      };

      setMessages((prev) => [...prev, agentMsg]);
    } catch (err: any) {
      clearTimeout(stepTimer1);
      clearTimeout(stepTimer2);
      clearTimeout(stepTimer3);
      setLiveStep(null);

      const errorMsg: ChatMessage = {
        id: `err-${Date.now()}`,
        sender: 'agent',
        text: `Error contacting backend: ${err.message || 'Unknown network error'}. Please verify the FastAPI backend is running on port 8000.`,
        timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
        status: 'failed',
      };
      setMessages((prev) => [...prev, errorMsg]);
    } finally {
      setIsLoading(false);
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  const samplePrompts = [
    {
      title: 'Programming Laptop',
      desc: 'Under ₹80,000 with 16GB RAM',
      query: 'I need a laptop for programming under ₹80,000 with at least 16GB RAM.',
      mode: 'none',
    },
    {
      title: 'Budget Student',
      desc: 'Lightweight under ₹45,000',
      query: 'Recommend a good student laptop under ₹45,000 for coursework.',
      mode: 'none',
    },
    {
      title: 'Gaming & Performance',
      desc: 'Dedicated GPU under ₹1,20,000',
      query: 'Find a gaming laptop with high-end dedicated GPU under ₹1,20,000.',
      mode: 'none',
    },
    {
      title: 'Controlled Failure Test',
      desc: 'Injected tool error for debugging',
      query: 'Search for laptops under ₹70,000 for coding.',
      mode: 'wrong_tool',
    },
  ];

  return (
    <div style={styles.container}>
      {/* Messages Scroll Area */}
      <div style={styles.messagesContainer}>
        {messages.length === 0 ? (
          /* Empty / Welcome State */
          <div style={styles.welcomeContainer}>
            <div style={styles.heroBox}>
              <h1 style={styles.heroTitle}>Black Box</h1>
              <p style={styles.heroSubtitle}>
                Debug your AI agent by watching what it actually does.
              </p>
            </div>

            {/* Prompt Cards Grid */}
            <div style={styles.promptsGrid}>
              {samplePrompts.map((p, idx) => (
                <button
                  key={idx}
                  style={styles.promptCard}
                  onClick={() => {
                    setFailureMode(p.mode);
                    handleSend(p.query);
                  }}
                >
                  <div style={styles.promptCardHeader}>
                    <span style={styles.promptCardTitle}>{p.title}</span>
                    <Sparkles size={14} color="#2563EB" />
                  </div>
                  <div style={styles.promptCardDesc}>{p.desc}</div>
                </button>
              ))}
            </div>
          </div>
        ) : (
          <div style={styles.chatList}>
            {messages.map((msg) => (
              <div
                key={msg.id}
                style={{
                  ...styles.messageRow,
                  justifyContent: msg.sender === 'user' ? 'flex-end' : 'flex-start',
                }}
              >
                <div
                  style={{
                    ...styles.bubble,
                    backgroundColor:
                      msg.sender === 'user' ? 'var(--bg-tertiary)' : 'var(--bg-primary)',
                    border:
                      msg.sender === 'user'
                        ? '1px solid var(--border-color)'
                        : '1px solid var(--border-light)',
                    maxWidth: msg.sender === 'user' ? '70%' : '88%',
                  }}
                >
                  <div style={styles.messageHeader}>
                    <div style={styles.senderLabel}>
                      {msg.sender === 'user' ? (
                        <>
                          <User size={14} />
                          <span>You</span>
                        </>
                      ) : (
                        <>
                          <Bot size={14} color="#2563EB" />
                          <span>Black Box Agent</span>
                        </>
                      )}
                    </div>
                    <span style={styles.timestamp}>{msg.timestamp}</span>
                  </div>

                  <div style={styles.messageBody}>{msg.text}</div>

                  {/* Render Product Cards if available */}
                  {msg.products && msg.products.length > 0 && (
                    <div style={styles.productsContainer}>
                      <div style={styles.productsGrid}>
                        {msg.products.map((prod, idx) => (
                          <div key={idx} style={styles.productCard}>
                            <div style={styles.productImageWrapper}>
                              {prod.image_url ? (
                                <img
                                  src={prod.image_url}
                                  alt={prod.name}
                                  style={styles.productImage}
                                  onError={(e) => {
                                    // Fallback to placeholder box
                                    e.currentTarget.style.display = 'none';
                                    e.currentTarget.parentElement!.innerHTML =
                                      '<div style="display:flex;align-items:center;justify-content:center;height:100%;color:#94A3B8;"><svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><rect x="2" y="3" width="20" height="14" rx="2"/><line x1="2" y1="20" x2="22" y2="20"/></svg></div>';
                                  }}
                                />
                              ) : (
                                <div style={styles.productImagePlaceholder}>
                                  <LaptopIcon size={32} color="#94A3B8" />
                                </div>
                              )}
                            </div>
                            <div style={styles.productInfo}>
                              <div style={styles.productName} title={prod.name}>
                                {prod.name}
                              </div>
                              <div style={styles.productPrice}>
                                ₹{prod.price?.toLocaleString('en-IN')}
                              </div>
                              <div style={styles.productSpecs}>
                                <span>{prod.ram ? `${prod.ram} RAM` : '16GB RAM'}</span>
                                <span>·</span>
                                <span>{prod.storage || '512GB SSD'}</span>
                              </div>
                              <div style={styles.productSubSpecs}>
                                {prod.processor} {prod.gpu ? `· ${prod.gpu}` : ''}
                              </div>
                              <div style={styles.productBadge}>✓ Within budget</div>
                            </div>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}

                  {/* Execution Connection Status Bridge */}
                  {msg.runId && (
                    <div style={styles.executionBridge}>
                      {msg.status === 'success' ? (
                        <div style={styles.executionStatusSuccess}>
                          <CheckCircle2 size={15} color="#16A34A" />
                          <span>
                            Completed · {msg.trace?.steps?.length || 8} steps ·{' '}
                            {msg.duration ? `${msg.duration.toFixed(1)}s` : '1.8s'}
                          </span>
                          <button
                            onClick={() => onNavigateToRun(msg.runId!, 'trace')}
                            style={styles.inspectButton}
                          >
                            <span>Inspect trace</span>
                            <ArrowRight size={13} />
                          </button>
                        </div>
                      ) : (
                        <div style={styles.executionStatusFailed}>
                          <div style={{ display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
                            <AlertTriangle size={15} color="#DC2626" />
                            <span style={{ fontWeight: 600, color: '#DC2626' }}>
                              Execution failed · {msg.trace?.steps?.length || 8} steps
                            </span>
                          </div>
                          {msg.failedStep && (
                            <div style={styles.failedStepNotice}>
                              Potential issue detected: Step {msg.failedStep.step_number} ·{' '}
                              {msg.failedStep.stage_name}
                            </div>
                          )}
                          <div style={{ display: 'flex', gap: '0.5rem', marginTop: '0.35rem' }}>
                            <button
                              onClick={() => onNavigateToRun(msg.runId!, 'diagnosis')}
                              style={styles.diagnoseButton}
                            >
                              <span>Diagnose Root Cause</span>
                              <ArrowRight size={13} />
                            </button>
                            <button
                              onClick={() => onNavigateToRun(msg.runId!, 'trace')}
                              style={styles.inspectButtonSecondary}
                            >
                              View trace
                            </button>
                          </div>
                        </div>
                      )}
                    </div>
                  )}
                </div>
              </div>
            ))}

            {/* Live Agent Execution Indicator */}
            {isLoading && liveStep && (
              <div style={{ ...styles.messageRow, justifyContent: 'flex-start' }}>
                <div style={styles.liveExecutionBox}>
                  <Loader2 size={16} className="spin-animation" color="#2563EB" />
                  <span style={styles.liveStepText}>{liveStep}</span>
                </div>
              </div>
            )}

            <div ref={messagesEndRef} />
          </div>
        )}
      </div>

      {/* Input Area */}
      <div style={styles.inputSection}>
        {/* Controlled Failure Injection Drawer / Toggle */}
        <div style={styles.controlBar}>
          <div style={styles.controlBarLeft}>
            <Sliders size={13} color="var(--text-muted)" />
            <span style={styles.controlBarLabel}>Controlled Execution Mode:</span>
            <select
              value={failureMode}
              onChange={(e) => setFailureMode(e.target.value)}
              style={styles.modeSelect}
            >
              <option value="none">Normal Execution (No failure)</option>
              <option value="wrong_tool">Inject: Wrong Tool Selection (Step 4)</option>
              <option value="budget_overflow">Inject: Budget Constraint Violation (Step 5)</option>
              <option value="invalid_output">Inject: Specification Mismatch (Step 6)</option>
              <option value="timeout">Inject: Latency / Timeout Error (Step 3)</option>
            </select>
          </div>
          {failureMode !== 'none' && (
            <span style={styles.failureActiveWarning}>Controlled Failure Injection Active</span>
          )}
        </div>

        {/* Text Input Container */}
        <div style={styles.inputContainer}>
          <textarea
            ref={textareaRef}
            rows={2}
            value={inputText}
            onChange={(e) => setInputText(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="Ask the agent anything... (e.g. 'I need a laptop for programming under ₹80,000 with 16GB RAM')"
            style={styles.textarea}
            disabled={isLoading}
          />
          <button
            onClick={() => handleSend()}
            disabled={!inputText.trim() || isLoading}
            style={{
              ...styles.sendButton,
              opacity: !inputText.trim() || isLoading ? 0.5 : 1,
            }}
          >
            {isLoading ? <Loader2 size={16} /> : <Send size={16} />}
          </button>
        </div>
      </div>
    </div>
  );
};

const styles: Record<string, React.CSSProperties> = {
  container: {
    display: 'flex',
    flexDirection: 'column',
    height: '100vh',
    flex: 1,
    backgroundColor: 'var(--bg-primary)',
    position: 'relative',
    overflow: 'hidden',
  },
  messagesContainer: {
    flex: 1,
    overflowY: 'auto',
    display: 'flex',
    flexDirection: 'column',
    padding: '2rem 1.5rem',
  },
  welcomeContainer: {
    display: 'flex',
    flexDirection: 'column',
    alignItems: 'center',
    justifyContent: 'center',
    flex: 1,
    maxWidth: '760px',
    margin: '0 auto',
    width: '100%',
  },
  heroBox: {
    textAlign: 'center',
    marginBottom: '2.5rem',
  },
  heroTitle: {
    fontSize: '2.25rem',
    fontWeight: 700,
    color: 'var(--text-primary)',
    letterSpacing: '-0.02em',
    marginBottom: '0.5rem',
  },
  heroSubtitle: {
    fontSize: '1.05rem',
    color: 'var(--text-secondary)',
    lineHeight: 1.4,
  },
  promptsGrid: {
    display: 'grid',
    gridTemplateColumns: 'repeat(2, 1fr)',
    gap: '0.85rem',
    width: '100%',
  },
  promptCard: {
    textAlign: 'left',
    padding: '1rem 1.15rem',
    backgroundColor: 'var(--bg-secondary)',
    border: '1px solid var(--border-color)',
    borderRadius: '8px',
    display: 'flex',
    flexDirection: 'column',
    gap: '0.35rem',
    cursor: 'pointer',
    transition: 'border-color 0.15s ease, transform 0.1s ease',
  },
  promptCardHeader: {
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'space-between',
  },
  promptCardTitle: {
    fontSize: '0.9rem',
    fontWeight: 600,
    color: 'var(--text-primary)',
  },
  promptCardDesc: {
    fontSize: '0.8rem',
    color: 'var(--text-muted)',
  },
  chatList: {
    display: 'flex',
    flexDirection: 'column',
    gap: '1.5rem',
    maxWidth: '820px',
    width: '100%',
    margin: '0 auto',
  },
  messageRow: {
    display: 'flex',
    width: '100%',
  },
  bubble: {
    borderRadius: '10px',
    padding: '1.15rem 1.25rem',
    boxShadow: '0 1px 3px rgba(0,0,0,0.02)',
  },
  messageHeader: {
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'space-between',
    marginBottom: '0.6rem',
  },
  senderLabel: {
    display: 'flex',
    alignItems: 'center',
    gap: '0.4rem',
    fontSize: '0.8rem',
    fontWeight: 600,
    color: 'var(--text-secondary)',
  },
  timestamp: {
    fontSize: '0.72rem',
    color: 'var(--text-muted)',
  },
  messageBody: {
    fontSize: '0.92rem',
    lineHeight: 1.55,
    color: 'var(--text-primary)',
    whiteSpace: 'pre-wrap',
  },
  productsContainer: {
    marginTop: '1.25rem',
    paddingTop: '1rem',
    borderTop: '1px solid var(--border-color)',
  },
  productsGrid: {
    display: 'grid',
    gridTemplateColumns: 'repeat(auto-fit, minmax(220px, 1fr))',
    gap: '0.85rem',
  },
  productCard: {
    backgroundColor: '#FFFFFF',
    border: '1px solid var(--border-color)',
    borderRadius: '8px',
    overflow: 'hidden',
    display: 'flex',
    flexDirection: 'column',
    boxShadow: '0 1px 2px rgba(0,0,0,0.03)',
  },
  productImageWrapper: {
    height: '130px',
    backgroundColor: '#F8FAFC',
    borderBottom: '1px solid var(--border-color)',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    overflow: 'hidden',
  },
  productImage: {
    width: '100%',
    height: '100%',
    objectFit: 'cover',
  },
  productImagePlaceholder: {
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    height: '100%',
  },
  productInfo: {
    padding: '0.75rem',
    display: 'flex',
    flexDirection: 'column',
    gap: '0.25rem',
  },
  productName: {
    fontSize: '0.86rem',
    fontWeight: 600,
    color: 'var(--text-primary)',
    whiteSpace: 'nowrap',
    overflow: 'hidden',
    textOverflow: 'ellipsis',
  },
  productPrice: {
    fontSize: '0.95rem',
    fontWeight: 700,
    color: 'var(--text-primary)',
  },
  productSpecs: {
    display: 'flex',
    alignItems: 'center',
    gap: '0.35rem',
    fontSize: '0.75rem',
    color: 'var(--text-secondary)',
  },
  productSubSpecs: {
    fontSize: '0.73rem',
    color: 'var(--text-muted)',
  },
  productBadge: {
    marginTop: '0.35rem',
    fontSize: '0.72rem',
    fontWeight: 500,
    color: '#16A34A',
    backgroundColor: '#F0FDF4',
    border: '1px solid #BBF7D0',
    borderRadius: '4px',
    padding: '0.15rem 0.4rem',
    alignSelf: 'flex-start',
  },
  executionBridge: {
    marginTop: '1rem',
    paddingTop: '0.75rem',
    borderTop: '1px solid var(--border-color)',
  },
  executionStatusSuccess: {
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'space-between',
    fontSize: '0.78rem',
    color: 'var(--text-secondary)',
    backgroundColor: '#F0FDF4',
    border: '1px solid #DCFCE7',
    padding: '0.45rem 0.75rem',
    borderRadius: '6px',
  },
  executionStatusFailed: {
    display: 'flex',
    flexDirection: 'column',
    gap: '0.3rem',
    fontSize: '0.78rem',
    backgroundColor: '#FEF2F2',
    border: '1px solid #FECACA',
    padding: '0.65rem 0.85rem',
    borderRadius: '6px',
  },
  failedStepNotice: {
    color: '#991B1B',
    fontWeight: 500,
    fontSize: '0.78rem',
  },
  inspectButton: {
    display: 'flex',
    alignItems: 'center',
    gap: '0.3rem',
    fontSize: '0.76rem',
    fontWeight: 600,
    color: '#2563EB',
    backgroundColor: '#FFFFFF',
    border: '1px solid #BFDBFE',
    padding: '0.2rem 0.5rem',
    borderRadius: '4px',
  },
  diagnoseButton: {
    display: 'flex',
    alignItems: 'center',
    gap: '0.3rem',
    fontSize: '0.76rem',
    fontWeight: 600,
    color: '#FFFFFF',
    backgroundColor: '#DC2626',
    padding: '0.3rem 0.65rem',
    borderRadius: '4px',
  },
  inspectButtonSecondary: {
    fontSize: '0.76rem',
    fontWeight: 500,
    color: 'var(--text-secondary)',
    backgroundColor: '#FFFFFF',
    border: '1px solid var(--border-color)',
    padding: '0.3rem 0.65rem',
    borderRadius: '4px',
  },
  liveExecutionBox: {
    display: 'flex',
    alignItems: 'center',
    gap: '0.6rem',
    backgroundColor: 'var(--bg-secondary)',
    border: '1px solid var(--border-color)',
    borderRadius: '8px',
    padding: '0.65rem 1rem',
    fontSize: '0.84rem',
    color: 'var(--text-secondary)',
  },
  liveStepText: {
    fontWeight: 500,
    color: 'var(--accent-blue)',
  },
  inputSection: {
    backgroundColor: 'var(--bg-primary)',
    borderTop: '1px solid var(--border-color)',
    padding: '0.75rem 1.5rem 1.25rem 1.5rem',
    maxWidth: '850px',
    width: '100%',
    margin: '0 auto',
  },
  controlBar: {
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'space-between',
    marginBottom: '0.5rem',
    fontSize: '0.78rem',
  },
  controlBarLeft: {
    display: 'flex',
    alignItems: 'center',
    gap: '0.4rem',
  },
  controlBarLabel: {
    color: 'var(--text-muted)',
    fontWeight: 500,
  },
  modeSelect: {
    fontSize: '0.78rem',
    border: '1px solid var(--border-color)',
    borderRadius: '4px',
    padding: '0.15rem 0.45rem',
    backgroundColor: 'var(--bg-secondary)',
    color: 'var(--text-primary)',
    outline: 'none',
  },
  failureActiveWarning: {
    fontSize: '0.72rem',
    fontWeight: 600,
    color: '#D97706',
    backgroundColor: '#FEF3C7',
    padding: '0.1rem 0.45rem',
    borderRadius: '4px',
  },
  inputContainer: {
    display: 'flex',
    alignItems: 'flex-end',
    gap: '0.5rem',
    backgroundColor: 'var(--bg-secondary)',
    border: '1px solid var(--border-color)',
    borderRadius: '8px',
    padding: '0.5rem 0.75rem',
  },
  textarea: {
    flex: 1,
    border: 'none',
    backgroundColor: 'transparent',
    resize: 'none',
    outline: 'none',
    fontSize: '0.9rem',
    lineHeight: 1.4,
    color: 'var(--text-primary)',
  },
  sendButton: {
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    width: '34px',
    height: '34px',
    backgroundColor: 'var(--accent-blue)',
    color: '#FFFFFF',
    borderRadius: '6px',
    cursor: 'pointer',
  },
};
