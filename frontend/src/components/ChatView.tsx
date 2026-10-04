import React, { useState, useRef, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
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

/* Framer Motion variants for staggered animations */
const containerVariants = {
  hidden: { opacity: 0 },
  show: {
    opacity: 1,
    transition: { staggerChildren: 0.08, delayChildren: 0.15 },
  },
};

const cardVariants = {
  hidden: { opacity: 0, y: 16, scale: 0.96 },
  show: { opacity: 1, y: 0, scale: 1, transition: { duration: 0.35, ease: 'easeOut' as const } },
};

const messageVariants = {
  hidden: { opacity: 0, y: 10, scale: 0.98 },
  show: { opacity: 1, y: 0, scale: 1, transition: { duration: 0.3, ease: 'easeOut' as const } },
};

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
  budgetMax?: number;
  failedStep?: {
    step_number: number;
    stage_name: string;
  };
}

interface ChatViewProps {
  onNavigateToRun: (runId: string, view: NavView) => void;
  catalogue: Product[];
  onNewChat?: () => void;
}

export const ChatView: React.FC<ChatViewProps> = ({ onNavigateToRun, catalogue }) => {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [inputText, setInputText] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [failureMode, setFailureMode] = useState<string>('none');
  const [liveStep, setLiveStep] = useState<string | null>(null);

  // Stable conversation ID for the lifetime of this chat session, persisted across refreshes
  const conversationId = useRef<string>(
    (() => {
      const stored = sessionStorage.getItem('blackbox_conversation_id');
      if (stored) return stored;
      const fresh = `conv-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;
      sessionStorage.setItem('blackbox_conversation_id', fresh);
      return fresh;
    })()
  );

  const messagesEndRef = useRef<HTMLDivElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages, liveStep]);

  // Restore stored conversation history on refresh / mount
  useEffect(() => {
    const convId = conversationId.current;
    if (convId) {
      api
        .getConversationMessages(convId)
        .then((res) => {
          if (res.messages && res.messages.length > 0) {
            const restored: ChatMessage[] = res.messages.map((m) => ({
              id: `stored-${m.id}`,
              sender: m.role === 'user' ? 'user' : 'agent',
              text: m.content,
              timestamp: m.timestamp
                ? new Date(m.timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
                : '',
            }));
            setMessages(restored);
          }
        })
        .catch(() => {
          // Ignore if empty / brand new
        });
    }
  }, []);

  /** Build the history array from all previous messages to send with each request */
  const buildHistory = (currentMessages: ChatMessage[]): Array<{ role: string; content: string }> => {
    return currentMessages.map((m) => ({
      role: m.sender === 'user' ? 'user' : 'assistant',
      content: m.text,
    }));
  };

  /** Extract budget from user text (simple heuristic) */
  const extractBudget = (text: string): number | undefined => {
    const cleaned = text.replace(/,/g, '').replace(/₹/g, ' ');
    const matches = cleaned.match(/\b(\d{4,7})\b/g);
    if (!matches) return undefined;
    const candidates = matches.map(Number).filter((n) => n >= 5000 && n <= 5000000);
    return candidates.length > 0 ? Math.min(...candidates) : undefined;
  };

  const handleSend = async (customText?: string, overrideMode?: string) => {
    const textToSend = customText || inputText;
    if (!textToSend.trim() || isLoading) return;

    const activeMode = overrideMode !== undefined ? overrideMode : failureMode;
    const budgetMax = extractBudget(textToSend);

    const userMsgId = `msg-${Date.now()}`;
    const userMsg: ChatMessage = {
      id: userMsgId,
      sender: 'user',
      text: textToSend.trim(),
      timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
    };

    // Capture current messages before state update for history building
    const prevMessages = [...messages];

    setMessages((prev) => [...prev, userMsg]);
    setInputText('');
    setIsLoading(true);
    setLiveStep('Step 1: Input Received');

    // Simulate observable step progression before API responds
    const stepTimer1 = setTimeout(() => setLiveStep('Step 2: Parsing Intent & Budget'), 300);
    const stepTimer2 = setTimeout(() => setLiveStep('Step 3: Extracting Specifications'), 700);
    const stepTimer3 = setTimeout(() => setLiveStep('Step 4: Tool Execution & Verification'), 1100);

    try {
      const modeParam = activeMode === 'none' ? undefined : activeMode;

      // Build full history from all previous messages + current user message
      const history = buildHistory(prevMessages);

      const res = await api.runAgent({
        request: textToSend.trim(),
        failure_mode: modeParam,
        history,
        conversation_id: conversationId.current,
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
            step_number: (failed as any).step_number || 4,
            stage_name: (failed as any).stage || failed.step_type || 'Tool Execution',
          };
        }
      }

      // Prefer structured products from API; fall back to text-match against catalogue
      let foundProducts: Product[] = res.products && res.products.length > 0
        ? res.products
        : catalogue.filter((p) => res.final_response.toLowerCase().includes(p.name.toLowerCase())).slice(0, 4);

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
        budgetMax,
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
      desc: 'Under ₹80,000 · 16GB RAM',
      query: 'I need a laptop for programming under ₹80,000 with at least 16GB RAM.',
      mode: 'none',
    },
    {
      title: 'Smartphone Recommendation',
      desc: 'Best phone under ₹30,000',
      query: 'Which smartphone should I buy under ₹30,000 for everyday use and good camera?',
      mode: 'none',
    },
    {
      title: 'Gaming Setup',
      desc: 'Gaming laptop under ₹1,20,000',
      query: 'Find a gaming laptop with dedicated GPU under ₹1,20,000.',
      mode: 'none',
    },
    {
      title: 'Controlled Failure Test',
      desc: 'Injected tool error for debugging',
      query: 'Search for headphones with ANC under ₹15,000.',
      mode: 'wrong_tool',
    },
  ];

  return (
    <div style={styles.container}>
      {/* Messages Scroll Area */}
      <div style={styles.messagesContainer}>
        {messages.length === 0 ? (
          /* Empty / Welcome State */
          <motion.div
            style={styles.welcomeContainer}
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5, ease: [0.16, 1, 0.3, 1] }}
          >
            <motion.div
              style={styles.heroBox}
              initial={{ opacity: 0, scale: 0.95 }}
              animate={{ opacity: 1, scale: 1 }}
              transition={{ delay: 0.1, duration: 0.45 }}
            >
              <h1 style={styles.heroTitle}>Black Box</h1>
              <p style={styles.heroSubtitle}>
                Ask me anything — or shop for electronics and I'll find the best options for you.
              </p>
            </motion.div>

            {/* Prompt Cards Grid */}
            <motion.div
              style={styles.promptsGrid}
              variants={containerVariants}
              initial="hidden"
              animate="show"
            >
              {samplePrompts.map((p, idx) => (
                <motion.button
                  key={idx}
                  variants={cardVariants}
                  whileHover={{ scale: 1.03, borderColor: 'rgba(59, 130, 246, 0.4)', boxShadow: '0 8px 24px rgba(59, 130, 246, 0.12)' }}
                  whileTap={{ scale: 0.98 }}
                  style={styles.promptCard}
                  className="glass-card"
                  onClick={() => {
                    handleSend(p.query, p.mode);
                  }}
                >
                  <div style={styles.promptCardHeader}>
                    <span style={styles.promptCardTitle}>{p.title}</span>
                    <Sparkles size={14} color="#3B82F6" />
                  </div>
                  <div style={styles.promptCardDesc}>{p.desc}</div>
                </motion.button>
              ))}
            </motion.div>
          </motion.div>
        ) : (
          <div style={styles.chatList}>
            {messages.map((msg, msgIdx) => (
              <motion.div
                key={msg.id}
                variants={messageVariants}
                initial="hidden"
                animate="show"
                transition={{ delay: msgIdx < 5 ? msgIdx * 0.05 : 0 }}
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
                        {msg.products.map((prod, idx) => {
                          const withinBudget = msg.budgetMax == null || prod.price <= msg.budgetMax;
                          const ram = prod.ram_gb ? `${prod.ram_gb}GB RAM` : prod.ram ? `${prod.ram} RAM` : null;
                          const storage = prod.storage_gb ? `${prod.storage_gb}GB` : prod.storage || null;
                          const category = prod.category?.toUpperCase() || 'ELECTRONICS';
                          return (
                            <div key={idx} style={styles.productCard}>
                              <div style={styles.productImageWrapper}>
                                {prod.image_url ? (
                                  <img
                                    src={prod.image_url}
                                    alt={prod.name}
                                    style={styles.productImage}
                                    onError={(e) => {
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
                                <div style={{ fontSize: '0.68rem', fontWeight: 600, color: '#64748B', textTransform: 'uppercase', marginBottom: '2px' }}>
                                  {category}
                                </div>
                                <div style={styles.productName} title={prod.name}>
                                  {prod.name}
                                </div>
                                <div style={styles.productPrice}>
                                  ₹{prod.price?.toLocaleString('en-IN')}
                                </div>
                                {(ram || storage) && (
                                  <div style={styles.productSpecs}>
                                    {ram && <span>{ram}</span>}
                                    {ram && storage && <span>·</span>}
                                    {storage && <span>{storage}</span>}
                                  </div>
                                )}
                                {prod.processor && (
                                  <div style={styles.productSubSpecs}>
                                    {prod.processor}{prod.gpu ? ` · ${prod.gpu}` : ''}
                                  </div>
                                )}
                                {prod.why_it_fits && (
                                  <div style={{ fontSize: '0.72rem', color: '#475569', marginTop: '4px', lineHeight: 1.3 }}>
                                    {prod.why_it_fits}
                                  </div>
                                )}
                                <div style={{
                                  ...styles.productBadge,
                                  color: withinBudget ? '#16A34A' : '#DC2626',
                                  backgroundColor: withinBudget ? '#F0FDF4' : '#FEF2F2',
                                  borderColor: withinBudget ? '#BBF7D0' : '#FECACA',
                                }}>
                                  {withinBudget ? '✓ Within budget' : '⚠ Exceeds budget'}
                                </div>
                              </div>
                            </div>
                          );
                        })}
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
              </motion.div>
            ))}

            {/* Live Agent Execution Indicator */}
            <AnimatePresence>
              {isLoading && liveStep && (
                <motion.div
                  initial={{ opacity: 0, y: 8, scale: 0.97 }}
                  animate={{ opacity: 1, y: 0, scale: 1 }}
                  exit={{ opacity: 0, y: -4, scale: 0.97 }}
                  transition={{ duration: 0.25 }}
                  style={{ ...styles.messageRow, justifyContent: 'flex-start' }}
                >
                  <div style={styles.liveExecutionBox} className="pulse-glow">
                    <Loader2 size={16} className="spin-animation" color="#3B82F6" />
                    <span style={styles.liveStepText}>{liveStep}</span>
                  </div>
                </motion.div>
              )}
            </AnimatePresence>

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
          <motion.button
            whileHover={{ scale: 1.08, boxShadow: '0 0 12px rgba(59, 130, 246, 0.4)' }}
            whileTap={{ scale: 0.92 }}
            onClick={() => handleSend()}
            disabled={!inputText.trim() || isLoading}
            style={{
              ...styles.sendButton,
              opacity: !inputText.trim() || isLoading ? 0.5 : 1,
            }}
          >
            {isLoading ? <Loader2 size={16} className="spin-animation" /> : <Send size={16} />}
          </motion.button>
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
    fontSize: '2.5rem',
    fontWeight: 700,
    background: 'linear-gradient(135deg, #F8FAFC 0%, #3B82F6 50%, #8B5CF6 100%)',
    WebkitBackgroundClip: 'text',
    WebkitTextFillColor: 'transparent',
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
    padding: '1.1rem 1.25rem',
    backgroundColor: 'rgba(15, 23, 42, 0.5)',
    border: '1px solid rgba(255, 255, 255, 0.08)',
    borderRadius: '12px',
    display: 'flex',
    flexDirection: 'column',
    gap: '0.35rem',
    cursor: 'pointer',
    backdropFilter: 'blur(12px)',
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
    backgroundColor: 'rgba(15, 23, 42, 0.6)',
    border: '1px solid rgba(255, 255, 255, 0.08)',
    borderRadius: '10px',
    overflow: 'hidden',
    display: 'flex',
    flexDirection: 'column',
    boxShadow: '0 4px 12px rgba(0,0,0,0.2)',
    backdropFilter: 'blur(10px)',
    transition: 'all 0.25s cubic-bezier(0.16, 1, 0.3, 1)',
  },
  productImageWrapper: {
    height: '130px',
    backgroundColor: 'rgba(30, 41, 59, 0.5)',
    borderBottom: '1px solid rgba(255, 255, 255, 0.06)',
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
    backgroundColor: 'rgba(34, 197, 94, 0.1)',
    border: '1px solid rgba(34, 197, 94, 0.25)',
    padding: '0.45rem 0.75rem',
    borderRadius: '8px',
  },
  executionStatusFailed: {
    display: 'flex',
    flexDirection: 'column',
    gap: '0.3rem',
    fontSize: '0.78rem',
    backgroundColor: 'rgba(239, 68, 68, 0.1)',
    border: '1px solid rgba(239, 68, 68, 0.25)',
    padding: '0.65rem 0.85rem',
    borderRadius: '8px',
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
    color: '#3B82F6',
    backgroundColor: 'rgba(59, 130, 246, 0.1)',
    border: '1px solid rgba(59, 130, 246, 0.25)',
    padding: '0.2rem 0.5rem',
    borderRadius: '6px',
  },
  diagnoseButton: {
    display: 'flex',
    alignItems: 'center',
    gap: '0.3rem',
    fontSize: '0.76rem',
    fontWeight: 600,
    color: '#FFFFFF',
    background: 'linear-gradient(135deg, #DC2626, #EF4444)',
    padding: '0.3rem 0.65rem',
    borderRadius: '6px',
    boxShadow: '0 2px 8px rgba(220, 38, 38, 0.3)',
  },
  inspectButtonSecondary: {
    fontSize: '0.76rem',
    fontWeight: 500,
    color: 'var(--text-secondary)',
    backgroundColor: 'rgba(255, 255, 255, 0.05)',
    border: '1px solid var(--border-color)',
    padding: '0.3rem 0.65rem',
    borderRadius: '6px',
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
    width: '36px',
    height: '36px',
    background: 'linear-gradient(135deg, #3B82F6, #2563EB)',
    color: '#FFFFFF',
    borderRadius: '8px',
    cursor: 'pointer',
    boxShadow: '0 2px 8px rgba(59, 130, 246, 0.3)',
  },
};
