import { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Sidebar, NavView } from './components/Sidebar';
import { ChatView } from './components/ChatView';
import { RunsView } from './components/RunsView';
import { TraceView } from './components/TraceView';
import { DiagnosisView } from './components/DiagnosisView';
import { ReplayView } from './components/ReplayView';
import { ComparisonView } from './components/ComparisonView';
import { EvaluationView } from './components/EvaluationView';
import { CatalogueView } from './components/CatalogueView';
import { SettingsView } from './components/SettingsView';
import { api } from './api';
import { Product, SystemStatus } from './types';

/* Framer Motion page transition variants */
const pageVariants = {
  initial: { opacity: 0, y: 12, scale: 0.985, filter: 'blur(4px)' },
  animate: { opacity: 1, y: 0, scale: 1, filter: 'blur(0px)' },
  exit: { opacity: 0, y: -8, scale: 0.985, filter: 'blur(4px)' },
};

const pageTransition = {
  duration: 0.32,
  ease: 'easeOut' as const,
};

/* Animated background orb config */
const orbVariants = {
  float: {
    x: [0, 15, -10, 20, 0],
    y: [0, -20, 10, 15, 0],
    scale: [1, 1.05, 0.95, 1.02, 1],
    transition: {
      duration: 20,
      ease: 'easeInOut' as const,
      repeat: Infinity,
    },
  },
};

export function App() {
  const [currentView, setCurrentView] = useState<NavView>('chat');
  const [selectedRunId, setSelectedRunId] = useState<string | null>(null);
  const [alternativeRunId, setAlternativeRunId] = useState<string | null>(null);
  const [catalogue, setCatalogue] = useState<Product[]>([]);
  const [systemStatus, setSystemStatus] = useState<SystemStatus | null>(null);

  useEffect(() => {
    // Load catalogue and system status
    api
      .getCatalogue()
      .then((data) => setCatalogue(data))
      .catch((err) => console.error('Failed to load catalogue:', err));

    api
      .getSystemStatus()
      .then((status) => setSystemStatus(status))
      .catch((err) => console.error('Failed to load system status:', err));
  }, []);

  const handleNavigateToRun = (runId: string, targetView: NavView) => {
    setSelectedRunId(runId);
    setCurrentView(targetView);
  };

  const [chatSessionKey, setChatSessionKey] = useState<number>(Date.now());

  const handleSelectReplayRun = (origId: string, altId: string) => {
    setSelectedRunId(origId);
    setAlternativeRunId(altId);
  };

  const handleNewChat = () => {
    sessionStorage.removeItem('blackbox_conversation_id');
    setChatSessionKey(Date.now());
    setCurrentView('chat');
  };

  return (
    <div
      className="noise-overlay"
      style={{
        display: 'flex',
        width: '100vw',
        height: '100vh',
        overflow: 'hidden',
        backgroundColor: 'var(--bg-primary)',
        position: 'relative',
      }}
    >
      {/* Animated Background Ambient Orbs for Glassmorphism depth */}
      <motion.div
        className="bg-orb"
        variants={orbVariants}
        animate="float"
        style={{
          top: '-120px',
          left: '8%',
          width: '500px',
          height: '500px',
          background: 'radial-gradient(circle, rgba(59, 130, 246, 0.5) 0%, transparent 70%)',
        }}
      />
      <motion.div
        className="bg-orb"
        custom={1}
        variants={orbVariants}
        animate="float"
        style={{
          bottom: '-150px',
          right: '12%',
          width: '550px',
          height: '550px',
          background: 'radial-gradient(circle, rgba(139, 92, 246, 0.5) 0%, transparent 70%)',
        }}
      />
      <motion.div
        className="bg-orb"
        custom={2}
        variants={orbVariants}
        animate="float"
        style={{
          top: '40%',
          left: '50%',
          width: '350px',
          height: '350px',
          background: 'radial-gradient(circle, rgba(6, 182, 212, 0.3) 0%, transparent 70%)',
          opacity: 0.08,
        }}
      />

      {/* Persistent Left Sidebar */}
      <Sidebar
        currentView={currentView}
        onSelectView={(v) => setCurrentView(v)}
        onNewChat={handleNewChat}
        systemStatus={systemStatus}
        selectedRunId={selectedRunId}
      />

      {/* Main Content Area with Motion Transitions */}
      <main style={{ flex: 1, height: '100vh', overflow: 'hidden', position: 'relative', zIndex: 1 }}>
        <AnimatePresence mode="wait">
          <motion.div
            key={currentView === 'chat' ? `chat-${chatSessionKey}` : currentView}
            variants={pageVariants}
            initial="initial"
            animate="animate"
            exit="exit"
            transition={pageTransition}
            style={{ width: '100%', height: '100%' }}
          >
            {currentView === 'chat' && (
              <ChatView
                key={chatSessionKey}
                onNavigateToRun={handleNavigateToRun}
                catalogue={catalogue}
                onNewChat={handleNewChat}
              />
            )}

            {currentView === 'runs' && (
              <RunsView onSelectRun={handleNavigateToRun} />
            )}

            {currentView === 'trace' && (
              <TraceView
                selectedRunId={selectedRunId}
                onSelectRunId={setSelectedRunId}
                onNavigateToView={(v) => setCurrentView(v)}
              />
            )}

            {currentView === 'diagnosis' && (
              <DiagnosisView
                selectedRunId={selectedRunId}
                onSelectRunId={setSelectedRunId}
                onNavigateToView={(v) => setCurrentView(v)}
              />
            )}

            {currentView === 'replay' && (
              <ReplayView
                selectedRunId={selectedRunId}
                onSelectRunId={setSelectedRunId}
                onNavigateToView={(v) => setCurrentView(v)}
                onSelectReplayRun={handleSelectReplayRun}
              />
            )}

            {currentView === 'comparison' && (
              <ComparisonView
                originalRunId={selectedRunId}
                alternativeRunId={alternativeRunId}
                onSelectRuns={(orig, alt) => {
                  setSelectedRunId(orig);
                  setAlternativeRunId(alt);
                }}
              />
            )}

            {currentView === 'evaluation' && <EvaluationView />}

            {currentView === 'catalogue' && <CatalogueView products={catalogue} />}

            {currentView === 'settings' && <SettingsView systemStatus={systemStatus} />}
          </motion.div>
        </AnimatePresence>
      </main>
    </div>
  );
}

export default App;
