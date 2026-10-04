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
    <div style={{ display: 'flex', width: '100vw', height: '100vh', overflow: 'hidden', backgroundColor: 'var(--bg-primary)', position: 'relative' }}>
      {/* Background Ambient Orbs for Glassmorphism */}
      <div className="bg-orb glow-blue" style={{ top: '-100px', left: '10%', width: '400px', height: '400px', background: 'radial-gradient(circle, #3B82F6 0%, transparent 70%)' }} />
      <div className="bg-orb glow-purple" style={{ bottom: '-100px', right: '15%', width: '450px', height: '450px', background: 'radial-gradient(circle, #8B5CF6 0%, transparent 70%)' }} />

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
            initial={{ opacity: 0, y: 8, scale: 0.99 }}
            animate={{ opacity: 1, y: 0, scale: 1 }}
            exit={{ opacity: 0, y: -8, scale: 0.99 }}
            transition={{ duration: 0.22, ease: [0.16, 1, 0.3, 1] }}
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

