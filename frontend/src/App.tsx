import { useState, useEffect } from 'react';
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

  const handleSelectReplayRun = (origId: string, altId: string) => {
    setSelectedRunId(origId);
    setAlternativeRunId(altId);
  };

  const handleNewChat = () => {
    setCurrentView('chat');
  };

  return (
    <div style={{ display: 'flex', width: '100vw', height: '100vh', overflow: 'hidden' }}>
      {/* Persistent Left Sidebar */}
      <Sidebar
        currentView={currentView}
        onSelectView={(v) => setCurrentView(v)}
        onNewChat={handleNewChat}
        systemStatus={systemStatus}
        selectedRunId={selectedRunId}
      />

      {/* Main Content Area */}
      <main style={{ flex: 1, height: '100vh', overflow: 'hidden', position: 'relative' }}>
        {currentView === 'chat' && (
          <ChatView onNavigateToRun={handleNavigateToRun} catalogue={catalogue} />
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
      </main>
    </div>
  );
}

export default App;
