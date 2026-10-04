import React from 'react';
import { motion } from 'framer-motion';
import {
  MessageSquare,
  ListFilter,
  Layers,
  Activity,
  RotateCcw,
  GitCompare,
  BarChart2,
  Package,
  Plus,
  Settings,
  Terminal,
} from 'lucide-react';
import { SystemStatus } from '../types';

export type NavView =
  | 'chat'
  | 'runs'
  | 'trace'
  | 'diagnosis'
  | 'replay'
  | 'comparison'
  | 'evaluation'
  | 'catalogue'
  | 'settings';

interface SidebarProps {
  currentView: NavView;
  onSelectView: (view: NavView) => void;
  onNewChat: () => void;
  systemStatus: SystemStatus | null;
  selectedRunId: string | null;
}

export const Sidebar: React.FC<SidebarProps> = ({
  currentView,
  onSelectView,
  onNewChat,
  systemStatus,
  selectedRunId,
}) => {
  const navItems: { id: NavView; label: string; icon: React.ReactNode; badge?: string }[] = [
    { id: 'chat', label: 'Chat', icon: <MessageSquare size={17} /> },
    { id: 'runs', label: 'Runs', icon: <ListFilter size={17} /> },
    {
      id: 'trace',
      label: 'Trace',
      icon: <Layers size={17} />,
      badge: selectedRunId ? 'Active' : undefined,
    },
    { id: 'diagnosis', label: 'Diagnosis', icon: <Activity size={17} /> },
    { id: 'replay', label: 'Replay', icon: <RotateCcw size={17} /> },
    { id: 'comparison', label: 'Comparison', icon: <GitCompare size={17} /> },
    { id: 'evaluation', label: 'Evaluation', icon: <BarChart2 size={17} /> },
    { id: 'catalogue', label: 'Catalogue', icon: <Package size={17} /> },
  ];

  return (
    <aside style={styles.container} className="glass-panel">
      {/* Brand Header */}
      <div style={styles.brandContainer}>
        <div style={styles.brandIconBox}>
          <Terminal size={18} color="#3B82F6" />
        </div>
        <div>
          <div style={styles.brandTitle}>BLACK BOX</div>
          <div style={styles.brandSubtitle}>AI Agent Debugging</div>
        </div>
      </div>

      {/* New Chat Button */}
      <div style={styles.newChatSection}>
        <motion.button
          whileHover={{ scale: 1.02 }}
          whileTap={{ scale: 0.98 }}
          onClick={() => {
            onNewChat();
            onSelectView('chat');
          }}
          style={styles.newChatButton}
          title="Start fresh conversation"
        >
          <Plus size={16} />
          <span>New Chat</span>
        </motion.button>
      </div>

      {/* Navigation List */}
      <nav style={styles.navList}>
        {navItems.map((item) => {
          const isActive = currentView === item.id;
          return (
            <motion.button
              key={item.id}
              whileHover={{ x: 3 }}
              whileTap={{ scale: 0.98 }}
              onClick={() => onSelectView(item.id)}
              style={{
                ...styles.navButton,
                backgroundColor: isActive ? 'var(--bg-active)' : 'transparent',
                color: isActive ? 'var(--text-primary)' : 'var(--text-secondary)',
                fontWeight: isActive ? 600 : 400,
                border: isActive ? '1px solid rgba(59, 130, 246, 0.3)' : '1px solid transparent',
              }}
            >
              <span
                style={{
                  ...styles.iconWrapper,
                  color: isActive ? 'var(--accent-blue)' : 'var(--text-muted)',
                }}
              >
                {item.icon}
              </span>
              <span style={{ flex: 1, textAlign: 'left' }}>{item.label}</span>
              {item.badge && (
                <span
                  style={{
                    fontSize: '0.68rem',
                    padding: '0.1rem 0.4rem',
                    borderRadius: '4px',
                    backgroundColor: 'var(--accent-blue-light)',
                    color: 'var(--accent-blue)',
                    fontWeight: 600,
                  }}
                >
                  {item.badge}
                </span>
              )}
            </motion.button>
          );
        })}
      </nav>

      {/* Settings / Footer */}
      <div style={styles.footerSection}>
        <motion.button
          whileHover={{ x: 3 }}
          whileTap={{ scale: 0.98 }}
          onClick={() => onSelectView('settings')}
          style={{
            ...styles.navButton,
            backgroundColor: currentView === 'settings' ? 'var(--bg-active)' : 'transparent',
            color: currentView === 'settings' ? 'var(--text-primary)' : 'var(--text-muted)',
            border: currentView === 'settings' ? '1px solid rgba(59, 130, 246, 0.3)' : '1px solid transparent',
          }}
        >
          <Settings size={16} />
          <span>Settings</span>
        </motion.button>
      </div>
    </aside>
  );
};

const styles: Record<string, React.CSSProperties> = {
  container: {
    width: '240px',
    minWidth: '240px',
    height: '100vh',
    backgroundColor: 'var(--bg-secondary)',
    borderRight: '1px solid var(--border-color)',
    display: 'flex',
    flexDirection: 'column',
    padding: '1rem 0.75rem',
    userSelect: 'none',
  },
  brandContainer: {
    display: 'flex',
    alignItems: 'center',
    gap: '0.65rem',
    padding: '0.25rem 0.5rem 1rem 0.5rem',
  },
  brandIconBox: {
    width: '32px',
    height: '32px',
    borderRadius: '6px',
    backgroundColor: '#EFF6FF',
    border: '1px solid #BFDBFE',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
  },
  brandTitle: {
    fontSize: '0.95rem',
    fontWeight: 700,
    letterSpacing: '0.04em',
    color: 'var(--text-primary)',
  },
  brandSubtitle: {
    fontSize: '0.72rem',
    color: 'var(--text-muted)',
    fontWeight: 400,
  },
  newChatSection: {
    marginBottom: '1rem',
  },
  newChatButton: {
    width: '100%',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    gap: '0.5rem',
    backgroundColor: 'var(--bg-primary)',
    border: '1px solid var(--border-color)',
    borderRadius: '6px',
    padding: '0.55rem 0.75rem',
    fontSize: '0.85rem',
    fontWeight: 500,
    color: 'var(--text-primary)',
    boxShadow: '0 1px 2px rgba(0,0,0,0.03)',
  },
  navList: {
    display: 'flex',
    flexDirection: 'column',
    gap: '0.2rem',
    flex: 1,
    overflowY: 'auto',
  },
  navButton: {
    display: 'flex',
    alignItems: 'center',
    gap: '0.65rem',
    width: '100%',
    padding: '0.45rem 0.65rem',
    borderRadius: '6px',
    fontSize: '0.85rem',
  },
  iconWrapper: {
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
  },
  divider: {
    height: '1px',
    backgroundColor: 'var(--border-color)',
    margin: '0.75rem 0.5rem',
  },
  footerSection: {
    paddingTop: '0.25rem',
  },
};
