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

/* Staggered nav item animation */
const navContainerVariants = {
  show: {
    transition: { staggerChildren: 0.04, delayChildren: 0.1 },
  },
};

const navItemVariants = {
  hidden: { opacity: 0, x: -12 },
  show: { opacity: 1, x: 0, transition: { duration: 0.25, ease: 'easeOut' as const } },
};

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
    <motion.aside
      initial={{ x: -20, opacity: 0 }}
      animate={{ x: 0, opacity: 1 }}
      transition={{ duration: 0.4, ease: [0.16, 1, 0.3, 1] }}
      style={styles.container}
      className="glass-panel"
    >
      {/* Brand Header */}
      <motion.div
        initial={{ opacity: 0, y: -8 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.15, duration: 0.35 }}
        style={styles.brandContainer}
      >
        <motion.div
          style={styles.brandIconBox}
          whileHover={{ scale: 1.08, rotate: 3 }}
          transition={{ type: 'spring', stiffness: 400 }}
        >
          <Terminal size={18} color="#3B82F6" />
        </motion.div>
        <div>
          <div style={styles.brandTitle}>BLACK BOX</div>
          <div style={styles.brandSubtitle}>AI Agent Debugging</div>
        </div>
      </motion.div>

      {/* New Chat Button */}
      <div style={styles.newChatSection}>
        <motion.button
          whileHover={{ scale: 1.02, boxShadow: '0 0 15px rgba(59, 130, 246, 0.2)' }}
          whileTap={{ scale: 0.97 }}
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
      <motion.nav
        style={styles.navList}
        variants={navContainerVariants}
        initial="hidden"
        animate="show"
      >
        {navItems.map((item) => {
          const isActive = currentView === item.id;
          return (
            <motion.button
              key={item.id}
              variants={navItemVariants}
              whileHover={{ x: 4, backgroundColor: isActive ? 'var(--bg-active)' : 'rgba(255,255,255,0.04)' }}
              whileTap={{ scale: 0.97 }}
              onClick={() => onSelectView(item.id)}
              style={{
                ...styles.navButton,
                backgroundColor: isActive ? 'var(--bg-active)' : 'transparent',
                color: isActive ? 'var(--text-primary)' : 'var(--text-secondary)',
                fontWeight: isActive ? 600 : 400,
                border: isActive ? '1px solid rgba(59, 130, 246, 0.3)' : '1px solid transparent',
              }}
            >
              {/* Active indicator bar */}
              {isActive && (
                <motion.div
                  layoutId="activeIndicator"
                  style={styles.activeBar}
                  transition={{ type: 'spring', stiffness: 500, damping: 35 }}
                />
              )}
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
                <motion.span
                  initial={{ scale: 0 }}
                  animate={{ scale: 1 }}
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
                </motion.span>
              )}
            </motion.button>
          );
        })}
      </motion.nav>

      {/* System Status Indicator */}
      {systemStatus && (
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.5 }}
          style={styles.statusPill}
        >
          <div
            style={{
              width: '6px',
              height: '6px',
              borderRadius: '50%',
              backgroundColor: systemStatus.status === 'healthy' ? '#22C55E' : '#EF4444',
              boxShadow: systemStatus.status === 'healthy'
                ? '0 0 6px rgba(34, 197, 94, 0.5)'
                : '0 0 6px rgba(239, 68, 68, 0.5)',
            }}
          />
          <span style={{ fontSize: '0.7rem', color: 'var(--text-muted)' }}>
            {systemStatus.status === 'healthy' ? 'All systems online' : 'System degraded'}
          </span>
        </motion.div>
      )}

      {/* Settings / Footer */}
      <div style={styles.footerSection}>
        <motion.button
          whileHover={{ x: 4 }}
          whileTap={{ scale: 0.97 }}
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
    </motion.aside>
  );
};

const styles: Record<string, React.CSSProperties> = {
  container: {
    width: '240px',
    minWidth: '240px',
    height: '100vh',
    backgroundColor: 'rgba(11, 17, 32, 0.85)',
    borderRight: '1px solid var(--border-color)',
    display: 'flex',
    flexDirection: 'column',
    padding: '1rem 0.75rem',
    userSelect: 'none',
    backdropFilter: 'blur(20px) saturate(180%)',
    zIndex: 10,
  },
  brandContainer: {
    display: 'flex',
    alignItems: 'center',
    gap: '0.65rem',
    padding: '0.25rem 0.5rem 1rem 0.5rem',
  },
  brandIconBox: {
    width: '34px',
    height: '34px',
    borderRadius: '8px',
    background: 'linear-gradient(135deg, rgba(59, 130, 246, 0.15) 0%, rgba(139, 92, 246, 0.15) 100%)',
    border: '1px solid rgba(59, 130, 246, 0.25)',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
  },
  brandTitle: {
    fontSize: '0.95rem',
    fontWeight: 700,
    letterSpacing: '0.06em',
    color: 'var(--text-primary)',
    background: 'linear-gradient(135deg, #F8FAFC 0%, #94A3B8 100%)',
    WebkitBackgroundClip: 'text',
    WebkitTextFillColor: 'transparent',
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
    background: 'linear-gradient(135deg, rgba(59, 130, 246, 0.12) 0%, rgba(139, 92, 246, 0.12) 100%)',
    border: '1px solid rgba(59, 130, 246, 0.2)',
    borderRadius: '8px',
    padding: '0.55rem 0.75rem',
    fontSize: '0.85rem',
    fontWeight: 500,
    color: 'var(--text-primary)',
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
    borderRadius: '8px',
    fontSize: '0.85rem',
    position: 'relative',
    overflow: 'hidden',
  },
  activeBar: {
    position: 'absolute',
    left: 0,
    top: '20%',
    bottom: '20%',
    width: '3px',
    borderRadius: '0 3px 3px 0',
    background: 'var(--gradient-blue-purple)',
  },
  iconWrapper: {
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
  },
  statusPill: {
    display: 'flex',
    alignItems: 'center',
    gap: '0.45rem',
    padding: '0.4rem 0.65rem',
    marginBottom: '0.5rem',
    borderRadius: '6px',
    backgroundColor: 'rgba(255, 255, 255, 0.03)',
    border: '1px solid var(--border-light)',
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
