import { useEffect } from 'react';
import { useDashboardStore } from './store';
import { Header } from './components/layout/Header';
import {
  SessionTreeTab,
  OverviewTab,
  TextMonitorTab,
  ChatTab,
  WeightsTab,
  TransactionsTab,
  EnvironmentTab,
  ArchitectureTab,
  SessionsTab,
} from './components/tabs';
import { motion, AnimatePresence } from 'framer-motion';

function App() {
  const activeTab = useDashboardStore((state) => state.activeTab);
  const initialize = useDashboardStore((state) => state.initialize);

  useEffect(() => {
    void initialize();
  }, [initialize]);

  return (
    <div className="min-h-screen bg-surface flex flex-col">
      <Header />

      <main className="flex-1 p-6 overflow-auto">
        <AnimatePresence mode="wait">
          <motion.div
            key={activeTab}
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -10 }}
            transition={{ duration: 0.15 }}
            className="tab-content"
          >
            {activeTab === 'session-tree' && <SessionTreeTab />}
            {activeTab === 'overview' && <OverviewTab />}
            {activeTab === 'text-monitor' && <TextMonitorTab />}
            {activeTab === 'chat' && <ChatTab />}
            {activeTab === 'weights' && <WeightsTab />}
            {activeTab === 'transactions' && <TransactionsTab />}
            {activeTab === 'environment' && <EnvironmentTab />}
            {activeTab === 'architecture' && <ArchitectureTab />}
            {activeTab === 'sessions' && <SessionsTab />}
          </motion.div>
        </AnimatePresence>
      </main>

      {/* Footer */}
      <footer className="border-t border-surface-200 px-6 py-3 flex items-center justify-between text-xs text-text-muted">
        <div className="flex items-center gap-4">
          <span>TTT-SSM Phase 1 Dashboard</span>
          <span className="text-surface-300">|</span>
          <span>Test-Time Training Research Tool</span>
        </div>
        <div className="flex items-center gap-4">
          <span>
            <kbd className="px-1.5 py-0.5 bg-surface-100 rounded">←</kbd>
            <kbd className="px-1.5 py-0.5 bg-surface-100 rounded ml-1">→</kbd>
            <span className="ml-2">Navigate time</span>
          </span>
          <span className="text-surface-300">|</span>
          <span>
            <kbd className="px-1.5 py-0.5 bg-surface-100 rounded">Tab</kbd>
            <span className="ml-2">Switch tabs</span>
          </span>
        </div>
      </footer>
    </div>
  );
}

export default App;
