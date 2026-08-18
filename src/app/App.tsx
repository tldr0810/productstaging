/**
 * The shell loads shared app state, routes between staging and settings by
 * location.hash, and raises the password gate when the deployment requires it.
 */

import { useCallback, useEffect, useState } from 'react';
import type { AppState } from '../shared/types';
import { api, onUnauthorized } from './api';
import PasswordGate from './components/PasswordGate';
import SettingsView from './components/SettingsView';
import StageView from './components/StageView';

type Tab = 'stage' | 'settings';

const tabFromHash = (): Tab => (location.hash === '#settings' ? 'settings' : 'stage');

export default function App() {
  const [state, setState] = useState<AppState | null>(null);
  const [loadError, setLoadError] = useState('');
  const [tab, setTab] = useState<Tab>(tabFromHash);
  const [gateOpen, setGateOpen] = useState(false);

  const refreshState = useCallback(async () => {
    try {
      const next = await api<AppState>('/api/state');
      setState(next);
      setLoadError('');
      setGateOpen(next.adminRequired && !next.adminOk);
    } catch (error) {
      setLoadError(error instanceof Error ? error.message : String(error));
    }
  }, []);

  useEffect(() => {
    onUnauthorized(() => setGateOpen(true));
    void refreshState();
    return () => onUnauthorized(null);
  }, [refreshState]);

  useEffect(() => {
    const onHash = () => setTab(tabFromHash());
    window.addEventListener('hashchange', onHash);
    return () => window.removeEventListener('hashchange', onHash);
  }, []);

  if (loadError) {
    return (
      <main className="shell">
        <div className="notice error">
          Could not reach the API: {loadError}{' '}
          <button className="link" onClick={() => void refreshState()}>
            Retry
          </button>
        </div>
      </main>
    );
  }

  if (!state) {
    return (
      <main className="shell">
        <p className="muted">Loading…</p>
      </main>
    );
  }

  return (
    <main className="shell">
      <header className="topbar">
        <div className="brand">
          <span className="brand-mark" aria-hidden>
            ✦
          </span>
          <div>
            <h1>Product Staging</h1>
            <p className="muted">Turn a plain product photo into a lifestyle scene.</p>
          </div>
        </div>
        <nav className="tabs" aria-label="Pages">
          <a className={tab === 'stage' ? 'tab active' : 'tab'} href="#stage">
            Stage
          </a>
          <a className={tab === 'settings' ? 'tab active' : 'tab'} href="#settings">
            Settings
          </a>
        </nav>
      </header>

      {tab === 'stage' ? (
        <StageView />
      ) : (
        <SettingsView agents={state.agents} initialSession={state.connect.session} refreshState={refreshState} />
      )}

      {gateOpen && <PasswordGate onSubmitted={refreshState} />}
    </main>
  );
}
