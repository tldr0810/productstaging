/**
 * The shell: loads /api/state once, routes between the two tabs by location.hash
 * (#chat, #settings — no router dependency), and raises the password gate whenever
 * the deployment requires one and the stored password is missing or wrong.
 */

import StageView from './components/StageView';

export default function App() {
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
      </header>
      <StageView />
    </main>
  );
}
