import { useEffect, useRef, useState } from 'react';
import { api } from '../api';
import { stageInBrowser } from '../localStage';

interface StageResult {
  staged: string;
  cutout: string;
  mask: string;
  comparison: string;
  cutoutBackend: string;
  sceneBackend: string;
}

interface SceneResponse {
  scene: string;
  sceneBackend: string;
}

const PRESETS = {
  Kitchen: 'product on a marble kitchen counter, soft window light, realistic lifestyle photography',
  Desk: 'product on a warm walnut desk, soft morning light, shallow depth of field, realistic lifestyle photography',
  Outdoor: 'product on a sunny outdoor patio table, natural light, realistic lifestyle photography',
  Studio: 'product on a clean neutral studio surface, soft diffused light, realistic catalog photography',
};

const dataUrl = (value: string): string => `data:image/png;base64,${value}`;

export default function StageView() {
  const [file, setFile] = useState<File | null>(null);
  const [prompt, setPrompt] = useState(PRESETS.Kitchen);
  const [preview, setPreview] = useState('');
  const [result, setResult] = useState<StageResult | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');
  const inputRef = useRef<HTMLInputElement | null>(null);

  useEffect(() => {
    if (!file) {
      setPreview('');
      return;
    }
    const url = URL.createObjectURL(file);
    setPreview(url);
    return () => URL.revokeObjectURL(url);
  }, [file]);

  const chooseFile = (next: File | undefined) => {
    if (!next || !next.type.startsWith('image/')) {
      setError('Please choose a PNG, JPEG, or WebP image.');
      return;
    }
    if (next.size > 12 * 1024 * 1024) {
      setError('Images must be smaller than 12 MB.');
      return;
    }
    setFile(next);
    setResult(null);
    setError('');
  };

  const stage = async () => {
    if (!file || !prompt.trim() || busy) return;
    setBusy(true);
    setError('');
    try {
      const encoded = await new Promise<string>((resolve, reject) => {
        const reader = new FileReader();
        reader.onload = () => {
          const value = String(reader.result ?? '');
          resolve(value.includes(',') ? value.slice(value.indexOf(',') + 1) : value);
        };
        reader.onerror = () => reject(new Error('Could not read the image.'));
        reader.readAsDataURL(file);
      });
      try {
        const next = await api<StageResult | SceneResponse>('/api/stage', {
          method: 'POST',
          body: JSON.stringify({ image: encoded, filename: file.name, prompt: prompt.trim() }),
        });
        if ('scene' in next) {
          setResult(await stageInBrowser(file, prompt.trim(), next.scene, next.sceneBackend));
        } else {
          setResult(next);
        }
      } catch (cause) {
        // A deployment without the optional Python service still works locally in the browser.
        const fallback = await stageInBrowser(file, prompt.trim());
        setResult(fallback);
        setError(cause instanceof Error
          ? `${cause.message} The result below is only a browser preview.`
          : 'The AI staging service is unavailable. The result below is only a browser preview.');
      }
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : String(cause));
    } finally {
      setBusy(false);
    }
  };

  return (
    <section className="stage-layout">
      <div className="panel stage-controls">
        <div className="stage-heading">
          <div>
            <p className="eyebrow">PRODUCT STAGING</p>
            <h2>Place your product in a scene</h2>
          </div>
          <span className="fidelity-badge">Product pixels protected</span>
        </div>

        <button className="upload-zone" type="button" onClick={() => inputRef.current?.click()}>
          {preview ? <img src={preview} alt="Selected product" /> : <span className="upload-icon">＋</span>}
          <span>{file ? file.name : 'Choose a product photo'}</span>
          <small>PNG, JPEG, or WebP · up to 12 MB</small>
        </button>
        <input
          ref={inputRef}
          className="visually-hidden"
          type="file"
          accept="image/png,image/jpeg,image/webp"
          onChange={(event) => chooseFile(event.target.files?.[0])}
        />

        <label className="field-label" htmlFor="scene-prompt">Scene description</label>
        <textarea
          id="scene-prompt"
          className="stage-prompt"
          rows={4}
          value={prompt}
          onChange={(event) => setPrompt(event.target.value)}
          placeholder="e.g. on a rustic wooden table, soft morning light"
        />
        <div className="preset-row" aria-label="Scene presets">
          {Object.entries(PRESETS).map(([name, value]) => (
            <button className={prompt === value ? 'preset active' : 'preset'} type="button" key={name} onClick={() => setPrompt(value)}>
              {name}
            </button>
          ))}
        </div>
        {error && <div className="notice error">{error}</div>}
        <button className="button primary stage-button" type="button" onClick={() => void stage()} disabled={!file || !prompt.trim() || busy}>
          {busy ? 'Staging…' : 'Generate staged photo'}
        </button>
        <p className="muted small stage-note">The original product is cut out first. Only the background and a narrow edge blend are generated.</p>
      </div>

      <div className="panel result-panel">
        {!result && !busy && <div className="empty-result"><span>✦</span><p>Your staged result will appear here.</p></div>}
        {busy && <div className="empty-result"><span className="spinner" /><p>Building the scene…</p></div>}
        {result && (
          <>
            <div className="result-heading">
              <div>
                <p className="eyebrow">RESULT</p>
                <h2>{result.sceneBackend === 'browser-scene-fallback' ? 'Preview only' : 'Ready to share'}</h2>
              </div>
              <a className="button primary" href={dataUrl(result.staged)} download="staged-product.png">Download PNG</a>
            </div>
            {result.sceneBackend === 'browser-scene-fallback' && (
              <div className="notice warning">
                Browser preview only: your prompt is reduced to a basic scene category. Connect the Python AI service for prompt-aware backgrounds.
              </div>
            )}
            {result.sceneBackend === 'fallback-scene' && (
              <div className="notice warning">
                Python service is connected, but its diffusion model is unavailable. This background is still a procedural fallback.
              </div>
            )}
            <img className="result-hero" src={dataUrl(result.staged)} alt="Staged product" />
            <div className="result-grid">
              <ResultCard label="Before / after" source={dataUrl(result.comparison)} download="comparison.png" />
              <ResultCard label="Transparent cutout" source={dataUrl(result.cutout)} download="product-cutout.png" />
              <ResultCard label="Binary mask" source={dataUrl(result.mask)} download="product-mask.png" />
            </div>
            <p className="muted small backend-note">Cutout: {result.cutoutBackend} · Scene: {result.sceneBackend}</p>
          </>
        )}
      </div>
    </section>
  );
}

function ResultCard(props: { label: string; source: string; download: string }) {
  return (
    <div className="result-card">
      <img src={props.source} alt={props.label} />
      <div className="result-card-footer"><span>{props.label}</span><a href={props.source} download={props.download} aria-label={`Download ${props.label}`}>↓</a></div>
    </div>
  );
}
