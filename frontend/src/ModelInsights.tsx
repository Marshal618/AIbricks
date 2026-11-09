import { useEffect, useState, type FormEvent } from "react";
import * as api from "./api";
import type { ModelCard, ModelComparison, SavedModel } from "./types";

function message(error: unknown, fallback: string) {
  return error instanceof Error ? error.message : fallback;
}

function label(model: SavedModel) {
  const date = new Date(model.startedAt * 1000).toLocaleString(undefined, { dateStyle: "medium", timeStyle: "short" });
  return `${model.projectName} · ${model.datasetVersion !== null ? `v${model.datasetVersion}, ` : ""}${date}`;
}

function number(value: number) {
  return value.toLocaleString(undefined, { maximumFractionDigits: 2 });
}

// ------------------------------------------------------------------ comparing models

export function CompareModels({ datasetId, models, beginner }: { datasetId: string; models: SavedModel[]; beginner: boolean }) {
  const finished = models.filter((model) => model.status === "finished");
  const [first, setFirst] = useState(finished[1]?.id ?? "");
  const [second, setSecond] = useState(finished[0]?.id ?? "");
  const [result, setResult] = useState<ModelComparison | null>(null);
  const [problem, setProblem] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  // A model that has been deleted can't stay chosen.
  useEffect(() => {
    const ids = new Set(finished.map((model) => model.id));
    if (!ids.has(first)) setFirst(finished[1]?.id ?? "");
    if (!ids.has(second)) setSecond(finished[0]?.id ?? "");
  }, [finished, first, second]);

  if (finished.length < 2) return null;

  const run = async (event: FormEvent) => {
    event.preventDefault();
    setProblem(null);
    setResult(null);
    setBusy(true);
    try {
      setResult(await api.compareModels(datasetId, first, second));
    } catch (error) {
      setProblem(message(error, "The models couldn't be compared."));
    } finally {
      setBusy(false);
    }
  };

  const numbers = result?.task === "number";
  const tone = result?.verdict === "first" || result?.verdict === "second" ? "result-pass" : "notice";

  return (
    <section className="card">
      <h2>Compare two models</h2>
      <p className="muted small">
        {beginner
          ? "Checks both models on the same held-back examples, the ones neither learned from, and says whether one is really better or the gap could be luck."
          : "Paired comparison on the held-back examples both runs share: McNemar's exact test for categories, a sign test on absolute error for numbers."}
      </p>
      <form className="table-choices" onSubmit={(event) => void run(event)}>
        <label>
          Model A
          <select value={first} onChange={(event) => { setFirst(event.target.value); setResult(null); }}>
            {finished.map((model) => <option key={model.id} value={model.id}>{label(model)}</option>)}
          </select>
        </label>
        <label>
          Model B
          <select value={second} onChange={(event) => { setSecond(event.target.value); setResult(null); }}>
            {finished.map((model) => <option key={model.id} value={model.id}>{label(model)}</option>)}
          </select>
        </label>
        <div className="label-buttons" style={{ justifyContent: "flex-start", alignSelf: "end" }}>
          <button className="button button-small" type="submit" disabled={busy || !first || !second || first === second}>{busy ? "Comparing…" : "Compare"}</button>
        </div>
      </form>
      {first && first === second && <p className="muted small field-note">Choose two different models.</p>}
      {problem && <p className="notice field-note" role="alert">{problem}</p>}
      {result && (
        <div className={`result ${tone} field-note`} role="status">
          <p><strong>{result.summary}</strong></p>
          {result.firstScore !== undefined && result.secondScore !== undefined && (
            <table className="compare-table">
              <thead>
                <tr><th scope="col"></th><th scope="col">Model A</th><th scope="col">Model B</th></tr>
              </thead>
              <tbody>
                <tr>
                  <th scope="row">{numbers ? "Average error" : "Right"}</th>
                  <td>{numbers ? number(result.firstScore) : `${Math.round(result.firstScore * 100)}%`}</td>
                  <td>{numbers ? number(result.secondScore) : `${Math.round(result.secondScore * 100)}%`}</td>
                </tr>
                <tr>
                  <th scope="row">{numbers ? "Closer on" : "Only this one right"}</th>
                  <td>{result.onlyFirst}</td>
                  <td>{result.onlySecond}</td>
                </tr>
              </tbody>
            </table>
          )}
          <p className="muted small">
            {`Compared on ${result.shared.toLocaleString()} held-back ${result.shared === 1 ? "example" : "examples"} both models share.`}
            {result.bothRight !== undefined ? ` Both right on ${result.bothRight}, both wrong on ${result.neitherRight}.` : ""}
            {result.ties ? ` Equally close on ${result.ties}.` : ""}
          </p>
        </div>
      )}
    </section>
  );
}

// ---------------------------------------------------------------------- model cards

export function ModelCardView({ model, onClose }: { model: SavedModel; onClose: () => void }) {
  const [card, setCard] = useState<ModelCard | null>(null);
  const [use, setUse] = useState("");
  const [notFor, setNotFor] = useState("");
  const [problem, setProblem] = useState<string | null>(null);
  const [saved, setSaved] = useState(false);

  useEffect(() => {
    let live = true;
    api.fetchModelCard(model.id).then((next) => {
      if (!live) return;
      setCard(next);
      setUse(next.notes.intendedUse);
      setNotFor(next.notes.notFor);
    }).catch((error) => live && setProblem(message(error, "The model card couldn't load.")));
    return () => { live = false; };
  }, [model.id]);

  const changed = !!card && (use !== card.notes.intendedUse || notFor !== card.notes.notFor);

  const save = async (event: FormEvent) => {
    event.preventDefault();
    setProblem(null);
    try {
      setCard(await api.updateModelCard(model.id, { intendedUse: use, notFor }));
      setSaved(true);
    } catch (error) {
      setProblem(message(error, "The card couldn't be saved."));
    }
  };

  const download = () => {
    if (!card) return;
    const link = document.createElement("a");
    link.href = URL.createObjectURL(new Blob([card.markdown], { type: "text/markdown" }));
    link.download = "MODEL_CARD.md";
    link.click();
    URL.revokeObjectURL(link.href);
  };

  return (
    <section className="card model-card" aria-label={`Model card for ${model.projectName}`}>
      <div className="card-head">
        <h2>Model card: {model.projectName}</h2>
        <div className="label-buttons">
          <button className="button button-small" disabled={!card} onClick={download}>Download .md</button>
          <button className="button button-small button-quiet" onClick={onClose}>Close</button>
        </div>
      </div>
      {problem && <p className="notice field-note" role="alert">{problem}</p>}
      {!card && !problem && <p className="muted small">Loading…</p>}
      {card?.sections.map((section) => (
        <div key={section.title} className="model-card-section">
          <h3 className="small">{section.title}</h3>
          {section.editable ? (
            <form className="dataset-about" onSubmit={(event) => void save(event)}>
              <label>
                Use it for
                <textarea value={use} maxLength={2000} placeholder="What this model is meant for, and who will use it" onChange={(event) => { setUse(event.target.value); setSaved(false); }} />
              </label>
              <label>
                Don't use it for
                <textarea value={notFor} maxLength={2000} placeholder="Where it shouldn't be trusted, for example data unlike what it learned from" onChange={(event) => { setNotFor(event.target.value); setSaved(false); }} />
              </label>
              <div className="label-buttons" style={{ justifyContent: "flex-start", alignItems: "center" }}>
                <button className="button button-small" type="submit" disabled={!changed}>Save</button>
                {saved && !changed && <span className="muted small" role="status">Saved. Downloads include it.</span>}
              </div>
            </form>
          ) : (
            <ul>{section.lines.map((line) => <li key={line}>{line}</li>)}</ul>
          )}
        </div>
      ))}
    </section>
  );
}
