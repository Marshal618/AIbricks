import { useCallback, useEffect, useState, type FormEvent } from "react";
import * as api from "./api";
import type { DatasetPreview, DatasetSummary, SavedModel, VersionChanges, VersionOverview } from "./types";

const LICENCES = ["CC0", "CC BY 4.0", "CC BY-SA 4.0", "MIT", "Apache 2.0", "Private, not for sharing"];

function message(error: unknown, fallback: string) {
  return error instanceof Error ? error.message : fallback;
}

function when(seconds: number) {
  return new Date(seconds * 1000).toLocaleString(undefined, { dateStyle: "medium", timeStyle: "short" });
}

function amount(count: number, one: string, many = `${one}s`) {
  return `${count.toLocaleString()} ${count === 1 ? one : many}`;
}

function describe(changes: VersionChanges) {
  const parts = [];
  if (changes.added) parts.push(`${changes.added.toLocaleString()} added`);
  if (changes.removed) parts.push(`${changes.removed.toLocaleString()} removed`);
  if (changes.relabelled) parts.push(amount(changes.relabelled, "label") + " changed");
  if (changes.categoriesAdded.length) parts.push(`new: ${changes.categoriesAdded.join(", ")}`);
  if (changes.categoriesRemoved.length) parts.push(`gone: ${changes.categoriesRemoved.join(", ")}`);
  return parts.join(" · ") || "No changes";
}

// ------------------------------------------------------------------- overview

export function DatasetOverview({ dataset, beginner, onSaved, onLabel }: { dataset: DatasetSummary; beginner: boolean; onSaved: (next: DatasetSummary) => void; onLabel: () => void }) {
  const [preview, setPreview] = useState<DatasetPreview | null>(null);
  const [description, setDescription] = useState(dataset.description);
  const [licence, setLicence] = useState(dataset.licence);
  const [status, setStatus] = useState<string | null>(null);
  const [problem, setProblem] = useState<string | null>(null);

  useEffect(() => {
    api.fetchDatasetPreview(dataset.id).then(setPreview).catch((error) => setProblem(message(error, "The preview could not load.")));
  }, [dataset.id]);

  const save = async (event: FormEvent) => {
    event.preventDefault();
    setProblem(null);
    try {
      onSaved(await api.updateDatasetDetails(dataset.id, description, licence));
      setStatus("Saved.");
    } catch (error) {
      setProblem(message(error, "The details couldn't be saved."));
    }
  };

  const changed = description !== dataset.description || licence !== dataset.licence;
  const unknown = preview?.sources.filter((source) => !source.licence || source.licence.toLowerCase() === "unknown" || source.licence.toLowerCase().includes("not ")) ?? [];

  return (
    <>
      {problem && <div className="result result-fail" role="alert"><p>{problem}</p></div>}
      <section className="card">
        <h2>About this dataset</h2>
        <form className="dataset-about" onSubmit={(event) => void save(event)}>
          <label>
            Description
            <textarea value={description} maxLength={2000} placeholder="What's in this data, where it came from, and what it's good for" onChange={(event) => { setDescription(event.target.value); setStatus(null); }} />
          </label>
          <label>
            Licence
            <input type="text" list="licence-options" value={licence} maxLength={60} placeholder="Not stated" onChange={(event) => { setLicence(event.target.value); setStatus(null); }} />
            <datalist id="licence-options">{LICENCES.map((name) => <option key={name} value={name} />)}</datalist>
            {beginner && <span className="muted small">A licence says what other people may do with this data. Only choose one if you have the right to.</span>}
          </label>
          <div className="label-buttons" style={{ justifyContent: "flex-start", alignItems: "center" }}>
            <button className="button button-small" type="submit" disabled={!changed}>Save details</button>
            {status && !changed && <span className="muted small" role="status">{status}</span>}
          </div>
        </form>
      </section>

      {preview && preview.sources.length > 0 && (
        <section className="card">
          <h2>Data sources</h2>
          <ul className="project-list">
            {preview.sources.map((source) => (
              <li key={source.dataset}>
                <div className="project-text">
                  <a href={source.url} target="_blank" rel="noreferrer">{source.dataset}</a>
                  <span className="muted small">Licence: {source.licence || "unknown"} · {amount(source.added, "item")} imported</span>
                </div>
              </li>
            ))}
          </ul>
          {unknown.length > 0 && <p className="notice field-note">Check the licence on the source page before sharing this data or a model trained on it.</p>}
        </section>
      )}

      <section className="card">
        <h2>Preview</h2>
        {!dataset.labelled && (
          <div className="notice">
            <span>No labelled examples yet.</span>
            <button className="button button-small" onClick={onLabel}>Add and label data</button>
          </div>
        )}
        {dataset.labelled > 0 && preview?.categories.map((category) => (
          <div className="preview-category" key={category.name}>
            <h3>{category.name} <span className="muted">{(dataset.counts[category.name] ?? 0).toLocaleString()}</span></h3>
            {category.samples.length === 0 && <p className="muted small">Nothing in this category yet.</p>}
            {dataset.kind === "image"
              ? <ul className="preview-images">{category.samples.map((sample, index) => (
                  <li key={sample.id}><img src={api.datasetPreviewImage(dataset.id, category.name, sample.id)} alt={`${category.name} example ${index + 1}`} loading="lazy" /></li>
                ))}</ul>
              : <ul className="preview-text">{category.samples.map((sample) => <li key={sample.id}>{sample.text}</li>)}</ul>}
          </div>
        ))}
      </section>
    </>
  );
}

// ------------------------------------------------------------------- versions

export function DatasetVersions({ dataset, beginner, onRestored }: { dataset: DatasetSummary; beginner: boolean; onRestored: (next: DatasetSummary) => void }) {
  const [data, setData] = useState<VersionOverview | null>(null);
  const [note, setNote] = useState("");
  const [busy, setBusy] = useState(false);
  const [problem, setProblem] = useState<string | null>(null);
  const [status, setStatus] = useState<string | null>(null);
  const [comparison, setComparison] = useState<{ number: number; changes: VersionChanges } | null>(null);

  const load = useCallback(async () => setData(await api.fetchVersions(dataset.id)), [dataset.id]);

  useEffect(() => { load().catch((error) => setProblem(message(error, "The versions could not load."))); }, [load]);

  const attempt = async (action: () => Promise<void>, fallback: string) => {
    setBusy(true);
    setProblem(null);
    setStatus(null);
    setComparison(null);
    try {
      await action();
      await load();
    } catch (error) {
      setProblem(message(error, fallback));
    } finally {
      setBusy(false);
    }
  };

  const save = (event: FormEvent) => {
    event.preventDefault();
    void attempt(async () => {
      const saved = await api.saveVersion(dataset.id, note);
      setNote("");
      setStatus(`Saved v${saved.number}.`);
    }, "The version couldn't be saved.");
  };

  const restore = (number: number) => {
    if (!window.confirm(`Make this dataset match v${number}? Any unsaved changes are saved as a new version first, so you can come back to them.`)) return;
    void attempt(async () => {
      const reply = await api.restoreVersion(dataset.id, number);
      onRestored(reply.dataset);
      setStatus(reply.savedFirst ? `Restored v${number}. Your previous data is saved as v${reply.savedFirst}.` : `Restored v${number}.`);
    }, "That version couldn't be restored.");
  };

  const remove = (number: number) => {
    if (!window.confirm(`Delete v${number}? Examples that only this version still holds are erased for good. Models trained on it keep working, but you can't get that exact data back.`)) return;
    void attempt(async () => {
      await api.deleteVersion(dataset.id, number);
      setStatus(`Deleted v${number}.`);
    }, "That version couldn't be deleted.");
  };

  const compare = (number: number) => {
    if (comparison?.number === number) {
      setComparison(null);
      return;
    }
    setProblem(null);
    api.compareVersion(dataset.id, number)
      .then((changes) => setComparison({ number, changes }))
      .catch((error) => setProblem(message(error, "The comparison failed.")));
  };

  return (
    <>
      {problem && <div className="result result-fail" role="alert"><p>{problem}</p></div>}
      {status && <div className="notice notice-good" role="status"><span>{status}</span></div>}
      <section className="card">
        <h2>Save a version</h2>
        <p className="muted small">
          {beginner
            ? "A version is a saved copy of your data and labels that you can return to. One is also saved automatically whenever you train, so you can always tell which data made which model."
            : "Snapshot examples, labels, categories and credits. Training saves one automatically when the data has changed."}
        </p>
        {data && (data.unsaved
          ? <p className="small field-note"><strong>Since v{data.head}:</strong> {describe(data.unsaved)}</p>
          : data.head === null && <p className="small field-note">{data.canSave ? "No versions saved yet." : "Add some examples first."}</p>)}
        <form className="version-save" onSubmit={save}>
          <input type="text" value={note} maxLength={120} aria-label="What changed" placeholder="What changed? For example: fixed labels on blurry photos" onChange={(event) => setNote(event.target.value)} />
          <button className="button button-primary" type="submit" disabled={busy || !data?.canSave}>Save version</button>
        </form>
      </section>
      <section className="card">
        <h2>History</h2>
        {data?.versions.length === 0 && <p className="muted small">Nothing saved yet.</p>}
        <ul className="project-list">
          {data?.versions.map((version) => (
            <li key={version.number}>
              <div className="project-text">
                <span>
                  <span className="version-tag">v{version.number}</span>
                  {version.message || <span className="muted">No description</span>}
                  {version.number === data.head && <span className="version-badge">{data.canSave ? "based on this" : "current"}</span>}
                </span>
                <span className="muted small">
                  {when(version.createdAt)} · {amount(version.labelled, "labelled example")}{version.unsorted ? ` · ${version.unsorted.toLocaleString()} unsorted` : ""}
                  {version.changes ? ` · ${describe(version.changes)}` : ""}
                </span>
              </div>
              <div className="label-buttons">
                <button className="button button-small button-quiet" aria-label={`Compare v${version.number} with the data now`} aria-expanded={comparison?.number === version.number} onClick={() => compare(version.number)}>Compare</button>
                <button className="button button-small button-quiet" aria-label={`Restore v${version.number}`} disabled={busy} onClick={() => restore(version.number)}>Restore</button>
                <button className="button button-small button-quiet" aria-label={`Delete v${version.number}`} disabled={busy} onClick={() => remove(version.number)}>Delete</button>
              </div>
              {comparison?.number === version.number && (
                <div className="version-compare">
                  <strong>From v{version.number} to the data now:</strong> {describe(comparison.changes)}
                  {comparison.changes.counts.length > 0 && <div className="muted">{comparison.changes.counts.map((row) => `${row.name}: ${row.before.toLocaleString()} to ${row.after.toLocaleString()}`).join(" · ")}</div>}
                </div>
              )}
            </li>
          ))}
        </ul>
        <p className="muted small field-note">Removing an example from the dataset leaves it in older versions. To erase it completely, delete the versions that hold it.</p>
      </section>
    </>
  );
}

// --------------------------------------------------------------------- models

function score(model: SavedModel) {
  if (model.status === "running") return "Training";
  if (!model.best || model.best.accuracy === null) return model.status === "failed" ? "Failed" : "No score";
  return `${Math.round(model.best.accuracy * 100)}%`;
}

function megabytes(bytes: number) {
  return bytes >= 1024 * 1024 ? `${(bytes / (1024 * 1024)).toFixed(1)} MB` : `${Math.max(1, Math.round(bytes / 1024))} KB`;
}

export function DatasetModels({ dataset, beginner, onBuild }: { dataset: DatasetSummary; beginner: boolean; onBuild: () => void }) {
  const [models, setModels] = useState<SavedModel[] | null>(null);
  const [known, setKnown] = useState<Set<number>>(new Set());
  const [problem, setProblem] = useState<string | null>(null);

  const load = useCallback(async () => {
    const [rows, versions] = await Promise.all([api.fetchDatasetModels(dataset.id), api.fetchVersions(dataset.id)]);
    setModels(rows);
    setKnown(new Set(versions.versions.map((version) => version.number)));
  }, [dataset.id]);

  useEffect(() => { load().catch((error) => setProblem(message(error, "The models could not load."))); }, [load]);

  const remove = async (model: SavedModel) => {
    if (!window.confirm(`Delete this "${model.projectName}" model? Its record and trained file are removed. This can't be undone.`)) return;
    setProblem(null);
    try {
      await api.deleteSavedModel(model.id);
      await load();
    } catch (error) {
      setProblem(message(error, "The model couldn't be deleted."));
    }
  };

  return (
    <>
      {problem && <div className="result result-fail" role="alert"><p>{problem}</p></div>}
      <section className="card">
        <h2>Models trained on this data</h2>
        <p className="muted small">
          {beginner
            ? "Each model shows which version of the data trained it and how often it was right on held-back examples. Scores from small datasets are rough, and each run can hold back different examples."
            : "Score is accuracy on the held-back split of each run, so it's a rough comparison across versions. The 10 newest model files are kept."}
        </p>
        {models?.length === 0 && (
          <div className="notice field-note">
            <span>No models yet.</span>
            <button className="button button-small" disabled={!dataset.labelled} onClick={onBuild}>Build a model with this data</button>
          </div>
        )}
        <ul className="project-list">
          {models?.map((model) => (
            <li key={model.id}>
              <span className="model-score">{score(model)}</span>
              <div className="project-text">
                <strong>{model.projectName}</strong>
                <span className="muted small">
                  {model.datasetVersion !== null ? `Data v${model.datasetVersion}${known.has(model.datasetVersion) ? "" : " (deleted)"}` : "No data version"}
                  {" · "}{when(model.startedAt)}
                  {model.best ? ` · guessing would score ${Math.round(model.baseline * 100)}%` : ""}
                  {model.review ? ` · ${model.review.correct} of ${model.review.total} held-back right` : ""}
                  {model.hasModel ? ` · ${megabytes(model.modelBytes)}` : ""}
                  {model.modelRemoved ? " · model file removed to save space" : ""}
                </span>
              </div>
              <div className="label-buttons">
                {model.hasModel && model.status !== "running" && <a className="button button-small" href={`/api/train/${model.id}/download`} aria-label={`Download ${model.projectName} from ${when(model.startedAt)}`}>Download</a>}
                {model.status !== "running" && <button className="button button-small button-quiet" aria-label={`Delete ${model.projectName} from ${when(model.startedAt)}`} onClick={() => void remove(model)}>Delete</button>}
              </div>
            </li>
          ))}
        </ul>
      </section>
    </>
  );
}
