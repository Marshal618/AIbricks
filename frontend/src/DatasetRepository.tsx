import { useCallback, useEffect, useState, type FormEvent } from "react";
import * as api from "./api";
import { ExplainableText } from "./Explain";
import { ClipPreview, SoundMistakes, SoundTrainer, SoundTry } from "./Sound";
import { CompareModels, ModelCardView } from "./ModelInsights";
import { WebcamTry } from "./Webcam";
import { PreviewTable, TableResult, TableTrainer } from "./TableModels";
import type { DatasetPreview, DatasetSummary, PredictAnswer, SavedModel, TextMistake, VersionChanges, VersionOverview } from "./types";

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
        {preview?.table && <PreviewTable table={preview.table} />}
        {!dataset.labelled && (
          <div className="notice">
            <span>{dataset.kind === "table" ? "No data yet." : "No labelled examples yet."}</span>
            <button className="button button-small" onClick={onLabel}>{dataset.kind === "table" ? "Upload a CSV file" : "Add and label data"}</button>
          </div>
        )}
        {dataset.labelled > 0 && preview?.categories.map((category) => (
          <div className="preview-category" key={category.name}>
            <h3>{category.name} <span className="muted">{(dataset.counts[category.name] ?? 0).toLocaleString()}</span></h3>
            {category.samples.length === 0 && <p className="muted small">Nothing in this category yet.</p>}
            {dataset.kind === "sound"
              ? <ClipPreview dataset={dataset} category={category.name} clips={category.samples.map((sample) => sample.id)} />
              : dataset.kind === "image"
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
                  {when(version.createdAt)} · {amount(version.labelled, dataset.kind === "table" ? "row" : "labelled example")}{version.unsorted ? ` · ${version.unsorted.toLocaleString()} unsorted` : ""}
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

export function score(model: SavedModel) {
  if (model.status === "running") return "Training";
  if (model.metrics?.task === "number") return `±${(model.metrics.mae ?? 0).toLocaleString(undefined, { maximumFractionDigits: 2 })}`;
  if (!model.best || model.best.accuracy === null) return model.status === "failed" ? "Failed" : "No score";
  return `${Math.round(model.best.accuracy * 100)}%`;
}

function megabytes(bytes: number) {
  return bytes >= 1024 * 1024 ? `${(bytes / (1024 * 1024)).toFixed(1)} MB` : `${Math.max(1, Math.round(bytes / 1024))} KB`;
}

const TEXT_MINIMUM = 4;
const TEXT_SUGGESTED = 16;

function TextTrainer({ dataset, beginner, running, busyElsewhere, onStart, onStop }: { dataset: DatasetSummary; beginner: boolean; running: SavedModel | null; busyElsewhere: boolean; onStart: () => void; onStop: (id: string) => void }) {
  const short = dataset.categories.filter((name) => (dataset.counts[name] ?? 0) < TEXT_MINIMUM);
  const ready = dataset.categories.length >= 2 && short.length === 0;
  const few = ready && dataset.categories.some((name) => (dataset.counts[name] ?? 0) < TEXT_SUGGESTED);
  return (
    <section className="card">
      <div className="card-head">
        <h2>Train a text sorter</h2>
        {running
          ? <button className="button button-small" onClick={() => onStop(running.id)}>Stop</button>
          : <button className="button button-small button-primary" disabled={!ready || busyElsewhere} onClick={onStart}>Start training</button>}
      </div>
      <p className="muted small">
        {beginner
          ? "This adapts a small pretrained language model to your categories, so it works with few examples. Training runs on this computer and your text isn't sent anywhere."
          : "Uses the SetFit method: fine-tunes all-MiniLM-L6-v2 on contrastive pairs, then fits a linear classifier. Training is local."}
      </p>
      {running && (
        <div className="field-note" role="status">
          <progress className="train-progress" max={running.stepsTotal || 1} value={running.stepsDone} aria-label="Training progress" />
          <p className="muted small">
            {!running.device
              ? "Getting the base model ready. The first run downloads about 90 MB from Hugging Face. Your text stays here."
              : running.stage === "reviewing"
                ? "Checking the held-back examples…"
                : `Step ${running.stepsDone.toLocaleString()} of ${(running.stepsTotal || 0).toLocaleString()}, using ${running.device}. This can be left running.`}
          </p>
        </div>
      )}
      {!running && busyElsewhere && <p className="muted small field-note">Another model is training. Only one can train at a time.</p>}
      {!running && dataset.categories.length < 2 && <p className="muted small field-note">Add at least 2 categories on the Label tab.</p>}
      {!running && dataset.categories.length >= 2 && short.length > 0 && <p className="muted small field-note">Label at least {TEXT_MINIMUM} examples in: {short.join(", ")}.</p>}
      {!running && few && <p className="muted small field-note">You can train now, but scores are more trustworthy with at least {TEXT_SUGGESTED} examples per category.</p>}
    </section>
  );
}

function TryText({ model, beginner }: { model: SavedModel; beginner: boolean }) {
  const [text, setText] = useState("");
  const [answers, setAnswers] = useState<PredictAnswer[] | null>(null);
  const [asked, setAsked] = useState("");
  const [asking, setAsking] = useState(false);
  const [problem, setProblem] = useState<string | null>(null);

  const ask = async (event: FormEvent) => {
    event.preventDefault();
    setAsking(true);
    setProblem(null);
    setAnswers(null);
    try {
      setAnswers(await api.tryTextModel(model.id, text));
      setAsked(text);
    } catch (error) {
      setProblem(message(error, "The model couldn't answer."));
    } finally {
      setAsking(false);
    }
  };

  return (
    <section className="card">
      <h2>Try the newest model</h2>
      <form className="dataset-about" onSubmit={(event) => void ask(event)}>
        <label>
          <span className="visually-hidden">Text to sort</span>
          <textarea value={text} maxLength={5000} placeholder={beginner ? "Type something the model hasn't seen" : "Text to classify"} onChange={(event) => setText(event.target.value)} />
        </label>
        <div><button className="button button-small" type="submit" disabled={asking || !text.trim()}>{asking ? "Asking…" : "Ask the model"}</button></div>
      </form>
      {problem && <p className="small field-note" role="alert">{problem}</p>}
      {answers && (
        <ul className="project-list field-note" role="status">
          {answers.slice(0, 5).map((answer) => (
            <li key={answer.category}>
              <span className="model-score">{Math.round(answer.chance * 100)}%</span>
              <span className="project-text">{answer.category}</span>
            </li>
          ))}
        </ul>
      )}
      {answers && <div className="field-note"><ExplainableText key={asked} jobId={model.id} text={asked} beginner={beginner} /></div>}
    </section>
  );
}

function TextMistakes({ model, dataset, beginner, onChanged }: { model: SavedModel; dataset: DatasetSummary; beginner: boolean; onChanged: () => void }) {
  const [rows, setRows] = useState<TextMistake[] | null>(null);
  const [total, setTotal] = useState(0);
  const [problem, setProblem] = useState<string | null>(null);

  useEffect(() => {
    api.fetchTextMistakes(model.id)
      .then((reply) => { setRows(reply.mistakes); setTotal(reply.total); })
      .catch((error) => setProblem(message(error, "The mistakes could not load.")));
  }, [model.id]);

  const relabel = async (row: TextMistake, category: string) => {
    setProblem(null);
    try {
      await api.assignText(row.id, category, dataset.id);
      setRows((current) => current?.map((item) => item.id === row.id ? { ...item, label: category } : item) ?? null);
      onChanged();
    } catch (error) {
      setProblem(message(error, "That label couldn't be changed."));
    }
  };

  if (!rows && !problem) return null;
  return (
    <section className="card">
      <h2>Mistakes from the newest model</h2>
      {problem && <p className="small" role="alert">{problem}</p>}
      {rows?.length === 0 && <p className="muted small">{total ? "The texts it got wrong are no longer in this dataset." : "It got every held-back example right."}</p>}
      {rows && rows.length > 0 && (
        <>
          <p className="muted small">
            {beginner ? "Sometimes the model is right and the label is wrong. Fix any wrong labels here, then train again and see if the score improves." : "Corrections apply to the dataset and take effect from the next run."}
            {total > rows.length ? ` Showing ${rows.length} of ${total}.` : ""}
          </p>
          <ul className="project-list">
            {rows.map((row) => (
              <li key={row.id}>
                <div className="project-text">
                  <span>{row.text}</span>
                  <span className="muted small">Filed under {row.actual}, model guessed {row.guess} ({Math.round(row.chance * 100)}% sure)</span>
                  <div><ExplainableText jobId={model.id} text={row.text} beginner={beginner} /></div>
                </div>
                <label className="text-relabel">
                  <span className="visually-hidden">Label for this text</span>
                  <select value={row.label ?? ""} onChange={(event) => void relabel(row, event.target.value)}>
                    {row.label !== null && !dataset.categories.includes(row.label) && <option value={row.label}>{row.label}</option>}
                    {dataset.categories.map((name) => <option key={name} value={name}>{name}</option>)}
                  </select>
                </label>
              </li>
            ))}
          </ul>
        </>
      )}
    </section>
  );
}

export function DatasetModels({ dataset, beginner, onBuild, onChanged }: { dataset: DatasetSummary; beginner: boolean; onBuild: () => void; onChanged: () => void }) {
  const [models, setModels] = useState<SavedModel[] | null>(null);
  const [known, setKnown] = useState<Set<number>>(new Set());
  const [problem, setProblem] = useState<string | null>(null);
  const [busyElsewhere, setBusyElsewhere] = useState(false);
  const [cardFor, setCardFor] = useState<string | null>(null);

  const load = useCallback(async () => {
    const [rows, versions, latest] = await Promise.all([api.fetchDatasetModels(dataset.id), api.fetchVersions(dataset.id), api.fetchTraining()]);
    setModels(rows);
    setKnown(new Set(versions.versions.map((version) => version.number)));
    setBusyElsewhere(latest?.status === "running" && latest.dataset !== dataset.id);
  }, [dataset.id]);

  useEffect(() => { load().catch((error) => setProblem(message(error, "The models could not load."))); }, [load]);

  const running = models?.find((model) => model.status === "running") ?? null;
  const newestText = models?.find((model) => model.task === "text" && model.status !== "running" && model.hasModel) ?? null;

  // Training runs on the server, so this keeps up with it, and leaving the page doesn't stop it.
  useEffect(() => {
    if (!running && !busyElsewhere) return;
    const timer = window.setInterval(() => { load().catch(() => undefined); }, 1000);
    return () => window.clearInterval(timer);
  }, [running, busyElsewhere, load]);

  const startText = async () => {
    setProblem(null);
    try {
      await api.trainTextModel(dataset.id);
      await load();
      onChanged();
    } catch (error) {
      setProblem(message(error, "Training could not start."));
    }
  };

  const startSound = async () => {
    setProblem(null);
    try {
      await api.trainSoundModel(dataset.id);
      await load();
      onChanged();
    } catch (error) {
      setProblem(message(error, "Training could not start."));
    }
  };

  const stop = async (id: string) => {
    setProblem(null);
    try {
      await api.stopTraining(id);
      await load();
    } catch (error) {
      setProblem(message(error, "Training could not be stopped."));
    }
  };
  const failed = models?.[0]?.status === "failed" ? models[0] : null;
  const cardForModel = models?.find((model) => model.id === cardFor) ?? null;
  const newestTable = models?.find((model) => model.task === "table" && model.status !== "running" && model.hasModel) ?? null;
  const newestPhoto = models?.find((model) => model.task === "image" && model.status === "finished" && model.hasModel) ?? null;
  const newestSound = models?.find((model) => model.task === "sound" && model.status === "finished" && model.hasModel) ?? null;

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
      {failed && <div className="result result-fail" role="alert"><p><strong>The last training run failed.</strong> {failed.error}</p></div>}
      {dataset.kind === "text" && <TextTrainer dataset={dataset} beginner={beginner} running={running} busyElsewhere={busyElsewhere} onStart={() => void startText()} onStop={(id) => void stop(id)} />}
      {dataset.kind === "sound" && <SoundTrainer dataset={dataset} beginner={beginner} running={running} busyElsewhere={busyElsewhere} onStart={() => void startSound()} onStop={(id) => void stop(id)} />}
      {dataset.kind === "table" && <TableTrainer dataset={dataset} beginner={beginner} running={running} busyElsewhere={busyElsewhere} onStarted={load} onStop={(id) => void stop(id)} />}
      {newestTable && <TableResult key={newestTable.id} model={newestTable} beginner={beginner} />}
      {newestPhoto && (
        <section className="card">
          <h2>Try the newest model</h2>
          <p className="muted small">{`${newestPhoto.projectName}, trained on data v${newestPhoto.datasetVersion ?? "?"}. Hold things up to the camera and watch its guesses change.`}</p>
          <WebcamTry key={newestPhoto.id} jobId={newestPhoto.id} beginner={beginner} />
        </section>
      )}
      {newestSound && <SoundTry key={newestSound.id} model={newestSound} beginner={beginner} />}
      {newestSound && <SoundMistakes key={`m${newestSound.id}`} model={newestSound} dataset={dataset} beginner={beginner} onChanged={onChanged} />}
      {newestText && <TryText key={newestText.id} model={newestText} beginner={beginner} />}
      {newestText && <TextMistakes key={`m${newestText.id}`} model={newestText} dataset={dataset} beginner={beginner} onChanged={onChanged} />}
      <section className="card">
        <h2>Models trained on this data</h2>
        <p className="muted small">
          {beginner
            ? "Each model shows which version of the data trained it and how often it was right on held-back examples. Scores from small datasets are rough. To see if one model is really better, compare them below."
            : "Score is accuracy on each run's held-back split. Splits are chosen by content fingerprint, so runs share most held-back examples; use Compare for a paired test. The 10 newest model files are kept."}
        </p>
        {models?.length === 0 && dataset.kind === "image" && (
          <div className="notice field-note">
            <span>No models yet.</span>
            <button className="button button-small" disabled={!dataset.labelled} onClick={onBuild}>Build a model with this data</button>
          </div>
        )}
        {models?.length === 0 && dataset.kind !== "image" && <p className="muted small field-note">No models yet.</p>}
        <ul className="project-list">
          {models?.map((model) => (
            <li key={model.id}>
              <span className="model-score">{score(model)}</span>
              <div className="project-text">
                <strong>{model.projectName}</strong>
                <span className="muted small">
                  {model.datasetVersion !== null ? `Data v${model.datasetVersion}${known.has(model.datasetVersion) ? "" : " (deleted)"}` : "No data version"}
                  {" · "}{when(model.startedAt)}
                  {model.metrics?.task === "number"
                    ? ` · typical error; always guessing the average would be off by ${(model.metrics.baselineMae ?? 0).toLocaleString(undefined, { maximumFractionDigits: 2 })}`
                    : model.best ? ` · guessing would score ${Math.round((model.metrics?.baseline ?? model.baseline) * 100)}%` : ""}
                  {model.review ? ` · ${model.review.correct} of ${model.review.total} held-back right` : ""}
                  {model.hasModel ? ` · ${megabytes(model.modelBytes)}` : ""}
                  {model.modelRemoved ? " · model file removed to save space" : ""}
                </span>
              </div>
              <div className="label-buttons">
                {model.hasModel && model.status !== "running" && <a className="button button-small" href={`/api/train/${model.id}/download`} aria-label={`Download ${model.projectName} from ${when(model.startedAt)}`}>Download</a>}
                {model.status === "finished" && <button className="button button-small" aria-pressed={cardFor === model.id} aria-label={`Model card for ${model.projectName} from ${when(model.startedAt)}`} onClick={() => setCardFor(cardFor === model.id ? null : model.id)}>Card</button>}
                {model.status !== "running" && <button className="button button-small button-quiet" aria-label={`Delete ${model.projectName} from ${when(model.startedAt)}`} onClick={() => void remove(model)}>Delete</button>}
              </div>
            </li>
          ))}
        </ul>
      </section>
      {cardForModel && <ModelCardView key={`card-${cardForModel.id}`} model={cardForModel} onClose={() => setCardFor(null)} />}
      {models && <CompareModels datasetId={dataset.id} models={models} beginner={beginner} />}
    </>
  );
}
