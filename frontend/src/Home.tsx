import { useEffect, useState } from "react";
import * as api from "./api";
import { score } from "./DatasetRepository";
import type { DatasetSummary, Mode, ProjectRow, SystemInfo, WorkspaceModel } from "./types";

interface Props {
  mode: Mode;
  currentProjectId: string;
  onNewModel: () => void;
  onContinueModel: () => void;
  onOpenModel: (id: string) => void;
  onOpenData: () => void;
  onOpenDataset: (id: string, tab?: "models") => void;
}

const RECENT_SHOWN = 5;
const RUNS_SHOWN = 8;
const POLL_WHILE_TRAINING = 3000;

const KIND = {
  image: { name: "Images", className: "is-image" },
  text: { name: "Text", className: "is-text" },
  table: { name: "Spreadsheet", className: "is-table" },
  sound: { name: "Sound", className: "is-sound" },
  anomaly: { name: "Unusual photos", className: "is-image" },
  pose: { name: "Poses", className: "is-image" },
  detect: { name: "Find things", className: "is-image" },
} as const;

const STATUS: Record<string, { name: string; className: string; mark: string }> = {
  running: { name: "Training", className: "is-running", mark: "●" },
  finished: { name: "Finished", className: "is-finished", mark: "✓" },
  failed: { name: "Failed", className: "is-failed", mark: "✕" },
  stopped: { name: "Stopped", className: "is-stopped", mark: "■" },
};

function bytes(value: number) {
  if (value < 1024) return `${value} B`;
  const units = ["KB", "MB", "GB", "TB"];
  let amount = value / 1024;
  let unit = 0;
  while (amount >= 1024 && unit < units.length - 1) { amount /= 1024; unit += 1; }
  return `${amount.toFixed(amount < 10 ? 1 : 0)} ${units[unit]}`;
}

function ago(seconds: number) {
  const gone = Date.now() / 1000 - seconds;
  if (gone < 60) return "just now";
  if (gone < 3600) return `${Math.floor(gone / 60)} min ago`;
  if (gone < 86400) return `${Math.floor(gone / 3600)} h ago`;
  if (gone < 86400 * 7) return `${Math.floor(gone / 86400)} d ago`;
  return new Date(seconds * 1000).toLocaleDateString();
}

function KindTag({ kind }: { kind: keyof typeof KIND }) {
  return <span className={`kind-tag ${KIND[kind].className}`}>{KIND[kind].name}</span>;
}

function StatusTag({ run }: { run: WorkspaceModel }) {
  const status = STATUS[run.status] ?? { name: run.status, className: "", mark: "·" };
  const progress = run.status === "running" && run.stepsTotal ? Math.min(1, run.stepsDone / run.stepsTotal) : null;
  return (
    <span className={`status-tag ${status.className}`}>
      <span aria-hidden="true">{status.mark}</span> {status.name}
      {progress !== null && <span className="status-progress" role="progressbar" aria-label="Training progress" aria-valuenow={Math.round(progress * 100)} aria-valuemin={0} aria-valuemax={100}><span style={{ width: `${progress * 100}%` }} /></span>}
    </span>
  );
}

export function Home({ mode, currentProjectId, onNewModel, onContinueModel, onOpenModel, onOpenData, onOpenDataset }: Props) {
  const beginner = mode === "beginner";
  const [projects, setProjects] = useState<ProjectRow[] | null>(null);
  const [datasets, setDatasets] = useState<DatasetSummary[] | null>(null);
  const [runs, setRuns] = useState<WorkspaceModel[] | null>(null);
  const [system, setSystem] = useState<SystemInfo | null>(null);
  const [problem, setProblem] = useState<string | null>(null);

  useEffect(() => {
    let stopped = false;
    Promise.all([api.fetchProjects(), api.fetchDatasets(), api.fetchAllModels()])
      .then(([nextProjects, nextDatasets, nextRuns]) => {
        if (stopped) return;
        setProjects(nextProjects);
        setDatasets(nextDatasets);
        setRuns(nextRuns);
      })
      .catch((error) => {
        if (stopped) return;
        setProjects([]);
        setDatasets([]);
        setRuns([]);
        setProblem(error instanceof Error ? error.message : "Your work could not load.");
      });
    // The system check starts Python once, so it loads on its own and never holds up the rest.
    api.fetchSystem().then((next) => { if (!stopped) setSystem(next); }).catch(() => undefined);
    return () => { stopped = true; };
  }, []);

  // Keeps a training run's progress moving while it's on screen.
  const training = runs?.some((run) => run.status === "running") ?? false;
  useEffect(() => {
    if (!training) return;
    const timer = window.setInterval(() => { api.fetchAllModels().then(setRuns).catch(() => undefined); }, POLL_WHILE_TRAINING);
    return () => window.clearInterval(timer);
  }, [training]);

  const loaded = projects !== null && datasets !== null && runs !== null;
  const labelled = datasets?.reduce((sum, row) => sum + (row.kind === "table" ? 0 : row.labelled), 0) ?? 0;
  const rows = datasets?.reduce((sum, row) => sum + (row.kind === "table" ? row.labelled : 0), 0) ?? 0;
  const finished = runs?.filter((run) => run.status === "finished") ?? [];
  const running = runs?.filter((run) => run.status === "running").length ?? 0;
  const shownRuns = runs?.slice(0, RUNS_SHOWN) ?? [];

  const steps = [
    { done: (datasets?.length ?? 0) > 0, name: "Create a dataset", detail: "Images, text or a spreadsheet.", action: onOpenData, button: "Open data" },
    { done: labelled + rows >= 20, name: "Add and label examples", detail: "Find open data, or sort your own. 20 per category is a good start.", action: onOpenData, button: "Label" },
    { done: finished.length > 0, name: "Train a model", detail: "Build one from blocks, or train straight from a dataset's Models tab.", action: onNewModel, button: "Build" },
  ];
  const showSteps = beginner && loaded && steps.some((step) => !step.done);

  return (
    <div className="page-scroll">
      <div className="page-inner home">
        <header className="home-head">
          <div className="home-title">
            <h1>{beginner ? "Welcome back" : "Workspace"}</h1>
            <p className="muted">
              {beginner
                ? "Get your data ready, then build a model that learns from it. Everything stays on this computer."
                : "Datasets, model builds and training runs on this computer."}
            </p>
          </div>
          <div className="home-head-actions">
            <button className="button" onClick={onOpenData}>Label data</button>
            <button className="button button-primary" onClick={onNewModel}>New model</button>
          </div>
        </header>

        {problem && <div className="result result-fail" role="alert"><p>{problem}</p></div>}

        <dl className="stat-strip" aria-label="Workspace summary">
          <div className="stat">
            <dt>Datasets</dt>
            <dd>{loaded ? datasets.length.toLocaleString() : "–"}</dd>
            <span className="stat-note muted">{loaded ? `${labelled.toLocaleString()} labelled examples${rows ? `, ${rows.toLocaleString()} rows` : ""}` : " "}</span>
          </div>
          <div className="stat">
            <dt>Model builds</dt>
            <dd>{loaded ? projects.length.toLocaleString() : "–"}</dd>
            <span className="stat-note muted">Saved block designs</span>
          </div>
          <div className="stat">
            <dt>Trained models</dt>
            <dd>{loaded ? finished.length.toLocaleString() : "–"}</dd>
            <span className="stat-note muted">{running ? `${running} training now` : runs?.length ? `Last run ${ago(runs[0].startedAt)}` : "No runs yet"}</span>
          </div>
          <div className="stat">
            <dt>Compute</dt>
            <dd className="stat-text">{system ? system.deviceLabel : "–"}</dd>
            <span className="stat-note muted">{system ? `${system.cpus} CPU cores${system.torch ? ` · PyTorch ${system.torch}` : ""}` : "Checking…"}</span>
          </div>
        </dl>

        {showSteps && (
          <section className="card">
            <h2>Getting started</h2>
            <ol className="start-steps">
              {steps.map((step, index) => (
                <li key={step.name} className={step.done ? "is-done" : ""}>
                  <span className="start-mark" aria-hidden="true">{step.done ? "✓" : index + 1}</span>
                  <span className="project-text">
                    <strong>{step.name}</strong>
                    <span className="muted small">{step.done ? "Done" : step.detail}</span>
                  </span>
                  {!step.done && <button className="button button-small" onClick={step.action}>{step.button}</button>}
                </li>
              ))}
            </ol>
          </section>
        )}

        <section className="card">
          <div className="card-head">
            <h2>Recent training runs</h2>
            {running > 0 && <span className="muted small" role="status">Updating every few seconds</span>}
          </div>
          {runs?.length === 0 && (
            <p className="muted small">
              {beginner ? "Models you train appear here with their scores, so you can see which one is best." : "No runs yet. Runs from every dataset appear here with their held-back scores."}
            </p>
          )}
          {shownRuns.length > 0 && (
            <div className="run-scroll">
              <table className="run-table">
                <thead>
                  <tr>
                    <th scope="col">Model</th>
                    <th scope="col" className="is-dataset">Dataset</th>
                    <th scope="col">Status</th>
                    <th scope="col" className="is-number">Score</th>
                    <th scope="col" className="is-wide">Data</th>
                    <th scope="col" className="is-wide">Started</th>
                    <th scope="col"><span className="visually-hidden">Open</span></th>
                  </tr>
                </thead>
                <tbody>
                  {shownRuns.map((run) => (
                    <tr key={run.id}>
                      <td>
                        <span className="run-name">{run.projectName}</span>
                        <span className="run-id muted">{run.id.slice(0, 8)}</span>
                        <span className="run-phone-only small muted">
                          {run.datasetName
                            ? <button className="link-button" onClick={() => onOpenDataset(run.dataset, "models")}>{run.datasetName}</button>
                            : "Model builder images"} · {KIND[run.task].name}
                        </span>
                      </td>
                      <td className="is-dataset">
                        <span className="run-dataset">{run.datasetName ?? "Model builder images"}</span>
                        <KindTag kind={run.task} />
                      </td>
                      <td><StatusTag run={run} /></td>
                      <td className="is-number run-score">{run.status === "running" ? "–" : score(run)}</td>
                      <td className="is-wide mono muted">{run.datasetVersion !== null ? `v${run.datasetVersion}` : "–"}</td>
                      <td className="is-wide muted" title={new Date(run.startedAt * 1000).toLocaleString()}>{ago(run.startedAt)}</td>
                      <td className="run-open">
                        {run.datasetName && (
                          <button className="button button-small" aria-label={`Open ${run.projectName} on ${run.datasetName}`} onClick={() => onOpenDataset(run.dataset, "models")}>Open</button>
                        )}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
          {runs && runs.length > RUNS_SHOWN && <p className="muted small field-note">Showing the newest {RUNS_SHOWN}. Each dataset's Models tab lists all of its runs.</p>}
        </section>

        <div className="home-lists">
          <section className="card">
            <div className="card-head">
              <h2>Model builds</h2>
              <button className="button button-small button-quiet" onClick={onNewModel}>New</button>
            </div>
            {projects?.length === 0 && <p className="muted small">Models you build from blocks appear here.</p>}
            <ul className="project-list">
              {projects?.slice(0, RECENT_SHOWN).map((row) => (
                <li key={row.id}>
                  <div className="project-text">
                    <strong>{row.projectName || "Untitled model"}</strong>
                    <span className="muted small">
                      {row.components} {row.components === 1 ? "block" : "blocks"} · edited {ago(row.updatedAt)}
                    </span>
                  </div>
                  <button
                    className="button button-small"
                    aria-label={`Open ${row.projectName || "Untitled model"}`}
                    onClick={() => (row.id === currentProjectId ? onContinueModel() : onOpenModel(row.id))}
                  >
                    {row.id === currentProjectId ? "Continue" : "Open"}
                  </button>
                </li>
              ))}
            </ul>
          </section>
          <section className="card">
            <div className="card-head">
              <h2>Datasets</h2>
              <button className="button button-small button-quiet" onClick={onOpenData}>All datasets</button>
            </div>
            {datasets?.length === 0 && <p className="muted small">Datasets you create appear here.</p>}
            <ul className="project-list">
              {datasets?.slice(0, RECENT_SHOWN).map((row) => (
                <li key={row.id}>
                  <div className="project-text">
                    <span className="dataset-line"><strong>{row.name}</strong><KindTag kind={row.kind} /></span>
                    <span className="muted small">
                      {row.kind === "table"
                        ? `${row.labelled.toLocaleString()} rows${row.columns ? ` · ${row.columns.length} columns` : ""}`
                        : `${row.labelled.toLocaleString()} labelled · ${row.categories.length} ${row.categories.length === 1 ? "category" : "categories"}${row.unsorted ? ` · ${row.unsorted.toLocaleString()} unsorted` : ""}`}
                    </span>
                  </div>
                  <button className="button button-small" aria-label={`Open ${row.name}`} onClick={() => onOpenDataset(row.id)}>Open</button>
                </li>
              ))}
            </ul>
          </section>
        </div>

        {system && (
          <section className="card system-card" aria-label="This computer">
            <h2>This computer</h2>
            <dl className="system-grid">
              <div><dt>Training on</dt><dd>{system.deviceLabel} <span className="muted mono">({system.device})</span></dd></div>
              <div><dt>Software</dt><dd>PyTorch {system.torch ?? "?"} · Python {system.python}</dd></div>
              <div><dt>Platform</dt><dd>{system.platform} · {system.cpus} cores</dd></div>
              <div>
                <dt>Workspace</dt>
                <dd>
                  {bytes(system.storage.bytes)} used, {bytes(system.storage.freeBytes)} free
                  <span className="muted small"> · data {bytes(system.storage.parts.data)}, models {bytes(system.storage.parts.models)}, downloads {bytes(system.storage.parts.downloads)}</span>
                </dd>
              </div>
              <div className="system-path"><dt>Folder</dt><dd className="mono muted">{system.storage.path}</dd></div>
            </dl>
            {beginner && system.device === "cpu" && <p className="muted small field-note">No GPU was found, so training runs on the processor. Small models still train in minutes.</p>}
          </section>
        )}
      </div>
    </div>
  );
}
