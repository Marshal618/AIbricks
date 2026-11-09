import { useCallback, useEffect, useRef, useState, type ChangeEvent, type DragEvent } from "react";
import * as api from "./api";
import { useEditor } from "./context";
import { LossChart } from "./LossChart";
import { ExplainablePhoto } from "./Explain";
import { ModelApp } from "./ModelApp";
import { TrainingCheck } from "./TrainingCheck";
import { WebcamTry } from "./Webcam";
import type { DatasetInfo, DatasetSummary, FixKind, GraphPayload, PredictAnswer, TrainingJob, TrainingReview } from "./types";

const MODELS_KEPT = 10;

interface Props {
  graph: GraphPayload;
  onDesign: () => void;
  onTest: () => void;
  onExport: () => void;
  datasetId: string | null;
  onDatasetChange: (id: string | null) => void;
  onOpenData: (id: string | null) => void;
  onOpenDatasetTab: (id: string, tab: "models" | "label") => void;
  onApplyFix: (fix: FixKind) => void;
  onUseCategories: (names: string[]) => void;
}

interface Upload {
  done: number;
  total: number;
  added: number;
  duplicates: number;
  ignored: number;
  rejected: string[];
  running: boolean;
}

const UPLOAD_WORKERS = 4;
const MEANINGFUL_LEAD = 0.1;
const ROUGH_CHECK_COUNT = 20;
const MISTAKES_SHOWN = 60;

function percent(value: number) {
  return `${Math.round(value * 100)}%`;
}

function duration(seconds: number) {
  const whole = Math.max(Math.round(seconds), 0);
  if (whole < 60) return `${whole} sec`;
  const minutes = Math.floor(whole / 60);
  if (minutes < 60) return `${minutes} min ${whole % 60} sec`;
  return `${Math.floor(minutes / 60)} hr ${minutes % 60} min`;
}

export function TrainPage({ graph, onDesign, onTest, onExport, datasetId, onDatasetChange, onOpenData, onOpenDatasetTab, onApplyFix, onUseCategories }: Props) {
  const { analysis, mode } = useEditor();
  const beginner = mode === "beginner";
  const categories = analysis?.categories ?? null;
  const categoryKey = JSON.stringify(categories);
  const [dataset, setDataset] = useState<DatasetInfo | null>(null);
  const [datasets, setDatasets] = useState<DatasetSummary[] | null>(null);
  const [job, setJob] = useState<TrainingJob | null>(null);
  const [upload, setUpload] = useState<Upload | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [starting, setStarting] = useState(false);
  const folderInput = useRef<HTMLInputElement>(null);

  const refreshDatasets = useCallback(() => {
    api.fetchDatasets().then(setDatasets).catch(() => setDatasets([]));
  }, []);
  useEffect(refreshDatasets, [refreshDatasets]);

  const chosen = datasets?.find((row) => row.id === datasetId && row.kind === "image") ?? null;
  const chosenId = chosen?.id ?? null;

  const loadDataset = useCallback(async () => {
    if (!categories || !chosenId) {
      setDataset(null);
      return;
    }
    try {
      setDataset(await api.fetchDataset(categories, chosenId));
    } catch (problem) {
      setError(problem instanceof Error ? problem.message : "The image list could not load.");
    }
    // categoryKey stands in for the array, which is new on every check.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [categoryKey, chosenId]);

  useEffect(() => { void loadDataset(); }, [loadDataset]);

  // Training runs on the server, so leaving this page or reloading doesn't interrupt it.
  const running = job?.status === "running";
  useEffect(() => {
    let cancelled = false;
    const poll = async () => {
      try {
        const latest = await api.fetchTraining("image");
        if (!cancelled) setJob(latest);
      } catch {
        // The offline banner already covers this. The next tick tries again.
      }
    };
    void poll();
    if (!running) return () => { cancelled = true; };
    const timer = window.setInterval(poll, 1000);
    return () => { cancelled = true; window.clearInterval(timer); };
  }, [running]);

  const send = useCallback(async (items: { file: File; category: string }[], ignored: number) => {
    if (!items.length && !ignored) return;
    setError(null);
    const state: Upload = { done: 0, total: items.length, added: 0, duplicates: 0, ignored, rejected: [], running: true };
    setUpload({ ...state });
    const queue = [...items];
    const worker = async () => {
      for (let item = queue.shift(); item; item = queue.shift()) {
        try {
          const reply = await api.uploadImage(item.category, item.file, chosenId as string);
          if (reply.duplicate) state.duplicates += 1; else state.added += 1;
        } catch (problem) {
          state.rejected.push(`${item.file.name}: ${problem instanceof Error ? problem.message : "could not be read."}`);
        }
        state.done += 1;
        setUpload({ ...state });
      }
    };
    await Promise.all(Array.from({ length: UPLOAD_WORKERS }, worker));
    state.running = false;
    setUpload({ ...state });
    await loadDataset();
    refreshDatasets();
  }, [chosenId, loadDataset, refreshDatasets]);

  const addToCategory = (category: string, files: FileList | null) => {
    if (files) void send(Array.from(files).filter((file) => !file.name.startsWith(".")).map((file) => ({ file, category })), 0);
  };

  const addFolder = (event: ChangeEvent<HTMLInputElement>) => {
    const byName = new Map((categories ?? []).map((name) => [name.toLowerCase(), name]));
    const items: { file: File; category: string }[] = [];
    let ignored = 0;
    for (const file of Array.from(event.target.files ?? [])) {
      if (file.name.startsWith(".")) continue;
      const parts = file.webkitRelativePath.split("/");
      const category = byName.get((parts[parts.length - 2] ?? "").toLowerCase());
      if (category) items.push({ file, category }); else ignored += 1;
    }
    event.target.value = "";
    void send(items, ignored);
  };

  const clear = async (category: string) => {
    if (!window.confirm(`Remove all images from "${category}"? Your original files aren't touched.`)) return;
    try {
      await api.clearImages(category, chosenId as string);
    } catch (problem) {
      setError(problem instanceof Error ? problem.message : "The images could not be removed.");
    }
    await loadDataset();
  };

  const start = async () => {
    setError(null);
    setStarting(true);
    try {
      setJob(await api.startTraining(graph, chosenId as string));
    } catch (problem) {
      setError(problem instanceof Error ? problem.message : "Training could not start.");
    } finally {
      setStarting(false);
    }
  };

  const stop = async () => {
    if (!job) return;
    try {
      setJob(await api.stopTraining(job.id));
    } catch (problem) {
      setError(problem instanceof Error ? problem.message : "Training could not be stopped.");
    }
  };

  if (!analysis?.ok) {
    return (
      <Shell beginner={beginner}>
        <div className="notice">
          <span>Finish the model first. It needs to be free of problems before it can train.</span>
          <button className="button button-small" onClick={onDesign}>Go to Design</button>
        </div>
      </Shell>
    );
  }

  if (!categories) {
    return (
      <Shell beginner={beginner}>
        <section className="card">
          <h2>This model trains from the export</h2>
          <p className="muted">
            {analysis.labelKind === "text"
              ? "This block design trains from the downloaded project. To train a text sorter inside the app instead, open a text dataset in the Data labeller and use its Models tab. That model is separate from this design."
              : "Training inside the app covers models that sort images into categories. Other models train on your computer from the downloaded project."}
          </p>
          <div className="label-buttons train-other">
            {analysis.labelKind === "text" && <button className="button" onClick={() => onOpenData(null)}>Prepare text in the Data labeller</button>}
            <button className="button" onClick={onExport}>Go to Export</button>
          </div>
        </section>
      </Shell>
    );
  }

  const tested = analysis.tested;
  const stale = job !== null && (job.buildHash !== analysis.buildHash || job.dataset !== chosenId);
  const ready = chosen !== null && (dataset?.problems.length ?? 1) === 0;
  const mismatch = chosen !== null && JSON.stringify([...chosen.categories].sort()) !== JSON.stringify([...categories].sort());
  const imageDatasets = datasets?.filter((row) => row.kind === "image") ?? [];
  const counts = dataset?.counts ?? {};
  const suggested = dataset?.suggestedPerCategory ?? 20;
  const fewImages = dataset !== null && categories.some((name) => (counts[name] ?? 0) < suggested);
  const elapsedPerEpoch = job?.epochs.length ? job.elapsedSeconds / job.epochs.length : null;

  return (
    <Shell beginner={beginner}>
      {error && <div className="result result-fail" role="alert"><p>{error}</p></div>}

      <section className="card">
        <div className="card-head">
          <h2>{beginner ? "Step 1: Choose your data" : "Dataset"}</h2>
          {chosen && <button className="button button-small" onClick={() => onOpenData(chosen.id)}>Open in Data labeller</button>}
        </div>
        {datasets !== null && imageDatasets.length > 0 && (
          <label className="field train-dataset">
            <span className="visually-hidden">Dataset</span>
            <select value={chosenId ?? ""} disabled={running} onChange={(event) => onDatasetChange(event.target.value || null)}>
              <option value="">Choose a dataset…</option>
              {imageDatasets.map((row) => <option key={row.id} value={row.id}>{row.name} ({row.labelled.toLocaleString()} labelled)</option>)}
            </select>
          </label>
        )}
        {datasets !== null && !chosen && (
          <div className="train-no-data">
            <p className="muted small">{imageDatasets.length ? "Or start a new one for this model's categories." : "This model needs a dataset to learn from. Start one with this model's categories, then add images or find open data."}</p>
            <div className="label-buttons train-other">
              <button
                className="button button-small"
                onClick={() => {
                  setError(null);
                  api.createDataset(graph.projectName.slice(0, 60) || "My dataset", "image", categories)
                    .then((made) => { refreshDatasets(); onDatasetChange(made.id); })
                    .catch((problem) => setError(problem instanceof Error ? problem.message : "The dataset couldn't be created."));
                }}
              >
                Create a dataset for this model
              </button>
              <button className="button button-small" onClick={() => onOpenData(null)}>Open the Data labeller</button>
            </div>
          </div>
        )}
        {chosen && mismatch && (
          <div className="notice field-note">
            <span>This dataset's categories are {chosen.categories.join(", ") || "not set yet"}, but your model's are {categories.join(", ")}.</span>
            <span className="label-buttons">
              {chosen.categories.length >= 2 && <button className="button button-small" disabled={running} onClick={() => onUseCategories(chosen.categories)}>Use the dataset's in my model</button>}
              <button
                className="button button-small"
                disabled={running}
                onClick={() => {
                  api.addDatasetCategories(chosen.id, categories).then(refreshDatasets)
                    .catch((problem) => setError(problem instanceof Error ? problem.message : "The categories couldn't be added."));
                }}
              >
                Add my model's to the dataset
              </button>
            </span>
          </div>
        )}
        {chosen && (
          <>
            <div className="card-head train-add-head">
              <p className="muted small">Add more images here, or choose a folder with a subfolder for each category. They're copied to this computer and aren't sent anywhere.</p>
              <button className="button button-small" disabled={upload?.running} onClick={() => folderInput.current?.click()}>Choose a folder</button>
              <input
                ref={folderInput}
                className="visually-hidden"
                type="file"
                aria-label="Choose a folder with one subfolder per category"
                onChange={addFolder}
                tabIndex={-1}
                {...{ webkitdirectory: "" }}
              />
            </div>
            <ul className="train-rows">
              {categories.map((name) => (
                <CategoryRow
                  key={name}
                  name={name}
                  count={counts[name] ?? 0}
                  busy={upload?.running ?? false}
                  locked={running}
                  onAdd={(files) => addToCategory(name, files)}
                  onClear={() => void clear(name)}
                />
              ))}
            </ul>
            {upload && <UploadSummary upload={upload} />}
            {dataset && dataset.problems.length > 0 && <p className="small muted field-note">{dataset.problems.join(" ")}</p>}
            {ready && fewImages && <p className="small muted field-note">You can train now, but results are more trustworthy with at least {suggested} images per category.</p>}
          </>
        )}
      </section>

      <section className="card">
        <div className="card-head">
          <h2>{beginner ? "Step 2: Train your model" : "Training"}</h2>
          {running
            ? job?.stage === "training" && <button className="button button-small" onClick={() => void stop()}>Stop and keep best so far</button>
            : <button className="button button-small button-primary" disabled={!tested || !ready || starting || upload?.running} onClick={() => void start()}>
                {starting ? "Starting…" : job ? "Train again" : "Start training"}
              </button>}
        </div>
        {!tested && (
          <div className="notice">
            <span>Training unlocks after this exact model passes a test run.</span>
            <button className="button button-small" onClick={onTest}>Go to Test</button>
          </div>
        )}
        {tested && !job && <p className="muted small">Training uses this computer, so it may slow other apps. It keeps running if you leave this page.</p>}
        {tested && job && !running && <p className="muted small">Training again keeps this model. The newest {MODELS_KEPT} stay on the dataset's Models tab.</p>}
        {job && stale && !running && (
          <div className="notice field-note">These results came from an earlier version of this model or a different dataset. Train again to try the current one.</div>
        )}
        {job?.datasetVersion != null && <p className="muted small">{running ? "Using" : "Trained on"} version v{job.datasetVersion} of the dataset. Labels you change now won't affect this run.</p>}
        {job && <Progress job={job} beginner={beginner} perEpoch={elapsedPerEpoch} />}
      </section>

      {job && job.status !== "running" && job.checkup && !stale && (
        <TrainingCheck
          findings={job.checkup}
          available={chosenId ? ["augment", "slower", "longer", "pretrained", "more_photos", "experiment"] : ["augment", "slower", "longer"]}
          onFix={(fix) => {
            if (fix === "more_photos" && chosenId) onOpenDatasetTab(chosenId, "label");
            else if ((fix === "pretrained" || fix === "experiment") && chosenId) onOpenDatasetTab(chosenId, "models");
            else onApplyFix(fix);
          }}
        />
      )}
      {job && job.status !== "running" && job.hasModel && job.review && !stale && (
        <Mistakes
          jobId={job.id}
          review={job.review}
          categories={Object.keys(job.categories)}
          beginner={beginner}
          onChanged={(next) => { setJob(next); void loadDataset(); }}
        />
      )}
      {job && job.status !== "running" && job.hasModel && !stale && job.reviewError && (
        <div className="notice">The score is still valid, but the list of mistakes isn't available: {job.reviewError}</div>
      )}
      {job && job.status !== "running" && job.hasModel && !stale && (
        <TryModel job={job} beginner={beginner} />
      )}
    </Shell>
  );
}

function Shell({ beginner, children }: { beginner: boolean; children: React.ReactNode }) {
  return (
    <div className="page-scroll">
      <div className="page-inner page-narrow">
        <header className="page-head">
          <div>
            <h1>Train</h1>
            <p className="muted">
              {beginner
                ? "Teach your model with your own images, then try it out."
                : "Runs the generated train.py on your images, on this computer."}
            </p>
          </div>
        </header>
        {children}
      </div>
    </div>
  );
}

interface RowProps {
  name: string;
  count: number;
  busy: boolean;
  locked: boolean;
  onAdd: (files: FileList | null) => void;
  onClear: () => void;
}

function CategoryRow({ name, count, busy, locked, onAdd, onClear }: RowProps) {
  const input = useRef<HTMLInputElement>(null);
  const [over, setOver] = useState(false);
  const onDrop = (event: DragEvent) => {
    event.preventDefault();
    setOver(false);
    if (!busy) onAdd(event.dataTransfer.files);
  };
  return (
    <li
      className={`train-row${over ? " is-over" : ""}`}
      onDragOver={(event) => { event.preventDefault(); setOver(true); }}
      onDragLeave={() => setOver(false)}
      onDrop={onDrop}
    >
      <strong className="train-name">{name}</strong>
      <span className="muted small train-count">{count.toLocaleString()} {count === 1 ? "image" : "images"}</span>
      <span className="spacer" />
      {count > 0 && <button className="button button-small button-quiet" disabled={busy || locked} onClick={onClear}>Remove all</button>}
      <button className="button button-small" disabled={busy} onClick={() => input.current?.click()}>Add images</button>
      <input
        ref={input}
        className="visually-hidden"
        type="file"
        multiple
        accept="image/jpeg,image/png,image/bmp,image/webp"
        aria-label={`Add images to ${name}`}
        tabIndex={-1}
        onChange={(event) => { onAdd(event.target.files); event.target.value = ""; }}
      />
    </li>
  );
}

function UploadSummary({ upload }: { upload: Upload }) {
  if (upload.running) {
    return (
      <div className="field-note" role="status">
        <progress className="train-progress" max={upload.total} value={upload.done} aria-label="Images copied" />
        <p className="small muted">Checking and copying {upload.done} of {upload.total}…</p>
      </div>
    );
  }
  const parts = [`Added ${upload.added} ${upload.added === 1 ? "image" : "images"}.`];
  if (upload.duplicates) parts.push(`${upload.duplicates} ${upload.duplicates === 1 ? "was" : "were"} already here.`);
  if (upload.ignored) parts.push(`${upload.ignored} skipped because their folder doesn't match a category.`);
  return (
    <div className="field-note small" role="status">
      <p className="muted">{parts.join(" ")}</p>
      {upload.rejected.length > 0 && (
        <details className="train-rejected" open={upload.rejected.length <= 3}>
          <summary>{upload.rejected.length} {upload.rejected.length === 1 ? "file" : "files"} couldn't be used</summary>
          <ul>{upload.rejected.slice(0, 50).map((line) => <li key={line}>{line}</li>)}</ul>
        </details>
      )}
    </div>
  );
}

function Progress({ job, beginner, perEpoch }: { job: TrainingJob; beginner: boolean; perEpoch: number | null }) {
  const finishedEpochs = job.epochs.length;
  const running = job.status === "running";
  const left = perEpoch !== null ? perEpoch * (job.totalEpochs - finishedEpochs) : null;
  const points = job.epochs.flatMap((entry) => entry.checkLoss === null ? [] : [{ step: entry.epoch, loss: entry.checkLoss }]);
  return (
    <div className="train-progress-block">
      {running && (
        <div role="status">
          <progress className="train-progress" max={job.totalEpochs} value={finishedEpochs} aria-label="Training progress" />
          <p className="small muted">
            {job.stage === "reviewing"
              ? "Checking the images it got wrong…"
              : <>
                  {job.device ? `${beginner ? "Pass" : "Epoch"} ${Math.min(finishedEpochs + 1, job.totalEpochs)} of ${job.totalEpochs}` : "Loading your images…"}
                  {" · "}{duration(job.elapsedSeconds)} so far{left !== null ? ` · about ${duration(left)} left` : ""}
                </>}
          </p>
        </div>
      )}
      {job.status === "failed" && (
        <div className="result result-fail" role="alert">
          <div><strong>Training failed</strong><p>{job.error}</p></div>
        </div>
      )}
      {job.status === "stopped" && !job.hasModel && (
        <div className="notice">Training stopped before finishing a pass through your images, so no model was saved.</div>
      )}
      {!running && job.best && job.hasModel && <Verdict job={job} beginner={beginner} />}
      {points.length > 0 && (
        <LossChart points={points} totalSteps={Math.max(job.totalEpochs, 1)} caption="Loss on images held back from training" unit={beginner ? "Pass" : "Epoch"} />
      )}
      {beginner && points.length > 0 && <p className="muted small field-note">Loss measures how wrong the model is. A falling line means it is learning.</p>}
      {!beginner && job.epochs.length > 0 && (
        <details className="chart-table">
          <summary>Show every epoch</summary>
          <table>
            <thead><tr><th scope="col">Epoch</th><th scope="col">Training loss</th><th scope="col">Check loss</th><th scope="col">Accuracy</th></tr></thead>
            <tbody>
              {job.epochs.map((entry) => (
                <tr key={entry.epoch}>
                  <td>{entry.epoch}{job.best?.epoch === entry.epoch ? " (saved)" : ""}</td>
                  <td>{entry.trainLoss?.toFixed(4) ?? "invalid"}</td>
                  <td>{entry.checkLoss?.toFixed(4) ?? "invalid"}</td>
                  <td>{entry.accuracy === null ? "invalid" : percent(entry.accuracy)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </details>
      )}
    </div>
  );
}

function Verdict({ job, beginner }: { job: TrainingJob; beginner: boolean }) {
  const accuracy = job.best?.accuracy ?? 0;
  const useful = accuracy >= job.baseline + MEANINGFUL_LEAD;
  const rough = job.checkCount < ROUGH_CHECK_COUNT;
  return (
    <div className={`result ${useful ? "result-pass" : "result-fail"}`} role="status">
      <div>
        <strong>{percent(accuracy)} correct on {job.checkCount} images it never trained on</strong>
        <p>
          {useful
            ? `Always picking the largest category would score ${percent(job.baseline)}, so the model has learned something real.`
            : `Always picking the largest category would score ${percent(job.baseline)}, so this isn't meaningfully better than guessing. Try more images per category${beginner ? "." : ", a pretrained backbone, or a lower learning rate."}`}
          {rough ? ` With so few held-back images, treat this score as rough.` : ""}
          {job.status === "stopped" ? " You stopped training early, so this is the best version so far." : ""}
          {!beginner && job.best ? ` Saved from epoch ${job.best.epoch}, which was chosen using these same images, so the score can run slightly high.` : ""}
        </p>
      </div>
    </div>
  );
}

function Mistakes({ jobId, review, categories, beginner, onChanged }: { jobId: string; review: TrainingReview; categories: string[]; beginner: boolean; onChanged: (job: TrainingJob) => void }) {
  const shown = review.mistakes.slice(0, MISTAKES_SHOWN);
  const [problem, setProblem] = useState<string | null>(null);
  const fixes = review.mistakes.filter((mistake) => mistake.movedTo || mistake.removed).length;
  const fix = async (index: number, action: "move" | "remove" | "undo", category?: string) => {
    setProblem(null);
    try {
      onChanged(await api.fixMistake(jobId, index, action, category));
    } catch (error) {
      setProblem(error instanceof Error ? error.message : "That change couldn't be saved.");
    }
  };
  return (
    <section className="card">
      <h2>What it got wrong</h2>
      <ul className="review-categories">
        {Object.entries(review.categories).map(([name, score]) => (
          <li key={name}>
            <span className="try-name">{name}</span>
            <span className="muted small">{score.correct} of {score.total} correct</span>
          </li>
        ))}
      </ul>
      {problem && <div className="result result-fail field-note" role="alert"><p>{problem}</p></div>}
      {fixes > 0 && <p className="notice field-note">You've changed {fixes} {fixes === 1 ? "image" : "images"}. Train again to use the corrected labels.</p>}
      {review.mistakeCount === 0
        ? <p className="muted small field-note">It got every held-back image right. Try it on harder images below to find its limits.</p>
        : <>
            <p className="muted small field-note">
              Most confident mistakes come first. Look for a pattern, such as blurry photos, and use “Where did it look?” to check it was looking at the object, not the background. If an image is filed in the wrong category, move it here.
              {review.mistakeCount > shown.length ? ` Showing ${shown.length} of ${review.mistakeCount}.` : ""}
            </p>
            <ul className="mistake-grid">
              {shown.map((mistake, index) => (
                <li key={`${mistake.actual}/${mistake.file}`}>
                  <ExplainablePhoto className="is-thumb" jobId={jobId} src={api.mistakeImageUrl(jobId, index)} alt={`Filed under ${mistake.actual}, guessed ${mistake.guess}`} beginner={beginner} />
                  <span className="small"><span className="muted">Filed under</span> {mistake.actual}</span>
                  <span className="small"><span className="muted">Guessed</span> {mistake.guess} <span className="muted">{percent(mistake.chance)}</span></span>
                  {mistake.movedTo || mistake.removed
                    ? <span className="mistake-fix small">
                        <span className="tone-good">{mistake.removed ? "Removed" : `Moved to ${mistake.movedTo}`}</span>
                        <button className="button button-small button-quiet" onClick={() => void fix(index, "undo")}>Undo</button>
                      </span>
                    : <span className="mistake-fix small">
                        <select
                          aria-label={`Move this image out of ${mistake.actual}`}
                          value=""
                          onChange={(event) => { if (event.target.value) void fix(index, "move", event.target.value); }}
                        >
                          <option value="">Move to…</option>
                          {categories.filter((name) => name !== mistake.actual).map((name) => <option key={name} value={name}>{name}</option>)}
                        </select>
                        <button className="button button-small button-quiet" onClick={() => void fix(index, "remove")}>Remove</button>
                      </span>}
                </li>
              ))}
            </ul>
          </>}
    </section>
  );
}

function TryModel({ job, beginner }: { job: TrainingJob; beginner: boolean }) {
  const input = useRef<HTMLInputElement>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [chosen, setChosen] = useState<File | null>(null);
  const [answers, setAnswers] = useState<PredictAnswer[] | null>(null);
  const [problem, setProblem] = useState<string | null>(null);
  const [asking, setAsking] = useState(false);
  const [over, setOver] = useState(false);

  useEffect(() => () => { if (preview) URL.revokeObjectURL(preview); }, [preview]);

  const ask = async (file: File | undefined) => {
    if (!file) return;
    setPreview(URL.createObjectURL(file));
    setChosen(file);
    setAnswers(null);
    setProblem(null);
    setAsking(true);
    try {
      setAnswers(await api.predictImage(job.id, file));
    } catch (error) {
      setPreview(null);
      setProblem(error instanceof Error ? error.message : "The model could not read this image.");
    } finally {
      setAsking(false);
    }
  };

  return (
    <section className="card">
      <div className="card-head">
        <h2>{beginner ? "Step 3: Try it" : "Try the model"}</h2>
        <a className="button button-small" href={api.trainedDownloadUrl(job.id)} download>Download trained model</a>
      </div>
      <p className="muted small field-note-top">
        Use a photo that wasn't in your training images. The download holds the project, the model and a small web API, not your images.
      </p>
      <div
        className={`try-drop${over ? " is-over" : ""}`}
        onDragOver={(event) => { event.preventDefault(); setOver(true); }}
        onDragLeave={() => setOver(false)}
        onDrop={(event) => { event.preventDefault(); setOver(false); void ask(event.dataTransfer.files[0]); }}
      >
        {preview && chosen
          ? answers ? <ExplainablePhoto jobId={job.id} src={preview} file={chosen} alt="The image you chose" beginner={beginner} /> : <img src={preview} alt="The image you chose" />
          : <span className="muted small">Drop an image here</span>}
        <button className="button button-small" disabled={asking} onClick={() => input.current?.click()}>{preview ? "Try another image" : "Choose an image"}</button>
        <input
          ref={input}
          className="visually-hidden"
          type="file"
          accept="image/jpeg,image/png,image/bmp,image/webp"
          aria-label="Choose an image to try"
          tabIndex={-1}
          onChange={(event) => { void ask(event.target.files?.[0]); event.target.value = ""; }}
        />
      </div>
      <WebcamTry jobId={job.id} beginner={beginner} />
      <ModelApp jobId={job.id} beginner={beginner} />
      {asking && <p className="small muted field-note" role="status">Asking the model…</p>}
      {problem && <div className="result result-fail field-note" role="alert"><p>{problem}</p></div>}
      {answers && (
        <ul className="try-answers" role="status" aria-label="The model's answers">
          {answers.map((answer, index) => (
            <li key={answer.category} className={index === 0 ? "is-top" : ""}>
              <span className="try-name">{answer.category}</span>
              <span className="try-track" aria-hidden="true"><span style={{ width: `${answer.chance * 100}%` }} /></span>
              <span className="try-value">{(answer.chance * 100).toFixed(1)}%</span>
            </li>
          ))}
        </ul>
      )}
      {answers && <p className="muted small field-note">{beginner ? "These chances are the model's own confidence. It can be confident and still wrong." : "Softmax output. These values aren't calibrated probabilities."}</p>}
    </section>
  );
}
