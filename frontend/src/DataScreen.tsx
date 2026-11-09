import { useCallback, useEffect, useState, type FormEvent } from "react";
import * as api from "./api";
import { DatasetLabels } from "./LabelPage";
import { DatasetModels, DatasetOverview, DatasetVersions, score } from "./DatasetRepository";
import { DataHealthTab } from "./DataHealth";
import { SoundLabels } from "./Sound";
import { TableData } from "./TableModels";
import type { DatasetSummary, GraphPayload, Mode, SavedModel } from "./types";

const KIND_NAMES: Record<DatasetSummary["kind"], string> = { image: "Images", text: "Text", table: "Spreadsheet (CSV)", sound: "Sound" };

type Tab = "overview" | "label" | "health" | "versions" | "models";
// The three steps, in the order they're done, then the two places to look things up.
const STEPS: { id: Tab; name: string }[] = [
  { id: "label", name: "Add data" },
  { id: "health", name: "Check" },
  { id: "models", name: "Train" },
];
const SIDE: { id: Tab; name: string }[] = [
  { id: "overview", name: "About" },
  { id: "versions", name: "History" },
];
// A good start for each kind of data, per category. Training can begin with fewer.
const TARGET: Record<DatasetSummary["kind"], number> = { image: 20, text: 16, sound: 20, table: 0 };
const MINIMUM: Record<DatasetSummary["kind"], number> = { image: 2, text: 4, sound: 4, table: 0 };

function enoughData(dataset: DatasetSummary) {
  if (dataset.kind === "table") return dataset.labelled > 0;
  return dataset.categories.length >= 2 && dataset.categories.every((name) => (dataset.counts[name] ?? 0) >= MINIMUM[dataset.kind]);
}

/** One sentence and one button: what to do next on this dataset. */
function NextStep({ dataset, onTab }: { dataset: DatasetSummary; onTab: (tab: Tab) => void }) {
  const [newest, setNewest] = useState<SavedModel | null | undefined>(undefined);
  useEffect(() => {
    let live = true;
    api.fetchDatasetModels(dataset.id)
      .then((rows) => { if (live) setNewest(rows.find((row) => row.status === "finished" && row.hasModel) ?? null); })
      .catch(() => { if (live) setNewest(null); });
    return () => { live = false; };
  }, [dataset.id, dataset.updatedAt]);
  if (newest === undefined) return null;

  const target = TARGET[dataset.kind];
  const short = dataset.categories.filter((name) => (dataset.counts[name] ?? 0) < target);
  let step: number, text: string, button: string, tab: Tab, extra: { label: string; tab: Tab } | null = null, anchor: string | null = null;
  if (dataset.kind === "table" && !dataset.labelled) {
    [step, text, button, tab] = [1, "Upload a CSV file with one row per example.", "Add data", "label"];
  } else if (dataset.kind !== "table" && dataset.categories.length < 2) {
    [step, text, button, tab] = [1, "Name at least 2 categories: the answers your model will choose between.", "Add categories", "label"];
  } else if (!newest && short.length) {
    const list = short.slice(0, 3).map((name) => `${name} (${dataset.counts[name] ?? 0})`).join(", ");
    [step, text, button, tab] = [1, `Add examples. Aim for ${target} in each category. Still short: ${list}${short.length > 3 ? "…" : ""}.`, "Add data", "label"];
    if (enoughData(dataset)) extra = { label: "Train anyway", tab: "models" };
  } else if (!newest) {
    [step, text, button, tab] = [3, "Your data is ready. Train your first model.", "Train", "models"];
    if (dataset.kind === "image" || dataset.kind === "text") extra = { label: "Check labels first", tab: "health" };
  } else {
    const result = score(newest);
    [step, text, button, tab] = [3, `Your newest model scores ${result}. Try it, or let it find a better one.`, "Try it", "models"];
    anchor = "#newest-model";
  }
  return (
    <div className="next-step" role="status">
      <span className="step-number" aria-hidden="true">{step}</span>
      <p><span className="visually-hidden">Next step: </span>{text}</p>
      <div className="next-step-actions">
        {extra && <button className="button button-small button-quiet" onClick={() => onTab(extra!.tab)}>{extra.label}</button>}
        <button className="button button-small button-primary" onClick={() => {
          onTab(tab);
          if (anchor) window.setTimeout(() => document.querySelector(anchor!)?.scrollIntoView({ behavior: "smooth", block: "start" }), 50);
        }}>{button}</button>
      </div>
    </div>
  );
}

interface Props {
  mode: Mode;
  datasetId: string | null;
  /** The tab to open the dataset on, when it shouldn't be the usual one. */
  tab?: "models" | "label";
  onOpen: (id: string | null) => void;
  onBuild: (dataset: DatasetSummary) => void;
  onOpenDesign: (graph: GraphPayload, datasetId: string) => void;
}

function splitNames(text: string) {
  return text.split(",").map((name) => name.trim()).filter(Boolean);
}

export function DataScreen({ mode, datasetId, tab, onOpen, onBuild, onOpenDesign }: Props) {
  return datasetId
    ? <DatasetDetail key={datasetId} mode={mode} datasetId={datasetId} initialTab={tab} onBack={() => onOpen(null)} onBuild={onBuild} onOpenDesign={onOpenDesign} />
    : <DatasetList mode={mode} onOpen={onOpen} />;
}

function DatasetList({ mode, onOpen }: { mode: Mode; onOpen: (id: string) => void }) {
  const beginner = mode === "beginner";
  const [rows, setRows] = useState<DatasetSummary[] | null>(null);
  const [name, setName] = useState("");
  const [kind, setKind] = useState<DatasetSummary["kind"]>("image");
  const [categories, setCategories] = useState("");
  const [problem, setProblem] = useState<string | null>(null);

  useEffect(() => {
    api.fetchDatasets().then(setRows).catch((error) => {
      setRows([]);
      setProblem(error instanceof Error ? error.message : "Your datasets could not load.");
    });
  }, []);

  const create = async (event: FormEvent) => {
    event.preventDefault();
    setProblem(null);
    try {
      onOpen((await api.createDataset(name, kind, kind === "table" ? [] : splitNames(categories))).id);
    } catch (error) {
      setProblem(error instanceof Error ? error.message : "The dataset couldn't be created.");
    }
  };

  return (
    <div className="page-scroll">
      <div className="page-inner page-narrow">
        <header className="page-head">
          <div>
            <h1>Data labeller</h1>
            <p className="muted">
              {beginner
                ? "A dataset is a collection of examples sorted into categories. Models learn from these examples."
                : "Create labelled datasets from open data or your own files."}
            </p>
          </div>
        </header>
        {problem && <div className="result result-fail" role="alert"><p>{problem}</p></div>}
        <section className="card">
          <h2>New dataset</h2>
          <form className="dataset-form" onSubmit={(event) => void create(event)}>
            <label className="field">
              <span>Name</span>
              <input type="text" value={name} maxLength={60} required placeholder="For example: Cats and dogs" onChange={(event) => setName(event.target.value)} />
            </label>
            <fieldset className="dataset-kind">
              <legend>Type of data</legend>
              {(["image", "text", "sound", "table"] as const).map((value) => (
                <label key={value}>
                  <input type="radio" name="dataset-kind" checked={kind === value} onChange={() => { setKind(value); if (value === "sound" && !categories.trim()) setCategories("Background noise, "); }} />
                  {KIND_NAMES[value]}
                </label>
              ))}
            </fieldset>
            {kind !== "table" && <label className="field">
              <span>Categories (optional)</span>
              <input type="text" value={categories} placeholder={kind === "sound" ? "For example: Clap, Whistle, Background noise" : "For example: Cat, Dog"} onChange={(event) => setCategories(event.target.value)} />
              <span className="muted small">Separate them with commas. You can also add them later, or take them from an open dataset.</span>
            </label>}
            <div><button className="button button-primary" type="submit" disabled={!name.trim()}>Create dataset</button></div>
          </form>
        </section>
        <section className="card">
          <h2>Your datasets</h2>
          {rows?.length === 0 && <p className="muted small">Nothing here yet.</p>}
          <ul className="project-list">
            {rows?.map((row) => (
              <li key={row.id}>
                <div className="project-text">
                  <strong>{row.name}</strong>
                  <span className="muted small">
                    {row.kind === "table"
                      ? `Spreadsheet · ${row.labelled.toLocaleString()} rows · ${row.columns?.length ?? 0} columns`
                      : `${KIND_NAMES[row.kind]} · ${row.categories.length} ${row.categories.length === 1 ? "category" : "categories"} · ${row.labelled.toLocaleString()} labelled${row.unsorted ? ` · ${row.unsorted.toLocaleString()} unsorted` : ""}`}
                  </span>
                </div>
                <button className="button button-small" aria-label={`Open ${row.name}`} onClick={() => onOpen(row.id)}>Open</button>
              </li>
            ))}
          </ul>
        </section>
      </div>
    </div>
  );
}

function DatasetDetail({ mode, datasetId, initialTab, onBack, onBuild, onOpenDesign }: {
  mode: Mode; datasetId: string; initialTab?: Tab; onBack: () => void; onBuild: (dataset: DatasetSummary) => void; onOpenDesign: (graph: GraphPayload, datasetId: string) => void;
}) {
  const beginner = mode === "beginner";
  const [dataset, setDataset] = useState<DatasetSummary | null>(null);
  const [missing, setMissing] = useState(false);
  const [problem, setProblem] = useState<string | null>(null);
  const [newCategory, setNewCategory] = useState("");
  const [tab, setTab] = useState<Tab | null>(initialTab ?? null);

  const load = useCallback(async () => {
    try {
      const loaded = await api.fetchSavedDataset(datasetId);
      setDataset(loaded);
      // It opens on the step you're at: adding data until there's enough, then training.
      setTab((open) => open ?? (enoughData(loaded) ? "models" : "label"));
    } catch {
      setMissing(true);
    }
  }, [datasetId]);

  useEffect(() => { void load(); }, [load]);

  const run = async (action: () => Promise<DatasetSummary>) => {
    setProblem(null);
    try {
      setDataset(await action());
    } catch (error) {
      setProblem(error instanceof Error ? error.message : "That change couldn't be saved.");
    }
  };

  const remove = async () => {
    if (!dataset || !window.confirm(`Delete "${dataset.name}" and all of its copied images, labels and saved versions? Your original files and trained models aren't touched. This can't be undone.`)) return;
    try {
      await api.deleteSavedDataset(datasetId);
      onBack();
    } catch (error) {
      setProblem(error instanceof Error ? error.message : "The dataset couldn't be deleted.");
    }
  };

  const addCategory = async (event: FormEvent) => {
    event.preventDefault();
    const names = splitNames(newCategory);
    if (!names.length) return;
    await run(() => api.addDatasetCategories(datasetId, names));
    setNewCategory("");
  };

  const renameCategory = (old: string) => {
    const next = window.prompt(`New name for "${old}"`, old)?.trim();
    if (next && next !== old) void run(() => api.renameDatasetCategory(datasetId, old, next));
  };

  const removeCategory = (name: string) => {
    const warning = dataset?.kind === "sound"
      ? `Remove the category "${name}" and its clips? They're saved in a version first, so you can restore them from Versions.`
      : `Remove the category "${name}"? Its examples go back to being unsorted.`;
    if (window.confirm(warning)) {
      void run(() => api.removeDatasetCategory(datasetId, name));
    }
  };

  if (missing) {
    return (
      <div className="page-scroll">
        <div className="page-inner page-narrow">
          <div className="notice">
            <span>This dataset no longer exists.</span>
            <button className="button button-small" onClick={onBack}>All datasets</button>
          </div>
        </div>
      </div>
    );
  }
  if (!dataset) return <div className="splash">Loading…</div>;

  return (
    <div className="page-scroll">
      <div className="page-inner page-narrow">
        <button className="link-button dataset-back" onClick={onBack}>All datasets</button>
        <header className="page-head">
          <div className="dataset-title">
            <label>
              <span className="visually-hidden">Dataset name</span>
              <input
                className="dataset-name"
                defaultValue={dataset.name}
                maxLength={60}
                onBlur={(event) => {
                  const next = event.target.value.trim();
                  if (next && next !== dataset.name) void run(() => api.renameSavedDataset(datasetId, next));
                  else event.target.value = dataset.name;
                }}
              />
            </label>
            <p className="muted small">
              {dataset.kind === "table"
                ? `Spreadsheet · ${dataset.labelled.toLocaleString()} rows · ${dataset.columns?.length ?? 0} columns`
                : dataset.kind === "sound"
                  ? `Sound · ${dataset.labelled.toLocaleString()} one-second ${dataset.labelled === 1 ? "clip" : "clips"}`
                  : `${KIND_NAMES[dataset.kind]} · ${dataset.labelled.toLocaleString()} labelled · ${dataset.unsorted.toLocaleString()} unsorted`}
            </p>
          </div>
          <div className="label-buttons">
            <button className="button button-small button-quiet" onClick={() => void remove()}>Delete dataset</button>
          </div>
        </header>
        {problem && <div className="result result-fail" role="alert"><p>{problem}</p></div>}

        <NextStep dataset={dataset} onTab={setTab} />

        <nav className="dataset-tabs" aria-label="Dataset sections">
          {STEPS.map((item, index) => (
            <button key={item.id} className={`is-step${tab === item.id ? " is-active" : ""}`} aria-current={tab === item.id ? "page" : undefined} onClick={() => setTab(item.id)}>
              <span className="tab-number" aria-hidden="true">{index + 1}</span>{item.name}
            </button>
          ))}
          <span className="tab-gap" />
          {SIDE.map((item) => (
            <button key={item.id} className={tab === item.id ? "is-active" : ""} aria-current={tab === item.id ? "page" : undefined} onClick={() => setTab(item.id)}>{item.name}</button>
          ))}
        </nav>

        {tab === "overview" && <DatasetOverview dataset={dataset} beginner={beginner} onSaved={setDataset} onLabel={() => setTab("label")} />}
        {tab === "health" && <DataHealthTab dataset={dataset} beginner={beginner} onChanged={() => void load()} />}
        {tab === "versions" && <DatasetVersions dataset={dataset} beginner={beginner} onRestored={setDataset} />}
        {tab === "models" && <DatasetModels dataset={dataset} beginner={beginner} onBuild={() => onBuild(dataset)} onChanged={() => void load()} onOpenDesign={(graph) => onOpenDesign(graph, dataset.id)} />}

        {tab === "label" && dataset.kind === "table" && <TableData dataset={dataset} beginner={beginner} onChanged={setDataset} onTrain={() => setTab("models")} />}

        {tab === "label" && dataset.kind !== "table" && <h2 className="section-title">Categories</h2>}
        {tab === "label" && dataset.kind !== "table" && <section className="card">
          <h2>{beginner ? "The answers your model will choose between" : "Categories"}</h2>
          {dataset.categories.length === 0 && <p className="muted small">{beginner ? "Categories are the answers your model will learn to give, like Cat or Dog." : "No categories yet."}</p>}
          <ul className="category-list">
            {dataset.categories.map((name) => (
              <li key={name}>
                <span className="try-name"><strong>{name}</strong> <span className="muted small">{(dataset.counts[name] ?? 0).toLocaleString()}</span></span>
                <button className="button button-small button-quiet" aria-label={`Rename ${name}`} onClick={() => renameCategory(name)}>Rename</button>
                <button className="button button-small button-quiet" aria-label={`Remove ${name}`} onClick={() => removeCategory(name)}>Remove</button>
              </li>
            ))}
          </ul>
          <form className="find-search field-note" onSubmit={(event) => void addCategory(event)}>
            <input type="text" value={newCategory} placeholder="Add categories, separated by commas" aria-label="Add categories" onChange={(event) => setNewCategory(event.target.value)} />
            <button className="button" type="submit" disabled={!newCategory.trim()}>Add</button>
          </form>
        </section>}

        {tab === "label" && dataset.kind === "sound" && <SoundLabels dataset={dataset} beginner={beginner} onChanged={() => void load()} />}

        {tab === "label" && (dataset.kind === "image" || dataset.kind === "text") && <DatasetLabels
          dataset={dataset}
          onChanged={() => void load()}
          onUseCategories={(names) => void run(() => api.addDatasetCategories(datasetId, names))}
          onBuild={() => onBuild(dataset)}
        />}
      </div>
    </div>
  );
}
