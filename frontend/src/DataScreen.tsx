import { useCallback, useEffect, useState, type FormEvent } from "react";
import * as api from "./api";
import { DatasetLabels } from "./LabelPage";
import type { DatasetSummary, Mode } from "./types";

interface Props {
  mode: Mode;
  datasetId: string | null;
  onOpen: (id: string | null) => void;
  onBuild: (dataset: DatasetSummary) => void;
}

function splitNames(text: string) {
  return text.split(",").map((name) => name.trim()).filter(Boolean);
}

export function DataScreen({ mode, datasetId, onOpen, onBuild }: Props) {
  return datasetId
    ? <DatasetDetail key={datasetId} mode={mode} datasetId={datasetId} onBack={() => onOpen(null)} onBuild={onBuild} />
    : <DatasetList mode={mode} onOpen={onOpen} />;
}

function DatasetList({ mode, onOpen }: { mode: Mode; onOpen: (id: string) => void }) {
  const beginner = mode === "beginner";
  const [rows, setRows] = useState<DatasetSummary[] | null>(null);
  const [name, setName] = useState("");
  const [kind, setKind] = useState<"image" | "text">("image");
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
      onOpen((await api.createDataset(name, kind, splitNames(categories))).id);
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
              {(["image", "text"] as const).map((value) => (
                <label key={value}>
                  <input type="radio" name="dataset-kind" checked={kind === value} onChange={() => setKind(value)} />
                  {value === "image" ? "Images" : "Text"}
                </label>
              ))}
            </fieldset>
            <label className="field">
              <span>Categories (optional)</span>
              <input type="text" value={categories} placeholder="For example: Cat, Dog" onChange={(event) => setCategories(event.target.value)} />
              <span className="muted small">Separate them with commas. You can also add them later, or take them from an open dataset.</span>
            </label>
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
                    {row.kind === "image" ? "Images" : "Text"} · {row.categories.length} {row.categories.length === 1 ? "category" : "categories"} · {row.labelled.toLocaleString()} labelled{row.unsorted ? ` · ${row.unsorted.toLocaleString()} unsorted` : ""}
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

function DatasetDetail({ mode, datasetId, onBack, onBuild }: { mode: Mode; datasetId: string; onBack: () => void; onBuild: (dataset: DatasetSummary) => void }) {
  const beginner = mode === "beginner";
  const [dataset, setDataset] = useState<DatasetSummary | null>(null);
  const [missing, setMissing] = useState(false);
  const [problem, setProblem] = useState<string | null>(null);
  const [newCategory, setNewCategory] = useState("");

  const load = useCallback(async () => {
    try {
      setDataset(await api.fetchSavedDataset(datasetId));
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
    if (!dataset || !window.confirm(`Delete "${dataset.name}" and all of its copied images and labels? Your original files aren't touched. This can't be undone.`)) return;
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
    if (window.confirm(`Remove the category "${name}"? Its examples go back to being unsorted.`)) {
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
            <p className="muted small">{dataset.kind === "image" ? "Images" : "Text"} · {dataset.labelled.toLocaleString()} labelled · {dataset.unsorted.toLocaleString()} unsorted</p>
          </div>
          <div className="label-buttons">
            <button className="button button-small button-quiet" onClick={() => void remove()}>Delete dataset</button>
            <button className="button button-primary" disabled={!dataset.labelled} onClick={() => onBuild(dataset)}>Build a model with this data</button>
          </div>
        </header>
        {problem && <div className="result result-fail" role="alert"><p>{problem}</p></div>}

        <section className="card">
          <h2>Categories</h2>
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
        </section>

        <DatasetLabels
          dataset={dataset}
          onChanged={() => void load()}
          onUseCategories={(names) => void run(() => api.addDatasetCategories(datasetId, names))}
          onBuild={() => onBuild(dataset)}
        />
      </div>
    </div>
  );
}
