import { useEffect, useMemo, useState } from "react";
import * as api from "./api";
import type { OpenDataImport, OpenDataInfo, OpenDataResult, OpenDataSource } from "./types";

interface Props {
  datasetId: string;
  kind: "image" | "text";
  categories: string[];
  onImported: () => void;
  onUseCategories: (names: string[]) => void;
}

const POLL_MS = 800;
const DEFAULT_PER_CATEGORY = 50;
const MAX_PER_CATEGORY = 500;
const SKIP = "";

/** Licences that clearly allow use and sharing, including in commercial work. */
const PERMISSIVE = new Set(["mit", "apache-2.0", "bsd", "bsd-2-clause", "bsd-3-clause", "cc0-1.0", "cc-by-4.0", "cc-by-3.0", "cc-by-sa-4.0", "odc-by", "pddl", "unlicense"]);

function licenceNote(licence: string) {
  if (licence === "unknown" || licence === "other") return "No clear licence is listed, so check the dataset page before relying on it.";
  if (PERMISSIVE.has(licence)) return `Licence: ${licence}. Check its terms, such as giving credit, before sharing your model.`;
  return `Licence: ${licence}. It may limit how you can use the data, such as no commercial use, so read it first.`;
}

export function FindData({ datasetId, kind, categories, onImported, onUseCategories }: Props) {
  const [open, setOpen] = useState(false);
  const [query, setQuery] = useState("");
  const [results, setResults] = useState<OpenDataResult[] | null>(null);
  const [info, setInfo] = useState<OpenDataInfo | null>(null);
  const [mapping, setMapping] = useState<Record<string, string>>({});
  const [perCategory, setPerCategory] = useState(DEFAULT_PER_CATEGORY);
  const [job, setJob] = useState<OpenDataImport | null>(null);
  const [sources, setSources] = useState<OpenDataSource[]>([]);
  const [busy, setBusy] = useState(false);
  const [problem, setProblem] = useState<string | null>(null);

  const running = job?.status === "running";
  const categoryKey = JSON.stringify(categories);

  useEffect(() => {
    let stopped = false;
    let announced = false;
    const poll = async () => {
      try {
        const reply = await api.fetchOpenDataImport(datasetId);
        if (stopped) return;
        setJob(reply.job);
        setSources(reply.sources);
        if (reply.job?.status === "running") {
          onImported();
        } else if (running && !announced) {
          announced = true;
          onImported();
        }
      } catch {
        // The offline banner covers this. The next tick tries again.
      }
    };
    void poll();
    if (!running) return () => { stopped = true; };
    const timer = window.setInterval(poll, POLL_MS);
    return () => { stopped = true; window.clearInterval(timer); };
    // onImported changes every render, and this should only restart when an import starts or ends.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [running, datasetId]);

  useEffect(() => {
    if (!info) return;
    const byName = new Map((JSON.parse(categoryKey) as string[]).map((name) => [name.toLowerCase(), name]));
    setMapping(Object.fromEntries(info.labels.map((label) => [label, byName.get(label.toLowerCase()) ?? SKIP])));
  }, [info, categoryKey]);

  const search = async () => {
    setBusy(true);
    setProblem(null);
    setInfo(null);
    try {
      setResults(await api.searchOpenData(query, kind));
    } catch (error) {
      setResults(null);
      setProblem(error instanceof Error ? error.message : "The search didn't work.");
    } finally {
      setBusy(false);
    }
  };

  const choose = async (id: string) => {
    setBusy(true);
    setProblem(null);
    try {
      setInfo(await api.inspectOpenData(id, kind));
    } catch (error) {
      setInfo(null);
      setProblem(error instanceof Error ? error.message : "That dataset couldn't be read.");
    } finally {
      setBusy(false);
    }
  };

  const start = async () => {
    if (!info) return;
    setProblem(null);
    try {
      setJob(await api.startOpenDataImport(info.dataset, kind, mapping, perCategory, datasetId));
    } catch (error) {
      setProblem(error instanceof Error ? error.message : "The import couldn't start.");
    }
  };

  const matched = useMemo(() => Object.values(mapping).filter(Boolean).length, [mapping]);
  const validAmount = Number.isInteger(perCategory) && perCategory >= 1 && perCategory <= MAX_PER_CATEGORY;

  if (!open) {
    return (
      <section className="card">
        <div className="card-head">
          <h2>No data yet?</h2>
          <button className="button button-small" onClick={() => setOpen(true)}>Find open data</button>
        </div>
        <p className="muted small">Search free labelled datasets on Hugging Face and copy a sample into this dataset.</p>
        {running && <p className="muted small field-note" role="status">Importing {job?.dataset}…</p>}
      </section>
    );
  }

  return (
    <section className="card">
      <div className="card-head">
        <h2>Find open data</h2>
        <button className="button button-small button-quiet" onClick={() => setOpen(false)}>Close</button>
      </div>
      <form className="find-search" onSubmit={(event) => { event.preventDefault(); void search(); }}>
        <input
          type="search"
          value={query}
          maxLength={100}
          placeholder={kind === "image" ? "For example: cats, plant disease, x-ray" : "For example: spam, sentiment, reviews"}
          aria-label="Search open datasets"
          onChange={(event) => setQuery(event.target.value)}
        />
        <button className="button" type="submit" disabled={busy}>{busy && !info ? "Searching…" : "Search"}</button>
      </form>
      <p className="muted small field-note">Searches Hugging Face. Your search words are sent there, but your own data isn't. Downloaded data stays on this computer.</p>

      {problem && <div className="result result-fail field-note" role="alert"><p>{problem}</p></div>}
      {results?.length === 0 && <p className="muted small field-note">Nothing matched. Try a simpler word.</p>}
      {results && results.length > 0 && !info && (
        <ul className="find-results">
          {results.map((result) => (
            <li key={result.id}>
              <div className="project-text">
                <strong>{result.id}</strong>
                <span className="muted small">{result.downloads.toLocaleString()} downloads · licence: {result.licence}</span>
              </div>
              <button className="button button-small" disabled={busy} onClick={() => void choose(result.id)}>View</button>
            </li>
          ))}
        </ul>
      )}

      {info && (
        <div className="find-detail">
          <div className="card-head">
            <h3>{info.dataset}</h3>
            <button className="button button-small button-quiet" onClick={() => setInfo(null)}>Back to results</button>
          </div>
          <p className="small muted">
            {info.totalRows.toLocaleString()} examples in {info.labels.length} categories.{" "}
            <a href={info.url} target="_blank" rel="noreferrer noopener">Open the dataset page</a>
          </p>
          <p className={`small field-note ${PERMISSIVE.has(info.licence) ? "muted" : "find-caution"}`}>{licenceNote(info.licence)}</p>

          {info.preview.length > 0 && (
            <div className={info.kind === "image" ? "find-preview" : "find-preview-text"}>
              {info.preview.map((item, index) => item.image
                ? <img key={index} src={item.image} alt="Sample from this dataset" loading="lazy" referrerPolicy="no-referrer" />
                : <p key={index} className="small">{item.text}</p>)}
            </div>
          )}

          {matched === 0 && info.namesUsable && (
            <div className="notice field-note">
              <span>None of these categories match this dataset's.</span>
              <button className="button button-small" onClick={() => onUseCategories(info.labels)}>Add these {info.labels.length} categories</button>
            </div>
          )}

          <h3 className="find-heading">Match the categories</h3>
          <ul className="find-mapping">
            {info.labels.slice(0, 200).map((label) => (
              <li key={label}>
                <span className="try-name small">{label}</span>
                <select
                  aria-label={`Category for ${label}`}
                  value={mapping[label] ?? SKIP}
                  onChange={(event) => setMapping((current) => ({ ...current, [label]: event.target.value }))}
                >
                  <option value={SKIP}>Don't import</option>
                  {categories.map((name) => <option key={name} value={name}>{name}</option>)}
                </select>
              </li>
            ))}
          </ul>
          {info.labels.length > 200 && <p className="muted small field-note">Showing the first 200 of {info.labels.length} categories.</p>}

          <div className="find-start">
            <label className="small">
              <span className="muted">Examples per category </span>
              <input
                type="number"
                min={1}
                max={MAX_PER_CATEGORY}
                value={Number.isNaN(perCategory) ? "" : perCategory}
                onChange={(event) => setPerCategory(event.target.valueAsNumber)}
              />
            </label>
            <span className="spacer" />
            <button className="button button-primary" disabled={!matched || !validAmount || running} onClick={() => void start()}>
              {running ? "Importing…" : "Import"}
            </button>
          </div>
        </div>
      )}

      {job && <ImportStatus job={job} />}

      {sources.length > 0 && (
        <details className="chart-table field-note">
          <summary>Imported so far</summary>
          <ul className="plain-list small">
            {sources.map((source) => (
              <li key={source.dataset}>
                <a href={source.url} target="_blank" rel="noreferrer noopener">{source.dataset}</a>: {source.added.toLocaleString()} examples, licence {source.licence}
              </li>
            ))}
          </ul>
        </details>
      )}
    </section>
  );
}

function ImportStatus({ job }: { job: OpenDataImport }) {
  const wanted = Object.keys(job.counts).length * job.perCategory;
  const have = Object.values(job.counts).reduce((sum, value) => sum + value, 0);
  const summary = Object.entries(job.counts).map(([name, amount]) => `${name} ${amount}`).join(", ");
  const short = job.status === "finished" && have < wanted;
  return (
    <div className="find-status" role="status">
      {job.status === "running" && (
        <>
          <progress className="train-progress" max={wanted} value={have} aria-label="Import progress" />
          <p className="small muted">Importing from {job.dataset}: {have} of {wanted}.</p>
          <button className="button button-small" onClick={() => void api.cancelOpenDataImport()}>Stop</button>
        </>
      )}
      {job.status === "failed" && <div className="result result-fail"><p>{job.error}</p></div>}
      {job.status !== "running" && job.status !== "failed" && (
        <p className="small muted">
          {job.status === "cancelled" ? "Stopped. " : "Import finished. "}
          {summary}.{job.duplicates ? ` ${job.duplicates} ${job.duplicates === 1 ? "was" : "were"} already here.` : ""}
          {job.skipped ? ` ${job.skipped} couldn't be used.` : ""}
          {short ? " The dataset ran out of usable examples for some categories, or they weren't in the part that was read." : ""}
        </p>
      )}
    </div>
  );
}
