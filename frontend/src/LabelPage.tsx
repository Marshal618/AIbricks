import { useCallback, useEffect, useMemo, useRef, useState, type ChangeEvent, type ReactNode } from "react";
import * as api from "./api";
import { FindData } from "./FindData";
import { WebcamCapture } from "./Webcam";
import type { DatasetSummary, ImageQueue, LabelAssist, LabelItem, TextQueue } from "./types";

interface Props {
  dataset: DatasetSummary;
  onChanged: () => void;
  onUseCategories: (names: string[]) => void;
  onBuild: () => void;
}

const UPLOAD_WORKERS = 4;
const UNDO_DEPTH = 100;
const PRELOAD = 3;
const POLL_MS = 1000;
const SAVED_REFRESH_MS = 400;

interface Action {
  item: LabelItem;
  category: string | null;
}

interface Adapter {
  assign: (id: string, category: string) => Promise<void>;
  unassign: (id: string, category: string | null) => Promise<void>;
  remove: (id: string) => Promise<void>;
}

/** Asks the dataset's newest model to guess a label for everything unsorted, and refreshes when it's done. */
function useSuggestions(datasetId: string, suggesting: boolean, check: () => Promise<LabelAssist | null>, load: () => Promise<void>) {
  const [problem, setProblem] = useState<string | null>(null);
  useEffect(() => {
    if (!suggesting) return;
    const timer = window.setInterval(async () => {
      const next = await check().catch(() => null);
      if (next && !next.suggesting) void load();
    }, POLL_MS);
    return () => window.clearInterval(timer);
  }, [suggesting, check, load]);
  const suggest = async () => {
    setProblem(null);
    try {
      await api.startSuggestions(datasetId);
      await load();
    } catch (error) {
      setProblem(error instanceof Error ? error.message : "Suggestions could not start.");
    }
  };
  return { suggest, problem };
}

/** The card that offers suggestions, or says what's needed before the model can help. */
function ModelHelp({ assist, items, categories, counts, things, trainHint, onSuggest, onBuild }: {
  assist: LabelAssist; items: LabelItem[]; categories: string[]; counts: Record<string, number>; things: string;
  trainHint: string; onSuggest: () => void; onBuild?: () => void;
}) {
  const sameCategories = assist.modelCategories !== null && JSON.stringify(assist.modelCategories) === JSON.stringify([...categories].sort());
  const suggested = items.some((item) => item.suggestion);
  const smallest = Math.min(...categories.map((name) => counts[name] ?? 0));
  const differs = assist.modelCategories !== null && !sameCategories;
  return (
    <section className="card label-assist">
      <div className="card-head">
        <h2>Let the model help</h2>
        {sameCategories && (
          <button className="button button-small" disabled={assist.suggesting || assist.unsuggested === 0} onClick={onSuggest}>
            {assist.suggesting ? "Working…" : assist.unsuggested === 0 ? "Suggestions ready" : "Suggest labels"}
          </button>
        )}
      </div>
      <p className="muted small">
        {suggested
          ? `The ${things} the model is least sure about come first, because they teach it the most. Check each suggestion, since the model can be wrong.`
          : sameCategories
            ? `Your trained model can suggest a label for each of the ${things}, so you only need to confirm or correct it.`
            : differs
              ? `The newest model was trained with different categories. ${trainHint} again, and it can suggest labels.`
              : `Label about 20 ${things} per category, then ${trainHint.toLowerCase()}. It can then suggest the rest, and you only confirm or correct them.${smallest >= 20 ? "" : " It isn't ready yet."}`}
      </p>
      {!suggested && assist.coachOrder && <p className="small field-note">For now, the data coach has put the most useful {things} first, so each one you label teaches the most.</p>}
      {assist.suggestError && <p className="small tone-bad field-note">{assist.suggestError}</p>}
      {!sameCategories && smallest >= 20 && categories.length > 0 && onBuild && <button className="button button-small field-note" onClick={onBuild}>Build a model with this data</button>}
    </section>
  );
}

/** Everything for filling one dataset: open data, your own files, and labelling. */
export function DatasetLabels(props: Props) {
  return props.dataset.kind === "image" ? <ImageLabels {...props} /> : <TextLabels {...props} />;
}

function useKeys(handler: (event: KeyboardEvent) => void) {
  const latest = useRef(handler);
  latest.current = handler;
  useEffect(() => {
    const listener = (event: KeyboardEvent) => latest.current(event);
    window.addEventListener("keydown", listener);
    return () => window.removeEventListener("keydown", listener);
  }, []);
}

interface LabellerProps {
  items: LabelItem[];
  setItems: (update: (items: LabelItem[]) => LabelItem[]) => void;
  /** Items beyond the ones sent to the browser, since one queue holds at most a few thousand. */
  hidden: number;
  categories: string[];
  counts: Record<string, number>;
  adapter: Adapter;
  onChanged: () => void;
  /** Called shortly after labels save, so the totals elsewhere on the page catch up. */
  onSaved: () => void;
  onCount: (category: string, change: number) => void;
  render: (item: LabelItem) => ReactNode;
  preload?: (item: LabelItem) => void;
  doneMessage: ReactNode;
}

/** Shows one item at a time. Labels apply instantly on screen and reach the server in order. */
function Labeller({ items, setItems, hidden, categories, counts, adapter, onChanged, onSaved, onCount, render, preload, doneMessage }: LabellerProps) {
  const [history, setHistory] = useState<Action[]>([]);
  const [problem, setProblem] = useState<string | null>(null);
  const requests = useRef<Promise<unknown>>(Promise.resolve());
  const current = items[0];

  const saveTimer = useRef<number>();
  useEffect(() => () => window.clearTimeout(saveTimer.current), []);

  const send = (request: () => Promise<void>, recover: () => void) => {
    requests.current = requests.current.then(request).then(() => {
      window.clearTimeout(saveTimer.current);
      saveTimer.current = window.setTimeout(onSaved, SAVED_REFRESH_MS);
    }).catch((error) => {
      setProblem(error instanceof Error ? error.message : "That change couldn't be saved.");
      recover();
    });
  };

  const assign = (category: string) => {
    if (!current) return;
    const item = current;
    setProblem(null);
    setItems((list) => list.filter((entry) => entry.id !== item.id));
    setHistory((list) => [{ item, category }, ...list].slice(0, UNDO_DEPTH));
    onCount(category, 1);
    send(() => adapter.assign(item.id, category), () => { onChanged(); setHistory((list) => list.filter((entry) => entry.item.id !== item.id)); });
  };

  const skip = () => {
    if (items.length > 1) setItems((list) => [...list.slice(1), list[0]]);
  };

  const remove = () => {
    if (!current) return;
    const item = current;
    setProblem(null);
    setItems((list) => list.filter((entry) => entry.id !== item.id));
    setHistory((list) => [{ item, category: null }, ...list].slice(0, UNDO_DEPTH));
    send(() => adapter.remove(item.id), () => { onChanged(); setHistory((list) => list.filter((entry) => entry.item.id !== item.id)); });
  };

  const undo = () => {
    const [last, ...rest] = history;
    if (!last) return;
    setProblem(null);
    setHistory(rest);
    setItems((list) => [last.item, ...list]);
    if (last.category) onCount(last.category, -1);
    send(() => adapter.unassign(last.item.id, last.category), onChanged);
  };

  useKeys((event) => {
    if (event.metaKey || event.ctrlKey || event.altKey || event.repeat) return;
    const target = event.target as HTMLElement;
    if (["INPUT", "TEXTAREA", "SELECT"].includes(target.tagName) || document.querySelector(".dialog")) return;
    const key = event.key.toLowerCase();
    if (key === "enter" && target.tagName === "BUTTON") return; // Someone tabbed here, so press that button.
    const number = Number(key);
    if (number >= 1 && number <= Math.min(categories.length, 9)) assign(categories[number - 1]);
    else if (key === "enter" && current?.suggestion && categories.includes(current.suggestion.category)) assign(current.suggestion.category);
    else if (key === "s") skip();
    else if (key === "x") remove();
    else if (key === "z") undo();
    else return;
    event.preventDefault();
  });

  useEffect(() => {
    if (preload) items.slice(1, 1 + PRELOAD).forEach(preload);
  }, [items, preload]);

  // Fetch the next batch once this one is used up and its labels have saved.
  useEffect(() => {
    if (!items.length && hidden > 0) void requests.current.then(onChanged);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [items.length, hidden]);

  const remaining = items.length + hidden;
  const suggestion = current?.suggestion && categories.includes(current.suggestion.category) ? current.suggestion : null;

  return (
    <section
      className="card"
      // After a mouse click, Enter should accept the suggestion, not press that button again.
      onClick={(event) => { if (event.detail > 0 && document.activeElement instanceof HTMLElement) document.activeElement.blur(); }}
    >
      <div className="card-head">
        <h2>{current ? `${remaining.toLocaleString()} left to label` : "Nothing left to label"}</h2>
        <button className="button button-small" disabled={!history.length} onClick={undo}>Undo</button>
      </div>
      {problem && <div className="result result-fail" role="alert"><p>{problem}</p></div>}
      {current
        ? <>
            <div className="label-stage">{render(current)}</div>
            {suggestion && (
              <p className="label-suggestion small" role="status">
                The model thinks this is <strong>{suggestion.category}</strong> ({Math.round(suggestion.chance * 100)}% sure). Press Enter to accept, or choose another.
              </p>
            )}
            <div className="label-choices" role="group" aria-label="Choose a category">
              {categories.map((name, index) => (
                <button key={name} className={`button label-choice${suggestion?.category === name ? " is-suggested" : ""}`} onClick={() => assign(name)}>
                  {index < 9 && <kbd>{index + 1}</kbd>}<span>{name}</span>
                </button>
              ))}
            </div>
            <div className="label-actions small">
              <button className="button button-small button-quiet" onClick={skip} disabled={items.length < 2}><kbd>S</kbd> Skip for now</button>
              <button className="button button-small button-quiet" onClick={remove}><kbd>X</kbd> Not useful, remove</button>
              <span className="muted"><kbd>Z</kbd> undoes the last one</span>
            </div>
          </>
        : doneMessage}
      <ul className="label-counts small">
        {categories.map((name) => (
          <li key={name}><span className="muted">{name}</span> <strong>{(counts[name] ?? 0).toLocaleString()}</strong></li>
        ))}
      </ul>
    </section>
  );
}

function ImageLabels({ dataset, onChanged, onUseCategories, onBuild }: Props) {
  const datasetId = dataset.id;
  const categories = dataset.categories;
  const adapter: Adapter = useMemo(() => ({
    assign: (id, category) => api.assignImage(id, category, datasetId),
    unassign: (id, category) => api.unassignImage(id, category, datasetId),
    remove: (id) => api.removeImage(id, datasetId),
  }), [datasetId]);
  const preload = useCallback((item: LabelItem) => { new Image().src = api.unsortedImageUrl(item.id, datasetId); }, [datasetId]);
  const [queue, setQueue] = useState<ImageQueue | null>(null);
  const [items, setItems] = useState<LabelItem[]>([]);
  const [counts, setCounts] = useState<Record<string, number>>({});
  const [upload, setUpload] = useState<{ done: number; total: number; added: number; duplicates: number; rejected: string[] } | null>(null);
  const [problem, setProblem] = useState<string | null>(null);
  const filesInput = useRef<HTMLInputElement>(null);
  const folderInput = useRef<HTMLInputElement>(null);
  const categoryKey = JSON.stringify(categories);

  const load = useCallback(async () => {
    try {
      const [nextQueue, info] = await Promise.all([api.fetchImageQueue(datasetId), api.fetchDataset(JSON.parse(categoryKey), datasetId)]);
      setQueue(nextQueue);
      setItems(nextQueue.items);
      setCounts(info.counts);
    } catch (error) {
      setProblem(error instanceof Error ? error.message : "The images could not load.");
    }
  }, [categoryKey, datasetId]);

  useEffect(() => { void load(); }, [load]);

  const checkImages = useCallback(() => api.fetchImageQueue(datasetId), [datasetId]);
  const help = useSuggestions(datasetId, queue?.suggesting ?? false, checkImages, load);
  const pose = dataset.goal === "pose";

  const addFiles = async (event: ChangeEvent<HTMLInputElement>) => {
    const files = Array.from(event.target.files ?? []).filter((file) => !file.name.startsWith("."));
    event.target.value = "";
    if (!files.length) return;
    const state = { done: 0, total: files.length, added: 0, duplicates: 0, rejected: [] as string[] };
    setUpload({ ...state });
    const list = [...files];
    const worker = async () => {
      for (let file = list.shift(); file; file = list.shift()) {
        try {
          const reply = await api.addUnsortedImage(file, datasetId);
          if (reply.duplicate) state.duplicates += 1; else state.added += 1;
        } catch (error) {
          state.rejected.push(`${file.name}: ${error instanceof Error ? error.message : "could not be read."}`);
        }
        state.done += 1;
        setUpload({ ...state });
      }
    };
    await Promise.all(Array.from({ length: UPLOAD_WORKERS }, worker));
    await load();
    onChanged();
  };

  const clearAll = async () => {
    if (!window.confirm("Remove all unsorted images from AI Bricks? Your original files aren't touched.")) return;
    await api.clearUnsortedImages(datasetId).catch(() => undefined);
    await load();
    onChanged();
  };

  const addedFromCamera = useCallback((category: string) => {
    setCounts((current) => ({ ...current, [category]: (current[category] ?? 0) + 1 }));
  }, []);

  const uploading = upload !== null && upload.done < upload.total;
  const smallest = Math.min(...categories.map((name) => counts[name] ?? 0));

  return (
    <>
      {problem && <div className="result result-fail" role="alert"><p>{problem}</p></div>}
      <h2 className="section-title">Add examples</h2>
      <WebcamCapture datasetId={datasetId} categories={categories} onAdded={addedFromCamera} onDone={onChanged} />
      <FindData datasetId={datasetId} kind="image" categories={categories} onImported={() => { void load(); onChanged(); }} onUseCategories={onUseCategories} />
      <section className="card">
        <div className="card-head">
          <h2>Add photos from your computer</h2>
          <div className="label-buttons">
            {(queue?.total ?? 0) > 0 && <button className="button button-small button-quiet" disabled={uploading} onClick={() => void clearAll()}>Remove all</button>}
            <button className="button button-small" disabled={uploading} onClick={() => folderInput.current?.click()}>Choose a folder</button>
            <button className="button button-small" disabled={uploading} onClick={() => filesInput.current?.click()}>Add images</button>
          </div>
          <input ref={filesInput} className="visually-hidden" type="file" multiple accept="image/jpeg,image/png,image/bmp,image/webp" aria-label="Add unsorted images" tabIndex={-1} onChange={(event) => void addFiles(event)} />
          <input ref={folderInput} className="visually-hidden" type="file" aria-label="Choose a folder of unsorted images" tabIndex={-1} onChange={(event) => void addFiles(event)} {...{ webkitdirectory: "" }} />
        </div>
        <p className="muted small">They wait below until you sort them into a category. Nothing is uploaded anywhere.</p>
        {upload && uploading && (
          <div className="field-note" role="status">
            <progress className="train-progress" max={upload.total} value={upload.done} aria-label="Images copied" />
            <p className="small muted">Checking and copying {upload.done} of {upload.total}…</p>
          </div>
        )}
        {upload && !uploading && (
          <div className="field-note small" role="status">
            <p className="muted">Added {upload.added} {upload.added === 1 ? "image" : "images"}.{upload.duplicates ? ` ${upload.duplicates} ${upload.duplicates === 1 ? "was" : "were"} already here.` : ""}</p>
            {upload.rejected.length > 0 && (
              <details className="train-rejected" open={upload.rejected.length <= 3}>
                <summary>{upload.rejected.length} {upload.rejected.length === 1 ? "file" : "files"} couldn't be used</summary>
                <ul>{upload.rejected.slice(0, 50).map((line) => <li key={line}>{line}</li>)}</ul>
              </details>
            )}
          </div>
        )}
      </section>

      {queue && queue.total > 0 && (
        <ModelHelp
          assist={queue} items={items} categories={categories} counts={counts} things={pose ? "photos" : "images"}
          trainHint={pose ? "Train a pose model on the Train tab" : "Train a model on this dataset"}
          onSuggest={() => void help.suggest()} onBuild={pose ? undefined : onBuild}
        />
      )}
      {help.problem && <div className="result result-fail" role="alert"><p>{help.problem}</p></div>}

      <h2 className="section-title">Sort them</h2>
      {queue && categories.length === 0 && (
        <div className="notice">Add at least one category above, then you can start labelling.</div>
      )}
      {queue && categories.length > 0 && (
        <Labeller
          key={categoryKey}
          items={items}
          setItems={setItems}
          hidden={queue.total - queue.items.length}
          categories={categories}
          counts={counts}
          adapter={adapter}
          onChanged={() => { void load(); onChanged(); }}
          onSaved={onChanged}
          onCount={(category, change) => setCounts((current) => ({ ...current, [category]: (current[category] ?? 0) + change }))}
          preload={preload}
          render={(item) => <img key={item.id} className="label-image" src={api.unsortedImageUrl(item.id, datasetId)} alt="The image to label" />}
          doneMessage={
            <div className="label-done">
              <p className="muted">{smallest === 0 && queue.total === 0 && !upload ? "Add images above to start labelling." : "All sorted. Your labelled images are ready to train on."}</p>
              {smallest > 0 && !pose && <button className="button button-small" onClick={onBuild}>Build a model with this data</button>}
            </div>
          }
        />
      )}
    </>
  );
}

function TextLabels({ dataset, onChanged, onUseCategories, onBuild }: Props) {
  const datasetId = dataset.id;
  const categories = dataset.categories;
  const adapter: Adapter = useMemo(() => ({
    assign: (id, category) => api.assignText(id, category, datasetId),
    unassign: (id) => api.unassignText(id, datasetId),
    remove: (id) => api.removeText(id, datasetId),
  }), [datasetId]);
  const [queue, setQueue] = useState<TextQueue | null>(null);
  const [items, setItems] = useState<LabelItem[]>([]);
  const [counts, setCounts] = useState<Record<string, number>>({});
  const [message, setMessage] = useState<string | null>(null);
  const [problem, setProblem] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const input = useRef<HTMLInputElement>(null);
  const categoryKey = JSON.stringify(categories);

  const load = useCallback(async () => {
    try {
      const next = await api.fetchTextQueue(JSON.parse(categoryKey), datasetId);
      setQueue(next);
      setItems(next.items);
      setCounts(next.counts);
    } catch (error) {
      setProblem(error instanceof Error ? error.message : "The text could not load.");
    }
  }, [categoryKey, datasetId]);

  useEffect(() => { void load(); }, [load]);

  const checkText = useCallback(() => api.fetchTextQueue(JSON.parse(categoryKey), datasetId), [categoryKey, datasetId]);
  const help = useSuggestions(datasetId, queue?.suggesting ?? false, checkText, load);

  const addFile = async (event: ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    event.target.value = "";
    if (!file) return;
    setProblem(null);
    setMessage(null);
    setBusy(true);
    try {
      const reply = await api.addUnsortedText(file, datasetId);
      const parts = [`Added ${reply.added.toLocaleString()} ${reply.added === 1 ? "text" : "texts"}${reply.column ? ` from the "${reply.column}" column` : ""}.`];
      if (reply.duplicates) parts.push(`${reply.duplicates.toLocaleString()} ${reply.duplicates === 1 ? "was" : "were"} already here.`);
      if (reply.tooLong) parts.push(`${reply.tooLong.toLocaleString()} skipped for being over 5,000 characters.`);
      setMessage(parts.join(" "));
      await load();
      onChanged();
    } catch (error) {
      setProblem(error instanceof Error ? error.message : "That file could not be read.");
    } finally {
      setBusy(false);
    }
  };

  const clearAll = async () => {
    if (!window.confirm("Remove all text and its labels from AI Bricks? Your original files aren't touched.")) return;
    await api.clearUnsortedText(datasetId).catch(() => undefined);
    setMessage(null);
    await load();
    onChanged();
  };

  const labelled = Object.values(counts).reduce((sum, value) => sum + value, 0);

  return (
    <>
      {problem && <div className="result result-fail" role="alert"><p>{problem}</p></div>}
      <h2 className="section-title">Add examples</h2>
      <FindData datasetId={datasetId} kind="text" categories={categories} onImported={() => { void load(); onChanged(); }} onUseCategories={onUseCategories} />
      <section className="card">
        <div className="card-head">
          <h2>Your text</h2>
          <div className="label-buttons">
            {((queue?.total ?? 0) > 0 || labelled > 0) && <button className="button button-small button-quiet" disabled={busy} onClick={() => void clearAll()}>Remove all</button>}
            <button className="button button-small" disabled={busy} onClick={() => input.current?.click()}>{busy ? "Reading…" : "Add a file"}</button>
          </div>
          <input ref={input} className="visually-hidden" type="file" accept=".txt,.csv,text/plain,text/csv" aria-label="Add a text or CSV file" tabIndex={-1} onChange={(event) => void addFile(event)} />
        </div>
        <p className="muted small">Use a .txt file with one example per line, or a .csv file with a column called "text". It stays on this computer.</p>
        {message && <p className="muted small field-note" role="status">{message}</p>}
      </section>

      {queue && queue.total > 0 && (
        <ModelHelp
          assist={queue} items={items} categories={categories} counts={counts} things="texts"
          trainHint="Train a text model on the Train tab" onSuggest={() => void help.suggest()}
        />
      )}
      {help.problem && <div className="result result-fail" role="alert"><p>{help.problem}</p></div>}

      <h2 className="section-title">Sort them</h2>
      {queue && categories.length === 0 && (
        <div className="notice">Add at least one category above, then you can start labelling.</div>
      )}
      {queue && categories.length > 0 && (
        <Labeller
          key={categoryKey}
          items={items}
          setItems={setItems}
          hidden={queue.total - queue.items.length}
          categories={categories}
          counts={counts}
          adapter={adapter}
          onChanged={() => { void load(); onChanged(); }}
          onSaved={onChanged}
          onCount={(category, change) => setCounts((current) => ({ ...current, [category]: (current[category] ?? 0) + change }))}
          render={(item) => <p className="label-text">{item.text}</p>}
          doneMessage={<p className="muted label-done">{queue.total === 0 && labelled === 0 ? "Add a file above to start labelling." : "All sorted. Download your labels below."}</p>}
        />
      )}

      <section className="card">
        <div className="card-head">
          <h2>Use your labels</h2>
          {labelled > 0
            ? <a className="button button-small" href={api.textExportUrl(categories, datasetId)} download="texts.csv">Download texts.csv</a>
            : <button className="button button-small" disabled>Download texts.csv</button>}
        </div>
        <p className="muted small">
          Train a text model on the Train tab, or download your labels for a project built in the Model builder: put this file at <code>data/texts.csv</code> inside it, then run <code>python train.py</code>.
          Labels save as you go, so you can stop and come back.
        </p>
        {(queue?.otherLabels ?? 0) > 0 && <p className="muted small field-note">{queue?.otherLabels.toLocaleString()} labelled texts use a category this dataset doesn't have now, so they're left out of the file.</p>}
        <button className="button button-small field-note" onClick={onBuild}>Build a model with this data</button>
      </section>
    </>
  );
}
