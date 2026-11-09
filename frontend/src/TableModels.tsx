import { useCallback, useEffect, useRef, useState, type FormEvent } from "react";
import * as api from "./api";
import type { DatasetSummary, PredictAnswer, SavedModel, TableAnalysis, TableSchema } from "./types";

function message(error: unknown, fallback: string) {
  return error instanceof Error ? error.message : fallback;
}

function plain(value: number) {
  return value.toLocaleString(undefined, { maximumFractionDigits: 2 });
}

export function PreviewTable({ table }: { table: { columns: string[]; rows: string[][] } }) {
  return (
    <div className="table-scroll" tabIndex={0} role="region" aria-label="First rows of the table">
      <table>
        <thead><tr>{table.columns.map((name) => <th key={name} scope="col" title={name}>{name}</th>)}</tr></thead>
        <tbody>
          {table.rows.map((row, index) => (
            <tr key={index}>{row.map((cell, column) => <td key={column} title={cell}>{cell}</td>)}</tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

export function TableData({ dataset, beginner, onChanged, onTrain }: { dataset: DatasetSummary; beginner: boolean; onChanged: (next: DatasetSummary) => void; onTrain: () => void }) {
  const input = useRef<HTMLInputElement>(null);
  const [table, setTable] = useState<{ columns: string[]; rows: string[][] } | null>(null);
  const [busy, setBusy] = useState(false);
  const [problem, setProblem] = useState<string | null>(null);
  const [status, setStatus] = useState<string | null>(null);

  const loadPreview = useCallback(async () => {
    setTable((await api.fetchDatasetPreview(dataset.id)).table ?? null);
  }, [dataset.id]);

  useEffect(() => { loadPreview().catch((error) => setProblem(message(error, "The table could not load."))); }, [loadPreview]);

  const send = async (file: File | undefined) => {
    if (!file) return;
    setBusy(true);
    setProblem(null);
    setStatus(null);
    try {
      const reply = await api.uploadTable(dataset.id, file);
      onChanged(reply.dataset);
      await loadPreview();
      const parts = [`Added ${reply.dataset.labelled.toLocaleString()} rows.`];
      if (reply.damagedRows) parts.push(`Skipped ${reply.damagedRows.toLocaleString()} damaged rows with the wrong number of columns.`);
      if (reply.savedFirst) parts.push(`Your previous data is saved as v${reply.savedFirst}.`);
      setStatus(parts.join(" "));
    } catch (error) {
      setProblem(message(error, "This file couldn't be read."));
    } finally {
      setBusy(false);
    }
  };

  return (
    <>
      {problem && <div className="result result-fail" role="alert"><p>{problem}</p></div>}
      {status && <div className="notice notice-good" role="status"><span>{status}</span></div>}
      <section className="card">
        <div className="card-head">
          <h2>Your table</h2>
          <div className="label-buttons">
            <button className={`button button-small${table ? "" : " button-primary"}`} disabled={busy} onClick={() => input.current?.click()}>{busy ? "Reading…" : table ? "Replace with a new file" : "Upload a CSV file"}</button>
            {table && <button className="button button-small button-primary" onClick={onTrain}>Train a model</button>}
          </div>
        </div>
        <input
          ref={input}
          className="visually-hidden"
          type="file"
          accept=".csv,.tsv,text/csv"
          aria-label="Choose a CSV file"
          tabIndex={-1}
          onChange={(event) => { void send(event.target.files?.[0]); event.target.value = ""; }}
        />
        <p className="muted small">
          {beginner
            ? "Use a table with a header row and one example on each row, such as a spreadsheet saved as CSV. It's copied to this computer and isn't sent anywhere."
            : "CSV with a header row, up to 50 MB and 200,000 rows. Comma, semicolon and tab separators work. Stored locally."}
          {table ? " Replacing the file saves any unsaved data as a version first." : ""}
        </p>
        {table && <PreviewTable table={table} />}
        {table && dataset.labelled > table.rows.length && <p className="muted small field-note">Showing the first {table.rows.length} of {dataset.labelled.toLocaleString()} rows.</p>}
      </section>
    </>
  );
}

export function TableTrainer({ dataset, beginner, running, busyElsewhere, onStarted, onStop }: { dataset: DatasetSummary; beginner: boolean; running: SavedModel | null; busyElsewhere: boolean; onStarted: () => Promise<void>; onStop: (id: string) => void }) {
  const [analysis, setAnalysis] = useState<TableAnalysis | null>(null);
  const [chosen, setChosen] = useState<Set<string>>(new Set());
  const [problem, setProblem] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const request = useRef(0);

  const read = useCallback(async (target?: string, task?: "category" | "number") => {
    const mine = ++request.current;
    setLoading(true);
    setProblem(null);
    try {
      const reply = await api.fetchTableAnalysis(dataset.id, target, task);
      if (mine !== request.current) return;
      setAnalysis(reply);
      setChosen(new Set(reply.columns.filter((column) => column.use).map((column) => column.name)));
    } catch (error) {
      if (mine === request.current) setProblem(message(error, "The table could not be checked."));
    } finally {
      if (mine === request.current) setLoading(false);
    }
  }, [dataset.id]);

  useEffect(() => { if (dataset.labelled) void read(); }, [dataset.labelled, read]);

  if (!dataset.labelled) {
    return <section className="card"><h2>Predict a column</h2><p className="muted small">Upload a CSV file on the Data tab first.</p></section>;
  }

  const start = async () => {
    if (!analysis) return;
    setProblem(null);
    try {
      await api.trainTableModel(dataset.id, analysis.target, analysis.task, analysis.columns.filter((column) => chosen.has(column.name)).map((column) => column.name));
      await onStarted();
    } catch (error) {
      setProblem(message(error, "Training could not start."));
    }
  };

  const toggle = (name: string) => setChosen((current) => {
    const next = new Set(current);
    if (next.has(name)) next.delete(name); else next.add(name);
    return next;
  });

  const left = analysis?.columns.filter((column) => !chosen.has(column.name) && column.warning).length ?? 0;
  const ready = !!analysis && analysis.problems.length === 0 && chosen.size > 0 && !loading;

  return (
    <section className="card">
      <div className="card-head">
        <h2>Predict a column</h2>
        {running
          ? <button className="button button-small" onClick={() => onStop(running.id)}>Stop</button>
          : <button className="button button-small button-primary" disabled={!ready || busyElsewhere} onClick={() => void start()}>Start training</button>}
      </div>
      {problem && <p className="small" role="alert">{problem}</p>}
      {running && (
        <div role="status">
          <progress className="train-progress" max={running.stepsTotal || 3} value={running.stepsDone} aria-label="Training progress" />
          <p className="muted small">Training on this computer. This can be left running.</p>
        </div>
      )}
      {!running && busyElsewhere && <p className="muted small">Another model is training. Only one can train at a time.</p>}
      {analysis && (
        <>
          <div className="table-choices">
            <label>
              What do you want to predict?
              <select value={analysis.target} disabled={!!running} onChange={(event) => void read(event.target.value)}>
                {analysis.headers.map((name) => <option key={name} value={name}>{name}</option>)}
              </select>
            </label>
            <label>
              It's a
              <select value={analysis.task} disabled={!!running} onChange={(event) => void read(analysis.target, event.target.value as "category" | "number")}>
                <option value="category">Category, like yes or no</option>
                <option value="number">Number, like a price</option>
              </select>
            </label>
          </div>
          {analysis.problems.map((text) => <p className="notice field-note" key={text}>{text}</p>)}
          {analysis.notes.map((text) => <p className="muted small field-note" key={text}>{text}</p>)}
          <h3 className="field-note small">Learn from these columns</h3>
          {beginner && left > 0 && <p className="muted small">Some are left out because they could make the score look better than the model really is. You can tick them if you disagree.</p>}
          <ul className="table-columns">
            {analysis.columns.map((column) => (
              <li key={column.name}>
                <label>
                  <input type="checkbox" checked={chosen.has(column.name)} disabled={!!running || column.kind === "empty" || column.kind === "date"} onChange={() => toggle(column.name)} />
                  <span className="table-column-text">
                    <span>{column.name} <span className="muted small">{column.kind}{column.kind === "category" ? `, ${column.distinct.toLocaleString()} values` : ""}{column.missing ? `, ${column.missing.toLocaleString()} blank` : ""}</span></span>
                    {column.warning && <span className="table-column-warning">{column.warning}</span>}
                  </span>
                </label>
              </li>
            ))}
          </ul>
        </>
      )}
    </section>
  );
}

function TableTry({ model, beginner }: { model: SavedModel; beginner: boolean }) {
  const [schema, setSchema] = useState<TableSchema | null>(null);
  const [values, setValues] = useState<Record<string, string>>({});
  const [value, setValue] = useState<number | null>(null);
  const [answers, setAnswers] = useState<PredictAnswer[] | null>(null);
  const [asking, setAsking] = useState(false);
  const [problem, setProblem] = useState<string | null>(null);

  useEffect(() => {
    api.fetchTableSchema(model.id).then(setSchema).catch((error) => setProblem(message(error, "The model's columns could not load.")));
  }, [model.id]);

  const ask = async (event: FormEvent) => {
    event.preventDefault();
    setAsking(true);
    setProblem(null);
    setValue(null);
    setAnswers(null);
    try {
      const reply = await api.tryTableModel(model.id, values);
      if (reply.value !== undefined) setValue(reply.value);
      if (reply.answers) setAnswers(reply.answers);
    } catch (error) {
      setProblem(message(error, "The model couldn't answer."));
    } finally {
      setAsking(false);
    }
  };

  if (!schema) return problem ? <p className="small" role="alert">{problem}</p> : null;
  return (
    <>
      <h3 className="field-note small">Try it</h3>
      {beginner && <p className="muted small">Fill in what you know. Blank values are fine.</p>}
      <form onSubmit={(event) => void ask(event)}>
        <div className="table-try">
          {schema.features.map((feature) => (
            <label key={feature.name}>
              {feature.name}
              {feature.choices
                ? (
                  <select value={values[feature.name] ?? ""} onChange={(event) => setValues({ ...values, [feature.name]: event.target.value })}>
                    <option value="">Blank</option>
                    {feature.choices.map((choice) => <option key={choice} value={choice}>{choice}</option>)}
                  </select>
                )
                : <input type="text" inputMode="decimal" value={values[feature.name] ?? ""} onChange={(event) => setValues({ ...values, [feature.name]: event.target.value })} />}
            </label>
          ))}
        </div>
        <div className="field-note"><button className="button button-small" type="submit" disabled={asking}>{asking ? "Asking…" : "Predict"}</button></div>
      </form>
      {problem && <p className="small field-note" role="alert">{problem}</p>}
      <div role="status">
        {value !== null && <p className="table-answer">{schema.target}: about {plain(value)}</p>}
        {answers && <p className="table-answer">{schema.target}: {answers[0].category} <span className="muted small">{Math.round(answers[0].chance * 100)}% likely</span></p>}
      </div>
    </>
  );
}

export function TableResult({ model, beginner }: { model: SavedModel; beginner: boolean }) {
  const metrics = model.metrics;
  if (!metrics) return null;
  const top = metrics.importance[0]?.share || 1;
  return (
    <section className="card">
      <h2>Newest model</h2>
      {metrics.task === "number"
        ? (
          <p className="small">
            On {metrics.total.toLocaleString()} rows held back from training, it's typically off by <strong>{plain(metrics.mae ?? 0)}</strong>. Always guessing the average would be off by {plain(metrics.baselineMae ?? 0)}.
            {!beginner && ` R² ${(metrics.r2 ?? 0).toFixed(3)}.`}
          </p>
        )
        : (
          <p className="small">
            It got <strong>{(metrics.correct ?? 0).toLocaleString()} of {metrics.total.toLocaleString()}</strong> held-back rows right ({Math.round((metrics.accuracy ?? 0) * 100)}%). Always guessing the most common answer would get {Math.round((metrics.baseline ?? 0) * 100)}%.
          </p>
        )}
      {metrics.total < 50 && <p className="muted small field-note">Only {metrics.total} rows were held back, so treat this as a rough guide.</p>}
      {metrics.repeatedRows > 0 && <p className="muted small field-note">{metrics.repeatedRows.toLocaleString()} exact repeated rows were removed first, so they couldn't inflate the score.</p>}
      {metrics.warnings.map((text) => <p className="notice field-note" key={text}>{text}</p>)}
      {metrics.importance.length > 0 && (
        <>
          <h3 className="field-note small">Which columns mattered most</h3>
          {beginner && <p className="muted small">How much worse the model got when each column was shuffled. This shows what the model relies on, not what causes the answer.</p>}
          <ul className="importance">
            {metrics.importance.slice(0, 12).map((row) => (
              <li key={row.name} title={`${row.name}: ${Math.round(row.share * 100)}%`}>
                <span className="importance-name">{row.name}</span>
                <span className="importance-track" aria-hidden="true"><span style={{ width: `${(row.share / top) * 100}%` }} /></span>
                <span className="importance-value">{Math.round(row.share * 100)}%</span>
              </li>
            ))}
          </ul>
        </>
      )}
      <TableTry model={model} beginner={beginner} />
    </section>
  );
}
