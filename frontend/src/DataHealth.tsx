import { useCallback, useEffect, useState } from "react";
import * as api from "./api";
import type { DataHealth, DatasetSummary, HealthIssue, HealthPair } from "./types";
import { DataCoachCard } from "./DataCoach";

function message(error: unknown, fallback: string) {
  return error instanceof Error ? error.message : fallback;
}

const STATE_TEXT: Record<HealthIssue["state"], string> = { open: "", fixed: "Fixed", removed: "Removed", kept: "Kept as it was" };

function Thumb({ dataset, item, label }: { dataset: DatasetSummary; item: string; label: string }) {
  return <img src={api.datasetPreviewImage(dataset.id, label, item)} alt={`Example filed under ${label}`} loading="lazy" />;
}

export function DataHealthTab({ dataset, beginner, onChanged, onOpenLabels }: { dataset: DatasetSummary; beginner: boolean; onChanged: () => void; onOpenLabels: () => void }) {
  const [health, setHealth] = useState<DataHealth | null>(null);
  const [problem, setProblem] = useState<string | null>(null);
  const [showDone, setShowDone] = useState(false);
  const [busy, setBusy] = useState(false);

  const load = useCallback(async () => setHealth(await api.fetchHealth(dataset.id)), [dataset.id]);
  useEffect(() => { load().catch((error) => setProblem(message(error, "Data health could not load."))); }, [load]);

  const running = health?.job?.status === "running";
  useEffect(() => {
    if (!running) return;
    const timer = window.setInterval(() => { load().catch(() => undefined); }, 1000);
    return () => window.clearInterval(timer);
  }, [running, load]);

  const start = async () => {
    setProblem(null);
    try {
      await api.startHealthCheck(dataset.id);
      await load();
    } catch (error) {
      setProblem(message(error, "The check could not start."));
    }
  };

  const act = async (body: Parameters<typeof api.reviewHealth>[1]) => {
    setProblem(null);
    setBusy(true);
    try {
      setHealth(await api.reviewHealth(dataset.id, body));
      onChanged();
    } catch (error) {
      setProblem(message(error, "That change couldn't be saved."));
    } finally {
      setBusy(false);
    }
  };

  if (!health) return problem ? <div className="result result-fail" role="alert"><p>{problem}</p></div> : <p className="muted small">Loading…</p>;

  if (health.kind === "table") {
    const table = health.table;
    return (
      <section className="card">
        <h2>Table health</h2>
        {!table && <p className="muted small">Upload a CSV file on the Data tab first.</p>}
        {table && (
          <ul className="project-list">
            <li><span className="project-text">{table.rows.toLocaleString()} rows and {table.columns} columns</span></li>
            <li><span className="project-text">{table.repeatedRows ? `${table.repeatedRows.toLocaleString()} rows are exact repeats of another row. They're removed before training.` : "No repeated rows."}</span></li>
            <li><span className="project-text">{Math.round(table.blankShare * 1000) / 10}% of cells are blank. The model handles blanks, but fewer is better.</span></li>
            <li><span className="project-text">{table.constantColumns.length ? `Same value in every row, so no use for predicting: ${table.constantColumns.join(", ")}.` : "No columns with the same value in every row."}</span></li>
          </ul>
        )}
        <p className="muted small field-note">The Models tab also checks each column when you choose what to predict.</p>
      </section>
    );
  }

  const level = health.level ?? 0;
  const check = health.check;
  const issues = check?.issues.filter((issue) => showDone || issue.state === "open") ?? [];
  const pairs = check?.duplicates.filter((pair) => pair.state !== "same" && (showDone || pair.state === "open")) ?? [];
  const reviewed = (check?.issues.filter((issue) => issue.state !== "open").length ?? 0) + (check?.duplicates.filter((pair) => pair.state === "kept").length ?? 0);

  return (
    <>
      {problem && <div className="result result-fail" role="alert"><p>{problem}</p></div>}
      <section className="card">
        <div className="health-level">
          <strong>{level ? `Level ${level}: ${health.levelName}` : health.levelName}</strong>
          <span className="muted small">of 4</span>
        </div>
        <div className="health-meter" role="img" aria-label={`Level ${level} of 4`}>
          {[1, 2, 3, 4].map((step) => <span key={step} className={step <= level ? "is-on" : ""} />)}
        </div>
        <div className="health-meter-labels" aria-hidden="true">
          {health.levels?.slice(1).map((name, index) => <span key={name} className={index + 1 <= level ? "is-on" : ""}>{name}</span>)}
        </div>
        {check?.passedShare != null && (
          <p className="health-share">
            <strong>{(Math.floor(check.passedShare * 1000) / 10).toLocaleString()}%</strong> of labels passed the check
          </p>
        )}
        {check?.passedShare != null && (
          <p className="muted small">
            {beginner
              ? "This shows how many labels the check found no reason to doubt. It can miss some mistakes, and about half of what it flags turns out to be fine, so it's a guide, not a guarantee."
              : "Share not currently flagged. Flags are confident-learning candidates: in tests about half to most were real errors, and some errors go unflagged."}
          </p>
        )}
        {(health.nextSteps?.length ?? 0) > 0 && (
          <>
            <h3 className="field-note small">{level === 0 ? "To get started" : `To reach level ${level + 1}`}</h3>
            <ul className="health-steps">{health.nextSteps?.map((step) => <li key={step}>{step}</li>)}</ul>
          </>
        )}
        {level === 4 && <p className="small field-note">Every flagged label has been reviewed. Run the check again after adding more data.</p>}
      </section>

      {(dataset.kind === "image" || dataset.kind === "text") && (!dataset.goal || dataset.goal === "sort") && (
        <DataCoachCard dataset={dataset} beginner={beginner} onOpenLabels={onOpenLabels} />
      )}

      {(health.conflicts?.length ?? 0) > 0 && (
        <section className="card">
          <h2>Filed under more than one category</h2>
          <p className="muted small">{dataset.kind === "image" ? "The same photo is in two categories, which teaches the model two answers for one picture." : dataset.kind === "sound" ? "The same clip is in two categories, which teaches the model two answers for one sound." : "The same text, apart from capitals and spaces, has different labels."}</p>
          <ul className="health-list">
            {health.conflicts?.map((conflict) => (
              <li key={conflict.ids.join()}>
                {dataset.kind === "image" && <Thumb dataset={dataset} item={conflict.ids[0]} label={conflict.labels[0]} />}
                <span className="health-text">Filed under {conflict.labels.join(" and ")}</span>
                <span className="label-buttons">
                  {dataset.kind === "image"
                    ? conflict.labels.map((label) => (
                      <button key={label} className="button button-small" disabled={busy} onClick={() => void act({ action: "drop", id: conflict.ids[0], category: conflict.labels.find((other) => other !== label) })}>Keep in {label} only</button>
                    ))
                    : <span className="muted small">Fix these on the Label tab.</span>}
                </span>
              </li>
            ))}
          </ul>
        </section>
      )}

      {dataset.kind === "sound" && (
        <section className="card">
          <h2>Label check</h2>
          <p className="muted small">The label check isn't available for sound yet. Listen to your clips on the Label tab, and to the mistakes a trained model makes on the Models tab, to find wrong ones.</p>
        </section>
      )}
      {dataset.kind !== "sound" && <section className="card">
        <div className="card-head">
          <h2>Label check</h2>
          <button className="button button-small button-primary" disabled={running || level === 0} onClick={() => void start()}>{running ? "Checking…" : check ? "Check again" : "Check labels"}</button>
        </div>
        <p className="muted small">
          {beginner
            ? "Each example is shown to a model that learned from all the others, but never saw that one. If it's confidently sure the example belongs somewhere else, the label is flagged for you to look at. Nothing changes unless you choose."
            : `Out-of-fold predictions from a pretrained ${dataset.kind === "image" ? "ResNet-18" : "MiniLM"} embedding with a logistic regression head, flagged by the confident learning rule. Runs locally.`}
        </p>
        {running && health.job && (
          <div role="status" className="field-note">
            <progress className="train-progress" max={health.job.stepsTotal || 3} value={health.job.stepsDone} aria-label="Check progress" />
            <p className="muted small">{health.job.stepsDone === 0 ? `Getting ready to check ${health.job.count.toLocaleString()} examples. The first check downloads a model of about 45–90 MB.` : `Checking ${health.job.count.toLocaleString()} examples…`}</p>
          </div>
        )}
        {health.job?.status === "failed" && <p className="notice field-note">{health.job.error}</p>}
        {check && !running && (
          <p className="small field-note">
            Last checked {new Date(check.ranAt * 1000).toLocaleString(undefined, { dateStyle: "medium", timeStyle: "short" })}.
            {" "}{check.issues.length ? `${check.issues.length} label${check.issues.length === 1 ? "" : "s"} flagged, ${check.openIssues} still to review.` : "No labels flagged."}
            {check.unchecked > 0 && ` ${check.unchecked} example${check.unchecked === 1 ? " has" : "s have"} been added or changed since.`}
          </p>
        )}
        {reviewed > 0 && (
          <label className="check-row field-note small">
            <input type="checkbox" checked={showDone} onChange={(event) => setShowDone(event.target.checked)} /> Show ones already reviewed ({reviewed})
          </label>
        )}
        {issues.length > 0 && (
          <ul className="health-list">
            {issues.map((issue) => (
              <li key={issue.id}>
                {dataset.kind === "image" && issue.labels[0] && <Thumb dataset={dataset} item={issue.id} label={issue.labels[0]} />}
                <div className="health-text">
                  {issue.text && <div>{issue.text}</div>}
                  <div className="muted small">Labelled <strong>{issue.given}</strong>. The check thinks it's <strong>{issue.suggested}</strong> ({Math.round(issue.confidence * 100)}% sure).</div>
                  {issue.state !== "open" && <div className="health-state">{STATE_TEXT[issue.state]}{issue.state === "fixed" ? `: now ${issue.labels.join(", ")}` : ""}</div>}
                </div>
                {issue.state === "open" && (
                  <span className="label-buttons">
                    <button className="button button-small button-primary" disabled={busy} onClick={() => void act({ action: "move", id: issue.id, category: issue.suggested })}>Move to {issue.suggested}</button>
                    <button className="button button-small" disabled={busy} onClick={() => void act({ action: "keep", id: issue.id, category: issue.given })}>Keep {issue.given}</button>
                    <button className="button button-small button-quiet" disabled={busy} onClick={() => void act({ action: "remove", id: issue.id })}>Remove</button>
                  </span>
                )}
              </li>
            ))}
          </ul>
        )}
      </section>}

      {check && pairs.length > 0 && (
        <section className="card">
          <h2>Near-duplicates with different labels</h2>
          <p className="muted small">These look almost the same but have different labels. Usually one label is wrong, or they really are different.</p>
          <ul className="health-list">
            {pairs.map((pair) => <PairRow key={`${pair.first}|${pair.second}`} pair={pair} dataset={dataset} busy={busy} onAct={act} />)}
          </ul>
        </section>
      )}
    </>
  );
}

function PairRow({ pair, dataset, busy, onAct }: { pair: HealthPair; dataset: DatasetSummary; busy: boolean; onAct: (body: Parameters<typeof api.reviewHealth>[1]) => void }) {
  const side = (item: string, labels: string[], text?: string | null) => (
    <figure>
      {dataset.kind === "image" && labels[0] && <Thumb dataset={dataset} item={item} label={labels[0]} />}
      {text && <span>{text}</span>}
      <figcaption className="muted">{labels.join(", ")}</figcaption>
    </figure>
  );
  return (
    <li>
      <div className="health-pair">
        {side(pair.first, pair.firstLabels, pair.firstText)}
        {side(pair.second, pair.secondLabels, pair.secondText)}
      </div>
      <span className="muted small">{Math.round(pair.similarity * 100)}% alike</span>
      {pair.state === "open"
        ? (
          <span className="label-buttons">
            <button className="button button-small" disabled={busy} onClick={() => onAct({ action: "keep-pair", pair: [pair.first, pair.second] })}>They're different, keep both</button>
            <button className="button button-small button-quiet" disabled={busy} onClick={() => onAct({ action: "remove", id: pair.second })}>Remove the second</button>
          </span>
        )
        : <span className="health-state">Kept</span>}
    </li>
  );
}
