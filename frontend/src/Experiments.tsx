import { useCallback, useEffect, useState } from "react";
import * as api from "./api";
import type { DatasetSummary, Experiment, ExperimentEffect, ExperimentOption, GraphPayload } from "./types";
import { Why } from "./Ui";

const POLL_MS = 1500;
const ROUND_ONE = ["start", "augment", "wider", "pretrained", "slower"];

function message(error: unknown, fallback: string) {
  return error instanceof Error ? error.message : fallback;
}

function percent(value: number | null) {
  return value === null ? "–" : `${Math.round(value * 100)}%`;
}

const MEANING: Record<ExperimentEffect["meaning"], { mark: string; className: string; text: string }> = {
  helped: { mark: "✓", className: "is-good", text: "Helped" },
  hurt: { mark: "✕", className: "is-bad", text: "Made it worse" },
  "no clear difference": { mark: "=", className: "", text: "No clear difference" },
  "too few photos to tell": { mark: "?", className: "", text: "Too few photos to tell" },
};

/** Tries several models one change at a time, shows what each change did, and keeps the best. */
export function FindBetterModel({ dataset, beginner, busyElsewhere, onOpenDesign, onFinished }: {
  dataset: DatasetSummary; beginner: boolean; busyElsewhere: boolean;
  onOpenDesign: (graph: GraphPayload) => void; onFinished: () => void;
}) {
  const [experiment, setExperiment] = useState<Experiment | null>(null);
  const [budgets, setBudgets] = useState<number[]>([5, 10, 20, 30]);
  const [labels, setLabels] = useState<Record<string, string>>({});
  const [minutes, setMinutes] = useState(10);
  const [problem, setProblem] = useState<string | null>(null);
  const [loaded, setLoaded] = useState(false);

  const load = useCallback(async () => {
    const reply = await api.fetchExperiment(dataset.id);
    setExperiment(reply.experiment);
    setBudgets(reply.budgets);
    setLabels(Object.fromEntries(Object.entries(reply.changes).map(([key, value]) => [key, value.label])));
    setLoaded(true);
    return reply.experiment;
  }, [dataset.id]);

  useEffect(() => { load().catch((error) => setProblem(message(error, "The experiment couldn't load."))); }, [load]);

  const running = experiment?.status === "running";
  useEffect(() => {
    if (!running) return;
    const timer = window.setInterval(() => {
      load().then((next) => { if (next && next.status !== "running") onFinished(); }).catch(() => undefined);
    }, POLL_MS);
    return () => window.clearInterval(timer);
  }, [running, load, onFinished]);

  const start = async () => {
    setProblem(null);
    try {
      setExperiment(await api.startExperiment(dataset.id, minutes));
    } catch (error) {
      setProblem(message(error, "The experiment couldn't start."));
    }
  };

  const stop = async () => {
    setProblem(null);
    try {
      await api.stopExperiment(dataset.id);
    } catch (error) {
      setProblem(message(error, "The experiment couldn't be stopped."));
    }
  };

  if (!loaded && !problem) return null;
  const winner = experiment?.options.find((option) => option.round === 2 && option.key === experiment.winner) ?? null;

  return (
    <section className="card find-better" aria-label="Find a better model">
      <div className="card-head">
        <h2>Find a better model</h2>
        {running && <button className="button button-small" onClick={() => void stop()}>Stop</button>}
      </div>
      <p className="muted small">
        {beginner
          ? "Tries five versions of a model, changing one thing each time, then trains the best ones for longer and keeps the winner."
          : "Successive halving over one-change variants, compared with McNemar's test on the shared held-back photos."}
      </p>
      <Why>
        <p>Each version changes one thing from a simple starting point: varying the training photos, a wider model, starting from a pretrained model, or learning more slowly. Changing one thing at a time shows which changes actually help with your photos.</p>
        <p>Every version is checked on the same held-back photos, and the comparison says whether a difference is real or could be luck. The best two, and a mix of the changes that helped, then train three times as long. Only the winner is kept, as a normal model you can try, download or open as blocks.</p>
      </Why>

      {!running && !experiment && startControls()}
      {problem && <p className="notice field-note" role="alert">{problem}</p>}

      {experiment && (running || experiment.options.length > 0) && (
        <Progress experiment={experiment} labels={labels} />
      )}
      {running && <p className="muted small field-note">Keep this dataset as it is until it finishes, so every version learns from the same photos.</p>}

      {experiment && !running && (
        <div className="find-result field-note" role="status">
          {experiment.status === "failed" && <p className="notice">It stopped with a problem: {experiment.error}</p>}
          {experiment.status === "stopped" && <p className="muted small">Stopped before it finished{winner ? ", but it kept the best model so far." : ". Nothing was kept."}</p>}
          {winner && <Winner winner={winner} experiment={experiment} onOpenDesign={onOpenDesign} />}
          {experiment.effects.length > 0 && <Effects effects={experiment.effects} />}
          {experiment.skipped.length > 0 && (
            <p className="muted small">Skipped to stay within {experiment.minutes} minutes: {experiment.skipped.join(", ")}. Give it more time to try these too.</p>
          )}
        </div>
      )}
      {!running && experiment && startControls()}
    </section>
  );

  function startControls() {
    return (
        <div className="find-start field-note">
          <fieldset className="chip-group">
            <legend className="small">Time to spend</legend>
            {budgets.map((value) => (
              <label key={value} className={`chip${minutes === value ? " is-on" : ""}`}>
                <input type="radio" name="experiment-minutes" checked={minutes === value} onChange={() => setMinutes(value)} />
                {value} min
              </label>
            ))}
          </fieldset>
          <button className="button button-primary" disabled={busyElsewhere} onClick={() => void start()}>
            {experiment ? "Run again" : "Start"}
          </button>
          {busyElsewhere && <span className="muted small">Another model is training.</span>}
        </div>
    );
  }
}

function Progress({ experiment, labels }: { experiment: Experiment; labels: Record<string, string> }) {
  const done = (key: string, round: number) => experiment.options.find((option) => option.key === key && option.round === round);
  const roundTwo = experiment.options.filter((option) => option.round === 2);
  const name = (key: string) => key === "start" ? "Starting point" : labels[key] ?? key;
  const running = experiment.status === "running";
  return (
    <ol className="find-steps field-note">
      <li className="find-round">
        <span className="small"><strong>Round 1</strong> <span className="muted">· every version, the same short training</span></span>
        <ul>
          {ROUND_ONE.map((key) => {
            const option = done(key, 1);
            const skipped = experiment.skipped.includes(labels[key] ?? "");
            const now = running && experiment.round === 1 && !option && experiment.currentLabel === name(key);
            return <Row key={key} label={name(key)} option={option} now={now} skipped={skipped} progress={now ? experiment.progress : null} />;
          })}
        </ul>
      </li>
      {(experiment.round === 2 || roundTwo.length > 0) && (
        <li className="find-round">
          <span className="small"><strong>Round 2</strong> <span className="muted">· the most promising, trained for longer</span></span>
          <ul>
            {roundTwo.map((option) => <Row key={option.jobId} label={option.label} option={option} now={false} skipped={false} progress={null} winner={option.key === experiment.winner} />)}
            {running && experiment.round === 2 && experiment.current && (
              <Row label={experiment.currentLabel ?? "Training"} now skipped={false} progress={experiment.progress ?? null} />
            )}
          </ul>
        </li>
      )}
    </ol>
  );
}

function Row({ label, option, now, skipped, progress, winner = false }: {
  label: string; option?: ExperimentOption; now: boolean; skipped: boolean; progress: Experiment["progress"] | null; winner?: boolean;
}) {
  const state = option ? "done" : now ? "now" : skipped ? "skipped" : "waiting";
  return (
    <li className={`find-row is-${state}${winner ? " is-winner" : ""}`}>
      <span className="find-mark" aria-hidden="true">{option ? (winner ? "★" : "✓") : now ? "●" : skipped ? "–" : "○"}</span>
      <span className="find-label">{label}</span>
      <span className="find-value small">
        {option
          ? <>{percent(option.accuracy)} <span className="muted">· {Math.round(option.seconds)} s</span></>
          : now
            ? <span className="muted">Training{progress ? `, epoch ${progress.epoch} of ${progress.epochs}` : "…"}</span>
            : <span className="muted">{skipped ? "Skipped" : "Waiting"}</span>}
      </span>
    </li>
  );
}

function Winner({ winner, experiment, onOpenDesign }: { winner: ExperimentOption; experiment: Experiment; onOpenDesign: (graph: GraphPayload) => void }) {
  const verdict = experiment.verdict;
  return (
    <div className="find-winner">
      <p className="find-winner-score"><span className="find-big">{percent(winner.accuracy)}</span> on held-back photos</p>
      <p><strong>Best: {winner.label}</strong>, trained for {winner.epochs} epochs.</p>
      {verdict && (
        <p className="muted small">
          {verdict.result === "first"
            ? `It's clearly better than the runner-up, ${verdict.against}.`
            : verdict.result === "too-few"
              ? `There are too few held-back photos to say it beats ${verdict.against} for sure.`
              : `It isn't clearly better than ${verdict.against}; either would do.`}
        </p>
      )}
      <div className="label-buttons" style={{ justifyContent: "flex-start", flexWrap: "wrap" }}>
        <button className="button button-small button-primary" onClick={() => onOpenDesign(winner.graph)}>Open its blocks</button>
        <span className="muted small">It's now your newest model, so you can try it and make an app below.</span>
      </div>
    </div>
  );
}

function Effects({ effects }: { effects: ExperimentEffect[] }) {
  return (
    <div className="find-effects">
      <h3 className="small">What each change did</h3>
      <ul>
        {effects.map((effect) => {
          const meaning = MEANING[effect.meaning];
          return (
            <li key={effect.key}>
              <span className="find-effect-name">
                <strong>{effect.label}</strong>
                <span className="muted small">{effect.description}</span>
              </span>
              <span className="find-points">{effect.points > 0 ? "+" : ""}{effect.points} points</span>
              <span className={`find-meaning ${meaning.className}`}><span aria-hidden="true">{meaning.mark}</span> {meaning.text}</span>
            </li>
          );
        })}
      </ul>
      <p className="muted small">Each was compared with the starting point after the same short training. Points are the change in held-back accuracy.</p>
      {effects.some((effect) => effect.meaning === "too few photos to tell") && (
        <p className="notice">
          <span>Only {effects[0].shared ?? "a few"} photos were held back, too few to tell real differences from luck. Add more photos, aiming for 50 or more in each category, for clearer answers.</span>
        </p>
      )}
    </div>
  );
}
