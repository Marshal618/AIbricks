import { useCallback, useEffect, useRef, useState } from "react";
import * as api from "./api";
import type { CoachPoint, CoachResult, DataCoach, DatasetSummary } from "./types";
import { Why } from "./Ui";

const WIDTH = 620;
const HEIGHT = 230;
const MARGIN = { top: 14, right: 14, bottom: 30, left: 44 };
const SHOWN_NEXT = 12;

function message(error: unknown, fallback: string) {
  return error instanceof Error ? error.message : fallback;
}

const percent = (value: number) => `${Math.round(value * 100)}%`;
const range = (point: CoachPoint) => point.low != null && point.high != null && Math.round(point.low * 100) !== Math.round(point.high * 100)
  ? `${percent(point.low)} to ${percent(point.high)}` : null;

/** The stand-in's accuracy as data grows: what was measured, the fitted curve, and where it leads. */
function CurveChart({ curve, things }: { curve: NonNullable<CoachResult["curve"]>; things: string }) {
  const svg = useRef<SVGSVGElement>(null);
  const [hover, setHover] = useState<CoachPoint | null>(null);
  // The curve starts where the measurements do, rather than guessing back towards no data at all.
  const fitted = curve.fitted.filter((point) => point.examples >= (curve.measured[0]?.examples ?? 0));
  const innerWidth = WIDTH - MARGIN.left - MARGIN.right;
  const innerHeight = HEIGHT - MARGIN.top - MARGIN.bottom;
  const most = fitted[fitted.length - 1]?.examples ?? curve.now.examples * 4;
  const values = [...curve.measured.map((point) => point.accuracy), ...fitted.map((point) => point.low ?? point.accuracy)];
  const bottom = Math.max(0, Math.floor(Math.min(...values) * 10) / 10);
  const x = (examples: number) => MARGIN.left + (examples / most) * innerWidth;
  const y = (accuracy: number) => MARGIN.top + (1 - (accuracy - bottom) / (1 - bottom)) * innerHeight;
  const line = (points: CoachPoint[]) => points.map((point, index) => `${index ? "L" : "M"}${x(point.examples).toFixed(1)} ${y(point.accuracy).toFixed(1)}`).join(" ");
  const before = fitted.filter((point) => point.examples <= curve.now.examples);
  const after = fitted.filter((point) => point.examples >= curve.now.examples);
  const band = after.length > 1
    ? `${after.map((point, index) => `${index ? "L" : "M"}${x(point.examples).toFixed(1)} ${y(point.high ?? point.accuracy).toFixed(1)}`).join(" ")} ${[...after].reverse().map((point) => `L${x(point.examples).toFixed(1)} ${y(point.low ?? point.accuracy).toFixed(1)}`).join(" ")} Z`
    : "";
  const ticks = [bottom, (bottom + 1) / 2, 1];

  const onMove = (event: React.PointerEvent<SVGSVGElement>) => {
    const box = svg.current?.getBoundingClientRect();
    if (!box || !fitted.length) return;
    const examples = (((event.clientX - box.left) / box.width) * WIDTH - MARGIN.left) / innerWidth * most;
    setHover(fitted.reduce((best, point) => (Math.abs(point.examples - examples) < Math.abs(best.examples - examples) ? point : best), fitted[0]));
  };

  return (
    <figure className="loss-chart coach-chart">
      <figcaption>Expected share right, by how many {things} are labelled</figcaption>
      <div className="chart-wrap">
        <svg ref={svg} viewBox={`0 0 ${WIDTH} ${HEIGHT}`} role="img" onPointerMove={onMove} onPointerLeave={() => setHover(null)}
          aria-label={`About ${percent(curve.now.accuracy)} right now with ${curve.now.examples} labelled; about ${percent(curve.more[1].accuracy)} expected with ${curve.more[1].examples}.`}>
          {ticks.map((value) => (
            <g key={value}>
              <line className="grid" x1={MARGIN.left} x2={WIDTH - MARGIN.right} y1={y(value)} y2={y(value)} />
              <text className="tick" x={MARGIN.left - 6} y={y(value) + 4} textAnchor="end">{percent(value)}</text>
            </g>
          ))}
          <text className="tick" x={MARGIN.left} y={HEIGHT - 8}>0</text>
          <text className="tick" x={WIDTH - MARGIN.right} y={HEIGHT - 8} textAnchor="end">{most.toLocaleString()} labelled</text>
          {band && <path className="coach-band" d={band} />}
          <path className="loss-line" d={line(before)} />
          <path className="loss-line coach-ahead" d={line(after)} />
          <line className="coach-now" x1={x(curve.now.examples)} x2={x(curve.now.examples)} y1={MARGIN.top} y2={MARGIN.top + innerHeight} />
          <text className="tick coach-now-label" x={x(curve.now.examples) + 5} y={MARGIN.top + innerHeight - 6}>Now</text>
          {curve.measured.map((point) => <circle key={point.examples} className="coach-dot" cx={x(point.examples)} cy={y(point.accuracy)} r={4} />)}
          {hover && (
            <>
              <line className="crosshair" x1={x(hover.examples)} x2={x(hover.examples)} y1={MARGIN.top} y2={MARGIN.top + innerHeight} />
              <circle className="loss-dot" cx={x(hover.examples)} cy={y(hover.accuracy)} r={4} />
            </>
          )}
        </svg>
        {hover && (
          <div className="chart-tip" style={{ left: `${Math.min((x(hover.examples) / WIDTH) * 100, 70)}%` }}>
            {hover.examples.toLocaleString()} labelled: about <strong>{percent(hover.accuracy)}</strong>{range(hover) ? ` (${range(hover)})` : ""}
          </div>
        )}
      </div>
      <p className="muted small">Dots are what the coach measured on smaller shares of your data. The line is the curve fitted through them, dashed where it looks ahead, with the shaded likely range.</p>
      <details className="chart-table">
        <summary>Show values as a table</summary>
        <table>
          <thead><tr><th scope="col">Labelled</th><th scope="col">Share right</th><th scope="col">Likely range</th></tr></thead>
          <tbody>
            {curve.measured.map((point) => <tr key={`m${point.examples}`}><td>{point.examples} (measured)</td><td>{percent(point.accuracy)}</td><td /></tr>)}
            {[curve.now, ...curve.more].map((point) => <tr key={`p${point.examples}`}><td>{point.examples} (expected)</td><td>{percent(point.accuracy)}</td><td>{range(point) ?? ""}</td></tr>)}
          </tbody>
        </table>
      </details>
    </figure>
  );
}

const HEADLINES: Record<CoachResult["verdict"], string> = {
  helps: "More labelled examples should help a lot.",
  some: "More labelled examples should help a little.",
  levelling: "It's levelling off: more of the same won't help much.",
  unknown: "There isn't enough to tell yet.",
};

/** How much more data would help, and which examples to label next. */
export function DataCoachCard({ dataset, beginner, onOpenLabels }: { dataset: DatasetSummary; beginner: boolean; onOpenLabels: () => void }) {
  const [coach, setCoach] = useState<DataCoach | null>(null);
  const [problem, setProblem] = useState<string | null>(null);
  const things = dataset.kind === "image" ? "photos" : "texts";

  const load = useCallback(async () => setCoach(await api.fetchCoach(dataset.id)), [dataset.id]);
  useEffect(() => { load().catch((error) => setProblem(message(error, "The data coach couldn't load."))); }, [load, dataset.updatedAt]);

  const running = coach?.job?.status === "running";
  useEffect(() => {
    if (!running) return;
    const timer = window.setInterval(() => { load().catch(() => undefined); }, 1000);
    return () => window.clearInterval(timer);
  }, [running, load]);

  const start = async () => {
    setProblem(null);
    try {
      await api.startCoach(dataset.id);
      await load();
    } catch (error) {
      setProblem(message(error, "The data coach couldn't start."));
    }
  };

  if (!coach) return problem ? <section className="card"><p className="small tone-bad" role="alert">{problem}</p></section> : null;
  const result = coach.result;
  const curve = result?.curve ?? null;
  const doubled = curve?.more[1];
  const smallest = Math.min(...dataset.categories.map((name) => dataset.counts[name] ?? 0));

  return (
    <section className="card coach">
      <div className="card-head">
        <h2>How much more data?</h2>
        <button className="button button-small" onClick={() => void start()} disabled={running}>
          {running ? "Working…" : result ? "Ask again" : "Ask the coach"}
        </button>
      </div>
      {!result && !running && (
        <p className="muted small">
          The coach tries a quick model on smaller and smaller shares of your labelled {things} and watches how fast it improves.
          From that it estimates what more {things} would bring, and picks which unlabelled {things} to label next.
          {smallest < coach.minPerCategory && ` It needs at least ${coach.minPerCategory} labelled ${things} in each of 2 categories.`}
        </p>
      )}
      {running && coach.job && (
        <div className="field-note" role="status">
          <progress max={coach.job.stepsTotal} value={coach.job.stepsDone} />
          <p className="muted small">Step {coach.job.stepsDone} of {coach.job.stepsTotal}. The first time downloads a small pretrained model.</p>
        </div>
      )}
      {coach.job?.status === "failed" && <p className="small tone-bad" role="alert">{coach.job.error}</p>}
      {problem && <p className="small tone-bad" role="alert">{problem}</p>}

      {result && (
        <>
          {result.stale && <p className="small field-note notice-inline">The labels have changed a lot since this ran ({result.labelled} then, {coach.labelledNow} now). Ask again for a fresh estimate.</p>}
          <p className="coach-headline"><strong>{HEADLINES[result.verdict]}</strong></p>
          {curve && doubled && (
            <div className="coach-numbers">
              <div><span className="muted small">Now, with {curve.now.examples} labelled</span><strong>about {percent(curve.now.accuracy)}</strong></div>
              <div aria-hidden="true" className="coach-arrow">→</div>
              <div>
                <span className="muted small">With twice as many ({doubled.examples})</span>
                <strong>about {percent(doubled.accuracy)}</strong>
                {range(doubled) && <span className="muted small">likely {range(doubled)}</span>}
              </div>
            </div>
          )}
          {result.rough && <p className="muted small">With fewer than 60 labelled {things}, treat this as a rough guide. It gets steadier as you add more.</p>}
          {result.verdict === "levelling" && (
            <ul className="health-steps">
              <li>Check the labels: run the label check below. Wrong labels put a ceiling on any model.</li>
              <li>Add different {things}, not more of the same: {dataset.kind === "image" ? "new places, lighting, angles and backgrounds" : "new topics, wordings and lengths"}.</li>
              <li>Try a bigger or pretrained model in the Model builder, or Find a better model.</li>
            </ul>
          )}
          {result.focus.length > 0 && result.verdict !== "levelling" && (
            <p className="small field-note">Add more <strong>{result.focus.join(", ")}</strong> first: {result.focus.length === 1 ? "that category looks like it'd" : "those categories look like they'd"} gain the most.</p>
          )}
          {curve && <CurveChart curve={curve} things={things} />}

          <h3 className="field-note small">Each category</h3>
          <table className="coach-table">
            <thead><tr><th scope="col">Category</th><th scope="col">Labelled</th><th scope="col">Right now</th><th scope="col">If doubled</th></tr></thead>
            <tbody>
              {result.categories.map((row) => (
                <tr key={row.category} className={result.focus.includes(row.category) ? "is-focus" : ""}>
                  <th scope="row">{row.category}</th>
                  <td>{row.count}</td>
                  <td>{row.accuracy == null ? "–" : percent(row.accuracy)}</td>
                  <td>{row.gainIfDoubled == null ? "–" : row.gainIfDoubled < 0.005 ? "little change" : `+${Math.round(row.gainIfDoubled * 100)} ${Math.round(row.gainIfDoubled * 100) === 1 ? "point" : "points"}`}</td>
                </tr>
              ))}
            </tbody>
          </table>

          <h3 className="field-note small">Label these next</h3>
          {result.next.length === 0 && <p className="muted small">There are no unlabelled {things} waiting. Add some on the Label tab, then ask again.</p>}
          {result.next.length > 0 && (
            <>
              <p className="muted small">
                The quick model is least sure about these, and they differ from what's labelled, so each teaches the most. The Label tab now shows them first.
              </p>
              {dataset.kind === "image" ? (
                <ul className="coach-next">
                  {result.next.slice(0, SHOWN_NEXT).map((row) => (
                    <li key={row.id}><img src={api.unsortedImageUrl(row.id, dataset.id)} alt={`Unlabelled photo, maybe ${row.guess}`} loading="lazy" /></li>
                  ))}
                </ul>
              ) : (
                <ul className="coach-next-text">
                  {result.next.slice(0, SHOWN_NEXT).map((row) => <li key={row.id}>{row.text}</li>)}
                </ul>
              )}
              <button className="button button-small field-note" onClick={onOpenLabels}>Start labelling these</button>
            </>
          )}
          <Why label={beginner ? "How does it know?" : "Method"}>
            {beginner ? (
              <>
                A pretrained model describes each of your {things}, and a quick, simple model learns your categories from those
                descriptions. The coach trains it on 10%, 25%, 50% and more of your labels, always testing on {things} it didn't learn from.
                Models usually improve along a smooth curve as data grows, so it fits that curve and follows it forward. The model you
                design may score higher or lower than this quick one, but it tends to improve in a similar way. When we tested this on
                a public dataset, its guesses were usually within a few points, and its picks reached the same score with about 40% fewer
                labels than picking at random.
              </>
            ) : (
              <>
                Logistic regression on frozen pretrained embeddings (ResNet-18 or MiniLM), 5-fold, at 7 training shares with repeats.
                Error is fitted as a + b·n<sup>−c</sup> (a ≥ 0) and extrapolated; the range comes from per-fold fits. Next-to-label uses
                margin uncertainty weighted by distance to the labelled and picked set (core-set style greedy). Estimates describe the
                stand-in model, not your architecture.
              </>
            )}
          </Why>
        </>
      )}
    </section>
  );
}
