import { useCallback, useEffect, useRef, useState, type FormEvent } from "react";
import * as api from "./api";
import type { DatasetSummary, ForecastAnalysis, ForecastDetails, SavedModel } from "./types";
import { Why } from "./Ui";

const WIDTH = 620;
const HEIGHT = 250;
const MARGIN = { top: 14, right: 14, bottom: 30, left: 56 };

function message(error: unknown, fallback: string) {
  return error instanceof Error ? error.message : fallback;
}

export function plain(value: number) {
  const size = Math.abs(value);
  return value.toLocaleString(undefined, { maximumFractionDigits: size >= 100 ? 0 : size >= 1 ? 2 : 4 });
}

function plural(count: number, word: string) {
  return `${count.toLocaleString()} ${word}${count === 1 ? "" : "s"}`;
}

interface ChartData {
  times: string[];
  real: (number | null)[];
  guess: (number | null)[];
  low: (number | null)[];
  high: (number | null)[];
}

/** Real readings as a solid line, and guesses as a dashed line with the likely range shaded behind them. */
export function ForecastChart({ data, caption, label, guessLabel = "Forecast" }: { data: ChartData; caption: string; label: string; guessLabel?: string }) {
  const svg = useRef<SVGSVGElement>(null);
  const [pointer, setHover] = useState<number | null>(null);
  const count = data.times.length;
  // A pointer left over from a longer chart is ignored.
  const hover = pointer !== null && pointer < count ? pointer : null;
  const shown = [...data.real, ...data.guess, ...data.low, ...data.high].filter((value): value is number => value !== null && Number.isFinite(value));
  if (!count || !shown.length) return null;
  let bottom = Math.min(...shown);
  let top = Math.max(...shown);
  if (top === bottom) { top += 1; bottom -= 1; }
  const pad = (top - bottom) * 0.06;
  top += pad;
  bottom -= pad;
  const innerWidth = WIDTH - MARGIN.left - MARGIN.right;
  const innerHeight = HEIGHT - MARGIN.top - MARGIN.bottom;
  const x = (index: number) => MARGIN.left + (index / Math.max(count - 1, 1)) * innerWidth;
  const y = (value: number) => MARGIN.top + (1 - (value - bottom) / (top - bottom)) * innerHeight;
  const line = (values: (number | null)[]) => {
    let path = "";
    let open = false;
    values.forEach((value, index) => {
      if (value === null) { open = false; return; }
      path += `${open ? "L" : "M"}${x(index).toFixed(1)} ${y(value).toFixed(1)}`;
      open = true;
    });
    return path;
  };
  const banded = data.low.map((value, index) => (value !== null && data.high[index] !== null ? index : -1)).filter((index) => index >= 0);
  const band = banded.length
    ? `M${banded.map((index) => `${x(index).toFixed(1)} ${y(data.high[index]!).toFixed(1)}`).join("L")}L${[...banded].reverse().map((index) => `${x(index).toFixed(1)} ${y(data.low[index]!).toFixed(1)}`).join("L")}Z`
    : "";
  const firstGuess = data.guess.findIndex((value) => value !== null);
  const lastReal = data.real.reduce<number>((last, value, index) => (value !== null ? index : last), -1);

  const onMove = (event: React.PointerEvent<SVGSVGElement>) => {
    const box = svg.current?.getBoundingClientRect();
    if (!box) return;
    const px = ((event.clientX - box.left) / box.width) * WIDTH;
    setHover(Math.max(0, Math.min(count - 1, Math.round(((px - MARGIN.left) / innerWidth) * (count - 1)))));
  };
  const ticks = [bottom + pad, (top + bottom) / 2, top - pad];
  const tableRows = data.times.map((time, index) => ({ time, index })).filter(({ index }) => data.guess[index] !== null);

  return (
    <figure className="loss-chart forecast-chart">
      <figcaption>{caption}</figcaption>
      <div className="forecast-legend" aria-hidden="true">
        <span><span className="legend-line is-real" />{label}</span>
        {firstGuess >= 0 && <span><span className="legend-line is-guess" />{guessLabel}</span>}
        {banded.length > 0 && <span><span className="legend-band" />Likely range</span>}
      </div>
      <div className="chart-wrap">
        <svg
          ref={svg}
          viewBox={`0 0 ${WIDTH} ${HEIGHT}`}
          role="img"
          aria-label={`${caption}. ${count} points from ${data.times[0]} to ${data.times[count - 1]}.`}
          onPointerMove={onMove}
          onPointerLeave={() => setHover(null)}
        >
          {ticks.map((value) => (
            <g key={value}>
              <line className="grid" x1={MARGIN.left} x2={WIDTH - MARGIN.right} y1={y(value)} y2={y(value)} />
              <text className="tick" x={MARGIN.left - 6} y={y(value) + 4} textAnchor="end">{plain(value)}</text>
            </g>
          ))}
          <text className="tick" x={MARGIN.left} y={HEIGHT - 9}>{data.times[0]}</text>
          <text className="tick" x={WIDTH - MARGIN.right} y={HEIGHT - 9} textAnchor="end">{data.times[count - 1]}</text>
          {firstGuess > 0 && firstGuess < count - 1 && <line className="forecast-split" x1={x(firstGuess)} x2={x(firstGuess)} y1={MARGIN.top} y2={MARGIN.top + innerHeight} />}
          {band && <path className="forecast-band" d={band} />}
          <path className="forecast-real" d={line(data.real)} />
          <path className="forecast-guess" d={line(data.guess)} />
          {lastReal >= 0 && <circle className="forecast-dot is-real" cx={x(lastReal)} cy={y(data.real[lastReal]!)} r={3.5} />}
          {hover !== null && (
            <>
              <line className="crosshair" x1={x(hover)} x2={x(hover)} y1={MARGIN.top} y2={MARGIN.top + innerHeight} />
              {data.real[hover] !== null && <circle className="forecast-dot is-real" cx={x(hover)} cy={y(data.real[hover]!)} r={4.5} />}
              {data.guess[hover] !== null && <circle className="forecast-dot is-guess" cx={x(hover)} cy={y(data.guess[hover]!)} r={4.5} />}
            </>
          )}
        </svg>
        {hover !== null && (
          <div className="chart-tip" style={x(hover) / WIDTH > 0.5 ? { right: `${(1 - x(hover) / WIDTH) * 100}%` } : { left: `${(x(hover) / WIDTH) * 100}%` }}>
            {data.times[hover]}
            {data.real[hover] !== null && <> · {label}: <strong>{plain(data.real[hover]!)}</strong></>}
            {data.guess[hover] !== null && <> · {guessLabel}: <strong>{plain(data.guess[hover]!)}</strong></>}
            {data.low[hover] !== null && data.high[hover] !== null && data.guess[hover] !== null && <> ({plain(data.low[hover]!)} to {plain(data.high[hover]!)})</>}
          </div>
        )}
      </div>
      {tableRows.length > 0 && (
        <details className="chart-table">
          <summary>Show values as a table</summary>
          <table>
            <thead><tr><th scope="col">Time</th>{data.real.some((value, index) => value !== null && data.guess[index] !== null) && <th scope="col">{label}</th>}<th scope="col">{guessLabel}</th><th scope="col">Likely range</th></tr></thead>
            <tbody>
              {tableRows.map(({ time, index }) => (
                <tr key={index}>
                  <td>{time}</td>
                  {data.real.some((value, row) => value !== null && data.guess[row] !== null) && <td>{data.real[index] === null ? "" : plain(data.real[index]!)}</td>}
                  <td>{plain(data.guess[index]!)}</td>
                  <td>{data.low[index] !== null ? `${plain(data.low[index]!)} to ${plain(data.high[index]!)}` : ""}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </details>
      )}
    </figure>
  );
}

const nothing = (count: number) => Array<number | null>(count).fill(null);

function aheadChart(details: ForecastDetails): ChartData {
  const { history, forecast } = details;
  const past = history.values.length;
  const end = history.values[past - 1];
  return {
    times: [...history.times, ...forecast.times],
    real: [...history.values, ...nothing(forecast.times.length)],
    guess: [...nothing(past - 1), end, ...forecast.values],
    low: [...nothing(past), ...forecast.low],
    high: [...nothing(past), ...forecast.high],
  };
}

function backChart(details: ForecastDetails): ChartData {
  const { history, backtest } = details;
  const before = history.values.length - backtest.values.length;
  return {
    times: history.times,
    real: history.values,
    guess: [...nothing(before), ...backtest.values],
    low: [...nothing(before), ...backtest.low],
    high: [...nothing(before), ...backtest.high],
  };
}

export function ForecastTrainer({ dataset, beginner, running, busyElsewhere, onStarted, onStop }: { dataset: DatasetSummary; beginner: boolean; running: SavedModel | null; busyElsewhere: boolean; onStarted: () => Promise<void>; onStop: (id: string) => void }) {
  const [analysis, setAnalysis] = useState<ForecastAnalysis | null>(null);
  const [horizonText, setHorizonText] = useState("");
  const [problem, setProblem] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const request = useRef(0);

  const read = useCallback(async (choice: { when?: string; value?: string; horizon?: number; combine?: "mean" | "sum" } = {}) => {
    const mine = ++request.current;
    setLoading(true);
    setProblem(null);
    try {
      const reply = await api.fetchForecastAnalysis(dataset.id, choice);
      if (mine !== request.current) return;
      setAnalysis(reply);
      setHorizonText(String(reply.horizon || ""));
    } catch (error) {
      if (mine === request.current) setProblem(message(error, "The table could not be checked."));
    } finally {
      if (mine === request.current) setLoading(false);
    }
  }, [dataset.id]);

  useEffect(() => { if (dataset.labelled) void read(); }, [dataset.labelled, read]);

  if (!dataset.labelled) {
    return <section className="card"><h2>Forecast a column</h2><p className="muted small">Upload a CSV file on the Data tab first.</p></section>;
  }

  const current = (change: { when?: string; value?: string; horizon?: number; combine?: "mean" | "sum" }) => analysis
    ? { when: analysis.when, value: analysis.value ?? undefined, horizon: analysis.horizon || undefined, combine: analysis.combine, ...change }
    : change;

  const horizon = Number(horizonText);
  const horizonOk = Number.isInteger(horizon) && horizon >= 1 && !!analysis && horizon <= Math.max(analysis.maxHorizon, 1);

  const start = async () => {
    if (!analysis?.value) return;
    setProblem(null);
    try {
      await api.trainForecastModel(dataset.id, { when: analysis.when, value: analysis.value, horizon, combine: analysis.combine });
      await onStarted();
    } catch (error) {
      setProblem(message(error, "Training could not start."));
    }
  };

  const ready = !!analysis?.value && analysis.problems.length === 0 && horizonOk && horizon === analysis.horizon && !loading;
  const step = analysis?.stepName || "step";

  return (
    <section className="card">
      <div className="card-head">
        <h2>Forecast a column</h2>
        {running
          ? <button className="button button-small" onClick={() => onStop(running.id)}>Stop</button>
          : <button className="button button-small button-primary" disabled={!ready || busyElsewhere} onClick={() => void start()}>Start training</button>}
      </div>
      {beginner && <p className="muted small">It learns from the past readings in one column to guess the next ones, like next week's sales or tomorrow's temperature.</p>}
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
              What do you want to forecast?
              <select value={analysis.value ?? ""} disabled={!!running || !analysis.valueColumns.length} onChange={(event) => void read(current({ value: event.target.value, horizon: undefined }))}>
                {!analysis.valueColumns.length && <option value="">No number columns</option>}
                {analysis.valueColumns.map((name) => <option key={name} value={name}>{name}</option>)}
              </select>
            </label>
            <label>
              When each reading was taken
              <select value={analysis.when} disabled={!!running} onChange={(event) => void read(current({ when: event.target.value, horizon: undefined }))}>
                {analysis.timeColumns.map((name) => <option key={name} value={name}>{name}</option>)}
                <option value="">No dates: rows are in time order</option>
              </select>
            </label>
            <label>
              How far ahead, in {step}s
              <input
                type="number"
                min={1}
                max={Math.max(analysis.maxHorizon, 1)}
                value={horizonText}
                disabled={!!running}
                onChange={(event) => setHorizonText(event.target.value)}
                onBlur={() => { if (horizonOk && horizon !== analysis.horizon) void read(current({ horizon })); }}
                onKeyDown={(event) => { if (event.key === "Enter" && horizonOk) void read(current({ horizon })); }}
              />
            </label>
            {analysis.duplicates && (
              <label>
                Readings at the same time are
                <select value={analysis.combine} disabled={!!running} onChange={(event) => void read(current({ combine: event.target.value as "mean" | "sum" }))}>
                  <option value="mean">Averaged, like temperatures</option>
                  <option value="sum">Added up, like sales</option>
                </select>
              </label>
            )}
          </div>
          {!horizonOk && horizonText !== "" && <p className="small field-note" role="alert">Choose a whole number from 1 to {Math.max(analysis.maxHorizon, 1)}.</p>}
          {analysis.readings > 0 && (
            <p className="muted small field-note">
              {plural(analysis.readings, "reading")}, one every {step}, from {analysis.start} to {analysis.end}. {analysis.seasonText}.
            </p>
          )}
          {analysis.problems.map((text) => <p className="notice field-note" key={text}>{text}</p>)}
          {analysis.notes.map((text) => <p className="muted small field-note" key={text}>{text}</p>)}
          {analysis.recent.values.length > 1 && (
            <div className="field-note">
              <ForecastChart
                caption={analysis.recent.values.length < analysis.readings ? `The newest ${analysis.recent.values.length} readings` : "Your readings"}
                label={analysis.value ?? "Reading"}
                data={{ times: analysis.recent.times, real: analysis.recent.values, guess: nothing(analysis.recent.values.length), low: nothing(analysis.recent.values.length), high: nothing(analysis.recent.values.length) }}
              />
            </div>
          )}
          <Why>
            {beginner
              ? "The newest fifth of the readings is kept back, and the model never learns from it. Afterwards it forecasts across that stretch, and those guesses are compared with what really happened, and with two simple guesses: the same as last time, and the same as last season."
              : "Rolling-origin evaluation on the last 20% of the series (up to 200 origins). The 15% before it picks the epoch count, the method and the 10–90% band. The final model is refit on all readings."}
          </Why>
        </>
      )}
    </section>
  );
}

function ForecastTry({ model, details, beginner }: { model: SavedModel; details: ForecastDetails; beginner: boolean }) {
  const [text, setText] = useState("");
  const [answer, setAnswer] = useState<{ pasted: number[]; values: number[]; low: number[]; high: number[] } | null>(null);
  const [asking, setAsking] = useState(false);
  const [problem, setProblem] = useState<string | null>(null);
  const needed = details.method === "model" ? details.context : details.method === "season" ? details.season : 1;

  const ask = async (event: FormEvent) => {
    event.preventDefault();
    setProblem(null);
    setAnswer(null);
    const parts = text.split(/[\s,;]+/).filter(Boolean);
    const values = parts.map((part) => Number(part));
    if (!values.length || values.some((value) => !Number.isFinite(value))) {
      setProblem(`Paste numbers only, separated by commas, spaces or new lines.${parts.find((part) => !Number.isFinite(Number(part))) ? ` “${parts.find((part) => !Number.isFinite(Number(part)))?.slice(0, 20)}” isn't a number.` : ""}`);
      return;
    }
    setAsking(true);
    try {
      setAnswer({ pasted: values, ...(await api.forecastFrom(model.id, values)) });
    } catch (error) {
      setProblem(message(error, "The model couldn't forecast from these."));
    } finally {
      setAsking(false);
    }
  };

  return (
    <>
      <h3 className="field-note small">Try it on other readings</h3>
      <p className="muted small">
        {beginner ? "Paste readings in time order, oldest first, " : "Paste readings oldest first, "}
        at least {needed}, one every {details.stepName}. It forecasts the next {plural(details.horizon, details.stepName)}.
      </p>
      <form onSubmit={(event) => void ask(event)}>
        <textarea className="forecast-input" rows={3} value={text} aria-label="Readings to forecast from" placeholder="12.5, 13.1, 12.8, …" onChange={(event) => setText(event.target.value)} />
        <div className="field-note"><button className="button button-small" type="submit" disabled={asking || !text.trim()}>{asking ? "Forecasting…" : "Forecast"}</button></div>
      </form>
      {problem && <p className="small field-note" role="alert">{problem}</p>}
      {answer && (
        <div role="status" className="field-note">
          <ForecastChart
            caption="Your readings and its forecast"
            label="Your readings"
            data={(() => {
              const pasted = answer.pasted.slice(-Math.min(Math.max(5 * details.horizon, 35), 300));
              const past = pasted.length;
              return {
                times: [...pasted.map((_, index) => `${index - past + 1}`), ...answer.values.map((_, index) => `+${index + 1}`)],
                real: [...pasted, ...nothing(answer.values.length)],
                guess: [...nothing(past - 1), pasted[past - 1], ...answer.values],
                low: [...nothing(past), ...answer.low],
                high: [...nothing(past), ...answer.high],
              };
            })()}
          />
        </div>
      )}
    </>
  );
}

export function ForecastResult({ model, beginner }: { model: SavedModel; beginner: boolean }) {
  const [details, setDetails] = useState<ForecastDetails | null>(null);
  const [view, setView] = useState<"ahead" | "back">("ahead");
  const [problem, setProblem] = useState<string | null>(null);
  const metrics = model.metrics;

  useEffect(() => {
    api.fetchForecast(model.id).then(setDetails).catch((error) => setProblem(message(error, "The forecast could not load.")));
  }, [model.id]);

  if (!metrics || metrics.task !== "forecast") return null;
  const step = metrics.stepName ?? "step";
  const horizon = metrics.horizon ?? 1;
  const simple = metrics.method === "last" ? "the same as last time" : "the same as last season";
  const better = metrics.maeLast ? 1 - (metrics.mae ?? 0) / Math.min(metrics.maeLast, metrics.maeSeason ?? Infinity) : 0;
  const perStep = metrics.maePerStep ?? [];

  return (
    <section className="card">
      <h2>Newest model</h2>
      <p className="small">
        Forecasting up to {plural(horizon, step)} ahead, it's typically off by <strong>{plain(metrics.mae ?? 0)}</strong>
        {metrics.typical ? `, where a typical reading is ${plain(metrics.typical)}` : ""}.
        {" "}Guessing the same as last time would be off by {plain(metrics.maeLast ?? 0)}
        {metrics.maeSeason != null ? `, and the same as last season by ${plain(metrics.maeSeason)}` : ""}.
        {metrics.method === "model" && better > 0.05 && ` That's ${Math.round(better * 100)}% better than the best simple guess.`}
      </p>
      {metrics.method !== "model" && <p className="notice field-note">Forecasts use a simple guess, {simple}, because it did better than the model on the readings just before the check.</p>}
      {metrics.warnings.filter((text) => metrics.method === "model" || !text.startsWith("A simple guess")).map((text) => <p className="notice field-note" key={text}>{text}</p>)}
      {metrics.inside != null && (
        <p className="muted small field-note">
          The shaded likely range held the real reading {Math.round(metrics.inside * 100)}% of the time on the recent readings{beginner ? ". It aims for 8 in 10." : " (target 80%)."}
          {perStep.length > 1 && ` It's off by ${plain(perStep[0])} one ${step} ahead and ${plain(perStep[perStep.length - 1])} at ${plural(perStep.length, step)}.`}
        </p>
      )}
      {problem && <p className="small" role="alert">{problem}</p>}
      {details && (
        <>
          <div className="forecast-views field-note" role="group" aria-label="Which chart to show">
            <button className={`button button-small${view === "ahead" ? " is-active" : ""}`} aria-pressed={view === "ahead"} onClick={() => setView("ahead")}>The forecast</button>
            <button className={`button button-small${view === "back" ? " is-active" : ""}`} aria-pressed={view === "back"} onClick={() => setView("back")}>How it did recently</button>
          </div>
          {view === "ahead"
            ? <ForecastChart caption={`${details.value}: the next ${plural(details.horizon, details.stepName)}`} label={details.value} data={aheadChart(details)} />
            : <ForecastChart caption={`If it had forecast from ${details.backtest.from}`} label="What happened" guessLabel="It would have said" data={backChart(details)} />}
          {view === "back" && <p className="muted small">The model that made this guess never learned from these readings.</p>}
          <ForecastTry model={model} details={details} beginner={beginner} />
        </>
      )}
    </section>
  );
}
