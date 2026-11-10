import { useEffect, useMemo, useState } from "react";
import * as api from "./api";
import type { DatasetSummary, PoseAnswer, SavedModel, TrainingMistake } from "./types";
import { Why } from "./Ui";
import { CameraView, grab, useCamera } from "./Webcam";

export const POSE_MINIMUM = 2;
export const POSE_TOTAL = 10;
export const POSE_SUGGESTED = 20;
const MISTAKES_SHOWN = 24;
const SEEN = 0.5;

// Which points join up, as MediaPipe numbers them.
const HAND_LINES: [number, number][] = [
  [0, 1], [1, 2], [2, 3], [3, 4], [0, 5], [5, 6], [6, 7], [7, 8], [5, 9], [9, 10], [10, 11], [11, 12],
  [9, 13], [13, 14], [14, 15], [15, 16], [13, 17], [0, 17], [17, 18], [18, 19], [19, 20],
];
const BODY_LINES: [number, number][] = [
  [0, 1], [1, 2], [2, 3], [3, 7], [0, 4], [4, 5], [5, 6], [6, 8], [9, 10],
  [11, 12], [11, 13], [13, 15], [15, 17], [15, 19], [15, 21], [17, 19], [12, 14], [14, 16], [16, 18], [16, 20], [16, 22], [18, 20],
  [11, 23], [12, 24], [23, 24], [23, 25], [24, 26], [25, 27], [26, 28], [27, 29], [28, 30], [29, 31], [30, 32], [27, 31], [28, 32],
];

function message(error: unknown, fallback: string) {
  return error instanceof Error ? error.message : fallback;
}

const what = (dataset: { points?: "body" | "hands" }) => (dataset.points === "hands" ? "hands" : "body");

export function poseReady(dataset: DatasetSummary) {
  const amounts = dataset.categories.map((name) => dataset.counts[name] ?? 0);
  return amounts.length >= 2 && amounts.every((amount) => amount >= POSE_MINIMUM) && amounts.reduce((a, b) => a + b, 0) >= POSE_TOTAL;
}

/** The points it found, joined up. Lines keep their width however the picture is stretched. */
export function PointsLayer({ points, kind, mirrored = false }: { points: PoseAnswer["points"]; kind: "body" | "hands"; mirrored?: boolean }) {
  const lines = kind === "hands" ? HAND_LINES : BODY_LINES;
  return (
    <svg className={`pose-layer${mirrored ? " is-mirrored" : ""}`} viewBox="0 0 1 1" preserveAspectRatio="none" aria-hidden="true">
      {points.map((one, which) => (
        <g key={which}>
          {lines.map(([a, b]) => one[a] && one[b] && one[a][2] >= SEEN && one[b][2] >= SEEN && (
            <line key={`${a}-${b}`} className="pose-bone" x1={one[a][0]} y1={one[a][1]} x2={one[b][0]} y2={one[b][1]} />
          ))}
          {one.map(([x, y, seen], index) => seen >= SEEN && (
            // A line of no length with round ends draws a dot that stays round when stretched.
            <line key={index} className="pose-joint" x1={x} y1={y} x2={x} y2={y} />
          ))}
        </g>
      ))}
    </svg>
  );
}

/** What makes good photos for poses, said once at the top of Add data. */
export function PoseGuide({ dataset }: { dataset: DatasetSummary }) {
  const hands = what(dataset) === "hands";
  return (
    <section className="card">
      <h2>{hands ? "Photos of hand signs" : "Photos of poses"}</h2>
      <ul className="pose-tips muted small">
        <li>{hands ? "Keep your hand clearly in view, not cut off at the edge of the photo." : "Stand back so your body is in view, at least from the waist up."}</li>
        <li>The webcam is quickest: choose a category, then hold the button and move a little between shots.</li>
        <li>Mix it up: different people, distances, angles and rooms. The background doesn't matter, as it only learns from the points.</li>
        <li>Add a category for doing nothing in particular, so it has an answer when you aren't making any of the others.</li>
      </ul>
    </section>
  );
}

export function PoseTrainer({ dataset, beginner, running, busyElsewhere, onStart, onStop }: {
  dataset: DatasetSummary; beginner: boolean; running: SavedModel | null; busyElsewhere: boolean; onStart: () => void; onStop: (id: string) => void;
}) {
  const hands = what(dataset) === "hands";
  const ready = poseReady(dataset);
  const short = dataset.categories.filter((name) => (dataset.counts[name] ?? 0) < POSE_MINIMUM);
  const finding = running && running.stepsTotal > 0 && running.stepsDone < running.stepsTotal;
  return (
    <section className="card">
      <div className="card-head">
        <h2>{hands ? "Teach it your hand signs" : "Teach it your poses"}</h2>
        {running
          ? <button className="button button-small" onClick={() => onStop(running.id)}>Stop</button>
          : <button className="button button-small button-primary" disabled={!ready || busyElsewhere} onClick={onStart}>Start</button>}
      </div>
      <p className="muted small">
        {hands
          ? "It finds 21 points on each hand, every finger joint, then learns which sign those points make."
          : "It finds 33 points on the body, like shoulders, elbows and knees, then learns which pose those points make."}
      </p>
      <Why>
        {beginner
          ? "A ready-made model from Google, MediaPipe, finds the points on this computer. Because only the points are learned from, clothes, light and background don't get in the way, so a few dozen photos are often enough. A fifth of your photos are kept back to check it honestly."
          : `MediaPipe ${hands ? "hand landmarker (2 hands, 21 points each)" : "pose landmarker lite (33 points)"}, run on CPU. Points are centred on ${hands ? "the wrist" : "the shoulders"} and scaled by size, then a 2-layer MLP (128, 64) is trained with AdamW, class weights and coordinate jitter; the lowest held-back loss is kept.`}
      </Why>
      {running && (
        <div className="field-note" role="status">
          <progress className="train-progress" max={running.stepsTotal || 1} value={running.stepsDone} aria-label="Progress" />
          <p className="muted small">
            {!running.device
              ? `Getting ready. The first run downloads the model that finds the points, about ${hands ? 8 : 6} MB.`
              : finding
                ? `Finding the points in photo ${running.stepsDone} of ${running.stepsTotal}.`
                : `Learning from the points of ${running.trainCount} photos, checking on ${running.checkCount}.`}
          </p>
        </div>
      )}
      {!running && busyElsewhere && <p className="muted small field-note">Another model is training. Only one can train at a time.</p>}
      {!running && !ready && (
        <p className="muted small field-note">
          {dataset.categories.length < 2
            ? "Add at least 2 categories on the Add data tab."
            : short.length
              ? `Add at least ${POSE_MINIMUM} photos to: ${short.join(", ")}.`
              : `Add at least ${POSE_TOTAL} photos in total.`}
        </p>
      )}
    </section>
  );
}

function Answers({ answer, kind }: { answer: PoseAnswer; kind: "body" | "hands" }) {
  if (!answer.found) {
    return <p className="small pose-nothing" role="status">{kind === "hands" ? "No hands in view." : "No body in view."} It needs to see {kind === "hands" ? "a hand" : "you"} clearly to answer.</p>;
  }
  return (
    <ul className="try-answers" aria-live="polite" aria-label="The model's answers">
      {answer.answers.map((row, index) => (
        <li key={row.category} className={index === 0 ? "is-top" : ""}>
          <span className="try-name">{row.category}</span>
          <span className="try-track" aria-hidden="true"><span style={{ width: `${row.chance * 100}%` }} /></span>
          <span className="try-value">{(row.chance * 100).toFixed(0)}%</span>
        </li>
      ))}
    </ul>
  );
}

function LivePose({ jobId, kind, beginner }: { jobId: string; kind: "body" | "hands"; beginner: boolean }) {
  const [open, setOpen] = useState(false);
  const camera = useCamera(open);
  const [answer, setAnswer] = useState<PoseAnswer | null>(null);
  const [problem, setProblem] = useState<string | null>(null);
  const [rate, setRate] = useState<number | null>(null);

  // One frame at a time, so slow answers never pile up.
  useEffect(() => {
    if (!open || !camera.ready) return;
    let live = true;
    const loop = async () => {
      let frames = 0;
      let since = performance.now();
      while (live) {
        const video = camera.video.current;
        if (!video || document.hidden) { await new Promise((done) => setTimeout(done, 300)); continue; }
        const file = await grab(video);
        if (!file || !live) continue;
        try {
          const next = await api.poseAnswer(jobId, file);
          if (!live) return;
          setAnswer(next);
          setProblem(null);
          frames += 1;
          const now = performance.now();
          if (now - since > 1000) { setRate(frames / ((now - since) / 1000)); frames = 0; since = now; }
        } catch (error) {
          if (!live) return;
          setProblem(message(error, "The model couldn't answer."));
          await new Promise((done) => setTimeout(done, 1500));
        }
        await new Promise((done) => setTimeout(done, 40));
      }
    };
    void loop();
    return () => { live = false; };
  }, [open, camera.ready, camera.video, jobId]);

  useEffect(() => { if (!open) { setAnswer(null); setRate(null); } }, [open]);

  return (
    <div className="webcam-try field-note">
      <button className="button button-small" aria-pressed={open} onClick={() => setOpen(!open)}>{open ? "Turn camera off" : "Try it with your webcam"}</button>
      {open && (
        <div className="webcam-layout field-note">
          <CameraView {...camera}>{answer && <PointsLayer points={answer.points} kind={kind} mirrored />}</CameraView>
          <div className="webcam-side">
            {!answer && camera.ready && !problem && <p className="muted small">Loading the model…</p>}
            {answer && <Answers answer={answer} kind={kind} />}
            {problem && <p className="small tone-bad" role="alert">{problem}</p>}
            {answer && (
              <p className="muted small">
                {beginner
                  ? "The lines show the points it found. If they're in the wrong place, it will struggle too: move back, or into better light."
                  : `About ${Math.round(rate ?? 0)} answers a second, on this computer.`}
              </p>
            )}
          </div>
        </div>
      )}
    </div>
  );
}

function PosePhoto({ jobId, file, src, kind }: { jobId: string; file: File; src: string; kind: "body" | "hands" }) {
  const [answer, setAnswer] = useState<PoseAnswer | null>(null);
  const [problem, setProblem] = useState<string | null>(null);
  useEffect(() => {
    let live = true;
    api.poseAnswer(jobId, file)
      .then((next) => live && setAnswer(next))
      .catch((error) => live && setProblem(message(error, "The model couldn't check this photo.")));
    return () => { live = false; };
  }, [jobId, file]);
  return (
    <div className="pose-photo field-note">
      <div className="find-stage is-result">
        <img src={src} alt="The photo you chose" />
        {answer && <PointsLayer points={answer.points} kind={kind} />}
      </div>
      <div className="webcam-side">
        {!answer && !problem && <p className="muted small">Checking…</p>}
        {answer && <Answers answer={answer} kind={kind} />}
        {problem && <p className="small tone-bad" role="alert">{problem}</p>}
      </div>
    </div>
  );
}

export function PoseTry({ model, kind, beginner }: { model: SavedModel; kind: "body" | "hands"; beginner: boolean }) {
  const [chosen, setChosen] = useState<{ file: File; src: string } | null>(null);
  useEffect(() => () => { if (chosen) URL.revokeObjectURL(chosen.src); }, [chosen]);
  const review = model.review;
  return (
    <section className="card">
      <h2>Try it</h2>
      <p className="muted small">
        {model.projectName}{review ? ` · ${review.correct} of ${review.total} held-back photos right` : ""} · data v{model.datasetVersion ?? "?"}
      </p>
      <div className="label-buttons field-note" style={{ justifyContent: "flex-start", flexWrap: "wrap" }}>
        <label className="button button-small">
          Choose a photo
          <input className="visually-hidden" type="file" accept="image/jpeg,image/png,image/bmp,image/webp" onChange={(event) => {
            const file = event.target.files?.[0];
            if (file) setChosen({ file, src: URL.createObjectURL(file) });
            event.target.value = "";
          }} />
        </label>
      </div>
      {chosen && <PosePhoto key={chosen.src} jobId={model.id} file={chosen.file} src={chosen.src} kind={kind} />}
      <LivePose jobId={model.id} kind={kind} beginner={beginner} />
    </section>
  );
}

/** How it did on the held-back photos, and the ones it got wrong, which can be refiled or removed. */
export function PoseResults({ model, dataset, beginner, onChanged }: { model: SavedModel; dataset: DatasetSummary; beginner: boolean; onChanged: () => void }) {
  const [rows, setRows] = useState<TrainingMistake[] | null>(null);
  const [total, setTotal] = useState(0);
  const [problem, setProblem] = useState<string | null>(null);
  const categories = useMemo(() => Object.keys(model.categories), [model.categories]);
  const metrics = model.metrics;
  const kind = what(metrics ?? dataset);
  const nothing = (metrics?.nothingFoundTraining ?? 0) + (metrics?.nothingFoundChecking ?? 0);
  const per = model.review?.categories ?? {};

  useEffect(() => {
    api.fetchPoseMistakes(model.id)
      .then((reply) => { setRows(reply.mistakes); setTotal(reply.total); })
      .catch((error) => setProblem(message(error, "The mistakes couldn't load.")));
  }, [model.id]);

  const fix = async (index: number, action: "move" | "remove" | "undo", category?: string) => {
    setProblem(null);
    try {
      const job = await api.fixMistake(model.id, index, action, category);
      setRows(job.review?.mistakes ?? rows);
      onChanged();
    } catch (error) {
      setProblem(message(error, "That photo couldn't be changed."));
    }
  };

  return (
    <section className="card">
      <h2>How it did</h2>
      <p className="muted small">On photos it never learned from.</p>
      <table className="find-table field-note">
        <thead><tr><th scope="col">Category</th><th scope="col">Right</th></tr></thead>
        <tbody>
          {categories.map((name) => (
            <tr key={name}><th scope="row">{name}</th><td>{per[name] ? `${per[name].correct} of ${per[name].total}` : "–"}</td></tr>
          ))}
        </tbody>
      </table>
      {nothing > 0 && (
        <p className="small field-note pose-nothing">
          {nothing} {nothing === 1 ? "photo had" : "photos had"} no {kind === "hands" ? "hands" : "body"} it could see, so {nothing === 1 ? "it was" : "they were"} left out of learning
          {metrics?.nothingFoundChecking ? `, and ${metrics.nothingFoundChecking} held-back ${metrics.nothingFoundChecking === 1 ? "one counts" : "ones count"} as wrong` : ""}.
          {" "}{kind === "hands" ? "Make sure hands aren't cut off or blurry." : "Make sure people are in view, at least from the waist up."}
        </p>
      )}
      {problem && <p className="small tone-bad field-note" role="alert">{problem}</p>}
      {rows && rows.length === 0 && <p className="muted small field-note">It got every held-back photo right.</p>}
      {rows && rows.length > 0 && (
        <>
          <h3 className="field-note">Photos it got wrong</h3>
          <p className="muted small">
            {beginner
              ? "Most confident mistakes first. If a photo is in the wrong category, move it. Changes count from the next training."
              : `Most confident first.${total > MISTAKES_SHOWN ? ` Showing ${MISTAKES_SHOWN} of ${total}.` : ""} Fixes apply from the next run.`}
          </p>
          <ul className="mistake-grid">
            {rows.slice(0, MISTAKES_SHOWN).map((row, index) => (
              <li key={`${row.actual}/${row.file}`}>
                {row.removed
                  ? <div className="pose-removed muted small">Removed</div>
                  : <img className="pose-thumb" src={api.mistakeImageUrl(model.id, index)} alt={`Filed under ${row.movedTo ?? row.actual}, guessed ${row.guess}`} loading="lazy" />}
                <p className="small"><strong>{row.movedTo ?? row.actual}</strong>{row.movedTo ? " (moved)" : ""}</p>
                <p className="muted small">Guessed {row.guess}{row.chance ? `, ${Math.round(row.chance * 100)}% sure` : ""}</p>
                <div className="label-buttons" style={{ justifyContent: "flex-start", flexWrap: "wrap" }}>
                  {row.movedTo || row.removed
                    ? <button className="button button-small button-quiet" onClick={() => void fix(index, "undo")}>Undo</button>
                    : (
                      <>
                        <select className="pose-move" aria-label={`Move this photo, filed under ${row.actual}`} value="" onChange={(event) => event.target.value && void fix(index, "move", event.target.value)}>
                          <option value="">Move to…</option>
                          {categories.filter((name) => name !== row.actual).map((name) => <option key={name} value={name}>{name}</option>)}
                        </select>
                        <button className="button button-small button-quiet" onClick={() => void fix(index, "remove")}>Remove</button>
                      </>
                    )}
                </div>
              </li>
            ))}
          </ul>
        </>
      )}
    </section>
  );
}
