import { useEffect, useMemo, useState } from "react";
import * as api from "./api";
import { ExplainablePhoto } from "./Explain";
import type { DatasetSummary, SavedModel } from "./types";
import { Why } from "./Ui";
import { WebcamTry } from "./Webcam";

export const UNUSUAL_MINIMUM = 10;
export const UNUSUAL_SUGGESTED = 30;

function message(error: unknown, fallback: string) {
  return error instanceof Error ? error.message : fallback;
}

/** What to put in each category, said once at the top of Add data. */
export function UnusualGuide({ dataset }: { dataset: DatasetSummary }) {
  const normal = dataset.counts.Normal ?? 0;
  return (
    <section className="card unusual-guide">
      <h2>Two kinds of photo</h2>
      <ul className="unusual-kinds">
        <li><strong>Normal</strong><span className="muted small">Things that are fine, in every condition it will see: different light, angles and backgrounds. It learns only from these. {normal} so far; aim for {UNUSUAL_SUGGESTED}.</span></li>
        <li><strong>Unusual</strong><span className="muted small">Optional. A few photos with problems, used only to check how well it spots them. It never learns from them.</span></li>
      </ul>
    </section>
  );
}

export function UnusualTrainer({ dataset, beginner, running, busyElsewhere, onStart, onStop }: {
  dataset: DatasetSummary; beginner: boolean; running: SavedModel | null; busyElsewhere: boolean; onStart: () => void; onStop: (id: string) => void;
}) {
  const normal = dataset.counts.Normal ?? 0;
  return (
    <section className="card">
      <div className="card-head">
        <h2>Learn what normal looks like</h2>
        {running
          ? <button className="button button-small" onClick={() => onStop(running.id)}>Stop</button>
          : <button className="button button-small button-primary" disabled={normal < UNUSUAL_MINIMUM || busyElsewhere} onClick={onStart}>Start</button>}
      </div>
      <p className="muted small">It remembers the details of your normal photos, then flags anything that doesn't match them.</p>
      <Why>
        {beginner
          ? "A model that already learned from a million photos describes every small patch of each normal photo, and a sample of those descriptions is kept. A new photo is unusual when part of it is unlike anything in that memory. The line between normal and unusual is set from your training photos, so the held-back photos give an honest check."
          : "PatchCore (Roth et al., 2022): ResNet-18 layer 2 and 3 patch features, greedy coreset memory (10%), nearest-neighbour patch distance. The threshold comes from two-fold scores on training photos only."}
      </Why>
      {running && (
        <div className="field-note" role="status">
          <progress className="train-progress" max={running.stepsTotal || 1} value={running.stepsDone} aria-label="Progress" />
          <p className="muted small">{!running.device ? "Getting ready. The first run downloads a pretrained model of about 45 MB." : `Looked at ${running.stepsDone} of ${running.stepsTotal} photos, using ${running.device}.`}</p>
        </div>
      )}
      {!running && busyElsewhere && <p className="muted small field-note">Another model is training. Only one can train at a time.</p>}
      {!running && normal < UNUSUAL_MINIMUM && <p className="muted small field-note">Add at least {UNUSUAL_MINIMUM} normal photos on the Add data tab. It has {normal}.</p>}
    </section>
  );
}

/** Tries the model on a photo or the webcam, where the bright parts are what looked unusual. */
export function UnusualTry({ model, beginner }: { model: SavedModel; beginner: boolean }) {
  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  useEffect(() => () => { if (preview) URL.revokeObjectURL(preview); }, [preview]);
  return (
    <section className="card">
      <h2>Try it</h2>
      <p className="muted small">Choose a photo, or use the webcam with “Show where it looks” on. Bright parts are what looked unusual.</p>
      <div className="label-buttons field-note" style={{ justifyContent: "flex-start", flexWrap: "wrap" }}>
        <label className="button button-small">
          Choose a photo
          <input className="visually-hidden" type="file" accept="image/jpeg,image/png,image/bmp,image/webp" onChange={(event) => {
            const chosen = event.target.files?.[0];
            if (chosen) { setFile(chosen); setPreview(URL.createObjectURL(chosen)); }
            event.target.value = "";
          }} />
        </label>
      </div>
      {preview && file && <div className="try-drop field-note"><UnusualPhoto key={preview} jobId={model.id} src={preview} file={file} beginner={beginner} /></div>}
      <WebcamTry jobId={model.id} beginner={beginner} />
    </section>
  );
}

function UnusualPhoto({ jobId, src, file, beginner }: { jobId: string; src: string; file: File; beginner: boolean }) {
  const [verdict, setVerdict] = useState<{ label: string; ratio: number } | null>(null);
  const [problem, setProblem] = useState<string | null>(null);
  useEffect(() => {
    let live = true;
    api.explainImage(jobId, file)
      .then((reply) => {
        if (!live) return;
        const ratio = (reply as unknown as { unusualness?: number }).unusualness ?? 0;
        setVerdict({ label: reply.answers[0].category, ratio });
      })
      .catch((error) => live && setProblem(message(error, "The model couldn't check this photo.")));
    return () => { live = false; };
  }, [jobId, file]);
  return (
    <div className="unusual-photo">
      {verdict && (
        <p className={`unusual-verdict ${verdict.label === "Unusual" ? "is-unusual" : "is-normal"}`} role="status">
          <span aria-hidden="true">{verdict.label === "Unusual" ? "!" : "✓"}</span> {verdict.label}
          <span className="muted small"> · {verdict.ratio.toFixed(2)}× the line</span>
        </p>
      )}
      {problem && <p className="small tone-bad" role="alert">{problem}</p>}
      <ExplainablePhoto jobId={jobId} src={src} file={file} alt="The photo you chose" beginner={beginner} />
    </div>
  );
}

/** Moves the line between normal and unusual, showing what each position does to the held-back photos. */
export function UnusualLine({ model, beginner }: { model: SavedModel; beginner: boolean }) {
  const scores = useMemo(() => model.metrics?.scores ?? [], [model.metrics]);
  const [factor, setFactor] = useState(1);
  const [saved, setSaved] = useState(1);
  const [problem, setProblem] = useState<string | null>(null);
  useEffect(() => {
    api.fetchUnusualSettings(model.id).then((reply) => { setFactor(reply.factor); setSaved(reply.factor); }).catch(() => undefined);
  }, [model.id]);

  const normals = scores.filter((row) => row.actual === "Normal");
  const odd = scores.filter((row) => row.actual === "Unusual");
  const falseAlarms = normals.filter((row) => row.unusualness > factor).length;
  const caught = odd.filter((row) => row.unusualness > factor).length;
  const auroc = model.metrics?.auroc;

  const save = async () => {
    setProblem(null);
    try {
      setSaved((await api.setUnusualLine(model.id, factor)).factor);
    } catch (error) {
      setProblem(message(error, "The line couldn't be moved."));
    }
  };

  return (
    <section className="card">
      <h2>How strict should it be?</h2>
      <p className="muted small">Stricter flags more problems but also more false alarms. Here's what each position does to the held-back photos.</p>
      <div className="unusual-line field-note">
        <span className="small muted">Stricter</span>
        <input type="range" min={0.5} max={3} step={0.05} value={factor} aria-label="Line between normal and unusual" onChange={(event) => setFactor(Number(event.target.value))} />
        <span className="small muted">More relaxed</span>
      </div>
      <dl className="unusual-stats">
        <div><dt>False alarms</dt><dd>{falseAlarms} of {normals.length}</dd><span className="muted small">normal photos flagged</span></div>
        <div><dt>Caught</dt><dd>{odd.length ? `${caught} of ${odd.length}` : "–"}</dd><span className="muted small">{odd.length ? "unusual photos flagged" : "add Unusual photos to measure this"}</span></div>
        {auroc !== undefined && auroc !== null && (
          <div><dt>Separation</dt><dd>{Math.round(auroc * 100)}%</dd><span className="muted small">{beginner ? "how often an unusual photo scores above a normal one" : "AUROC on held-back photos"}</span></div>
        )}
      </dl>
      <div className="label-buttons" style={{ justifyContent: "flex-start", alignItems: "center" }}>
        <button className="button button-small button-primary" disabled={factor === saved} onClick={() => void save()}>Use this line</button>
        {factor !== saved && <button className="button button-small button-quiet" onClick={() => setFactor(saved)}>Undo</button>}
        {factor === saved && <span className="muted small">In use{saved !== 1 ? ` (${saved.toFixed(2)}× the starting line)` : ""}. The webcam, photos and app follow it.</span>}
      </div>
      {problem && <p className="small tone-bad field-note" role="alert">{problem}</p>}
      {normals.length < 10 && <p className="muted small field-note">These counts come from only {normals.length} held-back normal {normals.length === 1 ? "photo" : "photos"}, so they're rough.</p>}
    </section>
  );
}
