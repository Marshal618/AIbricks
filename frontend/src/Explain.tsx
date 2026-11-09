import { useEffect, useRef, useState } from "react";
import * as api from "./api";
import type { Heatmap, TextExplanation } from "./types";

function message(error: unknown, fallback: string) {
  return error instanceof Error ? error.message : fallback;
}

/**
 * Draws where the model looked over a photo: what it ignored is dimmed, and what it used glows amber.
 * Brightness carries the meaning, not hue, so it reads the same for colour-blind viewers.
 * Fills its positioned parent, which should show the whole photo the model saw.
 */
export function HeatmapOverlay({ heatmap, aspect, mirrored = false }: { heatmap: Heatmap; aspect: number; mirrored?: boolean }) {
  const canvas = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const target = canvas.current;
    const context = target?.getContext("2d");
    if (!target || !context) return;
    const small = document.createElement("canvas");
    small.width = heatmap.width;
    small.height = heatmap.height;
    const cells = small.getContext("2d")!.createImageData(heatmap.width, heatmap.height);
    heatmap.values.forEach((raw, index) => {
      const value = Math.max(0, Math.min(1, raw));
      cells.data.set([Math.round(255 * value), Math.round(176 * value), 0, Math.round(255 * (0.62 - 0.27 * value))], index * 4);
    });
    small.getContext("2d")!.putImageData(cells, 0, 0);
    // The model saw the whole photo squashed square, so the heatmap stretches back to the photo's shape.
    // The overlay is then cropped with the same object-fit as the photo beneath it.
    target.height = 320;
    target.width = Math.max(1, Math.round(320 * (aspect || 1)));
    context.imageSmoothingEnabled = true;
    context.imageSmoothingQuality = "high";
    context.clearRect(0, 0, target.width, target.height);
    context.drawImage(small, 0, 0, target.width, target.height);
  }, [heatmap, aspect]);

  return <canvas ref={canvas} className={`heatmap-overlay${mirrored ? " is-mirrored" : ""}`} aria-hidden="true" />;
}

export function HeatmapKey({ heatmap, beginner }: { heatmap: Heatmap; beginner: boolean }) {
  return (
    <p className="muted small heatmap-key">
      <span className="heatmap-swatch" aria-hidden="true" />
      {beginner && heatmap.method === "patchcore"
        ? "Bright areas look least like your normal photos. Everything dim matched something it had seen."
        : beginner
        ? `Bright areas are where it looked most to decide “${heatmap.category}”. Dimmed areas mattered little. This shows where it looked, not proof of why.`
        : heatmap.method === "patchcore"
          ? `Patch distance to the nearest remembered normal patch, ${heatmap.width}×${heatmap.height}; full brightness is 1.5× the line.`
        : heatmap.method === "gradcam"
          ? `Grad-CAM for “${heatmap.category}” on the last spatial layer (${heatmap.width}×${heatmap.height}). Shows where, not why.`
          : `Gradient saliency for “${heatmap.category}”, pooled to ${heatmap.width}×${heatmap.height}: this model has no spatial layer near its output.`}
    </p>
  );
}

/** A photo with a "Where did it look?" button that lays the heatmap over it. */
export function ExplainablePhoto({ jobId, src, file, alt, beginner, className }: {
  jobId: string; src: string; file?: Blob; alt: string; beginner: boolean; className?: string;
}) {
  const [heatmap, setHeatmap] = useState<Heatmap | null>(null);
  const [shown, setShown] = useState(false);
  const [busy, setBusy] = useState(false);
  const [problem, setProblem] = useState<string | null>(null);
  const [missing, setMissing] = useState(false);
  const [aspect, setAspect] = useState(1);

  useEffect(() => { setHeatmap(null); setShown(false); setProblem(null); }, [src, jobId]);

  const toggle = async () => {
    if (heatmap) { setShown(!shown); return; }
    setBusy(true);
    setProblem(null);
    try {
      let blob = file;
      if (!blob) {
        const reply = await fetch(src);
        if (!reply.ok) throw new Error("The image is no longer saved.");
        blob = await reply.blob();
      }
      setHeatmap((await api.explainImage(jobId, blob)).heatmap);
      setShown(true);
    } catch (error) {
      setProblem(message(error, "The model couldn't show where it looked."));
    } finally {
      setBusy(false);
    }
  };

  if (missing) return <span className="mistake-missing muted small">Image removed</span>;
  return (
    <div className={`explain-photo ${className ?? ""}`}>
      <div className="explain-frame">
        <img src={src} alt={alt} loading="lazy" onError={() => setMissing(true)} onLoad={(event) => setAspect(event.currentTarget.naturalWidth / event.currentTarget.naturalHeight)} />
        {heatmap && shown && <HeatmapOverlay heatmap={heatmap} aspect={aspect} />}
      </div>
      <button className="button button-small button-quiet explain-toggle" aria-pressed={shown} disabled={busy} onClick={() => void toggle()}>
        {busy ? "Looking…" : shown ? "Hide where it looked" : "Where did it look?"}
      </button>
      {heatmap && shown && <HeatmapKey heatmap={heatmap} beginner={beginner} />}
      {problem && <p className="small tone-bad" role="alert">{problem}</p>}
    </div>
  );
}

// ------------------------------------------------------------------------------- text

/** The text with each word shaded by how much it pushed the model toward, or away from, its answer. */
export function WordWeights({ text, explanation, beginner }: { text: string; explanation: TextExplanation; beginner: boolean }) {
  const strongest = Math.max(1e-6, ...explanation.words.map((word) => Math.abs(word.weight)));
  const pieces: JSX.Element[] = [];
  let at = 0;
  explanation.words.forEach((word, index) => {
    if (word.start > at) pieces.push(<span key={`gap-${index}`}>{text.slice(at, word.start)}</span>);
    const share = word.weight / strongest;
    const points = Math.round(word.weight * 100);
    const label = points === 0 ? "made little difference" : `${points > 0 ? "+" : ""}${points} points toward ${explanation.category}`;
    pieces.push(
      <mark
        key={index}
        className={`word-weight ${share >= 0 ? "is-for" : "is-against"}`}
        style={{ ["--weight" as string]: Math.abs(share).toFixed(3) }}
        title={label}
      >
        {text.slice(word.start, word.end)}
      </mark>,
    );
    at = word.end;
  });
  if (at < text.length) pieces.push(<span key="rest">{text.slice(at)}</span>);

  const ranked = [...explanation.words].sort((a, b) => b.weight - a.weight);
  const top = ranked.filter((word) => word.weight > 0.005).slice(0, 3).map((word) => `“${text.slice(word.start, word.end)}”`);
  const against = ranked.reverse().filter((word) => word.weight < -0.005).slice(0, 2).map((word) => `“${text.slice(word.start, word.end)}”`);

  return (
    <div className="word-weights field-note">
      <p className="word-weights-text">{pieces}</p>
      <p className="muted small">
        {top.length
          ? `${top.join(", ")} pushed it most toward ${explanation.category}.`
          : `No single word changed its answer much, so it's going on the text as a whole.`}
        {against.length ? ` ${against.join(" and ")} pushed against it.` : ""}
        {explanation.wordCount > explanation.words.length ? ` Only the first ${explanation.words.length} words were checked.` : ""}
      </p>
      <p className="muted small heatmap-key">
        <span className="word-swatch is-for" aria-hidden="true" /> toward {explanation.category}
        <span className="word-swatch is-against" aria-hidden="true" /> against
        <span>{beginner ? " · Each word was removed in turn to see how the answer changed." : " · Leave-one-word-out: drop in the top answer's chance with the word removed."}</span>
      </p>
    </div>
  );
}

/** A "Why?" button for one text that shows its word weights below it. */
export function ExplainableText({ jobId, text, beginner }: { jobId: string; text: string; beginner: boolean }) {
  const [explanation, setExplanation] = useState<TextExplanation | null>(null);
  const [shown, setShown] = useState(false);
  const [busy, setBusy] = useState(false);
  const [problem, setProblem] = useState<string | null>(null);

  const toggle = async () => {
    if (explanation) { setShown(!shown); return; }
    setBusy(true);
    setProblem(null);
    try {
      setExplanation(await api.explainText(jobId, text));
      setShown(true);
    } catch (error) {
      setProblem(message(error, "The model couldn't explain this."));
    } finally {
      setBusy(false);
    }
  };

  return (
    <>
      <button className="button button-small button-quiet" aria-pressed={shown} disabled={busy} onClick={() => void toggle()}>
        {busy ? "Checking…" : shown ? "Hide words" : "Which words mattered?"}
      </button>
      {problem && <p className="small tone-bad" role="alert">{problem}</p>}
      {explanation && shown && <WordWeights text={text} explanation={explanation} beginner={beginner} />}
    </>
  );
}
