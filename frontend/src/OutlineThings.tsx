import { useCallback, useEffect, useMemo, useRef, useState, type PointerEvent as ReactPointerEvent } from "react";
import * as api from "./api";
import type { Box, BoxedPhoto, DatasetSummary, FoundOutlines, OutlineMetrics, OutlinedPhoto, SavedModel } from "./types";
import { Why } from "./Ui";
import { CameraView, grab, useCamera } from "./Webcam";
import { ShapePhotoAdder, colourOf, plural } from "./Find";
import { ModelApp } from "./ModelApp";

export const OUTLINE_MINIMUM_PHOTOS = 10;
export const OUTLINE_MINIMUM_OUTLINES = 5;
export const OUTLINE_SUGGESTED = 30;
const STRIP_LIMIT = 80;
const CLOSE_ENOUGH = 0.02;

type Point = [number, number];

function message(error: unknown, fallback: string) {
  return error instanceof Error ? error.message : fallback;
}

function percent(value: number | null | undefined) {
  return value === null || value === undefined ? "–" : `${Math.round(value * 100)}%`;
}

function withBox(category: string, points: Point[]): Box {
  const xs = points.map((p) => p[0]), ys = points.map((p) => p[1]);
  const x = Math.min(...xs), y = Math.min(...ys);
  return { category, points, x, y, w: Math.max(...xs) - x, h: Math.max(...ys) - y };
}

/** True once a dataset has enough outlined photos to train on. */
export function outlineReady(dataset: DatasetSummary) {
  return dataset.categories.length >= 1 && dataset.labelled >= OUTLINE_MINIMUM_PHOTOS
    && dataset.categories.every((name) => (dataset.counts[name] ?? 0) >= OUTLINE_MINIMUM_OUTLINES);
}

/** Outlines drawn over whatever fills their positioned parent, each with its thing's name. */
export function OutlineLayer({ shapes, categories, selected = null, onSelect, onRemove, style = "solid" }: {
  shapes: Box[]; categories: string[]; selected?: number | null; onSelect?: (index: number) => void; onRemove?: (index: number) => void;
  style?: "solid" | "expected" | "preview";
}) {
  return (
    <>
      <svg className="outline-layer" viewBox="0 0 1 1" preserveAspectRatio="none" aria-hidden="true">
        {shapes.map((shape, index) => (
          <polygon
            key={index}
            className={`outline-shape is-${style}${selected === index ? " is-selected" : ""}`}
            points={(shape.points ?? []).map(([x, y]) => `${x},${y}`).join(" ")}
            style={{ ["--box" as string]: style === "expected" ? "#ffffff" : colourOf(categories, shape.category) }}
            onPointerDown={onSelect ? (event) => { event.stopPropagation(); onSelect(index); } : undefined}
          />
        ))}
      </svg>
      {style === "solid" && shapes.map((shape, index) => (
        <span key={index} className="find-box-tag outline-tag" style={{ left: `${shape.x * 100}%`, top: `${shape.y * 100}%`, ["--box" as string]: colourOf(categories, shape.category) }}>
          {shape.category}
          {onRemove && <button type="button" aria-label={`Remove this ${shape.category} outline`} onPointerDown={(event) => event.stopPropagation()} onClick={() => onRemove(index)}>×</button>}
        </span>
      ))}
    </>
  );
}

// ----------------------------------------------------------------- adding photos and outlining them

export function OutlineLabels({ dataset, beginner, onChanged }: { dataset: DatasetSummary; beginner: boolean; onChanged: () => void }) {
  const datasetId = dataset.id;
  const categories = dataset.categories;
  const [photos, setPhotos] = useState<BoxedPhoto[] | null>(null);
  const [currentId, setCurrentId] = useState<string | null>(null);
  const [draft, setDraft] = useState<Box[]>([]);
  const [thing, setThing] = useState(categories[0] ?? "");
  const [selected, setSelected] = useState<number | null>(null);
  const [mode, setMode] = useState<"click" | "draw">("click");
  const [clicks, setClicks] = useState<api.Click[]>([]);
  const [preview, setPreview] = useState<Point[] | null>(null);
  const [thinking, setThinking] = useState(false);
  const [corners, setCorners] = useState<Point[]>([]);
  const [problem, setProblem] = useState<string | null>(null);
  const stage = useRef<HTMLDivElement>(null);
  const asked = useRef(0);

  useEffect(() => { if (!categories.includes(thing)) setThing(categories[0] ?? ""); }, [categories, thing]);

  const load = useCallback(async (keep?: string | null) => {
    const reply = await api.fetchBoxes(datasetId);
    setPhotos(reply.photos);
    const next = reply.photos.find((photo) => photo.id === keep) ?? reply.photos.find((photo) => photo.boxes === null) ?? reply.photos[0] ?? null;
    setCurrentId(next?.id ?? null);
  }, [datasetId]);

  useEffect(() => { load().catch((error) => setProblem(message(error, "The photos couldn't load."))); }, [load]);

  const current = photos?.find((photo) => photo.id === currentId) ?? null;
  const reset = useCallback(() => { setClicks([]); setPreview(null); setCorners([]); asked.current += 1; setThinking(false); }, []);
  useEffect(() => {
    setSelected(null);
    reset();
    setDraft(current?.boxes ?? []);
  }, [current?.id, current?.boxes, reset]);

  const done = photos?.filter((photo) => photo.boxes !== null).length ?? 0;
  const total = photos?.length ?? 0;

  const save = useCallback(async (shapes: Box[]) => {
    if (!current) return;
    setProblem(null);
    try {
      const saved = await api.saveBoxes(datasetId, current.id, shapes);
      const updated = (photos ?? []).map((photo) => photo.id === current.id ? { ...photo, boxes: saved.boxes } : photo);
      setPhotos(updated);
      const index = updated.findIndex((photo) => photo.id === current.id);
      const next = [...updated.slice(index + 1), ...updated.slice(0, index)].find((photo) => photo.boxes === null);
      setCurrentId(next?.id ?? current.id);
      onChanged();
    } catch (error) {
      setProblem(message(error, "The outlines couldn't be saved."));
    }
  }, [current, datasetId, photos, onChanged]);

  const keep = useCallback(() => {
    const points = mode === "click" ? preview : corners.length >= 3 ? corners : null;
    if (!points || !thing) return false;
    setDraft((shapes) => [...shapes, withBox(thing, points)]);
    setSelected(draft.length);
    reset();
    return true;
  }, [mode, preview, corners, thing, draft.length, reset]);

  const ask = async (next: api.Click[]) => {
    if (!current) return;
    setClicks(next);
    if (!next.some((click) => click.inside)) { setPreview(null); return; }
    const ticket = ++asked.current;
    setThinking(true);
    try {
      const found = await api.outlineAt(datasetId, current.id, next);
      if (ticket !== asked.current) return;
      setPreview(found.points.length >= 3 ? found.points : null);
      setProblem(found.points.length >= 3 ? null : "Nothing was found there. Try clicking nearer the middle of the thing.");
    } catch (error) {
      if (ticket === asked.current) setProblem(message(error, "The outline couldn't be found."));
    } finally {
      if (ticket === asked.current) setThinking(false);
    }
  };

  const point = (event: ReactPointerEvent): Point => {
    const area = stage.current!.getBoundingClientRect();
    return [Math.min(1, Math.max(0, (event.clientX - area.left) / area.width)), Math.min(1, Math.max(0, (event.clientY - area.top) / area.height))];
  };

  const onStage = (event: ReactPointerEvent<HTMLDivElement>) => {
    if (!thing || (event.button !== 0 && event.button !== 2)) return;
    const [x, y] = point(event);
    setSelected(null);
    if (mode === "click") {
      // Shift, Alt or the right button marks a part to leave out.
      void ask([...clicks, { x, y, inside: !(event.shiftKey || event.altKey || event.button === 2) }]);
      return;
    }
    if (corners.length >= 3 && Math.hypot(x - corners[0][0], y - corners[0][1]) < CLOSE_ENOUGH) {
      keep();
      return;
    }
    setCorners([...corners, [x, y]]);
  };

  const removeShape = useCallback((index: number) => {
    setDraft((shapes) => shapes.filter((_, at) => at !== index));
    setSelected(null);
  }, []);

  const removePhoto = async () => {
    if (!current) return;
    setProblem(null);
    try {
      await api.removeImage(current.id, datasetId);
      await load();
      onChanged();
    } catch (error) {
      setProblem(message(error, "The photo couldn't be removed."));
    }
  };

  // Keys: 1 to 9 pick a thing, Enter keeps the outline (or saves the photo), Escape starts again,
  // Delete removes the chosen outline.
  useEffect(() => {
    const onKey = (event: KeyboardEvent) => {
      const target = event.target as HTMLElement;
      if (target.closest("input, textarea, select, [contenteditable]") || event.metaKey || event.ctrlKey || event.altKey) return;
      if (/^[1-9]$/.test(event.key) && categories[Number(event.key) - 1]) {
        const name = categories[Number(event.key) - 1];
        setThing(name);
        if (selected !== null) setDraft((shapes) => shapes.map((shape, at) => at === selected ? { ...shape, category: name } : shape));
      } else if (event.key === "Enter" && current && !target.closest("button")) {
        event.preventDefault();
        if (!keep() && !clicks.length && !corners.length) void save(draft);
      } else if (event.key === "Escape") {
        reset();
      } else if ((event.key === "Delete" || event.key === "Backspace") && selected !== null) {
        event.preventDefault();
        removeShape(selected);
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [categories, selected, current, draft, save, removeShape, keep, reset, clicks.length, corners.length]);

  const strip = useMemo(() => (photos ?? []).slice(0, STRIP_LIMIT), [photos]);
  const pending = mode === "click" ? preview !== null : corners.length >= 3;

  return (
    <>
      {problem && <div className="result result-fail" role="alert"><p>{problem}</p></div>}
      <ShapePhotoAdder datasetId={datasetId} beginner={beginner} onAdded={async () => { await load(currentId); onChanged(); }} onProblem={setProblem}
        title="Photos to outline things in" shapes="outlines" verb="outline" />

      <h2 className="section-title">Outline things</h2>
      <section className="card find-labeller">
        <div className="card-head">
          <h2>{total ? `${done} of ${total} photos done` : "Outline things"}</h2>
          <div className="label-buttons" role="radiogroup" aria-label="How to outline">
            <button className="button button-small" role="radio" aria-checked={mode === "click"} aria-pressed={mode === "click"} onClick={() => { setMode("click"); reset(); }}>Click to outline</button>
            <button className="button button-small" role="radio" aria-checked={mode === "draw"} aria-pressed={mode === "draw"} onClick={() => { setMode("draw"); reset(); }}>Draw by hand</button>
          </div>
        </div>
        {!categories.length && <p className="muted small">Name the things to outline above first.</p>}
        {categories.length > 0 && total === 0 && <p className="muted small">Add some photos above, then outline each thing in them.</p>}
        {categories.length > 0 && current && (
          <>
            <p className="muted small">
              {mode === "click"
                ? "Click on a thing and its outline appears. Click more parts to add them, or Shift-click to leave a part out. Keep it, then do the next one."
                : "Click around the edge of the thing, corner by corner. Click the first corner again, or press Enter, to close it."}
              {beginner && " Outline every one you can see, even small or partly hidden ones."}
            </p>
            <fieldset className="chip-group field-note">
              <legend className="small">Outlining</legend>
              {categories.map((name, index) => (
                <label key={name} className={`chip find-chip${thing === name ? " is-on" : ""}`} style={{ ["--box" as string]: colourOf(categories, name) }}>
                  <input type="radio" name="outline-thing" checked={thing === name} onChange={() => {
                    setThing(name);
                    if (selected !== null) setDraft((shapes) => shapes.map((shape, at) => at === selected ? { ...shape, category: name } : shape));
                  }} />
                  <span className="find-swatch" aria-hidden="true" />{name}{index < 9 && <kbd>{index + 1}</kbd>}
                </label>
              ))}
            </fieldset>
            <div
              ref={stage}
              className={`find-stage field-note outline-stage${thinking ? " is-thinking" : ""}`}
              onPointerDown={onStage}
              onContextMenu={(event) => event.preventDefault()}
              role="application"
              aria-label={`Photo to outline ${thing} in. ${plural(draft.length, "outline")} so far.`}
            >
              <img src={api.unsortedImageUrl(current.id, datasetId)} alt="" draggable={false} />
              <OutlineLayer shapes={draft} categories={categories} selected={selected} onSelect={setSelected} onRemove={removeShape} />
              {preview && <OutlineLayer shapes={[withBox(thing, preview)]} categories={categories} style="preview" />}
              {mode === "draw" && corners.length > 0 && (
                <svg className="outline-layer" viewBox="0 0 1 1" preserveAspectRatio="none" aria-hidden="true">
                  <polyline className="outline-shape is-preview" points={corners.map(([x, y]) => `${x},${y}`).join(" ")} style={{ ["--box" as string]: colourOf(categories, thing) }} />
                </svg>
              )}
              {mode === "draw" && corners.map(([x, y], index) => (
                <span key={index} className={`outline-corner${index === 0 ? " is-first" : ""}`} style={{ left: `${x * 100}%`, top: `${y * 100}%` }} aria-hidden="true" />
              ))}
              {mode === "click" && clicks.map((click, index) => (
                <span key={index} className={`outline-click${click.inside ? "" : " is-out"}`} style={{ left: `${click.x * 100}%`, top: `${click.y * 100}%` }} aria-hidden="true">{click.inside ? "+" : "−"}</span>
              ))}
            </div>
            <div className="label-buttons field-note find-actions">
              {pending || clicks.length || corners.length ? (
                <>
                  <button className="button button-primary" disabled={!pending} onClick={() => keep()}>Keep this {thing} <kbd>Enter</kbd></button>
                  <button className="button" onClick={reset}>Start again <kbd>Esc</kbd></button>
                  {thinking && <span className="muted small" role="status">Finding the outline…</span>}
                </>
              ) : (
                <>
                  <button className="button button-primary" onClick={() => void save(draft)} disabled={!draft.length}>
                    Save {plural(draft.length, "outline")} and next <kbd>Enter</kbd>
                  </button>
                  <button className="button" onClick={() => void save([])}>Nothing to outline here</button>
                </>
              )}
              <span className="tab-gap" />
              {current.boxes !== null && <span className="muted small">Done{current.boxes.length ? `, ${plural(current.boxes.length, "outline")}` : ", nothing here"}</span>}
              <button className="button button-small button-quiet" onClick={() => void removePhoto()}>Remove photo</button>
            </div>
            <Why>
              {beginner
                ? "Clicking uses MobileSAM, a small version of Meta's Segment Anything, running on this computer. It already knows where the edges of everyday things are. The first click on a photo takes a moment while it looks at the whole photo; later clicks are quick."
                : "MobileSAM (Apache 2.0, ONNX, run locally with ONNX Runtime): a TinyViT image encoder runs once per photo, then the prompt decoder turns your clicks into a mask, traced into a polygon of at most 200 corners. Shift, Alt or right-click adds a background point."}
            </Why>
            <ol className="find-strip field-note" aria-label="Photos">
              {strip.map((photo, index) => (
                <li key={photo.id}>
                  <button className={`${photo.id === current.id ? "is-current" : ""}${photo.boxes !== null ? " is-done" : ""}`} aria-current={photo.id === current.id} aria-label={`Photo ${index + 1}${photo.boxes !== null ? ", done" : ""}`} onClick={() => setCurrentId(photo.id)}>
                    <img src={api.unsortedImageUrl(photo.id, datasetId)} alt="" loading="lazy" />
                    {photo.boxes !== null && <span className="find-strip-mark" aria-hidden="true">✓</span>}
                  </button>
                </li>
              ))}
            </ol>
            {total > STRIP_LIMIT && <p className="muted small">Showing the first {STRIP_LIMIT} of {total}. Photos still to do come first.</p>}
          </>
        )}
      </section>
    </>
  );
}

// ----------------------------------------------------------------------------- training

export function OutlineTrainer({ dataset, beginner, running, busyElsewhere, onStart, onStop }: {
  dataset: DatasetSummary; beginner: boolean; running: SavedModel | null; busyElsewhere: boolean; onStart: () => void; onStop: (id: string) => void;
}) {
  const ready = outlineReady(dataset);
  const short = dataset.categories.filter((name) => (dataset.counts[name] ?? 0) < OUTLINE_MINIMUM_OUTLINES);
  return (
    <section className="card">
      <div className="card-head">
        <h2>Teach it to outline {dataset.categories.length === 1 ? dataset.categories[0] : "your things"}</h2>
        {running
          ? <button className="button button-small" onClick={() => onStop(running.id)}>Stop</button>
          : <button className="button button-small button-primary" disabled={!ready || busyElsewhere} onClick={onStart}>Start</button>}
      </div>
      <p className="muted small">It starts from a model that can already outline everyday things, and learns yours from your outlines.</p>
      <Why>
        {beginner
          ? "The starting model learned from thousands of outlined photos, so it already knows where edges and objects are. Its last steps are swapped for ones that know your things, then the whole model is adjusted on your photos. A fifth of your photos are kept back to check it honestly."
          : "LRASPP with a MobileNetV3-Large backbone (3.2M parameters), torchvision's COCO-with-VOC-labels weights, both classifier convolutions replaced. Full fine-tune at 320×320 with AdamW (lr 1e-3, cosine), class-weighted cross-entropy, flips and brightness changes, pretrained batch norm frozen. The checkpoint with the lowest held-back loss is kept."}
      </Why>
      {running && (
        <div className="field-note" role="status">
          <progress className="train-progress" max={running.stepsTotal || 1} value={running.stepsDone} aria-label="Progress" />
          <p className="muted small">
            {!running.device
              ? "Getting ready. The first run downloads the starting model, about 13 MB."
              : `Step ${running.stepsDone} of ${running.stepsTotal || "?"}, learning from ${running.trainCount} photos and checking on ${running.checkCount}, using ${running.device}.`}
          </p>
        </div>
      )}
      {!running && busyElsewhere && <p className="muted small field-note">Another model is training. Only one can train at a time.</p>}
      {!running && !ready && (
        <p className="muted small field-note">
          {!dataset.categories.length
            ? "Name at least one thing to outline on the Add data tab."
            : dataset.labelled < OUTLINE_MINIMUM_PHOTOS
              ? `Finish at least ${OUTLINE_MINIMUM_PHOTOS} photos on the Add data tab. ${dataset.labelled} done.`
              : `Outline each thing at least ${OUTLINE_MINIMUM_OUTLINES} times. Short: ${short.join(", ")}.`}
        </p>
      )}
    </section>
  );
}

// -------------------------------------------------------------------------------- trying

function Areas({ found, categories }: { found: FoundOutlines; categories: string[] }) {
  return (
    <ul className="find-counts" aria-live="polite" aria-label="How much of the photo each thing covers">
      {categories.map((name) => (
        <li key={name}>
          <span className="find-swatch" style={{ ["--box" as string]: colourOf(categories, name) }} aria-hidden="true" />
          <span className="try-name">{name}</span>
          <strong className="find-count">{(found.areas[name] ?? 0) < 0.005 && (found.areas[name] ?? 0) > 0 ? "<1%" : percent(found.areas[name] ?? 0)}</strong>
        </li>
      ))}
    </ul>
  );
}

function Mask({ found, mirrored = false }: { found: FoundOutlines; mirrored?: boolean }) {
  return <img className={`outline-mask${mirrored ? " is-mirrored" : ""}`} src={`data:image/png;base64,${found.mask}`} alt="" />;
}

function TryPhoto({ jobId, file, src, categories }: { jobId: string; file: File; src: string; categories: string[] }) {
  const [found, setFound] = useState<FoundOutlines | null>(null);
  const [problem, setProblem] = useState<string | null>(null);
  useEffect(() => {
    let live = true;
    api.findOutlines(jobId, file).then((reply) => live && setFound(reply)).catch((error) => live && setProblem(message(error, "The model couldn't check this photo.")));
    return () => { live = false; };
  }, [jobId, file]);
  return (
    <div className="webcam-layout field-note">
      <div className="find-stage is-result">
        <img src={src} alt="The photo you chose" />
        {found && <Mask found={found} />}
      </div>
      <div className="webcam-side">
        {!found && !problem && <p className="muted small">Looking…</p>}
        {found && <><p className="small"><strong>How much of the photo each covers</strong></p><Areas found={found} categories={categories} /></>}
        {problem && <p className="small tone-bad" role="alert">{problem}</p>}
      </div>
    </div>
  );
}

function LiveOutlines({ jobId, categories, beginner }: { jobId: string; categories: string[]; beginner: boolean }) {
  const [open, setOpen] = useState(false);
  const camera = useCamera(open);
  const [found, setFound] = useState<FoundOutlines | null>(null);
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
          const next = await api.findOutlines(jobId, file);
          if (!live) return;
          setFound(next);
          setProblem(null);
          frames += 1;
          const now = performance.now();
          if (now - since > 1000) { setRate(frames / ((now - since) / 1000)); frames = 0; since = now; }
        } catch (error) {
          if (!live) return;
          setProblem(message(error, "The model couldn't answer."));
          await new Promise((done) => setTimeout(done, 1500));
        }
        await new Promise((done) => setTimeout(done, 60));
      }
    };
    void loop();
    return () => { live = false; };
  }, [open, camera.ready, camera.video, jobId]);

  useEffect(() => { if (!open) { setFound(null); setRate(null); } }, [open]);

  return (
    <div className="webcam-try field-note">
      <button className="button button-small" aria-pressed={open} onClick={() => setOpen(!open)}>{open ? "Turn camera off" : "Try it with your webcam"}</button>
      {open && (
        <div className="webcam-layout field-note">
          <CameraView {...camera}>{found && <Mask found={found} mirrored />}</CameraView>
          <div className="webcam-side">
            {!found && camera.ready && !problem && <p className="muted small">Loading the model…</p>}
            {found && <Areas found={found} categories={categories} />}
            {problem && <p className="small tone-bad" role="alert">{problem}</p>}
            {found && (
              <p className="muted small">
                {beginner
                  ? "Move things around and try other backgrounds. Where it outlines badly, take photos like that and outline them."
                  : `About ${Math.round(rate ?? 0)} answers a second, on this computer.`}
              </p>
            )}
          </div>
        </div>
      )}
    </div>
  );
}

export function OutlineTry({ model, beginner }: { model: SavedModel; beginner: boolean }) {
  const categories = useMemo(() => Object.keys(model.categories), [model.categories]);
  const [chosen, setChosen] = useState<{ file: File; src: string } | null>(null);
  useEffect(() => () => { if (chosen) URL.revokeObjectURL(chosen.src); }, [chosen]);
  return (
    <section className="card">
      <h2>Try it</h2>
      <p className="muted small">{model.projectName} · outlines {categories.join(", ")} · data v{model.datasetVersion ?? "?"}</p>
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
      {chosen && <TryPhoto key={chosen.src} jobId={model.id} file={chosen.file} src={chosen.src} categories={categories} />}
      <LiveOutlines jobId={model.id} categories={categories} beginner={beginner} />
      <ModelApp jobId={model.id} beginner={beginner} example={`beep when ${categories[0] ?? "a leaf"} covers a fifth of the view`} />
    </section>
  );
}

// ------------------------------------------------------------------------------ results

function CheckedPhotoView({ photo, jobId, datasetId, categories }: { photo: OutlinedPhoto; jobId: string; datasetId: string; categories: string[] }) {
  const [missing, setMissing] = useState(false);
  if (missing) return null;
  const parts = [
    photo.missed.length ? `missed ${photo.missed.join(", ")}` : "",
    photo.extra.length ? `outlined ${photo.extra.join(", ")} where there was none` : "",
  ].filter(Boolean);
  return (
    <figure className="find-checked">
      <div className="find-stage is-result">
        <img src={api.unsortedImageUrl(photo.file, datasetId)} alt="" loading="lazy" onError={() => setMissing(true)} />
        <img className="outline-mask" src={api.outlineReviewUrl(jobId, photo.mask)} alt="" loading="lazy" />
        <OutlineLayer shapes={photo.real} categories={categories} style="expected" />
      </div>
      <figcaption className="small">{Math.round(photo.score * 100)}% match{parts.length ? ` · ${parts.join("; ")}` : ""}</figcaption>
    </figure>
  );
}

export function OutlineResults({ model, datasetId, beginner }: { model: SavedModel; datasetId: string; beginner: boolean }) {
  const metrics = model.metrics as unknown as OutlineMetrics | null;
  if (!metrics || metrics.task !== "outline") return null;
  const categories = Object.keys(model.categories);
  const good = metrics.good ?? 0.7;
  const worst = metrics.photosChecked.filter((photo) => photo.score < good);
  return (
    <section className="card">
      <h2>How well it outlines</h2>
      <p className="muted small">Checked on {plural(metrics.photos, "held-back photo")} it never learned from, comparing its outlines with yours.</p>
      <dl className="unusual-stats">
        <div><dt>{beginner ? "Match" : "Mean IoU"}</dt><dd>{percent(metrics.meanIoU)}</dd><span className="muted small">{beginner ? "how much its outlines and yours overlap" : "intersection over union, averaged over things"}</span></div>
        <div><dt>Photos outlined well</dt><dd>{metrics.allRight} of {metrics.photos}</dd><span className="muted small">at least {Math.round(good * 100)}% match</span></div>
        {!beginner && <div><dt>Pixel accuracy</dt><dd>{percent(metrics.pixelAccuracy)}</dd><span className="muted small">spots given the right thing, background included</span></div>}
      </dl>
      {categories.length > 1 && (
        <table className="find-table field-note">
          <thead><tr><th scope="col">Thing</th><th scope="col">Match</th><th scope="col">Photos outlined well</th></tr></thead>
          <tbody>
            {categories.map((name) => {
              const row = metrics.perCategory[name] ?? { iou: null, photos: 0, good: 0 };
              return <tr key={name}><th scope="row">{name}</th><td>{percent(row.iou)}</td><td>{row.good} of {row.photos}</td></tr>;
            })}
          </tbody>
        </table>
      )}
      {metrics.photos < 10 && <p className="muted small field-note">That's only a few photos, so these numbers are rough. More outlined photos give a steadier score.</p>}
      {worst.length > 0 && (
        <>
          <h3 className="small field-note">Where it outlined badly</h3>
          <p className="muted small heatmap-key">
            <span className="find-key is-expected" aria-hidden="true" /> your outline
            <span className="outline-key" aria-hidden="true" /> what it outlined
          </p>
          <div className="find-checked-grid">
            {worst.slice(0, 12).map((photo) => <CheckedPhotoView key={photo.file} photo={photo} jobId={model.id} datasetId={datasetId} categories={categories} />)}
          </div>
          <p className="muted small field-note">Add more photos like these, with outlines, and train again.</p>
        </>
      )}
      {worst.length === 0 && <p className="muted small field-note">Every held-back photo was outlined well. Try it on harder photos above to find its limits.</p>}
    </section>
  );
}
