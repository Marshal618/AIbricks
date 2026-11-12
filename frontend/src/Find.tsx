import { useCallback, useEffect, useMemo, useRef, useState, type ChangeEvent, type PointerEvent as ReactPointerEvent } from "react";
import * as api from "./api";
import type { Box, BoxedPhoto, CheckedPhoto, DatasetSummary, FoundBoxes, SavedModel } from "./types";
import { Why } from "./Ui";
import { CameraView, grab, useCamera } from "./Webcam";
import { ModelApp } from "./ModelApp";

export const FIND_MINIMUM_PHOTOS = 10;
export const FIND_MINIMUM_BOXES = 5;
export const FIND_SUGGESTED = 30;
const UPLOAD_WORKERS = 4;
const STRIP_LIMIT = 80;
// Distinct colours for each thing. Every box also carries its thing's name, so colour is never the only clue.
export const COLOURS = ["#e8590c", "#1c7ed6", "#2f9e44", "#ae3ec9", "#c2255c", "#0c8599", "#f08c00", "#5c7cfa", "#66a80f"];

function message(error: unknown, fallback: string) {
  return error instanceof Error ? error.message : fallback;
}

export function colourOf(categories: string[], name: string) {
  const index = categories.indexOf(name);
  return COLOURS[(index < 0 ? 0 : index) % COLOURS.length];
}

export function plural(count: number, one: string, many = `${one}s`) {
  return `${count} ${count === 1 ? one : many}`;
}

/** True once a dataset has enough boxed photos to train on. */
export function findReady(dataset: DatasetSummary) {
  return dataset.categories.length >= 1 && dataset.labelled >= FIND_MINIMUM_PHOTOS
    && dataset.categories.every((name) => (dataset.counts[name] ?? 0) >= FIND_MINIMUM_BOXES);
}

/** Boxes drawn over whatever fills their positioned parent. */
export function BoxLayer({ boxes, categories, selected = null, onSelect, onRemove, mirrored = false, style = "solid" }: {
  boxes: Box[]; categories: string[]; selected?: number | null; onSelect?: (index: number) => void; onRemove?: (index: number) => void;
  mirrored?: boolean; style?: "solid" | "dashed" | "expected";
}) {
  return (
    <>
      {boxes.map((box, index) => {
        const colour = style === "expected" ? "#ffffff" : colourOf(categories, box.category);
        const left = mirrored ? 1 - box.x - box.w : box.x;
        return (
          <div
            key={index}
            className={`find-box is-${style}${selected === index ? " is-selected" : ""}`}
            style={{ left: `${left * 100}%`, top: `${box.y * 100}%`, width: `${box.w * 100}%`, height: `${box.h * 100}%`, ["--box" as string]: colour }}
            onPointerDown={onSelect ? (event) => { event.stopPropagation(); onSelect(index); } : undefined}
          >
            {style !== "expected" && (
              <span className="find-box-tag">
                {box.category}{box.chance !== undefined ? ` ${Math.round(box.chance * 100)}%` : ""}
                {onRemove && (
                  <button type="button" aria-label={`Remove this ${box.category} box`} onPointerDown={(event) => event.stopPropagation()} onClick={() => onRemove(index)}>×</button>
                )}
              </span>
            )}
          </div>
        );
      })}
    </>
  );
}

// ----------------------------------------------------------------- adding photos and drawing boxes

/** Adds photos to a dataset whose photos get boxes or outlines: files, a folder, or the webcam. */
export function ShapePhotoAdder({ datasetId, beginner, onAdded, onProblem, title = "Photos to draw boxes on", shapes = "boxes", verb = "find" }: {
  datasetId: string; beginner: boolean; onAdded: () => Promise<void>; onProblem: (text: string) => void; title?: string; shapes?: string; verb?: string;
}) {
  const [upload, setUpload] = useState<{ done: number; total: number; added: number; rejected: string[] } | null>(null);
  const [cameraOpen, setCameraOpen] = useState(false);
  const [taken, setTaken] = useState(0);
  const camera = useCamera(cameraOpen);
  const filesInput = useRef<HTMLInputElement>(null);
  const folderInput = useRef<HTMLInputElement>(null);

  const addFiles = async (event: ChangeEvent<HTMLInputElement>) => {
    const files = Array.from(event.target.files ?? []).filter((file) => !file.name.startsWith("."));
    event.target.value = "";
    if (!files.length) return;
    const state = { done: 0, total: files.length, added: 0, rejected: [] as string[] };
    setUpload({ ...state });
    const list = [...files];
    const worker = async () => {
      for (let file = list.shift(); file; file = list.shift()) {
        try {
          if (!(await api.addUnsortedImage(file, datasetId)).duplicate) state.added += 1;
        } catch (error) {
          state.rejected.push(`${file.name}: ${message(error, "could not be read.")}`);
        }
        state.done += 1;
        setUpload({ ...state });
      }
    };
    await Promise.all(Array.from({ length: UPLOAD_WORKERS }, worker));
    await onAdded();
  };

  const snap = async () => {
    const video = camera.video.current;
    if (!video || !camera.ready) return;
    const file = await grab(video);
    if (!file) return;
    try {
      if (!(await api.addUnsortedImage(file, datasetId)).duplicate) setTaken((count) => count + 1);
      await onAdded();
    } catch (error) {
      onProblem(message(error, "The photo couldn't be saved."));
    }
  };

  const uploading = upload !== null && upload.done < upload.total;
  return (
    <>
      <h2 className="section-title">Add photos</h2>
      <section className="card">
        <div className="card-head">
          <h2>{title}</h2>
          <div className="label-buttons">
            <button className="button button-small" aria-pressed={cameraOpen} onClick={() => setCameraOpen(!cameraOpen)}>{cameraOpen ? "Turn camera off" : "Use the webcam"}</button>
            <button className="button button-small" disabled={uploading} onClick={() => folderInput.current?.click()}>Choose a folder</button>
            <button className="button button-small" disabled={uploading} onClick={() => filesInput.current?.click()}>Add images</button>
          </div>
          <input ref={filesInput} className="visually-hidden" type="file" multiple accept="image/jpeg,image/png,image/bmp,image/webp" aria-label="Add photos" tabIndex={-1} onChange={(event) => void addFiles(event)} />
          <input ref={folderInput} className="visually-hidden" type="file" aria-label="Choose a folder of photos" tabIndex={-1} onChange={(event) => void addFiles(event)} {...{ webkitdirectory: "" }} />
        </div>
        <p className="muted small">
          {beginner
            ? "Take photos the way the model will see things later: different places, light, distances, and some with several things in. Include a few with nothing in them."
            : `Photos are copied to this computer only. Empty photos marked “Nothing here” teach it what not to ${verb}.`}
        </p>
        {cameraOpen && (
          <div className="webcam-layout field-note">
            <CameraView {...camera} />
            <div className="webcam-side">
              <button className="button button-primary" disabled={!camera.ready} onClick={() => void snap()}>Take photo</button>
              <p className="muted small">{taken ? `${plural(taken, "photo")} taken. Move things around between photos.` : `Each photo joins the list below, ready for ${shapes}.`}</p>
            </div>
          </div>
        )}
        {upload && uploading && (
          <div className="field-note" role="status">
            <progress className="train-progress" max={upload.total} value={upload.done} aria-label="Photos copied" />
            <p className="small muted">Checking and copying {upload.done} of {upload.total}…</p>
          </div>
        )}
        {upload && !uploading && (
          <p className="small muted field-note" role="status">
            Added {plural(upload.added, "photo")}.{upload.rejected.length ? ` ${upload.rejected.length} couldn't be read: ${upload.rejected.slice(0, 3).join("; ")}` : ""}
          </p>
        )}
      </section>

    </>
  );
}


export function FindLabels({ dataset, beginner, onChanged }: { dataset: DatasetSummary; beginner: boolean; onChanged: () => void }) {
  const datasetId = dataset.id;
  const categories = dataset.categories;
  const [photos, setPhotos] = useState<BoxedPhoto[] | null>(null);
  const [currentId, setCurrentId] = useState<string | null>(null);
  const [draft, setDraft] = useState<Box[]>([]);
  const [suggested, setSuggested] = useState(false);
  const [thing, setThing] = useState(categories[0] ?? "");
  const [selected, setSelected] = useState<number | null>(null);
  const [drawing, setDrawing] = useState<{ x: number; y: number; x2: number; y2: number } | null>(null);
  const [problem, setProblem] = useState<string | null>(null);
  const [helper, setHelper] = useState<string | null>(null);
  const [helpOn, setHelpOn] = useState(true);
  const stage = useRef<HTMLDivElement>(null);

  useEffect(() => { if (!categories.includes(thing)) setThing(categories[0] ?? ""); }, [categories, thing]);

  const load = useCallback(async (keep?: string | null) => {
    const reply = await api.fetchBoxes(datasetId);
    setPhotos(reply.photos);
    const next = reply.photos.find((photo) => photo.id === keep) ?? reply.photos.find((photo) => photo.boxes === null) ?? reply.photos[0] ?? null;
    setCurrentId(next?.id ?? null);
    return reply.photos;
  }, [datasetId]);

  useEffect(() => { load().catch((error) => setProblem(message(error, "The photos couldn't load."))); }, [load]);

  // A finished model on this data can draw the first boxes, which you then fix.
  useEffect(() => {
    api.fetchDatasetModels(datasetId)
      .then((rows) => setHelper(rows.find((row) => row.task === "detect" && row.status === "finished" && row.hasModel)?.id ?? null))
      .catch(() => setHelper(null));
  }, [datasetId, dataset.updatedAt]);

  const current = photos?.find((photo) => photo.id === currentId) ?? null;
  useEffect(() => {
    setSelected(null);
    setDrawing(null);
    setSuggested(false);
    if (!current) { setDraft([]); return; }
    if (current.boxes !== null || !helper || !helpOn) { setDraft(current.boxes ?? []); return; }
    setDraft([]);
    let live = true;
    fetch(api.unsortedImageUrl(current.id, datasetId))
      .then((reply) => reply.blob())
      .then((blob) => api.findBoxes(helper, blob))
      .then((found) => {
        if (!live) return;
        const usable = found.boxes.filter((box) => categories.includes(box.category));
        setDraft(usable);
        setSuggested(usable.length > 0);
      })
      .catch(() => undefined);
    return () => { live = false; };
  // Only a change of photo, or of the helper, should redraw the suggestions.
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [current?.id, current?.boxes, helper, helpOn, datasetId]);

  const done = photos?.filter((photo) => photo.boxes !== null).length ?? 0;
  const total = photos?.length ?? 0;

  const save = useCallback(async (boxes: Box[]) => {
    if (!current) return;
    setProblem(null);
    try {
      const saved = await api.saveBoxes(datasetId, current.id, boxes);
      const updated = (photos ?? []).map((photo) => photo.id === current.id ? { ...photo, boxes: saved.boxes } : photo);
      setPhotos(updated);
      const index = updated.findIndex((photo) => photo.id === current.id);
      const next = [...updated.slice(index + 1), ...updated.slice(0, index)].find((photo) => photo.boxes === null);
      setCurrentId(next?.id ?? current.id);
      onChanged();
    } catch (error) {
      setProblem(message(error, "The boxes couldn't be saved."));
    }
  }, [current, datasetId, photos, onChanged]);

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

  const point = (event: ReactPointerEvent) => {
    const area = stage.current!.getBoundingClientRect();
    return {
      x: Math.min(1, Math.max(0, (event.clientX - area.left) / area.width)),
      y: Math.min(1, Math.max(0, (event.clientY - area.top) / area.height)),
    };
  };

  const startBox = (event: ReactPointerEvent<HTMLDivElement>) => {
    if (!thing || event.button !== 0) return;
    event.currentTarget.setPointerCapture(event.pointerId);
    const at = point(event);
    setSelected(null);
    setDrawing({ x: at.x, y: at.y, x2: at.x, y2: at.y });
  };
  const moveBox = (event: ReactPointerEvent<HTMLDivElement>) => {
    if (!drawing) return;
    const at = point(event);
    setDrawing({ ...drawing, x2: at.x, y2: at.y });
  };
  const endBox = () => {
    if (!drawing) return;
    const box = { category: thing, x: Math.min(drawing.x, drawing.x2), y: Math.min(drawing.y, drawing.y2), w: Math.abs(drawing.x2 - drawing.x), h: Math.abs(drawing.y2 - drawing.y) };
    setDrawing(null);
    if (box.w > 0.01 && box.h > 0.01) {
      setDraft((boxes) => [...boxes, box]);
      setSelected(draft.length);
    }
  };

  const removeBox = useCallback((index: number) => {
    setDraft((boxes) => boxes.filter((_, at) => at !== index));
    setSelected(null);
  }, []);

  // Keys: 1 to 9 pick a thing, Enter saves, Delete removes the chosen box.
  useEffect(() => {
    const onKey = (event: KeyboardEvent) => {
      const target = event.target as HTMLElement;
      if (target.closest("input, textarea, select, [contenteditable]") || event.metaKey || event.ctrlKey || event.altKey) return;
      if (/^[1-9]$/.test(event.key) && categories[Number(event.key) - 1]) {
        const name = categories[Number(event.key) - 1];
        setThing(name);
        if (selected !== null) setDraft((boxes) => boxes.map((box, at) => at === selected ? { ...box, category: name } : box));
      } else if (event.key === "Enter" && current && !target.closest("button")) {
        event.preventDefault();
        void save(draft);
      } else if ((event.key === "Delete" || event.key === "Backspace") && selected !== null) {
        event.preventDefault();
        removeBox(selected);
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [categories, selected, current, draft, save, removeBox]);

  const shown = drawing && { x: Math.min(drawing.x, drawing.x2), y: Math.min(drawing.y, drawing.y2), w: Math.abs(drawing.x2 - drawing.x), h: Math.abs(drawing.y2 - drawing.y) };
  const strip = useMemo(() => (photos ?? []).slice(0, STRIP_LIMIT), [photos]);

  return (
    <>
      {problem && <div className="result result-fail" role="alert"><p>{problem}</p></div>}
      <ShapePhotoAdder datasetId={datasetId} beginner={beginner} onAdded={async () => { await load(currentId); onChanged(); }} onProblem={setProblem} />

      <h2 className="section-title">Draw boxes</h2>
      <section className="card find-labeller">
        <div className="card-head">
          <h2>{total ? `${done} of ${total} photos done` : "Draw boxes"}</h2>
          {helper && (
            <label className="check-row small">
              <input type="checkbox" checked={helpOn} onChange={(event) => setHelpOn(event.target.checked)} />
              Let the model draw first
            </label>
          )}
        </div>
        {!categories.length && <p className="muted small">Name the things to find above first.</p>}
        {categories.length > 0 && total === 0 && <p className="muted small">Add some photos above, then draw a box around each thing in them.</p>}
        {categories.length > 0 && current && (
          <>
            <p className="muted small">
              Drag a box tightly around every {categories.length === 1 ? categories[0] : "thing"} you can see, even small or partly hidden ones. Then save.
            </p>
            <fieldset className="chip-group field-note">
              <legend className="small">Drawing</legend>
              {categories.map((name, index) => (
                <label key={name} className={`chip find-chip${thing === name ? " is-on" : ""}`} style={{ ["--box" as string]: colourOf(categories, name) }}>
                  <input type="radio" name="find-thing" checked={thing === name} onChange={() => {
                    setThing(name);
                    if (selected !== null) setDraft((boxes) => boxes.map((box, at) => at === selected ? { ...box, category: name } : box));
                  }} />
                  <span className="find-swatch" aria-hidden="true" />{name}{index < 9 && <kbd>{index + 1}</kbd>}
                </label>
              ))}
            </fieldset>
            <div
              ref={stage}
              className={`find-stage field-note${drawing ? " is-drawing" : ""}`}
              onPointerDown={startBox}
              onPointerMove={moveBox}
              onPointerUp={endBox}
              onPointerCancel={() => setDrawing(null)}
              role="application"
              aria-label={`Photo to draw ${thing} boxes on. ${plural(draft.length, "box", "boxes")} so far.`}
            >
              <img src={api.unsortedImageUrl(current.id, datasetId)} alt="" draggable={false} />
              <BoxLayer boxes={draft} categories={categories} selected={selected} onSelect={setSelected} onRemove={removeBox} style={suggested ? "dashed" : "solid"} />
              {shown && <div className="find-box is-drawing" style={{ left: `${shown.x * 100}%`, top: `${shown.y * 100}%`, width: `${shown.w * 100}%`, height: `${shown.h * 100}%`, ["--box" as string]: colourOf(categories, thing) }} />}
            </div>
            {suggested && <p className="muted small field-note">The dashed boxes are the model's guesses. Fix or remove any that are wrong, add any it missed, then save.</p>}
            <div className="label-buttons field-note find-actions">
              <button className="button button-primary" onClick={() => void save(draft)} disabled={!draft.length}>
                Save {plural(draft.length, "box", "boxes")} and next <kbd>Enter</kbd>
              </button>
              <button className="button" onClick={() => void save([])}>Nothing to find here</button>
              <span className="tab-gap" />
              {current.boxes !== null && <span className="muted small">Done{current.boxes.length ? `, ${plural(current.boxes.length, "box", "boxes")}` : ", nothing here"}</span>}
              <button className="button button-small button-quiet" onClick={() => void removePhoto()}>Remove photo</button>
            </div>
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

export function FindTrainer({ dataset, beginner, running, busyElsewhere, onStart, onStop }: {
  dataset: DatasetSummary; beginner: boolean; running: SavedModel | null; busyElsewhere: boolean; onStart: (detector: "accurate" | "fast") => void; onStop: (id: string) => void;
}) {
  const [detector, setDetector] = useState<"accurate" | "fast">("accurate");
  const ready = findReady(dataset);
  const short = dataset.categories.filter((name) => (dataset.counts[name] ?? 0) < FIND_MINIMUM_BOXES);
  return (
    <section className="card">
      <div className="card-head">
        <h2>Teach it to find {dataset.categories.length === 1 ? dataset.categories[0] : "your things"}</h2>
        {running
          ? <button className="button button-small" onClick={() => onStop(running.id)}>Stop</button>
          : <button className="button button-small button-primary" disabled={!ready || busyElsewhere} onClick={() => onStart(detector)}>Start</button>}
      </div>
      <p className="muted small">It starts from a model that can already find 80 everyday kinds of thing, and learns yours from your boxes.</p>
      {!running && (
        <fieldset className="dataset-kind find-detector field-note">
          <legend className="small">Which detector?</legend>
          <label><input type="radio" name={`detector-${dataset.id}`} checked={detector === "accurate"} onChange={() => setDetector("accurate")} />
            <span><strong>Accurate</strong> <span className="muted small">Best at small and crowded things. For this computer.</span></span></label>
          <label><input type="radio" name={`detector-${dataset.id}`} checked={detector === "fast"} onChange={() => setDetector("fast")} />
            <span><strong>Fast</strong> <span className="muted small">A fifth of the size, and can become a web app for phones. Misses more small things.</span></span></label>
        </fieldset>
      )}
      <Why>
        {beginner
          ? "The starting model learned from over 100,000 photos with boxes, so it already knows what edges, shapes and objects look like. Only its last step is swapped for one that names your things, then the whole model is adjusted on your photos. A fifth of your photos are kept back to check it honestly."
          : "Accurate: Faster R-CNN, MobileNetV3-Large 320 FPN, 19M parameters. Fast: SSDlite320, MobileNetV3-Large, 3.4M parameters, pretrained batch norm kept frozen. Both use COCO weights from torchvision with the class predictor replaced, a full fine-tune with SGD (lr 0.01, momentum 0.9, cosine) and random horizontal flips. The checkpoint with the lowest held-back loss is kept."}
      </Why>
      {running && (
        <div className="field-note" role="status">
          <progress className="train-progress" max={running.stepsTotal || 1} value={running.stepsDone} aria-label="Progress" />
          <p className="muted small">
            {!running.device
              ? "Getting ready. The first run downloads the starting model."
              : `Step ${running.stepsDone} of ${running.stepsTotal || "?"}, learning from ${running.trainCount} photos and checking on ${running.checkCount}, using ${running.device}.`}
          </p>
        </div>
      )}
      {!running && busyElsewhere && <p className="muted small field-note">Another model is training. Only one can train at a time.</p>}
      {!running && !ready && (
        <p className="muted small field-note">
          {!dataset.categories.length
            ? "Name at least one thing to find on the Add data tab."
            : dataset.labelled < FIND_MINIMUM_PHOTOS
              ? `Finish at least ${FIND_MINIMUM_PHOTOS} photos on the Add data tab. ${dataset.labelled} done.`
              : `Draw at least ${FIND_MINIMUM_BOXES} boxes around each thing. Short: ${short.join(", ")}.`}
        </p>
      )}
    </section>
  );
}

// -------------------------------------------------------------------------------- trying

function Counts({ found, categories }: { found: FoundBoxes; categories: string[] }) {
  return (
    <ul className="find-counts" aria-live="polite" aria-label="What it found">
      {categories.map((name) => (
        <li key={name}>
          <span className="find-swatch" style={{ ["--box" as string]: colourOf(categories, name) }} aria-hidden="true" />
          <span className="try-name">{name}</span>
          <strong className="find-count">{found.counts[name] ?? 0}</strong>
        </li>
      ))}
    </ul>
  );
}

function TryPhoto({ jobId, file, src, categories, sure }: { jobId: string; file: File; src: string; categories: string[]; sure: number }) {
  const [found, setFound] = useState<FoundBoxes | null>(null);
  const [problem, setProblem] = useState<string | null>(null);
  useEffect(() => {
    let live = true;
    setFound(null);
    api.findBoxes(jobId, file).then((reply) => live && setFound(reply)).catch((error) => live && setProblem(message(error, "The model couldn't check this photo.")));
    return () => { live = false; };
  }, [jobId, file, sure]);
  return (
    <div className="webcam-layout field-note">
      <div className="find-stage is-result">
        <img src={src} alt="The photo you chose" />
        {found && <BoxLayer boxes={found.boxes} categories={categories} />}
      </div>
      <div className="webcam-side">
        {!found && !problem && <p className="muted small">Looking…</p>}
        {found && <Counts found={found} categories={categories} />}
        {problem && <p className="small tone-bad" role="alert">{problem}</p>}
      </div>
    </div>
  );
}

function LiveBoxes({ jobId, categories, beginner }: { jobId: string; categories: string[]; beginner: boolean }) {
  const [open, setOpen] = useState(false);
  const camera = useCamera(open);
  const [found, setFound] = useState<FoundBoxes | null>(null);
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
          const next = await api.findBoxes(jobId, file);
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
          <CameraView {...camera}>{found && <BoxLayer boxes={found.boxes} categories={categories} mirrored />}</CameraView>
          <div className="webcam-side">
            {!found && camera.ready && !problem && <p className="muted small">Loading the model…</p>}
            {found && <Counts found={found} categories={categories} />}
            {problem && <p className="small tone-bad" role="alert">{problem}</p>}
            {found && (
              <p className="muted small">
                {beginner
                  ? "Move things around, try other backgrounds, and put several in view. If it misses some, take photos like that and draw boxes on them."
                  : `About ${Math.round(rate ?? 0)} answers a second, on this computer.`}
              </p>
            )}
          </div>
        </div>
      )}
    </div>
  );
}

export function FindTry({ model, beginner }: { model: SavedModel; beginner: boolean }) {
  const categories = useMemo(() => Object.keys(model.categories), [model.categories]);
  const [chosen, setChosen] = useState<{ file: File; src: string } | null>(null);
  const [sure, setSure] = useState(0.5);
  const [saved, setSaved] = useState(0.5);
  const [problem, setProblem] = useState<string | null>(null);
  const [detector, setDetector] = useState<"accurate" | "fast" | null>(null);
  useEffect(() => () => { if (chosen) URL.revokeObjectURL(chosen.src); }, [chosen]);
  useEffect(() => {
    api.fetchFindSettings(model.id).then((reply) => { setSure(reply.sure); setSaved(reply.sure); setDetector(reply.detector); }).catch(() => undefined);
  }, [model.id]);

  const keep = async (value: number) => {
    setProblem(null);
    try {
      setSaved((await api.setFindSure(model.id, value)).sure);
    } catch (error) {
      setProblem(message(error, "That couldn't be changed."));
    }
  };

  return (
    <section className="card">
      <h2>Try it</h2>
      <p className="muted small">{model.projectName} · finds {categories.join(", ")}{detector ? ` · ${detector} detector` : ""} · data v{model.datasetVersion ?? "?"}</p>
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
      {chosen && <TryPhoto key={chosen.src} jobId={model.id} file={chosen.file} src={chosen.src} categories={categories} sure={saved} />}
      <LiveBoxes jobId={model.id} categories={categories} beginner={beginner} />
      <div className="field-note">
        <label className="small" htmlFor={`sure-${model.id}`}><strong>How sure before it draws a box</strong></label>
        <div className="unusual-line">
          <span className="small muted">Shows more</span>
          <input id={`sure-${model.id}`} type="range" min={0.2} max={0.95} step={0.05} value={sure} onChange={(event) => setSure(Number(event.target.value))} onPointerUp={() => void keep(sure)} onKeyUp={() => void keep(sure)} />
          <span className="small muted">Only the surest</span>
        </div>
        <p className="muted small">{Math.round(sure * 100)}%{sure === saved ? " · in use for photos, the webcam and downloads" : " · let go to use it"}. Lower it if it misses things; raise it if it boxes things that aren't there.</p>
        {problem && <p className="small tone-bad" role="alert">{problem}</p>}
      </div>
      {detector === "fast" && <ModelApp jobId={model.id} beginner={beginner} example={`beep when it sees a ${categories[0] ?? "cup"}`} />}
      {detector === "accurate" && <p className="muted small field-note">To make a web app for phones, train again with the Fast detector. This one is too big for a web page.</p>}
    </section>
  );
}

// ------------------------------------------------------------------------------ results

function CheckedPhotoView({ photo, datasetId, categories }: { photo: CheckedPhoto; datasetId: string; categories: string[] }) {
  const [missing, setMissing] = useState(false);
  if (missing) return null;
  const parts = [
    photo.missed ? `missed ${photo.missed}` : "",
    photo.falseAlarms ? `${plural(photo.falseAlarms, "false alarm")}` : "",
  ].filter(Boolean);
  return (
    <figure className="find-checked">
      <div className="find-stage is-result">
        <img src={api.unsortedImageUrl(photo.file, datasetId)} alt="" loading="lazy" onError={() => setMissing(true)} />
        <BoxLayer boxes={photo.real} categories={categories} style="expected" />
        <BoxLayer boxes={photo.found} categories={categories} />
      </div>
      <figcaption className="small">{parts.length ? parts.join(", ") : <span className="muted">All found</span>}</figcaption>
    </figure>
  );
}

export function FindResults({ model, datasetId, beginner }: { model: SavedModel; datasetId: string; beginner: boolean }) {
  const metrics = model.metrics;
  if (!metrics || metrics.task !== "detect") return null;
  const categories = Object.keys(model.categories);
  const found = metrics.found ?? 0;
  const missed = metrics.missed ?? 0;
  const falseAlarms = metrics.falseAlarms ?? 0;
  const photos = metrics.photosChecked ?? [];
  const wrong = photos.filter((photo) => photo.missed || photo.falseAlarms);
  return (
    <section className="card">
      <h2>How well it finds things</h2>
      <p className="muted small">Checked on {plural(metrics.photos ?? 0, "held-back photo")} it never learned from, counting only boxes it was at least {Math.round(((metrics as { sure?: number }).sure ?? 0.5) * 100)}% sure of.</p>
      <dl className="unusual-stats">
        <div><dt>Found</dt><dd>{found} of {found + missed}</dd><span className="muted small">things in the photos</span></div>
        <div><dt>False alarms</dt><dd>{falseAlarms}</dd><span className="muted small">boxes around nothing, or the wrong thing</span></div>
        <div><dt>Photos all right</dt><dd>{metrics.allRight ?? 0} of {metrics.photos ?? 0}</dd><span className="muted small">nothing missed, no false alarms</span></div>
        {metrics.map50 !== undefined && metrics.map50 !== null && (
          <div><dt>{beginner ? "Overall" : "mAP@50"}</dt><dd>{Math.round(metrics.map50 * 100)}%</dd><span className="muted small">{beginner ? "the standard detection score, at every sureness" : "mean average precision, IoU ≥ 0.5"}</span></div>
        )}
      </dl>
      {categories.length > 1 && metrics.perCategory && (
        <table className="find-table field-note">
          <thead><tr><th scope="col">Thing</th><th scope="col">Found</th><th scope="col">Missed</th><th scope="col">False alarms</th></tr></thead>
          <tbody>
            {categories.map((name) => {
              const row = metrics.perCategory![name] ?? { found: 0, missed: 0, falseAlarms: 0 };
              return <tr key={name}><th scope="row">{name}</th><td>{row.found}</td><td>{row.missed}</td><td>{row.falseAlarms}</td></tr>;
            })}
          </tbody>
        </table>
      )}
      {(metrics.photos ?? 0) < 10 && <p className="muted small field-note">That's only a few photos, so these numbers are rough. More boxed photos give a steadier score.</p>}
      {wrong.length > 0 && (
        <>
          <h3 className="small field-note">Where it went wrong</h3>
          <p className="muted small heatmap-key">
            <span className="find-key is-expected" aria-hidden="true" /> what you boxed
            <span className="find-key" aria-hidden="true" /> what it found
          </p>
          <div className="find-checked-grid">
            {wrong.slice(0, 12).map((photo) => <CheckedPhotoView key={photo.file} photo={photo} datasetId={datasetId} categories={categories} />)}
          </div>
          <p className="muted small field-note">Add more photos like these, with boxes, and train again.</p>
        </>
      )}
    </section>
  );
}
