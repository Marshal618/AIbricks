import { useCallback, useEffect, useRef, useState, type KeyboardEvent, type ReactNode } from "react";
import * as api from "./api";
import { HeatmapKey, HeatmapOverlay } from "./Explain";
import type { Heatmap, PredictAnswer } from "./types";

const CAPTURES_PER_SECOND = 4;
const MAX_UPLOADS_AT_ONCE = 3;
const FRAME_SIDE = 512; // Big enough for any model here, small enough to send quickly.

function cameraProblem(error: unknown) {
  const name = error instanceof DOMException ? error.name : "";
  if (name === "NotAllowedError" || name === "SecurityError") return "The browser wasn't allowed to use the camera. Allow it in the address bar, then try again.";
  if (name === "NotFoundError" || name === "OverconstrainedError") return "No camera was found.";
  if (name === "NotReadableError") return "The camera is being used by another app. Close it, then try again.";
  return "The camera couldn't start.";
}

/** Starts the camera while `on` is true, and always stops it again, so the light never stays on. */
export function useCamera(on: boolean) {
  const video = useRef<HTMLVideoElement>(null);
  const [ready, setReady] = useState(false);
  const [problem, setProblem] = useState<string | null>(null);

  useEffect(() => {
    if (!on) return;
    if (!navigator.mediaDevices?.getUserMedia) {
      setProblem("This browser can't use a camera here.");
      return;
    }
    let stream: MediaStream | null = null;
    let live = true;
    setProblem(null);
    navigator.mediaDevices.getUserMedia({ video: { width: { ideal: 640 }, height: { ideal: 480 } }, audio: false })
      .then(async (next) => {
        if (!live) { next.getTracks().forEach((track) => track.stop()); return; }
        stream = next;
        if (video.current) {
          video.current.srcObject = next;
          await video.current.play().catch(() => undefined);
          setReady(true);
        }
      })
      .catch((error) => live && setProblem(cameraProblem(error)));
    return () => {
      live = false;
      setReady(false);
      stream?.getTracks().forEach((track) => track.stop());
    };
  }, [on]);

  return { video, ready, problem };
}

/** The current frame as a JPEG. The preview is mirrored like a mirror, but the photo isn't. */
export function grab(video: HTMLVideoElement): Promise<File | null> {
  const scale = Math.min(1, FRAME_SIDE / Math.max(video.videoWidth, video.videoHeight));
  const canvas = document.createElement("canvas");
  canvas.width = Math.round(video.videoWidth * scale);
  canvas.height = Math.round(video.videoHeight * scale);
  if (!canvas.width || !canvas.height) return Promise.resolve(null);
  canvas.getContext("2d")?.drawImage(video, 0, 0, canvas.width, canvas.height);
  return new Promise((resolve) => canvas.toBlob(
    (blob) => resolve(blob ? new File([blob], "webcam.jpg", { type: "image/jpeg" }) : null), "image/jpeg", 0.9,
  ));
}

export function CameraView({ video, ready, problem, children }: ReturnType<typeof useCamera> & { children?: ReactNode }) {
  return (
    <div className="webcam-frame">
      <video ref={video} className="webcam-video" muted playsInline aria-label="Camera preview" />
      {children}
      {!ready && !problem && <span className="webcam-note muted small">Starting the camera…</span>}
      {problem && <span className="webcam-note small" role="alert">{problem}</span>}
    </div>
  );
}

// ------------------------------------------------------------------ taking photos

export function WebcamCapture({ datasetId, categories, onAdded, onDone }: {
  datasetId: string; categories: string[]; onAdded: (category: string) => void; onDone: () => void;
}) {
  const [open, setOpen] = useState(false);
  const camera = useCamera(open);
  const [recording, setRecording] = useState<string | null>(null);
  const [taken, setTaken] = useState<Record<string, number>>({});
  const [problem, setProblem] = useState<string | null>(null);
  const timer = useRef<number | null>(null);
  const inFlight = useRef(0);

  const snap = useCallback(async (category: string) => {
    const video = camera.video.current;
    if (!video || !camera.ready || inFlight.current >= MAX_UPLOADS_AT_ONCE) return;
    const file = await grab(video);
    if (!file) return;
    inFlight.current += 1;
    try {
      const reply = await api.uploadImage(category, file, datasetId);
      if (!reply.duplicate) {
        setTaken((current) => ({ ...current, [category]: (current[category] ?? 0) + 1 }));
        onAdded(category);
      }
    } catch (error) {
      setProblem(error instanceof Error ? error.message : "The photo couldn't be saved.");
      stop();
    } finally {
      inFlight.current -= 1;
    }
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [camera.ready, datasetId, onAdded]);

  const start = (category: string) => {
    if (timer.current !== null || !camera.ready) return;
    setProblem(null);
    setRecording(category);
    void snap(category);
    timer.current = window.setInterval(() => void snap(category), 1000 / CAPTURES_PER_SECOND);
  };

  function stop() {
    if (timer.current === null) return;
    window.clearInterval(timer.current);
    timer.current = null;
    setRecording(null);
    onDone();
  }

  useEffect(() => () => { if (timer.current !== null) window.clearInterval(timer.current); }, []);
  useEffect(() => { if (!open) stop(); }); // Closing the camera ends any recording.

  const keyDown = (event: KeyboardEvent, category: string) => {
    if ((event.key === " " || event.key === "Enter") && !event.repeat) {
      event.preventDefault();
      start(category);
    }
  };

  const total = Object.values(taken).reduce((sum, count) => sum + count, 0);

  return (
    <section className="card">
      <div className="card-head">
        <h2>Take photos with your webcam</h2>
        <button className="button button-small" aria-pressed={open} onClick={() => setOpen(!open)}>{open ? "Turn camera off" : "Use webcam"}</button>
      </div>
      <p className="muted small">
        {open
          ? "Hold a button to take 4 photos a second straight into that category. Move the object around and change the angle, so the model learns it from every side."
          : "Hold things up to the camera to collect labelled photos quickly. The camera only runs while this is open, and photos stay on this computer."}
      </p>
      {open && (
        <div className="webcam-layout field-note">
          <CameraView {...camera} />
          <div className="webcam-side">
            {categories.length === 0 && <p className="notice">Add a category first.</p>}
            <div className="webcam-buttons">
              {categories.map((name) => (
                <button
                  key={name}
                  className={`button webcam-hold${recording === name ? " is-recording" : ""}`}
                  disabled={!camera.ready || (recording !== null && recording !== name)}
                  aria-label={`Hold to take photos of ${name}`}
                  onPointerDown={(event) => { event.currentTarget.setPointerCapture(event.pointerId); start(name); }}
                  onPointerUp={stop}
                  onPointerCancel={stop}
                  onKeyDown={(event) => keyDown(event, name)}
                  onKeyUp={stop}
                  onBlur={stop}
                  onContextMenu={(event) => event.preventDefault()}
                >
                  <span>{recording === name ? "Recording" : "Hold for"} {name}</span>
                  <span className="webcam-count">{taken[name] ?? 0}</span>
                </button>
              ))}
            </div>
            <p className="muted small" role="status">
              {total ? `${total} new ${total === 1 ? "photo" : "photos"} added this time.` : "Or press and hold Space on a focused button."}
            </p>
            {problem && <p className="small tone-bad" role="alert">{problem}</p>}
          </div>
        </div>
      )}
    </section>
  );
}

// ------------------------------------------------------------------ trying a model

export function WebcamTry({ jobId, beginner }: { jobId: string; beginner: boolean }) {
  const [open, setOpen] = useState(false);
  const camera = useCamera(open);
  const [answers, setAnswers] = useState<PredictAnswer[] | null>(null);
  const [problem, setProblem] = useState<string | null>(null);
  const [rate, setRate] = useState<number | null>(null);
  const [looking, setLooking] = useState(false);
  const [heatmap, setHeatmap] = useState<Heatmap | null>(null);
  const lookingNow = useRef(looking);
  lookingNow.current = looking;

  // Asks about one frame at a time, so slow answers never pile up.
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
          const explained = lookingNow.current;
          const next = explained ? await api.predictLiveExplained(jobId, file) : { answers: await api.predictLive(jobId, file), heatmap: null };
          if (!live) return;
          setAnswers(next.answers);
          setHeatmap(lookingNow.current ? next.heatmap : null);
          setProblem(null);
          frames += 1;
          const now = performance.now();
          if (now - since > 1000) { setRate(frames / ((now - since) / 1000)); frames = 0; since = now; }
        } catch (error) {
          if (!live) return;
          setProblem(error instanceof Error ? error.message : "The model couldn't answer.");
          await new Promise((done) => setTimeout(done, 1500));
        }
        await new Promise((done) => setTimeout(done, 80));
      }
    };
    void loop();
    return () => { live = false; };
  }, [open, camera.ready, camera.video, jobId]);

  useEffect(() => { if (!open) { setAnswers(null); setRate(null); setHeatmap(null); } }, [open]);
  useEffect(() => { if (!looking) setHeatmap(null); }, [looking]);
  const video = camera.video.current;
  const aspect = video?.videoWidth ? video.videoWidth / video.videoHeight : 4 / 3;

  return (
    <div className="webcam-try field-note">
      <button className="button button-small" aria-pressed={open} onClick={() => setOpen(!open)}>{open ? "Turn camera off" : "Try it with your webcam"}</button>
      {open && (
        <div className="webcam-layout field-note">
          <CameraView {...camera}>{heatmap && <HeatmapOverlay heatmap={heatmap} aspect={aspect} mirrored />}</CameraView>
          <div className="webcam-side">
            <label className="check-row small">
              <input type="checkbox" checked={looking} onChange={(event) => setLooking(event.target.checked)} />
              Show where it looks
            </label>
            {!answers && camera.ready && !problem && <p className="muted small">Loading the model…</p>}
            {answers && (
              <ul className="try-answers" aria-live="polite" aria-label="The model's live answers">
                {answers.map((answer, index) => (
                  <li key={answer.category} className={index === 0 ? "is-top" : ""}>
                    <span className="try-name">{answer.category}</span>
                    <span className="try-track" aria-hidden="true"><span style={{ width: `${answer.chance * 100}%` }} /></span>
                    <span className="try-value">{(answer.chance * 100).toFixed(0)}%</span>
                  </li>
                ))}
              </ul>
            )}
            {problem && <p className="small tone-bad" role="alert">{problem}</p>}
            {heatmap && <HeatmapKey heatmap={heatmap} beginner={beginner} />}
            {answers && (
              <p className="muted small">
                {beginner
                  ? "Try things it hasn't seen, and different backgrounds and lighting. If it's confidently wrong, take more photos like that and train again."
                  : `About ${Math.round(rate ?? 0)} answers a second. Frames are sent to this computer's API only.`}
              </p>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
