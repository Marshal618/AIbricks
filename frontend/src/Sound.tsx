import { useCallback, useEffect, useRef, useState, type ChangeEvent, type KeyboardEvent } from "react";
import * as api from "./api";
import { HeatmapOverlay } from "./Explain";
import type { DatasetSummary, Grid, Heatmap, SavedModel, SoundAnswer, SoundMistake } from "./types";

const RATE = 16000;
const KEPT_SECONDS = 3;
const SHORTEST_SECONDS = 0.25;
const BACKGROUND_SECONDS = 20;
const LISTEN_EVERY_MS = 400;
export const SOUND_MINIMUM = 4;
export const SOUND_SUGGESTED = 20;

function message(error: unknown, fallback: string) {
  return error instanceof Error ? error.message : fallback;
}

function micProblem(error: unknown) {
  const name = error instanceof DOMException ? error.name : "";
  if (name === "NotAllowedError" || name === "SecurityError") return "The browser wasn't allowed to use the microphone. Allow it in the address bar, then try again.";
  if (name === "NotFoundError") return "No microphone was found.";
  if (name === "NotReadableError") return "The microphone is being used by another app. Close it, then try again.";
  return "The microphone couldn't start.";
}

const isBackground = (name: string) => /background|noise|silence|nothing|quiet/i.test(name);

// ------------------------------------------------------------------ the microphone

// Runs on the audio thread and hands every block of samples to the page.
const TAP = `class Tap extends AudioWorkletProcessor {
  process(inputs) { const channel = inputs[0] && inputs[0][0]; if (channel) this.port.postMessage(channel.slice(0)); return true; }
}
registerProcessor("ai-bricks-tap", Tap);`;

function join(chunks: Float32Array[]) {
  const total = chunks.reduce((sum, chunk) => sum + chunk.length, 0);
  const joined = new Float32Array(total);
  let at = 0;
  for (const chunk of chunks) { joined.set(chunk, at); at += chunk.length; }
  return joined;
}

/** Turns the microphone on while `on` is true, keeping the last few seconds, and always turns it off again. */
export function useMicrophone(on: boolean) {
  const [ready, setReady] = useState(false);
  const [problem, setProblem] = useState<string | null>(null);
  const [level, setLevel] = useState(0);
  const rate = useRef(48000);
  const recent = useRef<{ chunks: Float32Array[]; total: number }>({ chunks: [], total: 0 });
  const taking = useRef<Float32Array[] | null>(null);

  useEffect(() => {
    if (!on) return;
    if (!navigator.mediaDevices?.getUserMedia || typeof AudioWorkletNode === "undefined") {
      setProblem("This browser can't use a microphone here.");
      return;
    }
    let live = true;
    let stream: MediaStream | null = null;
    let context: AudioContext | null = null;
    setProblem(null);
    (async () => {
      try {
        // The model should hear the room as it is, not with the browser's call-cleaning filters.
        stream = await navigator.mediaDevices.getUserMedia({ audio: { echoCancellation: false, noiseSuppression: false, autoGainControl: false }, video: false });
        if (!live) { stream.getTracks().forEach((track) => track.stop()); return; }
        context = new AudioContext();
        rate.current = context.sampleRate;
        const url = URL.createObjectURL(new Blob([TAP], { type: "application/javascript" }));
        try { await context.audioWorklet.addModule(url); } finally { URL.revokeObjectURL(url); }
        if (!live) return;
        const source = context.createMediaStreamSource(stream);
        const tap = new AudioWorkletNode(context, "ai-bricks-tap");
        const silent = context.createGain();
        silent.gain.value = 0;
        source.connect(tap).connect(silent).connect(context.destination);
        let peak = 0;
        let shown = 0;
        tap.port.onmessage = (event: MessageEvent<Float32Array>) => {
          const chunk = event.data;
          const kept = recent.current;
          kept.chunks.push(chunk);
          kept.total += chunk.length;
          while (kept.chunks.length > 1 && kept.total - kept.chunks[0].length >= KEPT_SECONDS * rate.current) kept.total -= kept.chunks.shift()!.length;
          taking.current?.push(chunk);
          for (let index = 0; index < chunk.length; index += 4) peak = Math.max(peak, Math.abs(chunk[index]));
          const now = performance.now();
          if (now - shown > 80) { setLevel(Math.min(1, peak)); peak = 0; shown = now; }
        };
        if (context.state === "suspended") await context.resume();
        setReady(true);
      } catch (error) {
        if (live) setProblem(micProblem(error));
      }
    })();
    return () => {
      live = false;
      setReady(false);
      setLevel(0);
      stream?.getTracks().forEach((track) => track.stop());
      void context?.close();
      recent.current = { chunks: [], total: 0 };
      taking.current = null;
    };
  }, [on]);

  /** The last `seconds` of sound, or null until that much has been heard. */
  const last = useCallback((seconds: number) => {
    const wanted = Math.round(seconds * rate.current);
    const kept = recent.current;
    if (kept.total < wanted) return null;
    const joined = join(kept.chunks);
    return joined.subarray(joined.length - wanted);
  }, []);
  const begin = useCallback(() => { taking.current = []; }, []);
  const end = useCallback(() => {
    const chunks = taking.current;
    taking.current = null;
    return chunks ? join(chunks) : new Float32Array(0);
  }, []);

  return { ready, problem, level, rate, last, begin, end };
}

/** Resamples to 16 kHz with the browser's own resampler, which filters properly. */
async function toModelRate(samples: Float32Array, from: number) {
  if (from === RATE) return samples;
  const frames = Math.max(1, Math.round((samples.length * RATE) / from));
  const offline = new OfflineAudioContext(1, frames, RATE);
  const buffer = offline.createBuffer(1, samples.length, from);
  buffer.copyToChannel(new Float32Array(samples), 0);
  const source = offline.createBufferSource();
  source.buffer = buffer;
  source.connect(offline.destination);
  source.start();
  return (await offline.startRendering()).getChannelData(0);
}

/** 16-bit mono WAV, the one format the server reads. */
function toWav(samples: Float32Array) {
  const view = new DataView(new ArrayBuffer(44 + samples.length * 2));
  const text = (at: number, value: string) => { for (let index = 0; index < value.length; index++) view.setUint8(at + index, value.charCodeAt(index)); };
  text(0, "RIFF"); view.setUint32(4, 36 + samples.length * 2, true); text(8, "WAVE");
  text(12, "fmt "); view.setUint32(16, 16, true); view.setUint16(20, 1, true); view.setUint16(22, 1, true);
  view.setUint32(24, RATE, true); view.setUint32(28, RATE * 2, true); view.setUint16(32, 2, true); view.setUint16(34, 16, true);
  text(36, "data"); view.setUint32(40, samples.length * 2, true);
  samples.forEach((value, index) => view.setInt16(44 + index * 2, Math.max(-1, Math.min(1, value)) * 0x7fff, true));
  return new Blob([view], { type: "audio/wav" });
}

async function recordingOf(samples: Float32Array, from: number) {
  return toWav(await toModelRate(samples, from));
}

function LevelMeter({ level, label }: { level: number; label: string }) {
  const shown = Math.round(Math.min(1, level * 1.6) * 100);
  return (
    <span className="level-meter" role="meter" aria-label={label} aria-valuemin={0} aria-valuemax={100} aria-valuenow={shown}>
      <span style={{ width: `${shown}%` }} className={level > 0.95 ? "is-loud" : ""} />
    </span>
  );
}

// ------------------------------------------------------------------ recording clips

export function SoundLabels({ dataset, beginner, onChanged }: { dataset: DatasetSummary; beginner: boolean; onChanged: () => void }) {
  const [open, setOpen] = useState(false);
  const mic = useMicrophone(open);
  const [recording, setRecordingState] = useState<string | null>(null);
  const recordingNow = useRef<string | null>(null);
  const setRecording = (value: string | null) => { recordingNow.current = value; setRecordingState(value); };
  const [countdown, setCountdown] = useState<number | null>(null);
  const [saving, setSaving] = useState(false);
  const [note, setNote] = useState<string | null>(null);
  const [problem, setProblem] = useState<string | null>(null);
  const [browsing, setBrowsing] = useState<string | null>(null);
  const [version, setVersion] = useState(0);
  const timer = useRef<number | null>(null);
  const upload = useRef<HTMLInputElement>(null);
  const [uploadTo, setUploadTo] = useState(dataset.categories[0] ?? "");

  useEffect(() => { if (!dataset.categories.includes(uploadTo)) setUploadTo(dataset.categories[0] ?? ""); }, [dataset.categories, uploadTo]);

  const save = async (category: string, recordingBlob: Blob, seconds: number) => {
    setSaving(true);
    try {
      const reply = await api.addRecording(dataset.id, category, recordingBlob);
      setNote(`${reply.added} ${reply.added === 1 ? "clip" : "clips"} added to ${category} from ${seconds.toFixed(1)} s of sound${reply.duplicates ? `, ${reply.duplicates} already there` : ""}.`);
      setVersion((value) => value + 1);
      onChanged();
    } catch (error) {
      setProblem(message(error, "The recording couldn't be saved."));
    } finally {
      setSaving(false);
    }
  };

  const start = (category: string) => {
    if (recordingNow.current || !mic.ready || saving) return;
    setProblem(null);
    setNote(null);
    setRecording(category);
    mic.begin();
  };

  const finish = async () => {
    const category = recordingNow.current;
    if (!category) return;
    if (timer.current !== null) { window.clearInterval(timer.current); timer.current = null; }
    setRecording(null);
    setCountdown(null);
    const samples = mic.end();
    const seconds = samples.length / mic.rate.current;
    if (seconds < SHORTEST_SECONDS) { setNote("Hold the button a little longer. Each second you hold becomes one clip."); return; }
    await save(category, await recordingOf(samples, mic.rate.current), seconds);
  };

  // A background category is easiest to fill with one long, steady recording.
  const recordFor = (category: string, seconds: number) => {
    start(category);
    if (recordingNow.current !== category) return;
    let left = seconds;
    setCountdown(left);
    timer.current = window.setInterval(() => {
      left -= 1;
      setCountdown(left);
      if (left <= 0) void finish();
    }, 1000);
  };

  useEffect(() => () => { if (timer.current !== null) window.clearInterval(timer.current); }, []);
  useEffect(() => { if (!open && recordingNow.current) void finish(); });

  const keyDown = (event: KeyboardEvent, category: string) => {
    if ((event.key === " " || event.key === "Enter") && !event.repeat) { event.preventDefault(); start(category); }
  };

  const uploadFiles = async (event: ChangeEvent<HTMLInputElement>) => {
    const files = [...(event.target.files ?? [])];
    event.target.value = "";
    if (!files.length || !uploadTo) return;
    setProblem(null);
    setNote(null);
    setSaving(true);
    let added = 0;
    try {
      for (const file of files) added += (await api.addRecording(dataset.id, uploadTo, file)).added;
      setNote(`${added} ${added === 1 ? "clip" : "clips"} added to ${uploadTo} from ${files.length} ${files.length === 1 ? "file" : "files"}.`);
      setVersion((value) => value + 1);
      onChanged();
    } catch (error) {
      setProblem(message(error, "The file couldn't be added."));
    } finally {
      setSaving(false);
    }
  };

  const hasBackground = dataset.categories.some(isBackground);

  return (
    <>
      <h2 className="section-title">Add examples</h2>
      <section className="card">
        <div className="card-head">
          <h2>Record sounds</h2>
          <button className="button button-small" aria-pressed={open} onClick={() => setOpen(!open)}>{open ? "Turn microphone off" : "Use microphone"}</button>
        </div>
        <p className="muted small">
          {beginner
            ? "Hold a category's button while making the sound. Every second you hold becomes one clip. Vary it a little each time: louder, softer, closer, further away."
            : "Hold to record. Recordings are resampled to 16 kHz mono and cut into 1-second clips; a final piece under 0.25 s is dropped."}
        </p>
        {!hasBackground && dataset.categories.length > 0 && (
          <p className="notice field-note">
            <span>Add a <strong>Background noise</strong> category with recordings of the room being quiet. Without it, the model has to pick one of your sounds even when nothing is happening.</span>
          </p>
        )}
        {open && (
          <div className="sound-recorder field-note">
            <div className="sound-mic">
              {!mic.ready && !mic.problem && <span className="muted small">Starting the microphone…</span>}
              {mic.problem && <span className="small tone-bad" role="alert">{mic.problem}</span>}
              {mic.ready && <><span className="muted small">Level</span><LevelMeter level={mic.level} label="Microphone level" /></>}
            </div>
            {dataset.categories.length === 0 && <p className="notice">Add a category first.</p>}
            <div className="webcam-buttons">
              {dataset.categories.map((name) => (
                <div key={name} className="sound-row">
                  <button
                    className={`button webcam-hold${recording === name ? " is-recording" : ""}`}
                    disabled={!mic.ready || saving || (recording !== null && recording !== name) || countdown !== null && recording === name}
                    aria-label={`Hold to record ${name}`}
                    onPointerDown={(event) => { event.currentTarget.setPointerCapture(event.pointerId); start(name); }}
                    onPointerUp={() => { if (countdown === null) void finish(); }}
                    onPointerCancel={() => { if (countdown === null) void finish(); }}
                    onKeyDown={(event) => keyDown(event, name)}
                    onKeyUp={() => { if (countdown === null) void finish(); }}
                    onContextMenu={(event) => event.preventDefault()}
                  >
                    <span>{recording === name ? (countdown !== null ? `Recording ${name}… ${countdown} s` : `Recording ${name}…`) : `Hold for ${name}`}</span>
                    <span className="webcam-count">{(dataset.counts[name] ?? 0).toLocaleString()}</span>
                  </button>
                  {isBackground(name) && (
                    <button className="button button-small" disabled={!mic.ready || saving || recording !== null} onClick={() => recordFor(name, BACKGROUND_SECONDS)}>
                      Record {BACKGROUND_SECONDS} s
                    </button>
                  )}
                </div>
              ))}
            </div>
          </div>
        )}
        {saving && <p className="muted small field-note" role="status">Saving…</p>}
        {note && <p className="small field-note" role="status">{note}</p>}
        {problem && <p className="small tone-bad field-note" role="alert">{problem}</p>}
        <div className="sound-upload field-note">
          <label className="small">
            Or add WAV files to
            <select value={uploadTo} onChange={(event) => setUploadTo(event.target.value)} disabled={!dataset.categories.length}>
              {dataset.categories.map((name) => <option key={name} value={name}>{name}</option>)}
            </select>
          </label>
          <button className="button button-small" disabled={!uploadTo || saving} onClick={() => upload.current?.click()}>Choose WAV files</button>
          <input ref={upload} className="visually-hidden" type="file" accept=".wav,audio/wav,audio/x-wav" multiple tabIndex={-1} aria-label="Choose WAV files" onChange={(event) => void uploadFiles(event)} />
        </div>
      </section>

      <h2 className="section-title">Check them</h2>
      <section className="card">
        <h2>Listen to your clips</h2>
        <p className="muted small">Remove clips that are silent by mistake or caught the wrong sound. Saved versions keep their own copy.</p>
        <div className="label-buttons field-note" style={{ justifyContent: "flex-start", flexWrap: "wrap" }}>
          {dataset.categories.map((name) => (
            <button key={name} className="button button-small" aria-pressed={browsing === name} onClick={() => setBrowsing(browsing === name ? null : name)}>
              {name} <span className="muted">{(dataset.counts[name] ?? 0).toLocaleString()}</span>
            </button>
          ))}
        </div>
        {browsing && <ClipList key={`${browsing}-${version}`} dataset={dataset} category={browsing} onChanged={onChanged} />}
      </section>
    </>
  );
}

function ClipList({ dataset, category, onChanged }: { dataset: DatasetSummary; category: string; onChanged: () => void }) {
  const [clips, setClips] = useState<string[] | null>(null);
  const [total, setTotal] = useState(0);
  const [problem, setProblem] = useState<string | null>(null);

  useEffect(() => {
    api.fetchClips(dataset.id, category).then((reply) => { setClips(reply.clips); setTotal(reply.total); })
      .catch((error) => setProblem(message(error, "The clips couldn't load.")));
  }, [dataset.id, category]);

  const remove = async (clip: string) => {
    setProblem(null);
    try {
      await api.removeClip(dataset.id, category, clip);
      setClips((current) => current?.filter((item) => item !== clip) ?? null);
      setTotal((value) => value - 1);
      onChanged();
    } catch (error) {
      setProblem(message(error, "The clip couldn't be removed."));
    }
  };

  return (
    <div className="field-note">
      {problem && <p className="small tone-bad" role="alert">{problem}</p>}
      {clips?.length === 0 && <p className="muted small">No clips in {category} yet.</p>}
      <ul className="clip-list">
        {clips?.map((clip, index) => (
          <li key={clip}>
            <audio controls preload="none" src={api.clipUrl(dataset.id, category, clip)} aria-label={`${category} clip ${index + 1}`} />
            <button className="button button-small button-quiet" aria-label={`Remove ${category} clip ${index + 1}`} onClick={() => void remove(clip)}>Remove</button>
          </li>
        ))}
      </ul>
      {clips && total > clips.length && <p className="muted small">Showing the newest {clips.length} of {total.toLocaleString()}.</p>}
    </div>
  );
}

export function ClipPreview({ dataset, category, clips }: { dataset: DatasetSummary; category: string; clips: string[] }) {
  return (
    <ul className="clip-list is-compact">
      {clips.slice(0, 4).map((clip, index) => (
        <li key={clip}><audio controls preload="none" src={api.clipUrl(dataset.id, category, clip)} aria-label={`${category} example ${index + 1}`} /></li>
      ))}
    </ul>
  );
}

// ------------------------------------------------------------------ training

export function SoundTrainer({ dataset, beginner, running, busyElsewhere, onStart, onStop }: {
  dataset: DatasetSummary; beginner: boolean; running: SavedModel | null; busyElsewhere: boolean; onStart: () => void; onStop: (id: string) => void;
}) {
  const short = dataset.categories.filter((name) => (dataset.counts[name] ?? 0) < SOUND_MINIMUM);
  const ready = dataset.categories.length >= 2 && short.length === 0;
  const few = ready && dataset.categories.some((name) => (dataset.counts[name] ?? 0) < SOUND_SUGGESTED);
  return (
    <section className="card">
      <div className="card-head">
        <h2>Train a sound sorter</h2>
        {running
          ? <button className="button button-small" onClick={() => onStop(running.id)}>Stop</button>
          : <button className="button button-small button-primary" disabled={!ready || busyElsewhere} onClick={onStart}>Start training</button>}
      </div>
      <p className="muted small">
        {beginner
          ? "Each clip is turned into a spectrogram, a picture of which pitches are loud at each moment. A model that already learned from millions of sounds reads that picture, and a small classifier learns your categories from what it notices. Your recordings stay on this computer."
          : "Frozen AST (AudioSet, position table cut to 1.28 s) as a feature extractor, linear head fitted with L-BFGS on each clip plus two time-shifted, gain-changed copies."}
      </p>
      {running && (
        <div className="field-note" role="status">
          <progress className="train-progress" max={running.stepsTotal || 1} value={running.stepsDone} aria-label="Training progress" />
          <p className="muted small">
            {!running.device
              ? "Getting the sound model ready. The first run downloads about 350 MB from Hugging Face. Your recordings stay here."
              : running.stage === "reviewing"
                ? "Checking the held-back clips…"
                : `Listened to ${running.stepsDone.toLocaleString()} of ${(running.stepsTotal || 0).toLocaleString()} clips, using ${running.device}.`}
          </p>
        </div>
      )}
      {!running && busyElsewhere && <p className="muted small field-note">Another model is training. Only one can train at a time.</p>}
      {!running && dataset.categories.length < 2 && <p className="muted small field-note">Add at least 2 categories on the Label tab.</p>}
      {!running && dataset.categories.length >= 2 && short.length > 0 && <p className="muted small field-note">Record at least {SOUND_MINIMUM} clips in: {short.join(", ")}.</p>}
      {!running && few && <p className="muted small field-note">You can train now, but scores are more trustworthy with at least {SOUND_SUGGESTED} clips per category.</p>}
    </section>
  );
}

// ------------------------------------------------------------------ trying a model

const RAMP = [[244, 246, 251], [143, 170, 224], [47, 95, 184], [20, 40, 92]];

function rampColour(value: number) {
  const scaled = Math.max(0, Math.min(1, value)) * (RAMP.length - 1);
  const low = Math.min(RAMP.length - 2, Math.floor(scaled));
  const share = scaled - low;
  return RAMP[low].map((channel, index) => Math.round(channel + (RAMP[low + 1][index] - channel) * share));
}

/** A spectrogram drawn light to dark blue, quiet to loud, with where the model listened laid over it. */
function Spectrogram({ spectrogram, heatmap }: { spectrogram: Grid; heatmap: Heatmap | null }) {
  const canvas = useRef<HTMLCanvasElement>(null);
  useEffect(() => {
    const target = canvas.current;
    const context = target?.getContext("2d");
    if (!target || !context) return;
    target.width = spectrogram.width;
    target.height = spectrogram.height;
    const pixels = context.createImageData(spectrogram.width, spectrogram.height);
    spectrogram.values.forEach((value, index) => pixels.data.set([...rampColour(value), 255], index * 4));
    context.putImageData(pixels, 0, 0);
  }, [spectrogram]);
  return (
    <figure className="spectrogram">
      <div className="spectrogram-frame">
        <canvas ref={canvas} className="spectrogram-canvas" role="img" aria-label="Spectrogram of the last second: time left to right, pitch low to high from bottom to top" />
        {heatmap && <HeatmapOverlay heatmap={heatmap} aspect={2} />}
      </div>
      <figcaption className="spectrogram-axes muted small"><span>← 1 second →</span><span>low pitch at the bottom, high at the top</span></figcaption>
    </figure>
  );
}

function Answers({ answer }: { answer: SoundAnswer }) {
  return (
    <ul className="try-answers" aria-live="polite" aria-label="The model's answers">
      {answer.answers.map((item, index) => (
        <li key={item.category} className={index === 0 ? "is-top" : ""}>
          <span className="try-name">{item.category}</span>
          <span className="try-track" aria-hidden="true"><span style={{ width: `${item.chance * 100}%` }} /></span>
          <span className="try-value">{(item.chance * 100).toFixed(0)}%</span>
        </li>
      ))}
    </ul>
  );
}

export function SoundTry({ model, beginner }: { model: SavedModel; beginner: boolean }) {
  const [listening, setListening] = useState(false);
  const mic = useMicrophone(listening);
  const [explain, setExplain] = useState(false);
  const [answer, setAnswer] = useState<SoundAnswer | null>(null);
  const [problem, setProblem] = useState<string | null>(null);
  const [asking, setAsking] = useState(false);
  const [source, setSource] = useState<"live" | "file" | null>(null);
  const explainNow = useRef(explain);
  explainNow.current = explain;
  const input = useRef<HTMLInputElement>(null);
  const lastFile = useRef<File | null>(null);

  // Asks about the last second, one request at a time, so slow answers never pile up.
  useEffect(() => {
    if (!listening || !mic.ready) return;
    let live = true;
    const loop = async () => {
      while (live) {
        const samples = mic.last(1);
        if (samples && !document.hidden) {
          try {
            const next = await api.listen(model.id, await recordingOf(samples, mic.rate.current), explainNow.current);
            if (!live) return;
            setAnswer(next);
            setSource("live");
            setProblem(null);
          } catch (error) {
            if (!live) return;
            setProblem(message(error, "The model couldn't answer."));
            await new Promise((done) => setTimeout(done, 1500));
          }
        }
        await new Promise((done) => setTimeout(done, LISTEN_EVERY_MS));
      }
    };
    void loop();
    return () => { live = false; };
  }, [listening, mic.ready, mic.last, mic.rate, model.id]);

  const askFile = async (file: File | undefined | null, withExplain = explain) => {
    if (!file) return;
    lastFile.current = file;
    setListening(false);
    setAsking(true);
    setProblem(null);
    try {
      setAnswer(await api.listen(model.id, file, withExplain));
      setSource("file");
    } catch (error) {
      setProblem(message(error, "The model couldn't read this recording."));
    } finally {
      setAsking(false);
    }
  };

  const toggleExplain = (next: boolean) => {
    setExplain(next);
    if (source === "file") void askFile(lastFile.current, next);
  };

  return (
    <section className="card">
      <h2>Try the newest model</h2>
      <p className="muted small">{`${model.projectName}, trained on data v${model.datasetVersion ?? "?"}. Make a sound and watch its guess, or try a WAV file.`}</p>
      <div className="label-buttons field-note" style={{ justifyContent: "flex-start", flexWrap: "wrap" }}>
        <button className="button button-small" aria-pressed={listening} onClick={() => setListening(!listening)}>{listening ? "Stop listening" : "Listen with the microphone"}</button>
        <button className="button button-small" disabled={asking} onClick={() => input.current?.click()}>Try a WAV file</button>
        <input ref={input} className="visually-hidden" type="file" accept=".wav,audio/wav,audio/x-wav" tabIndex={-1} aria-label="Choose a WAV file to try" onChange={(event) => { void askFile(event.target.files?.[0]); event.target.value = ""; }} />
        <label className="check-row small">
          <input type="checkbox" checked={explain} onChange={(event) => toggleExplain(event.target.checked)} />
          Show where it listens
        </label>
      </div>
      {listening && (
        <div className="sound-mic field-note">
          {!mic.ready && !mic.problem && <span className="muted small">Starting the microphone…</span>}
          {mic.problem && <span className="small tone-bad" role="alert">{mic.problem}</span>}
          {mic.ready && <><span className="muted small">Level</span><LevelMeter level={mic.level} label="Microphone level" /></>}
        </div>
      )}
      {asking && <p className="muted small field-note" role="status">Listening to the file…</p>}
      {problem && <p className="small tone-bad field-note" role="alert">{problem}</p>}
      {answer && (
        <div className="sound-result field-note">
          <Spectrogram spectrogram={answer.spectrogram} heatmap={explain ? answer.heatmap : null} />
          <div className="webcam-side">
            <Answers answer={answer} />
            {source === "file" && answer.clips > 1 && <p className="muted small">Averaged over {answer.clips} one-second clips. The picture shows the first second.</p>}
            {explain && answer.heatmap && (
              <p className="muted small heatmap-key">
                <span className="heatmap-swatch" aria-hidden="true" />
                {beginner
                  ? `Bright areas are the moments and pitches that pushed it most toward “${answer.heatmap.category}”. This shows where it listened, not proof of why.`
                  : `Gradient saliency for “${answer.heatmap.category}” on the input spectrogram, pooled to ${answer.heatmap.width}×${answer.heatmap.height}.`}
              </p>
            )}
            {beginner && <p className="muted small">If it's confidently wrong, record more clips like that sound, in that room, and train again.</p>}
          </div>
        </div>
      )}
    </section>
  );
}

// ------------------------------------------------------------------ mistakes

export function SoundMistakes({ model, dataset, beginner, onChanged }: { model: SavedModel; dataset: DatasetSummary; beginner: boolean; onChanged: () => void }) {
  const [rows, setRows] = useState<SoundMistake[] | null>(null);
  const [total, setTotal] = useState(0);
  const [problem, setProblem] = useState<string | null>(null);

  const load = useCallback(() => {
    api.fetchSoundMistakes(model.id).then((reply) => { setRows(reply.mistakes); setTotal(reply.total); })
      .catch((error) => setProblem(message(error, "The mistakes couldn't load.")));
  }, [model.id]);
  useEffect(load, [load]);

  const fix = async (row: SoundMistake, action: "move" | "remove" | "undo", category?: string) => {
    setProblem(null);
    try {
      await api.fixMistake(model.id, row.index, action, category);
      load();
      onChanged();
    } catch (error) {
      setProblem(message(error, "That change couldn't be saved."));
    }
  };

  if (!rows && !problem) return null;
  return (
    <section className="card">
      <h2>Mistakes from the newest model</h2>
      {problem && <p className="small tone-bad" role="alert">{problem}</p>}
      {rows?.length === 0 && <p className="muted small">It got every held-back clip right. Try it with the microphone in a different room to find its limits.</p>}
      {rows && rows.length > 0 && (
        <>
          <p className="muted small">
            {beginner ? "Listen to each one. Sometimes the model is right and the clip is filed in the wrong category, or it's just silence. Fix those here, then train again." : "Most confident first. Changes apply from the next run."}
            {total > rows.length ? ` Showing ${rows.length} of ${total}.` : ""}
          </p>
          <ul className="project-list">
            {rows.map((row) => (
              <li key={row.index} className="sound-mistake">
                <audio controls preload="none" src={api.mistakeImageUrl(model.id, row.index)} aria-label={`Clip filed under ${row.actual}, guessed ${row.guess}`} />
                <span className="project-text small">
                  <span><span className="muted">Filed under</span> {row.actual} · <span className="muted">guessed</span> {row.guess} <span className="muted">({Math.round(row.chance * 100)}% sure)</span></span>
                  {(row.movedTo || row.removed) && <span className="tone-good">{row.removed ? "Removed" : `Moved to ${row.movedTo}`}</span>}
                </span>
                {row.movedTo || row.removed
                  ? <button className="button button-small button-quiet" onClick={() => void fix(row, "undo")}>Undo</button>
                  : <span className="mistake-fix small">
                      <select aria-label={`Move this clip out of ${row.actual}`} value="" onChange={(event) => { if (event.target.value) void fix(row, "move", event.target.value); }}>
                        <option value="">Move to…</option>
                        {dataset.categories.filter((name) => name !== row.actual).map((name) => <option key={name} value={name}>{name}</option>)}
                      </select>
                      <button className="button button-small button-quiet" onClick={() => void fix(row, "remove")}>Remove</button>
                    </span>}
              </li>
            ))}
          </ul>
        </>
      )}
    </section>
  );
}
