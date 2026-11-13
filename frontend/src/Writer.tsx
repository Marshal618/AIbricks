import { useCallback, useEffect, useRef, useState, type ChangeEvent, type FormEvent } from "react";
import * as api from "./api";
import type { ChatTurn, DatasetSummary, SavedModel, WriterAnalysis } from "./types";
import { Why } from "./Ui";

function message(error: unknown, fallback: string) {
  return error instanceof Error ? error.message : fallback;
}

function percent(value: number | undefined) {
  return `${Math.round((value ?? 0) * 100)}%`;
}

/** The Add data tab of a "write like this text" dataset: text goes in by file or by pasting. */
export function WritingData({ dataset, beginner, onChanged }: { dataset: DatasetSummary; beginner: boolean; onChanged: () => void }) {
  const [pasted, setPasted] = useState("");
  const [note, setNote] = useState<string | null>(null);
  const [problem, setProblem] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const input = useRef<HTMLInputElement>(null);

  const send = async (file: File) => {
    setProblem(null);
    setNote(null);
    setBusy(true);
    try {
      const reply = await api.addUnsortedText(file, dataset.id);
      const parts = [`Added ${reply.added.toLocaleString()} ${reply.added === 1 ? "piece" : "pieces"} of text${reply.column ? ` from the "${reply.column}" column` : ""}.`];
      if (reply.duplicates) parts.push(`${reply.duplicates.toLocaleString()} ${reply.duplicates === 1 ? "was" : "were"} already here.`);
      setNote(parts.join(" "));
      onChanged();
      return true;
    } catch (error) {
      setProblem(message(error, "That text could not be added."));
      return false;
    } finally {
      setBusy(false);
    }
  };

  const addFile = async (event: ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    event.target.value = "";
    if (file) await send(file);
  };

  const addPasted = async (event: FormEvent) => {
    event.preventDefault();
    if (pasted.trim() && await send(new File([pasted], "pasted.txt", { type: "text/plain" }))) setPasted("");
  };

  const clearAll = async () => {
    if (!window.confirm("Remove all the text from this dataset? Your original files aren't touched, and earlier versions stay in History.")) return;
    await api.clearUnsortedText(dataset.id).catch(() => undefined);
    setNote(null);
    onChanged();
  };

  const words = dataset.words ?? 0;
  return (
    <>
      {problem && <div className="result result-fail" role="alert"><p>{problem}</p></div>}
      <h2 className="section-title">Add text</h2>
      <section className="card">
        <div className="card-head">
          <h2>Your text</h2>
          <div className="label-buttons">
            {dataset.labelled > 0 && <button className="button button-small button-quiet" disabled={busy} onClick={() => void clearAll()}>Remove all</button>}
            <button className="button button-small" disabled={busy} onClick={() => input.current?.click()}>{busy ? "Reading…" : "Add a file"}</button>
          </div>
          <input ref={input} className="visually-hidden" type="file" accept=".txt,.md,.csv,text/plain,text/csv,text/markdown" aria-label="Add a text file" tabIndex={-1} onChange={(event) => void addFile(event)} />
        </div>
        <p className="muted small">
          {beginner
            ? "Add writing in the style you want: stories, notes, emails, product descriptions, song lyrics. Use a .txt file, or paste it below. Each line or paragraph becomes one piece. It stays on this computer."
            : "A .txt or .md file, one piece per line or paragraph, or a .csv with a \"text\" column. Paragraphs over 5,000 characters are split at sentence ends."}
        </p>
        <form className="field-note" onSubmit={(event) => void addPasted(event)}>
          <textarea className="writing-input" rows={5} value={pasted} aria-label="Paste text" placeholder="Paste text here" onChange={(event) => setPasted(event.target.value)} />
          <div className="field-note"><button className="button button-small" type="submit" disabled={busy || !pasted.trim()}>Add this text</button></div>
        </form>
        {note && <p className="muted small field-note" role="status">{note}</p>}
        <p className="small field-note">
          <strong>{words.toLocaleString()}</strong> words in {dataset.labelled.toLocaleString()} {dataset.labelled === 1 ? "piece" : "pieces"}.
          {words < 1500 ? ` It needs at least 1,500 to train, and more is better.` : words < 20000 ? " Enough to train. More text, 20,000 words or so, teaches your style better." : " Plenty to learn your style from."}
        </p>
      </section>
    </>
  );
}

/** The Check tab of a writing dataset. Label checks don't apply, so it says what does. */
export function WritingCheck({ dataset }: { dataset: DatasetSummary }) {
  return (
    <section className="card">
      <h2>Before you train</h2>
      <p className="muted small">Label checks don't apply here, because a writer learns from all the text rather than from categories. Instead, check that:</p>
      <ul className="small writing-list">
        <li>it's text you're allowed to use, such as your own writing or text with an open licence,</li>
        <li>it doesn't hold anything private you wouldn't want it to repeat, like names, addresses or passwords. A writer can echo what it learned,</li>
        <li>it's the style you want. It copies mistakes and quirks as readily as good writing.</li>
      </ul>
      <p className="muted small field-note">This dataset has {(dataset.words ?? 0).toLocaleString()} words in {dataset.labelled.toLocaleString()} pieces.</p>
    </section>
  );
}

export function WriterTrainer({ dataset, beginner, running, busyElsewhere, onStarted, onStop }: {
  dataset: DatasetSummary; beginner: boolean; running: SavedModel | null; busyElsewhere: boolean; onStarted: () => Promise<void>; onStop: (id: string) => void;
}) {
  const [analysis, setAnalysis] = useState<WriterAnalysis | null>(null);
  const [model, setModel] = useState("SmolLM2 135M");
  const [strength, setStrength] = useState<"light" | "more">("light");
  const [length, setLength] = useState<"quick" | "balanced" | "thorough">("balanced");
  const [problem, setProblem] = useState<string | null>(null);

  useEffect(() => {
    api.fetchWriterAnalysis(dataset.id).then(setAnalysis).catch((error) => setProblem(message(error, "The text could not be checked.")));
  }, [dataset.id, dataset.updatedAt, running?.status]);

  const start = async () => {
    setProblem(null);
    try {
      await api.trainWriter(dataset.id, { model, strength, length });
      await onStarted();
    } catch (error) {
      setProblem(message(error, "Training could not start."));
    }
  };

  const chosen = analysis?.models.find((item) => item.name === model);
  const ready = !!analysis && analysis.problems.length === 0;
  return (
    <section className="card">
      <div className="card-head">
        <h2>Teach a writer your style</h2>
        {running
          ? <button className="button button-small" onClick={() => onStop(running.id)}>Stop</button>
          : <button className="button button-small button-primary" disabled={!ready || busyElsewhere} onClick={() => void start()}>Start training</button>}
      </div>
      <p className="muted small">
        {beginner
          ? "It starts from a small open language model that can already write and chat, then adjusts it to write like your text. It runs on this computer, and your text isn't sent anywhere."
          : "Fine-tunes the last few decoder layers of a small open causal language model on next-token prediction. Training and answers are local."}
      </p>
      {problem && <p className="small" role="alert">{problem}</p>}
      {running && (
        <div className="field-note" role="status">
          <progress className="train-progress" max={running.stepsTotal || 1} value={running.stepsDone} aria-label="Training progress" />
          <p className="muted small">
            {!running.device
              ? `Getting the writer ready. The first time, this downloads ${chosen?.download ?? "the model"} from Hugging Face. Your text stays here.`
              : running.stage === "reviewing"
                ? "Writing a few samples to compare…"
                : `Step ${running.stepsDone.toLocaleString()} of ${(running.stepsTotal || 0).toLocaleString()}, using ${running.device}. This can be left running.`}
          </p>
        </div>
      )}
      {!running && busyElsewhere && <p className="muted small field-note">Another model is training. Only one can train at a time.</p>}
      {analysis && (
        <>
          <div className="table-choices field-note">
            <label>
              Writer to start from
              <select value={model} disabled={!!running} onChange={(event) => setModel(event.target.value)}>
                {analysis.models.map((item) => (
                  <option key={item.name} value={item.name}>{item.name} ({item.downloaded ? "downloaded" : `${item.download} download`})</option>
                ))}
              </select>
            </label>
            <label>
              How much it changes
              <select value={strength} disabled={!!running} onChange={(event) => setStrength(event.target.value as "light" | "more")}>
                <option value="light">A little: its last 2 layers</option>
                <option value="more">More: its last 6 layers</option>
              </select>
            </label>
            <label>
              How long to train
              <select value={length} disabled={!!running} onChange={(event) => setLength(event.target.value as "quick" | "balanced" | "thorough")}>
                <option value="quick">Quick: once through the text</option>
                <option value="balanced">Balanced: twice through</option>
                <option value="thorough">Thorough: four times through</option>
              </select>
            </label>
          </div>
          <p className="muted small field-note">
            {analysis.words.toLocaleString()} words in {analysis.pieces.toLocaleString()} pieces.
            {" "}{model.startsWith("SmolLM2 135M") ? "The smallest writer: quickest to train, simplest answers." : "Bigger writers write better, but train more slowly and need more memory."}
          </p>
          {analysis.problems.map((text) => <p className="notice field-note" key={text}>{text}</p>)}
          {analysis.notes.map((text) => <p className="muted small field-note" key={text}>{text}</p>)}
          <Why>
            {beginner
              ? "About 1 piece of text in 10 is held back, and the writer never learns from it. Before and after training, it reads that text and guesses each next word, so you can see whether it got better at writing like you."
              : "Pieces are held back by content fingerprint (about 10%). Check loss and top-1 next-token accuracy are measured on them before training and after each pass; the best pass is kept, or the original if none beat it."}
          </Why>
        </>
      )}
    </section>
  );
}

interface Exchange {
  prompt: string;
  chat: boolean;
  adjusted: string;
  original?: string;
}

function WriterChat({ model, beginner }: { model: SavedModel; beginner: boolean }) {
  const [mode, setMode] = useState<"chat" | "write">("chat");
  const [compare, setCompare] = useState(true);
  const [text, setText] = useState("");
  const [exchanges, setExchanges] = useState<Exchange[]>([]);
  const [asking, setAsking] = useState(false);
  const [problem, setProblem] = useState<string | null>(null);
  const end = useRef<HTMLDivElement>(null);

  useEffect(() => { end.current?.scrollIntoView({ block: "nearest" }); }, [exchanges.length, asking]);

  const ask = async (event: FormEvent) => {
    event.preventDefault();
    const prompt = text.trim();
    if (!prompt || asking) return;
    setProblem(null);
    setAsking(true);
    const history: ChatTurn[] = mode === "chat"
      ? exchanges.filter((item) => item.chat).flatMap((item) => [{ role: "user" as const, content: item.prompt }, { role: "assistant" as const, content: item.adjusted }])
      : [];
    try {
      const reply = await api.writeWith(model.id, { prompt, chat: mode === "chat", history, compare, seed: exchanges.length });
      setExchanges((items) => [...items, { prompt, chat: mode === "chat", ...reply }]);
      setText("");
    } catch (error) {
      setProblem(message(error, "The writer couldn't answer."));
    } finally {
      setAsking(false);
    }
  };

  return (
    <>
      <h3 className="field-note small">Try it</h3>
      <div className="forecast-views" role="group" aria-label="How it should answer">
        <button className={`button button-small${mode === "chat" ? " is-active" : ""}`} aria-pressed={mode === "chat"} onClick={() => setMode("chat")}>Chat with it</button>
        <button className={`button button-small${mode === "write" ? " is-active" : ""}`} aria-pressed={mode === "write"} onClick={() => setMode("write")}>Carry on my words</button>
      </div>
      <label className="compat writer-compare">
        <input type="checkbox" checked={compare} onChange={(event) => setCompare(event.target.checked)} />
        <span>Show the writer as downloaded next to it</span>
      </label>
      {exchanges.length > 0 && (
        <div className="writer-log" aria-live="polite">
          {exchanges.map((item, index) => (
            <div className="writer-exchange" key={index}>
              <p className="writer-you"><span className="muted small">{item.chat ? "You" : "Your opening"}</span>{item.prompt}</p>
              <div className={item.original !== undefined ? "writer-answers is-pair" : "writer-answers"}>
                <div className="writer-answer">
                  <span className="muted small">{item.original !== undefined ? "Adjusted to your text" : "Writer"}</span>
                  <p>{item.adjusted || "(It stopped without writing anything.)"}</p>
                </div>
                {item.original !== undefined && (
                  <div className="writer-answer is-original">
                    <span className="muted small">As downloaded</span>
                    <p>{item.original || "(It stopped without writing anything.)"}</p>
                  </div>
                )}
              </div>
            </div>
          ))}
          <div ref={end} />
        </div>
      )}
      <form className="field-note" onSubmit={(event) => void ask(event)}>
        <textarea
          className="writing-input" rows={2} value={text} maxLength={2000}
          aria-label={mode === "chat" ? "Message to the writer" : "Opening words for the writer"}
          placeholder={mode === "chat" ? "Ask it something" : "Write the first few words and it carries on"}
          onChange={(event) => setText(event.target.value)}
          onKeyDown={(event) => { if (event.key === "Enter" && !event.shiftKey) { event.preventDefault(); event.currentTarget.form?.requestSubmit(); } }}
        />
        <div className="field-note label-buttons">
          <button className="button button-small button-primary" type="submit" disabled={asking || !text.trim()}>{asking ? "Writing…" : mode === "chat" ? "Send" : "Carry on"}</button>
          {exchanges.length > 0 && <button className="button button-small button-quiet" type="button" disabled={asking} onClick={() => setExchanges([])}>Start again</button>}
        </div>
      </form>
      {asking && exchanges.length === 0 && <p className="muted small field-note" role="status">Loading the writer. The first answer takes a few seconds longer.</p>}
      {problem && <p className="small field-note" role="alert">{problem}</p>}
      <p className="muted small field-note">
        {beginner
          ? "It's a small model, so it can get facts wrong, repeat itself or wander off the subject. Check what it writes before you use it."
          : "Sampling: temperature 0.7, top-k 40, repetition penalty 1.2. Both writers get the same random choices, so differences come from training."}
      </p>
    </>
  );
}

export function WriterResult({ model, beginner }: { model: SavedModel; beginner: boolean }) {
  const metrics = model.metrics;
  const [shown, setShown] = useState(0);
  const samples = metrics?.samples ?? [];
  const sample = samples[Math.min(shown, samples.length - 1)];
  const step = useCallback((change: number) => setShown((index) => (index + change + samples.length) % samples.length), [samples.length]);
  if (!metrics || metrics.task !== "writer") return null;
  const gain = (metrics.accuracyAfter ?? 0) - (metrics.accuracyBefore ?? 0);

  return (
    <section className="card">
      <h2>Newest writer</h2>
      <p className="small">
        On held-back text it never learned from, it guessed the next word right <strong>{percent(metrics.accuracyAfter)}</strong> of the time.
        {" "}The {metrics.model} it started from managed {percent(metrics.accuracyBefore)}.
        {gain >= 0.005 && ` That's ${Math.round(gain * 100)} points better at writing like your text.`}
      </p>
      {!beginner && metrics.lossBefore != null && metrics.lossAfter != null && (
        <p className="muted small field-note">
          Perplexity {Math.exp(metrics.lossAfter).toFixed(1)}, from {Math.exp(metrics.lossBefore).toFixed(1)}, on {(metrics.checkTokens ?? 0).toLocaleString()} held-back tokens.
          {" "}Best after pass {metrics.bestEpoch} of {model.totalEpochs ?? "?"}; last {metrics.adjustLayers} layers adjusted.
        </p>
      )}
      {metrics.warnings.map((text) => <p className="notice field-note" key={text}>{text}</p>)}
      {sample && (
        <div className="field-note">
          <div className="card-head">
            <h3 className="small">How it carries on from held-back text</h3>
            {samples.length > 1 && (
              <div className="label-buttons">
                <button className="button button-small button-quiet" aria-label="Previous example" onClick={() => step(-1)}>‹</button>
                <span className="muted small">{Math.min(shown, samples.length - 1) + 1} of {samples.length}</span>
                <button className="button button-small button-quiet" aria-label="Next example" onClick={() => step(1)}>›</button>
              </div>
            )}
          </div>
          <p className="writer-you"><span className="muted small">It was given</span>{sample.prompt}…</p>
          <div className="writer-answers is-pair is-three">
            <div className="writer-answer is-real"><span className="muted small">What you wrote</span><p>…{sample.real}</p></div>
            <div className="writer-answer"><span className="muted small">Adjusted writer</span><p>…{sample.after}</p></div>
            <div className="writer-answer is-original"><span className="muted small">As downloaded</span><p>…{sample.before}</p></div>
          </div>
        </div>
      )}
      <WriterChat model={model} beginner={beginner} />
    </section>
  );
}
