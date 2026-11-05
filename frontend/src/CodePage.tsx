import { Suspense, lazy, useEffect, useRef, useState } from "react";
import * as api from "./api";
import type { CodeEditorHandle } from "./CodeEditor";
import { useEditor } from "./context";
import type { GraphPayload, ParseReply } from "./types";

// The editor is large, so it only loads when someone opens it in Advanced mode.
const CodeEditor = lazy(() => import("./CodeEditor").then((module) => ({ default: module.CodeEditor })));

const FILE_NOTES: Record<string, string> = {
  "model.py": "Connects the components",
  "bricks.py": "One module per component type",
  "data.py": "Loads your data",
  "train.py": "Training loop",
  "generate.py": "Produces text from the model",
  "smoke_test.py": "Repeats the test run",
  "sample_data.py": "Generated test data",
  "requirements.txt": "Python packages",
  "README.md": "Project guide",
};
const ORDER = Object.keys(FILE_NOTES);

interface Props {
  graph: GraphPayload;
  draft: { text: string; base: string } | null;
  onDraft: (draft: { text: string; base: string } | null) => void;
  onApply: (reply: ParseReply) => void;
  onDesign: () => void;
  onAdvanced: () => void;
}

export function CodePage({ graph, draft, onDraft, onApply, onDesign, onAdvanced }: Props) {
  const { analysis, mode, catalogue } = useEditor();
  const advanced = mode === "advanced";
  const files = analysis?.files ?? {};
  const generatedModel = files["model.py"] as string | undefined;
  const names = ORDER.filter((name) => name in files);
  const [active, setActive] = useState("model.py");
  const [copied, setCopied] = useState(false);
  const [check, setCheck] = useState<ParseReply | null>(null);
  const [checking, setChecking] = useState(false);
  const [showReference, setShowReference] = useState(true);
  const editor = useRef<CodeEditorHandle>(null);
  const latestGraph = useRef(graph);
  latestGraph.current = graph;

  const baseText = generatedModel ?? catalogue.codeStarter;
  const editorText = draft?.text ?? baseText;
  const hasEdits = draft !== null && draft.text !== draft.base;
  const blocksMoved = draft !== null && draft.base !== baseText;
  const editing = advanced && active === "model.py";

  // Check the code as it is typed. This only parses it.
  useEffect(() => {
    if (!editing) return;
    const controller = new AbortController();
    setChecking(true);
    const timer = window.setTimeout(async () => {
      try {
        const reply = await api.parseModel(editorText, latestGraph.current, controller.signal);
        if (!controller.signal.aborted) setCheck(reply);
      } catch { /* the status bar already reports a lost server */ }
      if (!controller.signal.aborted) setChecking(false);
    }, 350);
    return () => { window.clearTimeout(timer); controller.abort(); };
  }, [editing, editorText]);

  const copy = async (text: string) => {
    try {
      await navigator.clipboard.writeText(text);
      setCopied(true);
      window.setTimeout(() => setCopied(false), 1500);
    } catch { /* clipboard unavailable; the code can still be selected */ }
  };

  if (!names.length && !advanced) {
    return (
      <div className="page-empty">
        <h2>No code yet</h2>
        <p className="muted">Your code appears here when the model has no open problems.</p>
        <button className="button" onClick={onDesign}>Go to Design</button>
      </div>
    );
  }

  const listed = names.length ? names : ["model.py"];
  const current = listed.includes(active) ? active : listed[0];
  const text = files[current] ?? "";
  const lines = text.replace(/\n$/, "").split("\n");
  const problems = check?.errors ?? [];

  return (
    <div className="code-page">
      <nav className="file-tree" aria-label="Project files">
        <h2 className="pane-title">Project files</h2>
        {listed.map((name) => (
          <button key={name} className={name === current ? "active" : ""} aria-current={name === current} onClick={() => setActive(name)}>
            <span className="mono">{name}{name === "model.py" && hasEdits ? " •" : ""}</span>
            <small>{FILE_NOTES[name]}{name === "model.py" && advanced ? " (editable)" : ""}</small>
          </button>
        ))}
      </nav>

      {current === "model.py" && advanced ? (
        <div className="code-viewer">
          <div className="code-head">
            <span className="mono">model.py</span>
            <span className={`small ${problems.length ? "tone-bad" : check ? "tone-good" : "muted"}`}>
              {checking ? "Checking…" : problems.length ? `${problems.length} ${problems.length === 1 ? "line" : "lines"} can't become blocks` : check ? "Ready to convert" : ""}
            </span>
            <span className="spacer" />
            <button className="button button-small" onClick={() => setShowReference(!showReference)}>{showReference ? "Hide" : "Show"} components</button>
            <button className="button button-small" disabled={!hasEdits} onClick={() => { onDraft(null); setCheck(null); }}>Discard edits</button>
            <button
              className="button button-small button-primary"
              disabled={(!hasEdits && generatedModel !== undefined) || checking || !check?.ok}
              onClick={() => { if (check?.ok) { onApply(check); onDraft(null); } }}
            >
              Convert to blocks
            </button>
          </div>
          {blocksMoved && hasEdits && (
            <div className="code-notice">The blocks changed after you started editing. Converting replaces those changes with this code.</div>
          )}
          <div className="editor-split">
            <Suspense fallback={<div className="code-editor muted empty-note">Loading the editor…</div>}>
              <CodeEditor
                ref={editor}
                value={editorText}
                problems={problems}
                onChange={(next) => { if (next !== editorText) onDraft({ text: next, base: draft?.base ?? baseText }); }}
              />
            </Suspense>
            {showReference && (
              <aside className="code-reference" aria-label="Available components">
                <h2 className="pane-title">Components you can use</h2>
                <ul>
                  {catalogue.codeReference.map((item) => <li key={item.name} className="mono">{item.signature}</li>)}
                </ul>
              </aside>
            )}
          </div>
          <div className="code-messages" aria-live="polite">
            {problems.map((problem, index) => (
              <button key={index} className="code-message is-error" onClick={() => problem.line && editor.current?.goToLine(problem.line)}>
                {problem.line && <span className="mono">Line {problem.line}</span>}
                <span>{problem.message}</span>
              </button>
            ))}
            {(check?.notes ?? []).map((note, index) => (
              <button key={index} className="code-message" onClick={() => note.line && editor.current?.goToLine(note.line)}>
                <span className="mono">Line {note.line}</span>
                <span>{note.message}</span>
              </button>
            ))}
            {!problems.length && !check?.notes.length && (
              <p className="muted small">Converting rebuilds the blocks from this code, then regenerates it. Your comments and formatting aren't kept. Custom Python can be added after export.</p>
            )}
          </div>
        </div>
      ) : (
        <div className="code-viewer">
          <div className="code-head">
            <span className="mono">{current}</span>
            <span className="muted small">{lines.length} lines</span>
            <span className="spacer" />
            {!advanced && current === "model.py" && <button className="button button-small" onClick={onAdvanced}>Edit in Advanced</button>}
            <button className="button button-small" onClick={() => copy(text)}>{copied ? "Copied" : "Copy"}</button>
          </div>
          <div className="code-scroll" tabIndex={0}>
            <table>
              <tbody>
                {lines.map((line, index) => (
                  <tr key={index} className={line.includes("PUT YOUR DATA HERE") ? "line-mark" : ""}>
                    <td className="line-no">{index + 1}</td>
                    <td className="line-text">{line || " "}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  );
}
