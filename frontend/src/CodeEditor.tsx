import { python } from "@codemirror/lang-python";
import { setDiagnostics, type Diagnostic } from "@codemirror/lint";
import { EditorView, basicSetup } from "codemirror";
import { forwardRef, useEffect, useImperativeHandle, useRef } from "react";

export interface EditorProblem {
  line: number | null;
  message: string;
}

export interface CodeEditorHandle {
  goToLine: (line: number) => void;
}

interface Props {
  value: string;
  problems: EditorProblem[];
  onChange: (value: string) => void;
}

/** A Python editor. The text is only ever sent to be parsed, never run. */
export const CodeEditor = forwardRef<CodeEditorHandle, Props>(function CodeEditor({ value, problems, onChange }, ref) {
  const host = useRef<HTMLDivElement>(null);
  const view = useRef<EditorView | null>(null);
  const latestChange = useRef(onChange);
  latestChange.current = onChange;

  useEffect(() => {
    const editor = new EditorView({
      doc: value,
      parent: host.current!,
      extensions: [
        basicSetup,
        python(),
        EditorView.lineWrapping,
        EditorView.contentAttributes.of({ "aria-label": "model.py code editor" }),
        EditorView.updateListener.of((update) => {
          if (update.docChanged) latestChange.current(update.state.doc.toString());
        }),
      ],
    });
    view.current = editor;
    return () => { editor.destroy(); view.current = null; };
  // The editor is created once. Later values arrive through the effect below.
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => {
    const editor = view.current;
    if (editor && editor.state.doc.toString() !== value) {
      editor.dispatch({ changes: { from: 0, to: editor.state.doc.length, insert: value } });
    }
  }, [value]);

  useEffect(() => {
    const editor = view.current;
    if (!editor) return;
    const doc = editor.state.doc;
    const diagnostics: Diagnostic[] = problems
      .filter((problem) => problem.line !== null && problem.line >= 1 && problem.line <= doc.lines)
      .map((problem) => {
        const line = doc.line(problem.line as number);
        return { from: line.from, to: line.to, severity: "error", message: problem.message };
      });
    editor.dispatch(setDiagnostics(editor.state, diagnostics));
  }, [problems, value]);

  useImperativeHandle(ref, () => ({
    goToLine(number: number) {
      const editor = view.current;
      if (!editor || number < 1 || number > editor.state.doc.lines) return;
      const line = editor.state.doc.line(number);
      editor.dispatch({ selection: { anchor: line.from }, effects: EditorView.scrollIntoView(line.from, { y: "center" }) });
      editor.focus();
    },
  }), []);

  return <div className="code-editor" ref={host} />;
});
