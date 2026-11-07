import { useCallback, useEffect, useRef, useState, type ChangeEvent } from "react";
import * as api from "./api";
import type { ProjectData, ProjectRow } from "./types";

interface Props {
  currentId: string;
  onOpen: (id: string, data: ProjectData) => void;
  onOpenFile: (data: unknown) => string | null;
  onSaveFile: () => void;
  onClose: () => void;
}

export function ProjectsDialog({ currentId, onOpen, onOpenFile, onSaveFile, onClose }: Props) {
  const [rows, setRows] = useState<ProjectRow[] | null>(null);
  const [problem, setProblem] = useState<string | null>(null);
  const fileInput = useRef<HTMLInputElement>(null);

  const load = useCallback(async () => {
    try {
      setRows(await api.fetchProjects());
    } catch (error) {
      setRows([]);
      setProblem(error instanceof Error ? error.message : "Your projects could not load.");
    }
  }, []);

  useEffect(() => { void load(); }, [load]);
  useEffect(() => {
    const onKey = (event: KeyboardEvent) => { if (event.key === "Escape") onClose(); };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose]);

  const open = async (id: string) => {
    try {
      onOpen(id, await api.fetchProject(id));
    } catch (error) {
      setProblem(error instanceof Error ? error.message : "That project could not open.");
    }
  };

  const remove = async (row: ProjectRow) => {
    if (!window.confirm(`Delete "${row.projectName}"? This can't be undone.`)) return;
    try {
      await api.deleteProject(row.id);
    } catch (error) {
      setProblem(error instanceof Error ? error.message : "That project could not be deleted.");
    }
    await load();
  };

  const readFile = async (event: ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    event.target.value = "";
    if (!file) return;
    let reason: string | null;
    try {
      reason = onOpenFile(JSON.parse(await file.text()));
    } catch {
      reason = "This file isn't an AI Bricks project.";
    }
    if (reason) setProblem(reason);
  };

  return (
    <div className="dialog-backdrop">
      <div className="dialog" role="dialog" aria-modal="true" aria-labelledby="projects-title">
        <header className="dialog-head">
          <div>
            <h1 id="projects-title">Projects</h1>
            <p className="muted">Your work saves automatically on this computer.</p>
          </div>
          <button className="button button-quiet" onClick={onClose}>Close</button>
        </header>
        <div className="dialog-body">
          {problem && <div className="result result-fail project-problem" role="alert"><p>{problem}</p></div>}
          {rows === null && <p className="muted">Loading…</p>}
          {rows?.length === 0 && !problem && <p className="muted">Nothing saved yet. Add a component and your model appears here.</p>}
          <ul className="project-list">
            {rows?.map((row) => (
              <li key={row.id} className={row.id === currentId ? "is-current" : ""}>
                <div className="project-text">
                  <strong>{row.projectName || "Untitled model"}</strong>
                  <span className="muted small">
                    {row.components} {row.components === 1 ? "component" : "components"} · {new Date(row.updatedAt * 1000).toLocaleString()}
                  </span>
                </div>
                {row.id === currentId
                  ? <span className="muted small">Open now</span>
                  : <>
                      <button className="button button-small button-quiet" aria-label={`Delete ${row.projectName}`} onClick={() => void remove(row)}>Delete</button>
                      <button className="button button-small" aria-label={`Open ${row.projectName}`} onClick={() => void open(row.id)}>Open</button>
                    </>}
              </li>
            ))}
          </ul>
        </div>
        <footer className="dialog-foot project-foot">
          <button className="button button-small" onClick={onSaveFile}>Save current model to a file</button>
          <button className="button button-small" onClick={() => fileInput.current?.click()}>Open a file</button>
          <input ref={fileInput} className="visually-hidden" type="file" accept=".json,application/json" aria-label="Open a project file" tabIndex={-1} onChange={(event) => void readFile(event)} />
        </footer>
      </div>
    </div>
  );
}
