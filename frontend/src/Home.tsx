import { useEffect, useState } from "react";
import * as api from "./api";
import type { DatasetSummary, Mode, ProjectRow } from "./types";

interface Props {
  mode: Mode;
  currentProjectId: string;
  onNewModel: () => void;
  onContinueModel: () => void;
  onOpenModel: (id: string) => void;
  onOpenData: () => void;
  onOpenDataset: (id: string) => void;
}

const RECENT_SHOWN = 5;

export function Home({ mode, currentProjectId, onNewModel, onContinueModel, onOpenModel, onOpenData, onOpenDataset }: Props) {
  const beginner = mode === "beginner";
  const [projects, setProjects] = useState<ProjectRow[] | null>(null);
  const [datasets, setDatasets] = useState<DatasetSummary[] | null>(null);
  const [problem, setProblem] = useState<string | null>(null);

  useEffect(() => {
    let stopped = false;
    Promise.all([api.fetchProjects(), api.fetchDatasets()])
      .then(([nextProjects, nextDatasets]) => {
        if (stopped) return;
        setProjects(nextProjects);
        setDatasets(nextDatasets);
      })
      .catch((error) => {
        if (stopped) return;
        setProjects([]);
        setDatasets([]);
        setProblem(error instanceof Error ? error.message : "Your work could not load.");
      });
    return () => { stopped = true; };
  }, []);

  return (
    <div className="page-scroll">
      <div className="page-inner home">
        <header className="home-head">
          <h1>{beginner ? "What would you like to do?" : "Home"}</h1>
          <p className="muted">
            {beginner
              ? "Get your data ready, then build a model that learns from it. You can start with either one."
              : "Prepare datasets and build models. Everything stays on this computer."}
          </p>
        </header>

        {problem && <div className="result result-fail" role="alert"><p>{problem}</p></div>}

        <div className="home-actions">
          <button className="home-action" onClick={onOpenData}>
            <strong>Label data</strong>
            <span className="muted">Find open datasets, or sort your own images and text by category.</span>
          </button>
          <button className="home-action" onClick={onNewModel}>
            <strong>Build a model</strong>
            <span className="muted">Connect blocks into a model, test it, train it and download the code.</span>
          </button>
        </div>

        <div className="home-lists">
          <section className="card">
            <h2>Models</h2>
            {projects?.length === 0 && <p className="muted small">Models you build appear here.</p>}
            <ul className="project-list">
              {projects?.slice(0, RECENT_SHOWN).map((row) => (
                <li key={row.id}>
                  <div className="project-text">
                    <strong>{row.projectName || "Untitled model"}</strong>
                    <span className="muted small">
                      {row.components} {row.components === 1 ? "component" : "components"} · {new Date(row.updatedAt * 1000).toLocaleDateString()}
                    </span>
                  </div>
                  <button
                    className="button button-small"
                    aria-label={`Open ${row.projectName || "Untitled model"}`}
                    onClick={() => (row.id === currentProjectId ? onContinueModel() : onOpenModel(row.id))}
                  >
                    {row.id === currentProjectId ? "Continue" : "Open"}
                  </button>
                </li>
              ))}
            </ul>
          </section>
          <section className="card">
            <h2>Datasets</h2>
            {datasets?.length === 0 && <p className="muted small">Datasets you create appear here.</p>}
            <ul className="project-list">
              {datasets?.slice(0, RECENT_SHOWN).map((row) => (
                <li key={row.id}>
                  <div className="project-text">
                    <strong>{row.name}</strong>
                    <span className="muted small">
                      {row.kind === "image" ? "Images" : "Text"} · {row.labelled.toLocaleString()} labelled{row.unsorted ? ` · ${row.unsorted.toLocaleString()} unsorted` : ""}
                    </span>
                  </div>
                  <button className="button button-small" aria-label={`Open ${row.name}`} onClick={() => onOpenDataset(row.id)}>Open</button>
                </li>
              ))}
            </ul>
          </section>
        </div>
      </div>
    </div>
  );
}
