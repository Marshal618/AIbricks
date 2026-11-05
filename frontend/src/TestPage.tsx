import type { Edge } from "@xyflow/react";
import { useEditor } from "./context";
import { LossChart, type LossPoint } from "./LossChart";
import { Outline } from "./Outline";
import type { BrickNodeType } from "./types";

export interface TestState {
  running: boolean;
  checks: Record<string, boolean>;
  points: LossPoint[];
  parameters: { total: number; trainable: number } | null;
  result: { passed: boolean; detail?: string } | null;
}

export const EMPTY_TEST: TestState = { running: false, checks: {}, points: [], parameters: null, result: null };

const CHECKS = [
  ["connections", "Forward pass completes", "Data flows through every component"],
  ["shapes", "Output shape matches the task", "The answer comes out in the right form"],
  ["loss", "Loss decreases on sample data", "The model learns from practice data"],
] as const;

const TOTAL_STEPS = 60;

interface Props {
  test: TestState;
  nodes: BrickNodeType[];
  edges: Edge[];
  onRun: () => void;
  onDesign: () => void;
}

export function TestPage({ test, nodes, edges, onRun, onDesign }: Props) {
  const { analysis, mode } = useEditor();
  const beginner = mode === "beginner";
  const ready = analysis?.ok ?? false;
  const tested = analysis?.tested ?? false;
  const started = test.running || test.result !== null;
  const problemCount = analysis?.issues.length ?? 0;

  return (
    <div className="page-scroll">
      <div className="page-inner">
        <header className="page-head">
          <div>
            <h1>Test run</h1>
            <p className="muted">
              {beginner
                ? "A quick practice run on made-up data, to prove your model works before you use real data. Your own data is never uploaded."
                : `Trains for ${TOTAL_STEPS} steps on generated sample data. Your own data is not required or uploaded.`}
            </p>
          </div>
          <button className={`button${tested ? "" : " button-primary"}`} disabled={!ready || test.running} onClick={onRun}>
            {test.running ? "Testing…" : started ? "Run again" : "Run test"}
          </button>
        </header>

        {!ready && (
          <div className="notice">
            <span>Resolve {problemCount} open {problemCount === 1 ? "problem" : "problems"} before testing.</span>
            <button className="button button-small" onClick={onDesign}>Go to Design</button>
          </div>
        )}
        {ready && !started && !tested && <div className="notice">This model has not been tested since its last change.</div>}
        {ready && !started && tested && <div className="notice notice-good">This exact model has already passed.</div>}

        {test.result && (
          <div className={`result ${test.result.passed ? "result-pass" : "result-fail"}`} role="status">
            <div>
              <strong>{test.result.passed ? "Test passed" : "Test failed"}</strong>
              <p>
                {test.result.passed
                  ? beginner
                    ? "Your model works and can learn. It hasn't seen your real data yet, so this doesn't tell you how accurate it will be. Export it and train it on your data to find out."
                    : "The model runs and its loss decreases on sample data. This does not indicate accuracy on your data, which only training and evaluation on that data can show."
                  : test.result.detail}
              </p>
            </div>
          </div>
        )}

        {started && (
          <div className="test-grid">
            <section className="card">
              <h2>Checks</h2>
              <ul className="check-list">
                {CHECKS.map(([key, technical, plain]) => {
                  const label = beginner ? plain : technical;
                  const state = test.checks[key];
                  const failedEarly = test.result && !test.result.passed && state === undefined;
                  return (
                    <li key={key} className={state === true ? "pass" : state === false ? "fail" : ""}>
                      <span className="check-mark" aria-hidden="true">{state === true ? "✓" : state === false ? "✕" : failedEarly ? "–" : "…"}</span>
                      {label}
                      <span className="visually-hidden">{state === true ? " passed" : state === false ? " failed" : " not finished"}</span>
                    </li>
                  );
                })}
              </ul>
              {test.parameters && !beginner && (
                <dl className="prop-table">
                  <dt>Total parameters</dt><dd className="mono">{test.parameters.total.toLocaleString()}</dd>
                  <dt>Trainable</dt><dd className="mono">{test.parameters.trainable.toLocaleString()}</dd>
                </dl>
              )}
            </section>
            <section className="card">
              <LossChart points={test.points} totalSteps={TOTAL_STEPS} />
              {beginner && <p className="muted small field-note">Loss measures how wrong the model is. A falling line means it is learning.</p>}
            </section>
          </div>
        )}

        <section className="card">
          <h2>Model summary</h2>
          <Outline nodes={nodes} edges={edges} />
        </section>
      </div>
    </div>
  );
}
