import type { ReactNode } from "react";

/** A short "Why?" that opens to the longer explanation, so screens stay to the point. */
export function Why({ children, label = "Why?" }: { children: ReactNode; label?: string }) {
  return (
    <details className="why">
      <summary>{label}</summary>
      <div className="why-body muted small">{children}</div>
    </details>
  );
}

/** A heading for one step of a screen, numbered like assembly instructions. */
export function StepHeading({ number, title, done, children }: { number?: number; title: string; done?: boolean; children?: ReactNode }) {
  return (
    <div className="step-heading">
      {number !== undefined && <span className={`step-number${done ? " is-done" : ""}`} aria-hidden="true">{done ? "✓" : number}</span>}
      <h2>{number !== undefined && <span className="visually-hidden">Step {number}{done ? ", done" : ""}: </span>}{title}</h2>
      {children && <div className="step-heading-side">{children}</div>}
    </div>
  );
}
