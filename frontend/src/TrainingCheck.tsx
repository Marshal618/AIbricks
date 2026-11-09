import type { CheckupFinding, FixKind } from "./types";
import { Why } from "./Ui";

const FIX_LABELS: Record<FixKind, string> = {
  augment: "Vary training photos",
  slower: "Learn more slowly",
  longer: "Train for longer",
  pretrained: "Try a pretrained model",
  more_photos: "Add more photos",
  experiment: "Find a better model",
};

const TONE: Record<CheckupFinding["kind"], { mark: string; className: string }> = {
  healthy: { mark: "✓", className: "is-good" },
  still_learning: { mark: "↗", className: "is-info" },
  memorising: { mark: "!", className: "is-warn" },
  too_simple: { mark: "!", className: "is-warn" },
  unstable: { mark: "!", className: "is-warn" },
};

/**
 * What the loss chart says, in plain words, with buttons for the fixes this screen can apply.
 * Fixes it can't apply here are listed as advice.
 */
export function TrainingCheck({ findings, onFix, available }: {
  findings: CheckupFinding[]; onFix?: (fix: FixKind) => void; available: FixKind[];
}) {
  if (!findings.length) return null;
  return (
    <section className="card training-check" aria-label="Training check">
      <h2>Training check</h2>
      <ul className="check-findings">
        {findings.map((finding) => {
          const usable = finding.fixes.filter((fix) => available.includes(fix) && onFix);
          const advice = finding.fixes.filter((fix) => !usable.includes(fix));
          return (
            <li key={finding.kind} className={TONE[finding.kind].className}>
              <span className="check-mark" aria-hidden="true">{TONE[finding.kind].mark}</span>
              <div className="check-body">
                <strong>{finding.title}</strong>
                <p className="muted small">{finding.detail}</p>
                {(usable.length > 0 || advice.length > 0) && (
                  <div className="check-fixes">
                    {usable.map((fix) => <button key={fix} className="button button-small" onClick={() => onFix?.(fix)}>{FIX_LABELS[fix]}</button>)}
                    {advice.length > 0 && <span className="muted small">{usable.length ? "Or: " : "Try: "}{advice.map((fix) => FIX_LABELS[fix].toLowerCase()).join(", ")}.</span>}
                  </div>
                )}
              </div>
            </li>
          );
        })}
      </ul>
      <Why>
        These are the checks an experienced person makes by looking at the loss chart: whether the model is memorising its
        training photos, is too simple to learn, took steps that were too big, or stopped before it finished learning.
      </Why>
    </section>
  );
}
