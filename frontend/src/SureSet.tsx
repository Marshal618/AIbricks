import { useState } from "react";
import type { SureSummary } from "./types";
import { Why } from "./Ui";

const NAMES: Record<string, string> = { "0.8": "8 in 10", "0.9": "9 in 10", "0.95": "19 in 20" };
const OFTEN: Record<string, string> = { "0.8": "8 times in 10", "0.9": "9 times in 10", "0.95": "19 times in 20" };
const STORE = "ai-bricks-sure-level";

function savedLevel(): string | null {
  try {
    return localStorage.getItem(STORE);
  } catch {
    return null;
  }
}

/** "A or B": every answer the model can't rule out, right about as often as chosen. */
export function SureSet({ answers, sure, beginner }: { answers: { category: string; chance: number }[]; sure: SureSummary | null | undefined; beginner: boolean }) {
  const levels = sure ? Object.keys(sure.levels).sort() : [];
  const [chosen, setChosen] = useState<string | null>(savedLevel);
  if (!sure || !levels.length || !answers.length) return null;
  const level = chosen && levels.includes(chosen) ? chosen : levels.includes("0.9") ? "0.9" : levels[levels.length - 1];
  const row = sure.levels[level];
  const ranked = [...answers].sort((a, b) => b.chance - a.chance);
  const members = ranked.filter((answer) => answer.chance >= row.cutoff - 1e-9);
  const set = members.length ? members : ranked.slice(0, 1);
  const everything = set.length === ranked.length && ranked.length > 1;
  const pick = (next: string) => {
    setChosen(next);
    try { localStorage.setItem(STORE, next); } catch { /* the choice just isn't remembered */ }
  };

  return (
    <div className="sure-set field-note" role="status">
      <p className="sure-line">
        <span className="muted small">Sure set</span>
        <strong>{set.map((answer) => answer.category).join(" or ")}</strong>
      </p>
      <p className="muted small">
        {everything
          ? "It can't rule any answer out for this one, so it isn't safe to trust its top guess here."
          : set.length === 1
            ? `Sure enough to name just one answer. Sets like this hold the right answer about ${OFTEN[level]}.`
            : `It can't choose between these. Sets like this hold the right answer about ${OFTEN[level]}.`}
      </p>
      {levels.length > 1 && (
        <fieldset className="chip-group sure-levels">
          <legend className="small muted">Right about</legend>
          {levels.map((value) => (
            <label key={value} className={`chip${value === level ? " is-on" : ""}`}>
              <input type="radio" name="sure-level" checked={value === level} onChange={() => pick(value)} />{NAMES[value] ?? value}
            </label>
          ))}
        </fieldset>
      )}
      <Why label={beginner ? "What's a sure set?" : "Method"}>
        {beginner ? (
          <>
            Instead of one guess, it names every answer it can't rule out. To set how strict to be, it looked at the{" "}
            {sure.examples} held-back examples it never trained on and how much chance it gave the right answer each time.
            On those, judging each one against the others, sets like this held the right answer {Math.round(row.right * 100)}% of the time
            and named {row.size.toFixed(1)} answers on average. Asking to be right more often makes the sets bigger. It holds for
            new examples that are like the ones you trained on, so it's a guide for unusual ones.
          </>
        ) : (
          <>
            Split conformal prediction on {sure.examples} held-back examples: score 1 − p(true), cut-off at the ⌈(n+1)(1−α)⌉-th
            smallest score, so p ≥ {row.cutoff.toFixed(3)} joins the set. Leave-one-out coverage {Math.round(row.right * 100)}%,
            mean set size {row.size.toFixed(2)}. Checkpoint selection used the same examples, so coverage is close to, not
            exactly, guaranteed.
          </>
        )}
      </Why>
    </div>
  );
}
