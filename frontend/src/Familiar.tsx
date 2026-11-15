import { useEffect, useState } from "react";
import * as api from "./api";
import type { FamiliarCheck, FamiliarStatus } from "./types";
import { message, Why } from "./Ui";

const WAIT_MS = 1500;

/** Says when a tried photo or text is unlike anything the model learned from, and shows the closest ones. */
export function FamiliarNote({ jobId, file, text, beginner }: { jobId: string; file?: File | null; text?: string; beginner: boolean }) {
  const [phase, setPhase] = useState<"checking" | "preparing" | "done" | "few" | "failed">("checking");
  const [status, setStatus] = useState<FamiliarStatus | null>(null);
  const [check, setCheck] = useState<FamiliarCheck | null>(null);
  const [problem, setProblem] = useState<string | null>(null);
  const things = file ? "photos" : "texts";

  useEffect(() => {
    if (!file && !text) return;
    let alive = true;
    setPhase("checking");
    setCheck(null);
    setProblem(null);
    (async () => {
      let state = await api.fetchFamiliar(jobId);
      if (state.status === "missing" || state.status === "failed") state = await api.prepareFamiliar(jobId);
      while (alive && state.status === "building") {
        setPhase("preparing");
        await new Promise((resolve) => setTimeout(resolve, WAIT_MS));
        state = await api.fetchFamiliar(jobId);
      }
      if (!alive) return;
      setStatus(state);
      if (state.status !== "ready") throw new Error(state.error ?? "The check couldn't be prepared.");
      if (!state.canWarn) { setPhase("few"); return; }
      const reply = file ? await api.checkFamiliarPhoto(jobId, file) : await api.checkFamiliarText(jobId, text ?? "");
      if (!alive) return;
      setCheck(reply);
      setPhase("done");
    })().catch((error) => {
      if (!alive) return;
      setProblem(message(error, "The check couldn't run."));
      setPhase("failed");
    });
    return () => { alive = false; };
  }, [jobId, file, text]);

  if (phase === "checking") return null;
  if (phase === "preparing") {
    return <p className="muted small field-note" role="status">Comparing with the {things} it learned from. The first time takes a little while…</p>;
  }
  if (phase === "failed") return <p className="muted small field-note">Couldn't check how familiar this is: {problem}</p>;
  if (phase === "few" || !check) {
    return (
      <p className="muted small field-note">
        {beginner
          ? `It has too few ${things} to tell when something is unlike them. With about 20 or more, it can warn you.`
          : `Familiarity check off: ${status?.usual ?? 0} reference distances, at least 19 needed.`}
      </p>
    );
  }

  const shown = check.unfamiliar ? check.nearest : check.nearest.slice(0, 1);
  const percent = Math.min(99, Math.round(check.farther * 100));
  return (
    <div className={`familiar field-note${check.unfamiliar ? " is-unfamiliar" : ""}`} role="status">
      {check.unfamiliar ? (
        <>
          <p className="familiar-head"><strong>This doesn't look like what it learned from</strong></p>
          <p className="small">
            It's farther from the {things} it learned from than {percent}% of the ones it was checked on, so its answer is a
            guess about something new. Add {things} like this one to the dataset if it should know them.
          </p>
        </>
      ) : (
        <p className="small familiar-head"><strong>Looks like what it learned from.</strong></p>
      )}
      <div className="familiar-nearest">
        <span className="muted small">{check.unfamiliar ? "Closest it learned from" : "Closest one"}</span>
        <ul>
          {shown.map((row) => (
            <li key={row.index}>
              {file
                ? <img src={api.familiarPhotoUrl(jobId, row.index)} alt={`A training photo labelled ${row.category}`} loading="lazy" />
                : <q className="small">{row.text ?? "(since deleted)"}</q>}
              <span className="small muted">{row.category}</span>
            </li>
          ))}
        </ul>
      </div>
      <Why label={beginner ? "How does it know?" : "Method"}>
        {beginner ? (
          <>
            A model always picks one of its answers, even for something it has never seen. So this compares the{" "}
            {file ? "photo" : "text"} with the {status?.known} {things} it learned from, using a ready-made model that describes{" "}
            {things}, and measures how far it is from the closest few. It also measured that for the {things} it learned from and the
            ones held back to check it, which shows how far familiar {things} usually are. When a {file ? "photo" : "text"} is
            farther than 19 in 20 of those, it warns you. So about 1 ordinary {file ? "photo" : "text"} in 20 gets a warning it
            didn't need.
          </>
        ) : (
          <>
            Deep k-nearest-neighbour distance (k = 5, Sun et al. 2022) on {file ? "ResNet-18" : "MiniLM"} embeddings, with a
            conformal p-value (Bates et al. 2023) against {check.usual} reference distances: held-back examples against the
            training set, plus leave-one-out training distances. Flagged at p ≤ {check.level}; this one has p ={" "}
            {check.pValue.toFixed(3)}. The false-alarm rate is about {Math.round(check.level * 100)}% for examples like the training
            data; detection depends on how different an example is.
          </>
        )}
      </Why>
    </div>
  );
}
