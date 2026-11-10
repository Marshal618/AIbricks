import { useCallback, useEffect, useState } from "react";
import * as api from "./api";
import type { FailureSearch, SavedModel } from "./types";
import { Why } from "./Ui";

function percent(value: number) {
  return `${Math.round(value * 100)}%`;
}

/** Finds groups of held-back photos the model gets wrong far more often, and says what to do about each. */
export function WhereItFails({ model, datasetId, beginner, onAddPhotos, onVary }: {
  model: SavedModel; datasetId: string; beginner: boolean; onAddPhotos: () => void; onVary: () => void;
}) {
  const [search, setSearch] = useState<FailureSearch | null>(null);
  const [problem, setProblem] = useState<string | null>(null);

  const load = useCallback(() => api.fetchFailures(model.id).then(setSearch), [model.id]);
  useEffect(() => { load().catch((error) => setProblem(error instanceof Error ? error.message : "This couldn't load.")); }, [load]);
  const running = search?.status === "running";
  useEffect(() => {
    if (!running) return;
    const timer = window.setInterval(() => { load().catch(() => undefined); }, 1500);
    return () => window.clearInterval(timer);
  }, [running, load]);

  const start = async () => {
    setProblem(null);
    try {
      setSearch(await api.findFailures(model.id));
    } catch (error) {
      setProblem(error instanceof Error ? error.message : "It couldn't start.");
    }
  };

  if (!search && !problem) return null;
  const done = search?.status === "finished";
  return (
    <section className="card where-fails" aria-label="Where it fails">
      <div className="card-head">
        <h2>Where it fails</h2>
        {!running && <button className="button button-small" onClick={() => void start()}>{done || search?.status === "failed" ? "Look again" : "Look for patterns in its mistakes"}</button>}
      </div>
      <p className="muted small">Finds kinds of photos it gets wrong much more often than others, like dark ones or close-ups.</p>
      <Why>
        {beginner
          ? "An open model that understands photos and words, CLIP, looks at every held-back photo. Similar photos are grouped, and any group the model gets wrong far more often than the rest is described with the everyday words that fit it best. The words are a good guess, so check them against the example photos. It runs on this computer; the first time downloads CLIP, about 600 MB."
          : "Domino-style slice discovery (Eyuboglu et al., 2022): CLIP ViT-B/32 image embeddings, error-weighted k-means on held-back photos, slices named by the phrase most aligned with the slice-minus-rest embedding difference."}
      </Why>
      {problem && <p className="notice field-note" role="alert">{problem}</p>}
      {running && <p className="muted small field-note" role="status">{search?.stage === "loading" ? "Getting CLIP ready. The first time downloads about 600 MB…" : "Looking at the held-back photos…"}</p>}
      {search?.status === "failed" && <p className="notice field-note">It stopped with a problem: {search.error}</p>}
      {done && search.enough === false && (
        <p className="muted small field-note">It got {search.mistakes} of {search.held} held-back photos wrong. That's too few mistakes to find patterns in, which is good news. Add more photos and train again to check more widely.</p>
      )}
      {done && search.enough && search.slices?.length === 0 && (
        <p className="muted small field-note">No group of photos stands out. Its {search.mistakes} mistakes are spread across different kinds of photo.</p>
      )}
      {done && search.slices && search.slices.length > 0 && (
        <ul className="fail-list field-note">
          {search.slices.map((slice) => (
            <li key={slice.name}>
              <div className="fail-head">
                <strong>Looks like: {slice.name.replace(/^an? /, "")}</strong>
                <span className="fail-rate"><strong>{slice.wrong} of {slice.size}</strong> wrong ({percent(slice.rate)}) <span className="muted">vs {percent(search.overall ?? 0)} overall</span></span>
              </div>
              <p className="muted small">Also close: {slice.also.map((text) => text.replace(/^an? /, "")).join(", ")}. Most often it took {slice.confusion.actual} for {slice.confusion.guess}.</p>
              <div className="fail-examples">
                {slice.examples.map((example) => (
                  <img key={example.file} src={api.datasetPhotoUrl(datasetId, example.category, example.file)} alt={`${example.category}, guessed ${example.guess}`} title={`${example.category}, guessed ${example.guess}`} loading="lazy" />
                ))}
              </div>
              <div className="fail-advice">
                <span className="small">{slice.advice}</span>
                <button className="button button-small" onClick={onAddPhotos}>Add photos</button>
                {slice.action === "vary" && <button className="button button-small button-quiet" onClick={onVary}>Change training</button>}
              </div>
            </li>
          ))}
        </ul>
      )}
    </section>
  );
}
