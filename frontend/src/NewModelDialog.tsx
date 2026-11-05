import { useEffect } from "react";
import { useEditor } from "./context";
import type { Kit } from "./types";

interface Props {
  hasModel: boolean;
  onKit: (kit: Kit) => void;
  onBlank: () => void;
  onCancel: () => void;
}

export function NewModelDialog({ hasModel, onKit, onBlank, onCancel }: Props) {
  const { catalogue, mode } = useEditor();
  const beginner = mode === "beginner";
  useEffect(() => {
    if (!hasModel) return;
    const onKey = (event: KeyboardEvent) => { if (event.key === "Escape") onCancel(); };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [hasModel, onCancel]);
  const domains = [...new Set(catalogue.kits.map((kit) => kit.domain))];
  return (
    <div className="dialog-backdrop">
      <div className="dialog" role="dialog" aria-modal="true" aria-labelledby="new-model-title">
        <header className="dialog-head">
          <div>
            <h1 id="new-model-title">{beginner ? "What do you want to build?" : "New model"}</h1>
            <p className="muted">{beginner ? "Pick a goal and we'll set up most of it for you." : "Start from a template or a blank canvas."}{hasModel && " This replaces the current model."}</p>
          </div>
          {hasModel && <button className="button button-quiet" onClick={onCancel}>Cancel</button>}
        </header>
        <div className="dialog-body">
          {domains.map((domain) => (
            <section key={domain}>
              <h2 className="pane-title">{domain}</h2>
              <div className="kit-list">
                {catalogue.kits.filter((kit) => kit.domain === domain).map((kit) => (
                  <button key={kit.id} className="kit-row" onClick={() => onKit(kit)}>
                    <span>
                      <strong>{beginner ? kit.goal : kit.name}</strong>
                      <span className="kit-desc">{beginner ? `Example: ${kit.name}` : kit.audience}</span>
                    </span>
                    {!beginner && <span className="kit-family">{kit.family}</span>}
                  </button>
                ))}
              </div>
            </section>
          ))}
        </div>
        <footer className="dialog-foot">
          <button className="button" onClick={onBlank}>{beginner ? "I'll start from scratch" : "Start with a blank canvas"}</button>
        </footer>
      </div>
    </div>
  );
}
