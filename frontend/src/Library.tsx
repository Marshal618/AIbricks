import { useEffect, useState } from "react";
import { useEditor } from "./context";
import { PortShape } from "./PortGlyph";
import type { Brick } from "./types";

export type LibraryFocus =
  | { state: "empty" }
  | { state: "open"; name: string; type: string }
  | { state: "complete" };

interface Props {
  onAdd: (brick: Brick) => void;
  focus: LibraryFocus;
}

export function Library({ onAdd, focus }: Props) {
  const { catalogue, bricks, mode } = useEditor();
  const beginner = mode === "beginner";
  const available = beginner ? catalogue.bricks.filter((brick) => brick.beginner) : catalogue.bricks;
  const hiddenCount = catalogue.bricks.length - available.length;
  const [query, setQuery] = useState("");
  const [showAll, setShowAll] = useState(false);
  const [toggled, setToggled] = useState<Record<string, boolean>>({});
  const [hovered, setHovered] = useState<string | null>(null);

  // Each new connector starts from a clean view: fitting groups open, the rest closed.
  const focusKey = focus.state === "open" ? focus.type : focus.state;
  useEffect(() => { setToggled({}); setShowAll(false); }, [focusKey]);

  const fits = (brick: Brick) => {
    if (focus.state === "empty") return brick.group === "socket";
    if (focus.state === "complete") return false;
    return brick.inputs.some((port) => port.accepts.includes(focus.type));
  };
  const search = query.trim().toLowerCase();
  const matches = (brick: Brick) => !search || `${brick.name} ${brick.description}`.toLowerCase().includes(search);
  const detail = hovered ? bricks[hovered] : null;
  const typeName = (type: string) => catalogue.portTypes[type]?.name ?? type;
  const anyVisible = available.some(matches);

  return (
    <div className="library">
      <div className="next-banner">
        {focus.state === "open" && (
          <>
            <PortShape type={focus.type} size={26} />
            <div className="next-text">
              <strong>{beginner ? `Next: add something that takes ${typeName(focus.type)}` : `Connect ${typeName(focus.type)}`}</strong>
              <span>{beginner ? catalogue.portTypes[focus.type]?.description : `from ${focus.name}`}</span>
            </div>
          </>
        )}
        {focus.state === "empty" && <div className="next-text"><strong>{beginner ? "First: what kind of data do you have?" : "Choose a data input"}</strong><span>Every model starts here</span></div>}
        {focus.state === "complete" && <div className="next-text"><strong>{beginner ? "Your model is connected" : "Nothing left to connect"}</strong><span>{beginner ? "Next, run the test" : "Select a component to branch from it"}</span></div>}
      </div>
      <div className="library-tools">
        <input
          type="search"
          placeholder="Search components"
          aria-label="Search components"
          value={query}
          onChange={(event) => setQuery(event.target.value)}
        />
        <label className="compat">
          <input type="checkbox" checked={showAll} onChange={(event) => setShowAll(event.target.checked)} />
          <span>Show all components</span>
        </label>
      </div>
      <div className="library-list">
        {catalogue.groups.map((group) => {
          const all = available.filter((brick) => brick.group === group.id && matches(brick));
          if (!all.length) return null;
          const fitting = all.filter(fits);
          const startsOpen = search !== "" || showAll || fitting.length > 0;
          const open = toggled[group.id] === undefined ? startsOpen : toggled[group.id];
          // Open groups list only what fits, unless there is nothing that fits or everything was asked for.
          const items = search || showAll || fitting.length === 0 ? all : fitting;
          return (
            <section key={group.id} className={fitting.length ? "" : "group-no-fit"}>
              <button
                className="group-header"
                aria-expanded={open}
                onClick={() => setToggled((state) => ({ ...state, [group.id]: !open }))}
              >
                <span className={`chevron${open ? " is-open" : ""}`} aria-hidden="true" />
                {group.name}
                <span className="group-count">{fitting.length ? `${fitting.length} fit` : "None fit"}</span>
              </button>
              {open && items.map((brick) => (
                <button
                  key={brick.id}
                  className={`lib-item${fits(brick) ? "" : " no-fit"}`}
                  draggable
                  onDragStart={(event) => {
                    event.dataTransfer.setData("application/ai-brick", brick.id);
                    event.dataTransfer.effectAllowed = "copy";
                  }}
                  onClick={() => onAdd(brick)}
                  onMouseEnter={() => setHovered(brick.id)}
                  onMouseLeave={() => setHovered(null)}
                  onFocus={() => setHovered(brick.id)}
                  onBlur={() => setHovered(null)}
                >
                  <span className="lib-name">{brick.name}</span>
                  <span className="lib-ports" aria-hidden="true">
                    {brick.inputs[0]?.accepts.map((type) => <PortShape key={type} type={type} />)}
                    {brick.inputs.length > 0 && brick.output && <span className="lib-arrow">→</span>}
                    {brick.output && <PortShape type={brick.output} />}
                  </span>
                </button>
              ))}
            </section>
          );
        })}
        {!anyVisible && <p className="muted empty-note">No matching components.</p>}
        {beginner && hiddenCount > 0 && <p className="muted small empty-note">Advanced mode has {hiddenCount} more components.</p>}
      </div>
      <div className="library-detail" aria-live="polite">
        {detail ? (
          <>
            <strong>{detail.name}</strong>
            <p>{beginner && detail.why ? detail.why : detail.description}</p>
            <p className="muted">
              {detail.inputs.length ? `In: ${detail.inputs[0].accepts.map(typeName).join(" or ")}` : "No input"}
              {" · "}
              {detail.output ? `Out: ${typeName(detail.output)}` : "Final output"}
            </p>
          </>
        ) : (
          <p className="muted">Click a component to connect it, or drag it onto the canvas.</p>
        )}
      </div>
    </div>
  );
}
