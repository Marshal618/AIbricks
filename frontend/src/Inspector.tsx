import { useEditor } from "./context";
import { PortShape } from "./PortGlyph";
import type { BrickNodeType, ParamValue, TrainingOverride } from "./types";

interface Props {
  node: BrickNodeType | undefined;
  training: string;
  override: TrainingOverride | null;
  nodeCount: number;
  edgeCount: number;
  onTraining: (key: string) => void;
  onOverride: (override: TrainingOverride | null) => void;
  onChange: (nodeId: string, key: string, value: ParamValue) => void;
  onDelete: (nodeId: string) => void;
}

const PLAIN_TRAINING: Record<string, string> = {
  quick: "A short first try. Good for checking your data works.",
  balanced: "A sensible default for most projects.",
  thorough: "Takes longer, and can be more accurate with enough data.",
};
const TRAINING_FIELDS = [["epochs", "Epochs", "1"], ["batch_size", "Batch size", "1"], ["learning_rate", "Learning rate", "any"]] as const;

function ModelProperties({ training, override, nodeCount, edgeCount, onTraining, onOverride }: Pick<Props, "training" | "override" | "nodeCount" | "edgeCount" | "onTraining" | "onOverride">) {
  const { catalogue, analysis, mode } = useEditor();
  const profile = catalogue.training[training];
  const beginner = mode === "beginner";
  const isCustom = override !== null && Object.keys(override).length > 0;
  return (
    <div className="props">
      <h2>{beginner ? "Your model" : "Model properties"}</h2>
      <dl className="prop-table">
        <dt>Components</dt><dd>{nodeCount}</dd>
        {!beginner && <><dt>Connections</dt><dd>{edgeCount}</dd></>}
        <dt>{beginner ? "To fix" : "Open problems"}</dt><dd>{analysis?.issues.length ?? 0}</dd>
      </dl>
      <h3>{beginner ? "How long to train" : "Training profile"}</h3>
      <div className="segmented" role="radiogroup" aria-label="Training profile">
        {Object.entries(catalogue.training).map(([key, value]) => (
          <button
            key={key}
            role="radio"
            aria-checked={!isCustom && training === key}
            className={!isCustom && training === key ? "active" : ""}
            onClick={() => { onTraining(key); onOverride(null); }}
          >
            {value.name}
          </button>
        ))}
      </div>
      {beginner ? (
        <p className="muted small field-note">{isCustom ? "Custom values are set in Advanced mode." : PLAIN_TRAINING[training]}</p>
      ) : (
        <div className="training-fields">
          {TRAINING_FIELDS.map(([key, label, step]) => (
            <div className="field" key={key}>
              <label htmlFor={`training-${key}`}>{label}</label>
              <input
                id={`training-${key}`}
                type="number"
                step={step}
                value={String(override?.[key] ?? profile[key])}
                onChange={(event) => {
                  const number = event.target.valueAsNumber;
                  onOverride({ ...override, [key]: Number.isNaN(number) ? "" : number });
                }}
              />
            </div>
          ))}
          {isCustom && <p className="muted small">Custom values. Choose a profile above to reset them.</p>}
        </div>
      )}
      <p className="muted small field-note">
        {beginner ? "Select a component to see what it does." : "These values are written to the exported training script. Select a component to edit its settings."}
      </p>
    </div>
  );
}

export function Inspector({ node, training, override, nodeCount, edgeCount, onTraining, onOverride, onChange, onDelete }: Props) {
  const { bricks, analysis, catalogue, openBigBrick, mode } = useEditor();
  if (!node) return <ModelProperties training={training} override={override} nodeCount={nodeCount} edgeCount={edgeCount} onTraining={onTraining} onOverride={onOverride} />;
  const beginner = mode === "beginner";
  const brick = bricks[node.data.brickId];
  const issues = analysis?.issues.filter((issue) => issue.node === node.id) ?? [];
  const typeName = (type: string) => catalogue.portTypes[type]?.name ?? type;
  const groupLabel = catalogue.groups.find((group) => group.id === brick.group)?.label;
  const params = beginner ? brick.params.filter((param) => param.basic) : brick.params;
  const hiddenSettings = brick.params.length - params.length;
  return (
    <div className="props">
      <div className="prop-kicker">{groupLabel}</div>
      <h2>{brick.name}</h2>
      <p className="muted">{brick.description}</p>
      {beginner && brick.why && <p className="when-to-use">{brick.why}</p>}
      {issues.map((issue) => <p key={issue.message} className="issue-text" role="alert">{issue.message}</p>)}

      <h3>Data flow</h3>
      <dl className="prop-table">
        {brick.inputs.map((port) => (
          <div key={port.id} className="prop-row">
            <dt>{port.label || "Input"}</dt>
            <dd className="prop-port">{port.accepts.map((type) => <PortShape key={type} type={type} />)}{port.accepts.map(typeName).join(" or ")}</dd>
          </div>
        ))}
        {brick.output && <div className="prop-row"><dt>Output</dt><dd className="prop-port"><PortShape type={brick.output} />{typeName(brick.output)}</dd></div>}
        {analysis?.shapes[node.id] && <div className="prop-row"><dt>Shape</dt><dd className="mono">{analysis.shapes[node.id]}</dd></div>}
      </dl>

      {brick.parts.length > 0 && (
        <>
          <h3>Contains</h3>
          <ol className="contains">{brick.parts.map((part, index) => <li key={index}>{bricks[part.brick].name}</li>)}</ol>
          <button className="button" onClick={() => openBigBrick(node.id)}>Expand into layers</button>
        </>
      )}

      {params.length > 0 && <h3>Settings</h3>}
      {beginner && hiddenSettings > 0 && (
        <p className="muted small field-note">Using good defaults. {hiddenSettings} more {hiddenSettings === 1 ? "setting is" : "settings are"} in Advanced mode.</p>
      )}
      {params.map((param) => {
        const value = node.data.params[param.key] ?? param.default;
        const inputId = `param-${node.id}-${param.key}`;
        return (
          <div className="field" key={param.key}>
            {param.kind === "bool" ? (
              <label className="check-row" htmlFor={inputId}>
                <input id={inputId} type="checkbox" checked={Boolean(value)} onChange={(event) => onChange(node.id, param.key, event.target.checked)} />
                {param.label}
              </label>
            ) : (
              <label htmlFor={inputId}>{param.label}</label>
            )}
            {param.kind === "int" && (
              <input
                id={inputId}
                type="number"
                min={param.min ?? undefined}
                max={param.max ?? undefined}
                value={String(value)}
                onChange={(event) => {
                  const number = event.target.valueAsNumber;
                  onChange(node.id, param.key, Number.isNaN(number) ? "" : number);
                }}
              />
            )}
            {param.kind === "names" && (
              <input id={inputId} type="text" value={String(value)} onChange={(event) => onChange(node.id, param.key, event.target.value)} />
            )}
            {param.kind === "choice" && (
              <select
                id={inputId}
                value={String(value)}
                onChange={(event) => {
                  const choice = param.choices?.find((c) => String(c) === event.target.value);
                  if (choice !== undefined) onChange(node.id, param.key, choice);
                }}
              >
                {param.choices?.map((choice) => <option key={String(choice)} value={String(choice)}>{String(choice)}</option>)}
              </select>
            )}
            {param.help && <small className="muted">{param.help}</small>}
          </div>
        );
      })}
      <button className="button button-danger" onClick={() => onDelete(node.id)}>Delete component</button>
    </div>
  );
}
