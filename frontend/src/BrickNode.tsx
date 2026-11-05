import { Handle, Position, type NodeProps } from "@xyflow/react";
import { useEditor } from "./context";
import { PortShape } from "./PortGlyph";
import type { BrickNodeType } from "./types";

export function BrickNode({ id, data, selected }: NodeProps<BrickNodeType>) {
  const { bricks, catalogue, analysis, openBigBrick, connecting, nextNodeId } = useEditor();
  const brick = bricks[data.brickId];
  const issues = analysis?.issues.filter((issue) => issue.node === id) ?? [];
  const shape = analysis?.shapes[id];
  const portName = (type: string) => catalogue.portTypes[type]?.name ?? type;
  const groupLabel = catalogue.groups.find((group) => group.id === brick.group)?.label;

  // While a connection is being dragged, only connectors that fit stay lit.
  const inputState = (accepts: string[]) => {
    if (!connecting) return "";
    if (connecting.nodeId === id) return "port-dim";
    return connecting.side === "source" && accepts.some((type) => connecting.types.includes(type)) ? "port-fit" : "port-dim";
  };
  const outputState = () => {
    if (connecting) {
      if (connecting.nodeId === id) return "";
      return connecting.side === "target" && connecting.types.includes(brick.output as string) ? "port-fit" : "port-dim";
    }
    return nextNodeId === id ? "port-next" : "";
  };

  return (
    <div className={`brick${selected ? " is-selected" : ""}${issues.length ? " has-issue" : ""}`}>
      {brick.inputs.map((port, index) => (
        <Handle
          key={port.id}
          id={port.id}
          type="target"
          position={Position.Top}
          className={`port${port.accepts.length > 1 ? " port-multi" : ""} ${inputState(port.accepts)}`}
          style={brick.inputs.length > 1 ? { left: `${((index + 1) * 100) / (brick.inputs.length + 1)}%` } : undefined}
          title={`Takes ${port.accepts.map(portName).join(" or ")}`}
        >
          {port.accepts.map((type) => <PortShape key={type} type={type} size={22} />)}
        </Handle>
      ))}
      <div className="brick-body">
        {brick.inputs.length > 1 && (
          <div className="port-labels">
            {brick.inputs.map((port) => <span key={port.id}>{port.label}</span>)}
          </div>
        )}
        <div className="brick-group">{groupLabel}</div>
        <div className="brick-title">
          <strong>{brick.name}</strong>
          {brick.parts.length > 0 && (
            <button
              className="open-up nodrag"
              onClick={(event) => { event.stopPropagation(); openBigBrick(id); }}
              title="Replace this module with the layers inside it"
            >
              Expand
            </button>
          )}
        </div>
        <div className="brick-ports">
          {brick.inputs.length > 0 && (
            <span className="port-tag">
              {brick.inputs[0].accepts.map((type) => <PortShape key={type} type={type} size={12} />)}
              {brick.inputs[0].accepts.map(portName).join(" or ")}
            </span>
          )}
          {brick.inputs.length > 0 && brick.output && <span className="port-arrow" aria-hidden="true">→</span>}
          {brick.output && <span className="port-tag"><PortShape type={brick.output} size={12} />{portName(brick.output)}</span>}
        </div>
        {shape && <div className="brick-shape">{shape}</div>}
        {issues.length > 0 && <div className="brick-issue" role="alert">{issues[0].message}</div>}
      </div>
      {brick.output && (
        <Handle
          id="out"
          type="source"
          position={Position.Bottom}
          className={`port ${outputState()}`}
          title={`Gives ${portName(brick.output)}`}
        >
          <PortShape type={brick.output} size={22} />
        </Handle>
      )}
    </div>
  );
}
