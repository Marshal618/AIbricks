import type { Edge } from "@xyflow/react";
import { useEditor } from "./context";
import { executionOrder } from "./graphOrder";
import { PortShape } from "./PortGlyph";
import type { BrickNodeType } from "./types";

interface Props {
  nodes: BrickNodeType[];
  edges: Edge[];
  onSelect?: (id: string) => void;
}

/** The model as a table, in the order data flows through it. */
export function Outline({ nodes, edges, onSelect }: Props) {
  const { bricks, analysis, catalogue } = useEditor();
  if (!nodes.length) return <p className="muted empty-note">The model has no components yet.</p>;
  const groupLabel = (id: string) => catalogue.groups.find((group) => group.id === id)?.label ?? "";
  return (
    <table className="outline">
      <thead><tr><th scope="col">#</th><th scope="col">Component</th><th scope="col">Output</th></tr></thead>
      <tbody>
        {executionOrder(nodes, edges).map((node, index) => {
          const brick = bricks[node.data.brickId];
          const hasIssue = analysis?.issues.some((issue) => issue.node === node.id);
          return (
            <tr
              key={node.id}
              className={`${node.selected ? "is-selected " : ""}${hasIssue ? "has-issue " : ""}${onSelect ? "is-clickable" : ""}`}
              onClick={() => onSelect?.(node.id)}
            >
              <td className="num">{index + 1}</td>
              <td>
                <span className="outline-name">
                  {brick.output && <PortShape type={brick.output} size={12} />}
                  {onSelect
                    ? <button className="plain-button" onClick={(event) => { event.stopPropagation(); onSelect(node.id); }}>{brick.name}</button>
                    : brick.name}
                </span>
                <span className="outline-type">{groupLabel(brick.group)}</span>
              </td>
              <td className="mono">{hasIssue ? <span className="issue-inline">Needs attention</span> : (analysis?.shapes[node.id] ?? "–")}</td>
            </tr>
          );
        })}
      </tbody>
    </table>
  );
}
