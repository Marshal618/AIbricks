import type { Edge } from "@xyflow/react";
import { useEffect, useState } from "react";
import { useEditor } from "./context";
import type { BrickNodeType, Kit, KitCheck } from "./types";

function isDone(check: KitCheck, nodes: BrickNodeType[], edges: Edge[], tested: boolean, bricks: ReturnType<typeof useEditor>["bricks"]) {
  if (check.testPassed) return tested;
  if (check.brick) return nodes.some((node) => node.data.brickId === check.brick && edges.some((edge) => edge.target === node.id));
  if (check.paramChanged) {
    const params = bricks[check.paramChanged].params;
    return nodes.some((node) => node.data.brickId === check.paramChanged &&
      params.some((param) => node.data.params[param.key] !== undefined && node.data.params[param.key] !== param.default));
  }
  return false;
}

interface Props {
  kit: Kit | undefined;
  nodes: BrickNodeType[];
  edges: Edge[];
  onSelect: (id: string) => void;
}

export function Dock({ kit, nodes, edges, onSelect }: Props) {
  const { analysis, bricks, mode } = useEditor();
  const [tab, setTab] = useState<"problems" | "setup">(kit ? "setup" : "problems");
  const [open, setOpen] = useState(true);
  useEffect(() => { setTab(kit ? "setup" : "problems"); }, [kit?.id]);

  const issues = analysis?.issues ?? [];
  const done = kit?.steps.map((step) => isDone(step.check, nodes, edges, analysis?.tested ?? false, bricks)) ?? [];
  const next = done.indexOf(false);
  const nameOf = (id: string | null) => {
    const node = nodes.find((candidate) => candidate.id === id);
    return node ? bricks[node.data.brickId].name : "Model";
  };

  return (
    <div className={`dock${open ? "" : " is-closed"}`}>
      <div className="dock-tabs" role="tablist" aria-label="Design details">
        <button role="tab" aria-selected={tab === "problems"} className={tab === "problems" ? "active" : ""} onClick={() => { setTab("problems"); setOpen(true); }}>
          {mode === "beginner" ? "To fix" : "Problems"} <span className={`count${issues.length ? " count-bad" : ""}`}>{issues.length}</span>
        </button>
        {kit && (
          <button role="tab" aria-selected={tab === "setup"} className={tab === "setup" ? "active" : ""} onClick={() => { setTab("setup"); setOpen(true); }}>
            Template setup <span className="count">{done.filter(Boolean).length}/{kit.steps.length}</span>
          </button>
        )}
        <span className="spacer" />
        <button className="dock-toggle" onClick={() => setOpen(!open)} aria-expanded={open}>{open ? "Hide" : "Show"}</button>
      </div>
      {open && tab === "problems" && (
        <div className="dock-body">
          {!analysis ? (
            <p className="muted dock-empty">Checking the model…</p>
          ) : issues.length === 0 ? (
            <p className="muted dock-empty">{mode === "beginner" ? "Nothing to fix. Your model is ready to test." : "No problems. This model is ready to test."}</p>
          ) : (
            <table className="problems">
              <tbody>
                {issues.map((issue) => (
                  <tr key={`${issue.node}-${issue.message}`}>
                    <td className="problem-source">
                      {issue.node
                        ? <button className="link-strong" onClick={() => onSelect(issue.node!)}>{nameOf(issue.node)}</button>
                        : "Model"}
                    </td>
                    <td>{issue.message}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </div>
      )}
      {open && tab === "setup" && kit && (
        <div className="dock-body">
          <ol className="setup-steps">
            {kit.steps.map((step, index) => (
              <li key={step.text} className={done[index] ? "done" : index === next ? "next" : ""}>
                <span className="check-mark" aria-hidden="true">{done[index] ? "✓" : index + 1}</span>
                <span>{step.text}</span>
              </li>
            ))}
          </ol>
        </div>
      )}
    </div>
  );
}
