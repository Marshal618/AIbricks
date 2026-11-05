import type { Edge } from "@xyflow/react";
import type { BrickNodeType } from "./types";

/** Returns each node's depth: the longest path from any input to it. */
export function depths(nodes: BrickNodeType[], edges: Edge[]) {
  const incoming = new Map<string, string[]>();
  for (const edge of edges) incoming.set(edge.target, [...(incoming.get(edge.target) ?? []), edge.source]);
  const memo = new Map<string, number>();
  const visit = (id: string, path: Set<string>): number => {
    if (memo.has(id)) return memo.get(id)!;
    if (path.has(id)) return 0;
    path.add(id);
    const parents = incoming.get(id) ?? [];
    const depth = parents.length ? Math.max(...parents.map((parent) => visit(parent, path))) + 1 : 0;
    path.delete(id);
    memo.set(id, depth);
    return depth;
  };
  for (const node of nodes) visit(node.id, new Set());
  return memo;
}

/** Nodes in the order data reaches them, keeping left-to-right order within a level. */
export function executionOrder(nodes: BrickNodeType[], edges: Edge[]) {
  const depth = depths(nodes, edges);
  return [...nodes].sort((a, b) => (depth.get(a.id) ?? 0) - (depth.get(b.id) ?? 0) || a.position.x - b.position.x);
}
