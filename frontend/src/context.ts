import { createContext, useContext } from "react";
import type { Analysis, Brick, Catalogue, Mode } from "./types";

export interface EditorContextValue {
  catalogue: Catalogue;
  bricks: Record<string, Brick>;
  analysis: Analysis | null;
  mode: Mode;
  openBigBrick: (nodeId: string) => void;
  /** Set while a connection is being dragged. */
  connecting: { nodeId: string; side: "source" | "target"; types: string[] } | null;
  /** The component whose output the library is currently matching. */
  nextNodeId: string | null;
}

export const EditorContext = createContext<EditorContextValue | null>(null);

export function useEditor() {
  const value = useContext(EditorContext);
  if (!value) throw new Error("EditorContext is missing");
  return value;
}
