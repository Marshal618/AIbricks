import type { Node } from "@xyflow/react";

export type ParamValue = string | number | boolean;

export interface Param {
  key: string;
  label: string;
  kind: "int" | "choice" | "bool" | "names";
  default: ParamValue;
  help: string;
  min: number | null;
  max: number | null;
  choices: ParamValue[] | null;
  basic: boolean;
}

export interface Port {
  id: string;
  accepts: string[];
  label: string;
  optional: boolean;
}

export interface Brick {
  id: string;
  name: string;
  group: string;
  description: string;
  inputs: Port[];
  output: string | null;
  params: Param[];
  size: "small" | "big";
  beginner: boolean;
  why: string;
  parts: { brick: string; params: Record<string, ParamValue> }[];
}

export type KitCheck = { brick?: string; paramChanged?: string; testPassed?: boolean };

export interface Kit {
  id: string;
  name: string;
  domain: string;
  goal: string;
  family: string;
  audience: string;
  projectName: string;
  training: string;
  nodes: { id: string; type: string; x: number; y: number; params: Record<string, ParamValue> }[];
  edges: { source: string; target: string; targetHandle: string }[];
  steps: { text: string; check: KitCheck }[];
}

export interface Catalogue {
  portTypes: Record<string, { name: string; description: string }>;
  groups: { id: string; name: string; label: string }[];
  bricks: Brick[];
  training: Record<string, { name: string; epochs: number; batch_size: number; learning_rate: number }>;
  kits: Kit[];
  codeReference: { brick: string; name: string; signature: string }[];
  codeStarter: string;
}

export interface Issue {
  message: string;
  node: string | null;
}

export interface Analysis {
  ok: boolean;
  issues: Issue[];
  shapes: Record<string, string>;
  files: Record<string, string>;
  dataNeeds: string[];
  tested: boolean;
}

export type BrickData = { brickId: string; params: Record<string, ParamValue> };
export type BrickNodeType = Node<BrickData, "brick">;

export interface GraphPayload {
  projectName: string;
  training: string;
  trainingOverride: TrainingOverride | null;
  nodes: { id: string; type: string; params: Record<string, ParamValue> }[];
  edges: { source: string; target: string; targetHandle: string }[];
}

export type Mode = "beginner" | "advanced";

export type TrainingOverride = Partial<Record<"epochs" | "batch_size" | "learning_rate", number | string>>;

export interface ParseReply {
  ok: boolean;
  errors: { line: number | null; message: string }[];
  notes: { line: number | null; message: string }[];
  nodes: { id: string; type: string; params: Record<string, ParamValue> }[];
  edges: { source: string; target: string; targetHandle: string }[];
}

export type TestEvent =
  | { event: "model"; parameters: number; trainable: number }
  | { event: "check"; check: string; passed: boolean; detail?: string; first?: number; last?: number }
  | { event: "loss"; step: number; loss: number }
  | { event: "result"; passed: boolean; detail?: string };
