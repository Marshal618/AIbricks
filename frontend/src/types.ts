import type { Edge, Node } from "@xyflow/react";

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
  buildHash: string | null;
  categories: string[] | null;
  labelKind: "image" | "text" | null;
  labelCategories: string[] | null;
}

export interface DatasetSummary {
  id: string;
  name: string;
  kind: "image" | "text" | "table";
  categories: string[];
  counts: Record<string, number>;
  labelled: number;
  unsorted: number;
  updatedAt: number;
  description: string;
  licence: string;
  rows?: number;
  columns?: string[];
}

export interface TableColumn {
  name: string;
  kind: "number" | "category" | "date" | "empty";
  missing: number;
  distinct: number;
  use: boolean;
  warning: string | null;
}

export interface TableAnalysis {
  target: string;
  task: "category" | "number";
  targetKind: string;
  rows: number;
  totalRows: number;
  columns: TableColumn[];
  classes: Record<string, number> | null;
  problems: string[];
  notes: string[];
  headers: string[];
}

export interface TableMetrics {
  task: "category" | "number";
  repeatedRows: number;
  warnings: string[];
  importance: { name: string; share: number }[];
  total: number;
  accuracy?: number;
  baseline?: number;
  correct?: number;
  mae?: number;
  r2?: number;
  baselineMae?: number;
}

export interface TableSchema {
  task: "category" | "number";
  target: string;
  features: { name: string; kind: string; choices?: string[] }[];
  classes: string[] | null;
}

export interface VersionChanges {
  added: number;
  removed: number;
  relabelled: number;
  categoriesAdded: string[];
  categoriesRemoved: string[];
  counts: { name: string; before: number; after: number }[];
}

export interface DatasetVersion {
  number: number;
  message: string;
  automatic: boolean;
  createdAt: number;
  categories: string[];
  counts: Record<string, number>;
  labelled: number;
  unsorted: number;
  changes: VersionChanges | null;
  sources: OpenDataSource[];
}

export interface VersionOverview {
  versions: DatasetVersion[];
  head: number | null;
  unsaved: VersionChanges | null;
  canSave: boolean;
}

export interface TextMistake {
  id: string;
  text: string;
  label: string | null;
  actual: string;
  guess: string;
  chance: number;
}

export interface DatasetPreview {
  categories: { name: string; samples: { id: string; text?: string }[] }[];
  sources: OpenDataSource[];
  table?: { columns: string[]; rows: string[][] } | null;
}

export interface SavedModel {
  id: string;
  projectName: string;
  categories: Record<string, number>;
  baseline: number;
  status: TrainingJob["status"];
  best: TrainingEpoch | null;
  hasModel: boolean;
  modelRemoved: boolean;
  datasetVersion: number | null;
  modelBytes: number;
  startedAt: number;
  review: { total: number; correct: number } | null;
  task: "image" | "text" | "table";
  metrics: TableMetrics | null;
  device: string;
  stage: "training" | "reviewing";
  stepsDone: number;
  stepsTotal: number;
  error: string | null;
}

export interface OpenDataResult {
  id: string;
  downloads: number;
  licence: string;
}

export interface OpenDataInfo {
  dataset: string;
  kind: "image" | "text";
  licence: string;
  labels: string[];
  totalRows: number;
  preview: { image?: string; text?: string }[];
  namesUsable: boolean;
  url: string;
}

export interface OpenDataImport {
  dataset: string;
  status: "running" | "finished" | "cancelled" | "failed";
  perCategory: number;
  counts: Record<string, number>;
  added: number;
  duplicates: number;
  skipped: number;
  error: string | null;
}

export interface OpenDataSource {
  dataset: string;
  licence: string;
  added: number;
  url: string;
}

export interface LabelItem {
  id: string;
  text?: string;
  suggestion?: { category: string; chance: number } | null;
}

export interface ImageQueue {
  items: LabelItem[];
  total: number;
  unsuggested: number;
  suggesting: boolean;
  suggestError: string | null;
  modelCategories: string[] | null;
}

export interface TextQueue {
  items: LabelItem[];
  total: number;
  counts: Record<string, number>;
  otherLabels: number;
}

export interface DatasetInfo {
  counts: Record<string, number>;
  minPerCategory: number;
  minTotal: number;
  suggestedPerCategory: number;
  problems: string[];
}

export interface TrainingEpoch {
  epoch: number;
  trainLoss: number | null;
  checkLoss: number | null;
  accuracy: number | null;
}

export interface TrainingJob {
  id: string;
  projectName: string;
  buildHash: string;
  categories: Record<string, number>;
  baseline: number;
  status: "running" | "finished" | "stopped" | "failed";
  totalEpochs: number;
  trainCount: number;
  checkCount: number;
  device: string;
  epochs: TrainingEpoch[];
  best: TrainingEpoch | null;
  hasModel: boolean;
  error: string | null;
  elapsedSeconds: number;
  stage: "training" | "reviewing";
  dataset: string;
  datasetVersion: number | null;
  review: TrainingReview | null;
  reviewError: string | null;
}

export interface TrainingMistake {
  actual: string;
  guess: string;
  chance: number;
  file: string;
  movedTo?: string | null;
  removed?: boolean;
}

export interface TrainingReview {
  total: number;
  correct: number;
  categories: Record<string, { correct: number; total: number }>;
  mistakeCount: number;
  mistakes: TrainingMistake[];
}

export interface ProjectRow {
  id: string;
  projectName: string;
  components: number;
  updatedAt: number;
}

export interface ProjectData {
  projectName: string;
  training: string;
  trainingOverride: TrainingOverride | null;
  nodes: BrickNodeType[];
  edges: Edge[];
  kitId: string | null;
  datasetId?: string | null;
}

export interface PredictAnswer {
  category: string;
  chance: number;
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
