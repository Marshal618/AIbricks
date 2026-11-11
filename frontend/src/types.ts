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
  /** Shown only while this other on/off setting is on. */
  shown_when: string | null;
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
  kind: "image" | "text" | "table" | "sound";
  categories: string[];
  counts: Record<string, number>;
  labelled: number;
  unsorted: number;
  updatedAt: number;
  description: string;
  licence: string;
  rows?: number;
  columns?: string[];
  /** Photos only: "unusual" learns what normal looks like and spots what isn't; "find" draws boxes around things;
   * "pose" sorts the body poses or hand signs in photos. */
  goal?: "sort" | "unusual" | "find" | "pose";
  /** Sorting poses only: the points of a whole body, or of hands. */
  points?: "body" | "hands";
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
  task: "category" | "number" | "unusual" | "detect" | "pose";
  /** Sorting poses: photos where no body or hands could be seen, which it couldn't learn from or answer. */
  points?: "body" | "hands";
  nothingFoundTraining?: number;
  nothingFoundChecking?: number;
  trainedOn?: number;
  /** Finding things: boxes found, missed and found where there was nothing, on held-back photos. */
  photos?: number;
  found?: number;
  missed?: number;
  allRight?: number;
  map50?: number | null;
  perCategory?: Record<string, { found: number; missed: number; falseAlarms: number }>;
  photosChecked?: CheckedPhoto[];
  repeatedRows: number;
  warnings: string[];
  importance: { name: string; share: number }[];
  total: number;
  /** Spotting what's unusual: each held-back photo's unusualness, where 1 is the line. */
  scores?: { file: string; actual: "Normal" | "Unusual"; unusualness: number }[];
  auroc?: number | null;
  falseAlarms?: number;
  checkedNormal?: number;
  caught?: number;
  checkedUnusual?: number;
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
  review: { total: number; correct: number; categories?: Record<string, { correct: number; total: number }> } | null;
  task: "image" | "text" | "table" | "sound" | "anomaly" | "detect" | "pose";
  metrics: TableMetrics | null;
  totalEpochs?: number;
  trainCount?: number;
  checkCount?: number;
  device: string;
  stage: "training" | "reviewing";
  stepsDone: number;
  stepsTotal: number;
  error: string | null;
  checkup?: CheckupFinding[];
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

/** How the dataset's newest model can help label what's left. */
export interface LabelAssist {
  unsuggested: number;
  suggesting: boolean;
  suggestError: string | null;
  /** The categories of the model that makes suggestions, or null when there's no such model yet. */
  modelCategories: string[] | null;
}

export interface ImageQueue extends LabelAssist {
  items: LabelItem[];
  total: number;
}

export interface TextQueue extends LabelAssist {
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
  checkup?: CheckupFinding[];
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

export interface HealthIssue {
  id: string;
  given: string;
  suggested: string;
  confidence: number;
  givenChance: number;
  state: "open" | "fixed" | "removed" | "kept";
  labels: string[];
  text?: string;
}

export interface HealthPair {
  first: string;
  second: string;
  firstLabel: string;
  secondLabel: string;
  firstLabels: string[];
  secondLabels: string[];
  similarity: number;
  state: "open" | "kept" | "same";
  firstText?: string | null;
  secondText?: string | null;
}

export interface DataHealth {
  kind: "image" | "text" | "table";
  level?: number;
  levelName?: string;
  levels?: string[];
  nextSteps?: string[];
  labelled?: number;
  counts?: Record<string, number>;
  conflicts?: { ids: string[]; labels: string[] }[];
  check?: {
    ranAt: number;
    agreement: number | null;
    issues: HealthIssue[];
    duplicates: HealthPair[];
    unchecked: number;
    openIssues: number;
    openDuplicates: number;
    passedShare: number | null;
  } | null;
  table?: { rows: number; columns: number; repeatedRows: number; blankShare: number; constantColumns: string[] } | null;
  job: { status: "running" | "finished" | "failed"; stepsDone: number; stepsTotal: number; error: string | null; count: number } | null;
}

export interface ModelComparison {
  first: { id: string; name: string; version: number | null; startedAt: number };
  second: { id: string; name: string; version: number | null; startedAt: number };
  task: "category" | "number";
  shared: number;
  verdict: "too-few" | "same" | "first" | "second" | "unclear";
  summary: string;
  firstScore?: number;
  secondScore?: number;
  onlyFirst?: number;
  onlySecond?: number;
  bothRight?: number;
  neitherRight?: number;
  ties?: number;
  chanceOfLuck?: number;
}

export interface ModelCardNotes {
  intendedUse: string;
  notFor: string;
}

export interface ModelCard {
  jobId: string;
  notes: ModelCardNotes;
  sections: { title: string; lines: string[]; editable?: boolean }[];
  markdown: string;
}

export interface Heatmap {
  width: number;
  height: number;
  /** Row by row, 0 to 1, where 1 is where the model looked most. */
  values: number[];
  method: "gradcam" | "gradients" | "patchcore";
  category: string;
}

export interface ImageExplanation {
  answers: PredictAnswer[];
  heatmap: Heatmap;
}

export interface TextExplanation {
  answers: PredictAnswer[];
  category: string;
  /** How much the top answer's chance falls when this word is removed. Negative words pushed against it. */
  words: { start: number; end: number; weight: number }[];
  wordCount: number;
}

export interface WorkspaceModel extends SavedModel {
  dataset: string;
  datasetName: string | null;
  finishedAt: number | null;
}

export interface SystemInfo {
  device: "cuda" | "mps" | "cpu" | "unknown";
  deviceLabel: string;
  torch: string | null;
  python: string;
  platform: string;
  cpus: number;
  storage: { path: string; bytes: number; freeBytes: number; parts: { data: number; models: number; projects: number; downloads: number } };
}

export interface Grid {
  width: number;
  height: number;
  /** Row by row from the top, 0 to 1. */
  values: number[];
}

export interface SoundAnswer {
  answers: PredictAnswer[];
  /** The first second of the recording, high pitches at the top. */
  spectrogram: Grid;
  heatmap: Heatmap | null;
  clips: number;
}

export interface SoundMistake {
  index: number;
  actual: string;
  guess: string;
  chance: number;
  file: string;
  movedTo?: string | null;
  removed?: boolean;
}

export type FixKind = "augment" | "slower" | "longer" | "pretrained" | "more_photos" | "experiment";

export interface CheckupFinding {
  kind: "memorising" | "unstable" | "too_simple" | "still_learning" | "healthy";
  title: string;
  detail: string;
  fixes: FixKind[];
}

export interface ExperimentOption {
  key: string;
  label: string;
  changes: string[];
  round: number;
  epochs: number;
  jobId: string;
  seconds: number;
  accuracy: number | null;
  correct: number | null;
  total: number | null;
  deleted?: boolean;
  graph: GraphPayload;
}

export interface ExperimentEffect {
  key: string;
  label: string;
  description: string;
  points: number;
  meaning: "helped" | "hurt" | "no clear difference" | "too few photos to tell";
  chanceOfLuck: number | null;
  shared: number | null;
}

export interface Experiment {
  id: string;
  dataset: string;
  status: "running" | "finished" | "stopped" | "failed";
  minutes: number;
  startedAt: number;
  finishedAt: number | null;
  round: number;
  current: string | null;
  currentLabel?: string;
  progress?: { epoch: number; epochs: number } | null;
  options: ExperimentOption[];
  effects: ExperimentEffect[];
  winner: string | null;
  verdict: { against: string; result: string; chanceOfLuck: number | null } | null;
  error: string | null;
  skipped: string[];
}

export interface FailureSlice {
  name: string;
  also: string[];
  kind: string;
  size: number;
  wrong: number;
  rate: number;
  confusion: { actual: string; guess: string; times: number };
  examples: { file: string; category: string; guess: string }[];
  advice: string;
  action: "vary" | "photos";
}

export interface FailureSearch {
  status: "none" | "running" | "finished" | "failed";
  stage?: "loading" | "looking";
  enough?: boolean;
  held?: number;
  mistakes?: number;
  overall?: number;
  slices?: FailureSlice[];
  error?: string;
}

/** A box around one thing, as shares of the photo's width and height. */
export interface Box {
  category: string;
  x: number;
  y: number;
  w: number;
  h: number;
  /** How sure the model is, on boxes the model drew. */
  chance?: number;
}

export interface BoxedPhoto {
  id: string;
  /** null until the photo is done; an empty list means nothing to find in it. */
  boxes: Box[] | null;
}

export interface FoundBoxes {
  boxes: Box[];
  counts: Record<string, number>;
}

export interface CheckedPhoto {
  file: string;
  real: Box[];
  found: Box[];
  right: number;
  missed: number;
  falseAlarms: number;
}

/** What a pose model saw in one photo: the points it found, as 0-1 shares of the photo, and its answers. */
export interface PoseAnswer {
  found: boolean;
  answers: PredictAnswer[];
  /** One list per body or hand, of [x, y, how sure it is the point is in view]. */
  points: [number, number, number][][];
}
