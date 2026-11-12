import type { ForecastAnalysis, ForecastDetails, PoseAnswer, TrainingMistake, Box, BoxedPhoto, FoundBoxes, FoundOutlines, Experiment, FailureSearch, SoundAnswer, SoundMistake, SystemInfo, WorkspaceModel, ImageExplanation, TextExplanation, Analysis, Catalogue, DatasetInfo, DatasetPreview, DatasetSummary, DatasetVersion, GraphPayload, ImageQueue, ModelCard, ModelCardNotes, ModelComparison, OpenDataImport, OpenDataInfo, OpenDataResult, OpenDataSource, ParseReply, PredictAnswer, ProjectData, DataHealth, ProjectRow, SavedModel, TableAnalysis, TableSchema, TestEvent, TextMistake, TextQueue, TrainingJob, VersionChanges, VersionOverview } from "./types";

async function errorMessage(response: Response): Promise<string> {
  try {
    const body = await response.json();
    return body?.detail?.message ?? `The server replied with status ${response.status}.`;
  } catch {
    return `The server replied with status ${response.status}.`;
  }
}

function post(path: string, graph: GraphPayload, signal?: AbortSignal) {
  return fetch(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(graph),
    signal,
  });
}

export async function fetchCatalogue(): Promise<Catalogue> {
  const response = await fetch("/api/catalogue");
  if (!response.ok) throw new Error(await errorMessage(response));
  return response.json();
}

export async function analyse(graph: GraphPayload, signal?: AbortSignal): Promise<Analysis> {
  const response = await post("/api/analyse", graph, signal);
  if (!response.ok) throw new Error(await errorMessage(response));
  return response.json();
}

export async function parseModel(code: string, graph: GraphPayload, signal?: AbortSignal): Promise<ParseReply> {
  const response = await fetch("/api/parse-model", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ code, graph }),
    signal,
  });
  if (!response.ok) throw new Error(await errorMessage(response));
  return response.json();
}

export async function runTest(graph: GraphPayload, onEvent: (event: TestEvent) => void, signal?: AbortSignal) {
  const response = await post("/api/test-run", graph, signal);
  if (!response.ok || !response.body) throw new Error(await errorMessage(response));
  const reader = response.body.pipeThrough(new TextDecoderStream()).getReader();
  let buffer = "";
  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += value;
    const lines = buffer.split("\n");
    buffer = lines.pop() ?? "";
    for (const line of lines) if (line.trim()) onEvent(JSON.parse(line));
  }
}

export async function downloadProject(graph: GraphPayload) {
  const response = await post("/api/export", graph);
  if (!response.ok) throw new Error(await errorMessage(response));
  const disposition = response.headers.get("Content-Disposition") ?? "";
  const name = disposition.match(/filename="(.+)"/)?.[1] ?? "project.zip";
  const url = URL.createObjectURL(await response.blob());
  const link = document.createElement("a");
  link.href = url;
  link.download = name;
  link.click();
  URL.revokeObjectURL(url);
}

function sendFile(path: string, file: Blob) {
  return fetch(path, { method: "POST", headers: { "Content-Type": "application/octet-stream" }, body: file });
}

export async function fetchDataset(categories: string[], dataset: string): Promise<DatasetInfo> {
  const query = new URLSearchParams([["dataset", dataset], ...categories.map((name) => ["category", name])]);
  const response = await fetch(`/api/dataset?${query}`);
  if (!response.ok) throw new Error(await errorMessage(response));
  return response.json();
}

export async function uploadImage(category: string, file: File, dataset: string): Promise<{ duplicate: boolean; count: number }> {
  const response = await sendFile(`/api/dataset/images?${new URLSearchParams({ category, dataset })}`, file);
  if (!response.ok) throw new Error(await errorMessage(response));
  return response.json();
}

export async function clearImages(category: string, dataset: string) {
  const response = await fetch(`/api/dataset/images?${new URLSearchParams({ category, dataset })}`, { method: "DELETE" });
  if (!response.ok) throw new Error(await errorMessage(response));
}

export async function fetchTraining(kind?: "image" | "text" | "table"): Promise<TrainingJob | null> {
  const response = await fetch(kind ? `/api/train/current?${new URLSearchParams({ kind })}` : "/api/train/current");
  if (!response.ok) throw new Error(await errorMessage(response));
  return (await response.json()).job;
}

export async function startTraining(graph: GraphPayload, dataset: string): Promise<TrainingJob> {
  const response = await post(`/api/train?${new URLSearchParams({ dataset })}`, graph);
  if (!response.ok) throw new Error(await errorMessage(response));
  return (await response.json()).job;
}

export async function stopTraining(id: string): Promise<TrainingJob> {
  const response = await fetch(`/api/train/${id}`, { method: "DELETE" });
  if (!response.ok) throw new Error(await errorMessage(response));
  return (await response.json()).job;
}

export async function predictImage(id: string, file: File): Promise<PredictAnswer[]> {
  const response = await sendFile(`/api/train/${id}/predict`, file);
  if (!response.ok) throw new Error(await errorMessage(response));
  return (await response.json()).answers;
}

export function trainedDownloadUrl(id: string) {
  return `/api/train/${id}/download`;
}

export function mistakeImageUrl(id: string, index: number) {
  return `/api/train/${id}/mistakes/${index}`;
}

export async function fetchProjects(): Promise<ProjectRow[]> {
  const response = await fetch("/api/projects");
  if (!response.ok) throw new Error(await errorMessage(response));
  return (await response.json()).projects;
}

export async function fetchProject(id: string): Promise<ProjectData> {
  const response = await fetch(`/api/projects/${id}`);
  if (!response.ok) throw new Error(await errorMessage(response));
  return response.json();
}

export async function saveProject(id: string, data: ProjectData) {
  const response = await fetch(`/api/projects/${id}`, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(data),
  });
  if (!response.ok) throw new Error(await errorMessage(response));
}

export async function deleteProject(id: string) {
  const response = await fetch(`/api/projects/${id}`, { method: "DELETE" });
  if (!response.ok) throw new Error(await errorMessage(response));
}

async function sendJson(path: string, body: unknown = {}) {
  const response = await fetch(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!response.ok) throw new Error(await errorMessage(response));
  return response.json();
}

async function send(path: string, method: string) {
  const response = await fetch(path, { method });
  if (!response.ok) throw new Error(await errorMessage(response));
}

export async function fixMistake(id: string, index: number, action: "move" | "remove" | "undo", category?: string): Promise<TrainingJob> {
  return (await sendJson(`/api/train/${id}/mistakes/${index}/fix`, { action, category })).job;
}

export async function addUnsortedImage(file: File, dataset: string): Promise<{ duplicate: boolean }> {
  const response = await sendFile(`/api/label/images?${new URLSearchParams({ dataset })}`, file);
  if (!response.ok) throw new Error(await errorMessage(response));
  return response.json();
}

export async function fetchImageQueue(dataset: string): Promise<ImageQueue> {
  const response = await fetch(`/api/label/images?${new URLSearchParams({ dataset })}`);
  if (!response.ok) throw new Error(await errorMessage(response));
  return response.json();
}

export function unsortedImageUrl(id: string, dataset: string) {
  return `/api/label/images/${id}/file?${new URLSearchParams({ dataset })}`;
}

export async function assignImage(id: string, category: string, dataset: string) {
  await sendJson(`/api/label/images/${id}/assign?${new URLSearchParams({ dataset })}`, { category });
}

export async function unassignImage(id: string, category: string | null, dataset: string) {
  await sendJson(`/api/label/images/${id}/unassign?${new URLSearchParams({ dataset })}`, { category });
}

export async function removeImage(id: string, dataset: string) {
  await sendJson(`/api/label/images/${id}/remove?${new URLSearchParams({ dataset })}`);
}

export const clearUnsortedImages = (dataset: string) => send(`/api/label/images?${new URLSearchParams({ dataset })}`, "DELETE");

export async function startSuggestions(dataset: string) {
  await sendJson(`/api/label/suggest?${new URLSearchParams({ dataset })}`);
}

export async function addUnsortedText(file: File, dataset: string): Promise<{ added: number; duplicates: number; tooLong: number; column: string | null }> {
  const kind = file.name.toLowerCase().endsWith(".csv") ? "csv" : "lines";
  const response = await sendFile(`/api/label/text?${new URLSearchParams({ kind, dataset })}`, file);
  if (!response.ok) throw new Error(await errorMessage(response));
  return response.json();
}

export async function fetchTextQueue(categories: string[], dataset: string): Promise<TextQueue> {
  const query = new URLSearchParams([["dataset", dataset], ...categories.map((name) => ["category", name])]);
  const response = await fetch(`/api/label/text?${query}`);
  if (!response.ok) throw new Error(await errorMessage(response));
  return response.json();
}

export async function assignText(id: string, category: string, dataset: string) {
  await sendJson(`/api/label/text/${id}/assign?${new URLSearchParams({ dataset })}`, { category });
}

export async function unassignText(id: string, dataset: string) {
  await sendJson(`/api/label/text/${id}/unassign?${new URLSearchParams({ dataset })}`);
}

export async function removeText(id: string, dataset: string) {
  await sendJson(`/api/label/text/${id}/remove?${new URLSearchParams({ dataset })}`);
}

export const clearUnsortedText = (dataset: string) => send(`/api/label/text?${new URLSearchParams({ dataset })}`, "DELETE");

export function textExportUrl(categories: string[], dataset: string) {
  return `/api/label/text/export?${new URLSearchParams([["dataset", dataset], ...categories.map((name) => ["category", name])])}`;
}

export async function searchOpenData(query: string, kind: "image" | "text"): Promise<OpenDataResult[]> {
  const response = await fetch(`/api/open-data/search?${new URLSearchParams({ q: query, kind })}`);
  if (!response.ok) throw new Error(await errorMessage(response));
  return (await response.json()).results;
}

export async function inspectOpenData(dataset: string, kind: "image" | "text"): Promise<OpenDataInfo> {
  const response = await fetch(`/api/open-data/inspect?${new URLSearchParams({ dataset, kind })}`);
  if (!response.ok) throw new Error(await errorMessage(response));
  return response.json();
}

export async function startOpenDataImport(source: string, kind: "image" | "text", mapping: Record<string, string>, perCategory: number, dataset: string): Promise<OpenDataImport> {
  return (await sendJson(`/api/open-data/import?${new URLSearchParams({ dataset })}`, { dataset: source, kind, mapping, perCategory })).job;
}

export async function fetchOpenDataImport(dataset: string): Promise<{ job: OpenDataImport | null; sources: OpenDataSource[] }> {
  const response = await fetch(`/api/open-data/import?${new URLSearchParams({ dataset })}`);
  if (!response.ok) throw new Error(await errorMessage(response));
  return response.json();
}

export const cancelOpenDataImport = () => send("/api/open-data/import", "DELETE");

export async function fetchDatasets(): Promise<DatasetSummary[]> {
  const response = await fetch("/api/datasets");
  if (!response.ok) throw new Error(await errorMessage(response));
  return (await response.json()).datasets;
}

export async function fetchSavedDataset(id: string): Promise<DatasetSummary> {
  const response = await fetch(`/api/datasets/${id}`);
  if (!response.ok) throw new Error(await errorMessage(response));
  return response.json();
}

export async function createDataset(name: string, kind: DatasetSummary["kind"], categories: string[], goal: NonNullable<DatasetSummary["goal"]> = "sort", points: "body" | "hands" = "body"): Promise<DatasetSummary> {
  return sendJson("/api/datasets", { name, kind, categories, goal, points });
}

export async function trainPoseModel(id: string): Promise<void> {
  await sendJson(`/api/datasets/${id}/train-pose`);
}

export const fetchPoseMistakes = (job: string): Promise<{ mistakes: TrainingMistake[]; total: number }> => getJson(`/api/models/${job}/pose-mistakes`);

export async function poseAnswer(job: string, file: Blob): Promise<PoseAnswer> {
  const response = await sendFile(`/api/train/${job}/predict-live`, file);
  if (!response.ok) throw new Error(await errorMessage(response));
  return response.json();
}

export async function trainUnusualModel(id: string): Promise<void> {
  await sendJson(`/api/datasets/${id}/train-unusual`);
}

export async function setUnusualLine(job: string, factor: number): Promise<{ factor: number }> {
  return putJson(`/api/models/${job}/unusual`, { factor });
}

export const fetchUnusualSettings = (job: string): Promise<{ factor: number }> => getJson(`/api/models/${job}/unusual`);

async function putJson(path: string, body: unknown) {
  const response = await fetch(path, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!response.ok) throw new Error(await errorMessage(response));
  return response.json();
}

export const renameSavedDataset = (id: string, name: string): Promise<DatasetSummary> => putJson(`/api/datasets/${id}/name`, { name });

export const addDatasetCategories = (id: string, names: string[]): Promise<DatasetSummary> => sendJson(`/api/datasets/${id}/categories`, { names });

export const renameDatasetCategory = (id: string, old: string, next: string): Promise<DatasetSummary> => putJson(`/api/datasets/${id}/categories`, { old, new: next });

export async function removeDatasetCategory(id: string, name: string): Promise<DatasetSummary> {
  const response = await fetch(`/api/datasets/${id}/categories?${new URLSearchParams({ name })}`, { method: "DELETE" });
  if (!response.ok) throw new Error(await errorMessage(response));
  return response.json();
}

export const deleteSavedDataset = (id: string) => send(`/api/datasets/${id}`, "DELETE");

async function getJson(path: string) {
  const response = await fetch(path);
  if (!response.ok) throw new Error(await errorMessage(response));
  return response.json();
}

export const updateDatasetDetails = (id: string, description: string, licence: string): Promise<DatasetSummary> => putJson(`/api/datasets/${id}/details`, { description, licence });

export const fetchDatasetPreview = (id: string): Promise<DatasetPreview> => getJson(`/api/datasets/${id}/preview`);

export const datasetPreviewImage = (id: string, category: string, image: string) => `/api/datasets/${id}/preview/${image}?${new URLSearchParams({ category })}`;

export const fetchVersions = (id: string): Promise<VersionOverview> => getJson(`/api/datasets/${id}/versions`);

export const saveVersion = (id: string, message: string): Promise<DatasetVersion> => sendJson(`/api/datasets/${id}/versions`, { message });

export const compareVersion = (id: string, version: number): Promise<VersionChanges> => getJson(`/api/datasets/${id}/versions/${version}/compare`);

export const restoreVersion = (id: string, version: number): Promise<{ dataset: DatasetSummary; savedFirst: number | null }> => sendJson(`/api/datasets/${id}/versions/${version}/restore`);

export const deleteVersion = (id: string, version: number) => send(`/api/datasets/${id}/versions/${version}`, "DELETE");

export async function fetchDatasetModels(id: string): Promise<SavedModel[]> {
  return (await getJson(`/api/datasets/${id}/models`)).models;
}

export const deleteSavedModel = (job: string) => send(`/api/models/${job}`, "DELETE");

export async function trainTextModel(id: string): Promise<void> {
  await sendJson(`/api/datasets/${id}/train-text`);
}

export async function tryTextModel(job: string, text: string): Promise<PredictAnswer[]> {
  return (await sendJson(`/api/models/${job}/try`, { text })).answers;
}

export const fetchTextMistakes = (job: string): Promise<{ mistakes: TextMistake[]; total: number }> => getJson(`/api/models/${job}/mistakes`);

export async function uploadTable(id: string, file: File): Promise<{ dataset: DatasetSummary; damagedRows: number; savedFirst: number | null }> {
  const response = await sendFile(`/api/datasets/${id}/table`, file);
  if (!response.ok) throw new Error(await errorMessage(response));
  return response.json();
}

export function fetchTableAnalysis(id: string, target?: string, task?: "category" | "number"): Promise<TableAnalysis> {
  const query = new URLSearchParams();
  if (target) query.set("target", target);
  if (task) query.set("task", task);
  return getJson(`/api/datasets/${id}/table/analysis?${query}`);
}

export async function trainTableModel(id: string, target: string, task: "category" | "number", columns: string[]): Promise<void> {
  await sendJson(`/api/datasets/${id}/train-table`, { target, task, columns });
}

export const fetchTableSchema = (job: string): Promise<TableSchema> => getJson(`/api/models/${job}/schema`);

export const tryTableModel = (job: string, values: Record<string, string>): Promise<{ value?: number; answers?: PredictAnswer[] }> => sendJson(`/api/models/${job}/try-row`, { values });

export function fetchForecastAnalysis(id: string, choice: { when?: string; value?: string; horizon?: number; combine?: "mean" | "sum" } = {}): Promise<ForecastAnalysis> {
  const query = new URLSearchParams();
  if (choice.when !== undefined) query.set("when", choice.when);
  if (choice.value) query.set("value", choice.value);
  if (choice.horizon) query.set("horizon", String(choice.horizon));
  if (choice.combine) query.set("combine", choice.combine);
  return getJson(`/api/datasets/${id}/forecast/analysis?${query}`);
}

export async function trainForecastModel(id: string, choice: { when: string; value: string; horizon: number; combine: "mean" | "sum" }): Promise<void> {
  await sendJson(`/api/datasets/${id}/train-forecast`, choice);
}

export const fetchForecast = (job: string): Promise<ForecastDetails> => getJson(`/api/models/${job}/forecast`);

export const forecastFrom = (job: string, values: number[]): Promise<{ values: number[]; low: number[]; high: number[] }> => sendJson(`/api/models/${job}/forecast-from`, { values });

export const fetchHealth = (id: string): Promise<DataHealth> => getJson(`/api/datasets/${id}/health`);

export async function startHealthCheck(id: string): Promise<void> {
  await sendJson(`/api/datasets/${id}/health/check`);
}

export const reviewHealth = (id: string, body: { action: "keep" | "move" | "remove" | "drop" | "keep-pair"; id?: string; category?: string; pair?: string[] }): Promise<DataHealth> =>
  sendJson(`/api/datasets/${id}/health/review`, body);

export async function compareModels(dataset: string, first: string, second: string): Promise<ModelComparison> {
  return getJson(`/api/datasets/${dataset}/compare?${new URLSearchParams({ first, second })}`);
}

export const fetchModelCard = (job: string): Promise<ModelCard> => getJson(`/api/models/${job}/card`);

export const updateModelCard = (job: string, notes: ModelCardNotes): Promise<ModelCard> => putJson(`/api/models/${job}/card`, notes);

export async function predictLive(id: string, file: File): Promise<PredictAnswer[]> {
  const response = await sendFile(`/api/train/${id}/predict-live`, file);
  if (!response.ok) throw new Error(await errorMessage(response));
  return (await response.json()).answers;
}

export async function explainImage(id: string, file: Blob): Promise<ImageExplanation> {
  const response = await sendFile(`/api/train/${id}/explain`, file);
  if (!response.ok) throw new Error(await errorMessage(response));
  return response.json();
}

export async function predictLiveExplained(id: string, file: File): Promise<ImageExplanation> {
  const response = await sendFile(`/api/train/${id}/predict-live?explain=true`, file);
  if (!response.ok) throw new Error(await errorMessage(response));
  return response.json();
}

export async function explainText(job: string, text: string): Promise<TextExplanation> {
  const response = await fetch(`/api/models/${job}/explain`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ text }) });
  if (!response.ok) throw new Error(await errorMessage(response));
  return response.json();
}

export async function fetchAllModels(limit = 20): Promise<WorkspaceModel[]> {
  return (await getJson(`/api/models?limit=${limit}`)).models;
}

export const fetchSystem = (): Promise<SystemInfo> => getJson("/api/system");

export async function addRecording(dataset: string, category: string, recording: Blob): Promise<{ added: number; duplicates: number; count: number }> {
  const response = await sendFile(`/api/datasets/${dataset}/sounds?${new URLSearchParams({ category })}`, recording);
  if (!response.ok) throw new Error(await errorMessage(response));
  return response.json();
}

export const fetchClips = (dataset: string, category: string): Promise<{ clips: string[]; total: number }> =>
  getJson(`/api/datasets/${dataset}/sounds?${new URLSearchParams({ category })}`);

export const clipUrl = (dataset: string, category: string, clip: string) => `/api/datasets/${dataset}/sounds/${clip}?${new URLSearchParams({ category })}`;

export async function removeClip(dataset: string, category: string, clip: string) {
  const response = await fetch(clipUrl(dataset, category, clip), { method: "DELETE" });
  if (!response.ok) throw new Error(await errorMessage(response));
}

export async function trainSoundModel(id: string): Promise<void> {
  await sendJson(`/api/datasets/${id}/train-sound`);
}

export async function listen(job: string, recording: Blob, explain: boolean): Promise<SoundAnswer> {
  const response = await sendFile(`/api/models/${job}/listen?explain=${explain}`, recording);
  if (!response.ok) throw new Error(await errorMessage(response));
  return response.json();
}

export const fetchSoundMistakes = (job: string): Promise<{ mistakes: SoundMistake[]; total: number }> => getJson(`/api/models/${job}/sound-mistakes`);

export const modelAppUrl = (id: string, download = false) => `/api/train/${id}/app${download ? "?download=true" : ""}`;
export const piKitUrl = (id: string) => `/api/train/${id}/pi`;

export const fetchExperiment = (dataset: string): Promise<{ experiment: Experiment | null; budgets: number[]; changes: Record<string, { label: string; description: string }> }> =>
  getJson(`/api/datasets/${dataset}/experiments`);

export async function startExperiment(dataset: string, minutes: number): Promise<Experiment> {
  return (await sendJson(`/api/datasets/${dataset}/experiments`, { minutes })).experiment;
}

export async function stopExperiment(dataset: string) {
  const response = await fetch(`/api/datasets/${dataset}/experiments`, { method: "DELETE" });
  if (!response.ok) throw new Error(await errorMessage(response));
}

export async function previewAugment(dataset: string, size: number, channels: number, params: Record<string, boolean | number>): Promise<{ original: string; varied: string[] }> {
  return sendJson("/api/augment/preview", { dataset, size, channels, params });
}

export const fetchFailures = (job: string): Promise<FailureSearch> => getJson(`/api/models/${job}/slices`);

export async function findFailures(job: string): Promise<FailureSearch> {
  return sendJson(`/api/models/${job}/slices`);
}

export const datasetPhotoUrl = (dataset: string, category: string, file: string) => `/api/datasets/${dataset}/preview/${file}?${new URLSearchParams({ category })}`;

// ------------------------------------------------------------- finding things in photos

export const fetchBoxes = (dataset: string): Promise<{ photos: BoxedPhoto[]; done: number; total: number }> => getJson(`/api/datasets/${dataset}/boxes`);

export async function saveBoxes(dataset: string, image: string, boxes: Box[]): Promise<{ boxes: Box[] }> {
  return putJson(`/api/datasets/${dataset}/boxes/${image}`, { boxes: boxes.map(({ category, x, y, w, h, points }) => (points ? { category, points } : { category, x, y, w, h })) });
}

export const clearBoxes = (dataset: string, image: string) => send(`/api/datasets/${dataset}/boxes/${image}`, "DELETE");

export async function trainFindModel(id: string, detector: "accurate" | "fast" = "accurate"): Promise<void> {
  await sendJson(`/api/datasets/${id}/train-find?${new URLSearchParams({ detector })}`);
}

export const fetchFindSettings = (job: string): Promise<{ sure: number; categories: string[]; detector: "accurate" | "fast" }> => getJson(`/api/models/${job}/find`);

export const setFindSure = (job: string, sure: number): Promise<{ sure: number }> => putJson(`/api/models/${job}/find`, { sure });

// ------------------------------------------------------------- outlining things in photos

export interface Click { x: number; y: number; inside: boolean }

export const outlineAt = (dataset: string, image: string, clicks: Click[]): Promise<{ points: [number, number][]; box: { x: number; y: number; w: number; h: number } | null; score: number }> =>
  sendJson(`/api/datasets/${dataset}/outline/${image}`, { clicks });

export async function trainOutlineModel(id: string): Promise<void> {
  await sendJson(`/api/datasets/${id}/train-outline`);
}

export const outlineReviewUrl = (job: string, number: number) => `/api/models/${job}/outline-review/${number}`;

export async function findOutlines(job: string, file: Blob): Promise<FoundOutlines> {
  const response = await sendFile(`/api/train/${job}/predict-live`, file);
  if (!response.ok) throw new Error(await errorMessage(response));
  return response.json();
}

export async function findBoxes(job: string, file: Blob): Promise<FoundBoxes> {
  const response = await sendFile(`/api/train/${job}/predict-live`, file);
  if (!response.ok) throw new Error(await errorMessage(response));
  return response.json();
}
