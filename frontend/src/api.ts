import type { Analysis, Catalogue, DatasetInfo, GraphPayload, ImageQueue, ParseReply, PredictAnswer, ProjectData, ProjectRow, TestEvent, TextQueue, TrainingJob } from "./types";

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

function sendFile(path: string, file: File) {
  return fetch(path, { method: "POST", headers: { "Content-Type": "application/octet-stream" }, body: file });
}

export async function fetchDataset(categories: string[]): Promise<DatasetInfo> {
  const query = new URLSearchParams(categories.map((name) => ["category", name]));
  const response = await fetch(`/api/dataset?${query}`);
  if (!response.ok) throw new Error(await errorMessage(response));
  return response.json();
}

export async function uploadImage(category: string, file: File): Promise<{ duplicate: boolean; count: number }> {
  const response = await sendFile(`/api/dataset/images?${new URLSearchParams({ category })}`, file);
  if (!response.ok) throw new Error(await errorMessage(response));
  return response.json();
}

export async function clearImages(category: string) {
  const response = await fetch(`/api/dataset/images?${new URLSearchParams({ category })}`, { method: "DELETE" });
  if (!response.ok) throw new Error(await errorMessage(response));
}

export async function fetchTraining(): Promise<TrainingJob | null> {
  const response = await fetch("/api/train/current");
  if (!response.ok) throw new Error(await errorMessage(response));
  return (await response.json()).job;
}

export async function startTraining(graph: GraphPayload): Promise<TrainingJob> {
  const response = await post("/api/train", graph);
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

export async function addUnsortedImage(file: File): Promise<{ duplicate: boolean }> {
  const response = await sendFile("/api/label/images", file);
  if (!response.ok) throw new Error(await errorMessage(response));
  return response.json();
}

export async function fetchImageQueue(): Promise<ImageQueue> {
  const response = await fetch("/api/label/images");
  if (!response.ok) throw new Error(await errorMessage(response));
  return response.json();
}

export function unsortedImageUrl(id: string) {
  return `/api/label/images/${id}/file`;
}

export async function assignImage(id: string, category: string) {
  await sendJson(`/api/label/images/${id}/assign`, { category });
}

export async function unassignImage(id: string, category: string | null) {
  await sendJson(`/api/label/images/${id}/unassign`, { category });
}

export async function removeImage(id: string) {
  await sendJson(`/api/label/images/${id}/remove`);
}

export const clearUnsortedImages = () => send("/api/label/images", "DELETE");

export async function startSuggestions() {
  await sendJson("/api/label/suggest");
}

export async function addUnsortedText(file: File): Promise<{ added: number; duplicates: number; tooLong: number; column: string | null }> {
  const kind = file.name.toLowerCase().endsWith(".csv") ? "csv" : "lines";
  const response = await sendFile(`/api/label/text?kind=${kind}`, file);
  if (!response.ok) throw new Error(await errorMessage(response));
  return response.json();
}

export async function fetchTextQueue(categories: string[]): Promise<TextQueue> {
  const query = new URLSearchParams(categories.map((name) => ["category", name]));
  const response = await fetch(`/api/label/text?${query}`);
  if (!response.ok) throw new Error(await errorMessage(response));
  return response.json();
}

export async function assignText(id: string, category: string) {
  await sendJson(`/api/label/text/${id}/assign`, { category });
}

export async function unassignText(id: string) {
  await sendJson(`/api/label/text/${id}/unassign`);
}

export async function removeText(id: string) {
  await sendJson(`/api/label/text/${id}/remove`);
}

export const clearUnsortedText = () => send("/api/label/text", "DELETE");

export function textExportUrl(categories: string[]) {
  return `/api/label/text/export?${new URLSearchParams(categories.map((name) => ["category", name]))}`;
}
