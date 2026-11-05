import type { Analysis, Catalogue, GraphPayload, ParseReply, TestEvent } from "./types";

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
