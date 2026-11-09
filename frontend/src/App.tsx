import {
  Background, BackgroundVariant, Controls, ReactFlow, ReactFlowProvider, addEdge, useEdgesState, useNodesState, useReactFlow,
  type Connection, type Edge, type FinalConnectionState, type OnConnectStartParams,
} from "@xyflow/react";
import { useCallback, useEffect, useMemo, useRef, useState, type DragEvent } from "react";
import * as api from "./api";
import { BrickNode } from "./BrickNode";
import { CodePage } from "./CodePage";
import { DataScreen } from "./DataScreen";
import { EditorContext, type EditorContextValue } from "./context";
import { Dock } from "./Dock";
import { ExportPage } from "./ExportPage";
import { Home } from "./Home";
import { depths } from "./graphOrder";
import { Inspector } from "./Inspector";
import { Library, type LibraryFocus } from "./Library";
import { NewModelDialog } from "./NewModelDialog";
import { Outline } from "./Outline";
import { ProjectsDialog } from "./ProjectsDialog";
import { PortShape } from "./PortGlyph";
import { EMPTY_TEST, TestPage, type TestState } from "./TestPage";
import { TrainPage } from "./TrainPage";
import type { Analysis, Brick, BrickNodeType, Catalogue, DatasetSummary, FixKind, GraphPayload, Kit, Mode, ParamValue, ParseReply, ProjectData, TrainingOverride } from "./types";

const nodeTypes = { brick: BrickNode };
const STORAGE_KEY = "ai-bricks-build-v1";
const MODE_KEY = "ai-bricks-mode";
const SNAP_GAP = 64;
const SNAP_RADIUS = 130;
const FALLBACK_HEIGHT = 110;
const BRICK_WIDTH = 240;

interface Saved {
  projectName: string;
  training: string;
  trainingOverride?: TrainingOverride | null;
  nodes: BrickNodeType[];
  edges: Edge[];
  kitId: string | null;
  projectId?: string;
  datasetId?: string | null;
}

type DatasetTab = "models" | "label";
type Screen = { name: "home" } | { name: "data"; dataset: string | null; tab?: DatasetTab } | { name: "build" };

function screenFromHash(): Screen {
  const parts = window.location.hash.replace(/^#\/?/, "").split("/");
  if (parts[0] === "build") return { name: "build" };
  if (parts[0] === "data") return { name: "data", dataset: parts[1] || null, tab: parts[1] && (parts[2] === "models" || parts[2] === "label") ? parts[2] : undefined };
  return { name: "home" };
}

function hashOf(screen: Screen) {
  if (screen.name === "build") return "#/build";
  if (screen.name === "data") return screen.dataset ? `#/data/${screen.dataset}${screen.tab ? `/${screen.tab}` : ""}` : "#/data";
  return "#/";
}

const AUTOSAVE_DELAY = 800;
const PROJECT_FILE_FORMAT = "ai-bricks-project";

function loadSaved(): Saved | null {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    return raw ? JSON.parse(raw) : null;
  } catch {
    return null;
  }
}

function loadMode(): Mode {
  try {
    return localStorage.getItem(MODE_KEY) === "advanced" ? "advanced" : "beginner";
  } catch {
    return "beginner";
  }
}

function makeId() {
  return `b${Date.now().toString(36)}${Math.random().toString(36).slice(2, 6)}`;
}

function defaultParams(brick: Brick) {
  return Object.fromEntries(brick.params.map((param) => [param.key, param.default]));
}

function heightOf(node: BrickNodeType) {
  return node.measured?.height ?? FALLBACK_HEIGHT;
}

function Editor({ catalogue }: { catalogue: Catalogue }) {
  const bricks = useMemo(() => Object.fromEntries(catalogue.bricks.map((brick) => [brick.id, brick])), [catalogue]);
  const saved = useRef(loadSaved()).current;
  const flow = useReactFlow<BrickNodeType, Edge>();
  const [nodes, setNodes, onNodesChange] = useNodesState<BrickNodeType>(
    (saved?.nodes ?? []).filter((node) => bricks[node.data?.brickId]).map((node) => ({ ...node, selected: false })),
  );
  const [edges, setEdges, onEdgesChange] = useEdgesState<Edge>(saved?.edges ?? []);
  const [projectName, setProjectName] = useState(saved?.projectName ?? "Untitled model");
  const [training, setTraining] = useState(saved?.training ?? "balanced");
  const [trainingOverride, setTrainingOverride] = useState<TrainingOverride | null>(saved?.trainingOverride ?? null);
  const [mode, setModeState] = useState<Mode>(loadMode);
  const [codeDraft, setCodeDraft] = useState<{ text: string; base: string } | null>(null);
  const [tidyTick, setTidyTick] = useState(0);
  const [kitId, setKitId] = useState<string | null>(saved?.kitId ?? null);
  const [projectId, setProjectId] = useState(() => saved?.projectId ?? makeId());
  const [showProjects, setShowProjects] = useState(false);
  const [saveState, setSaveState] = useState<"saved" | "saving" | "failed" | "idle">("idle");
  const [screen, setScreen] = useState<Screen>(screenFromHash);
  const [showWelcome, setShowWelcome] = useState(screen.name === "build" && (!saved || saved.nodes.length === 0));
  const [datasetId, setDatasetId] = useState<string | null>(saved?.datasetId ?? null);
  const [pendingDataset, setPendingDataset] = useState<DatasetSummary | null>(null);
  const [analysis, setAnalysis] = useState<Analysis | null>(null);
  const [offline, setOffline] = useState(false);
  const [page, setPage] = useState<"design" | "code" | "test" | "train" | "export">("design");
  const [leftTab, setLeftTab] = useState<"library" | "outline">("library");
  const [toast, setToast] = useState<string | null>(null);
  const [connecting, setConnecting] = useState<EditorContextValue["connecting"]>(null);
  const [test, setTest] = useState<TestState>(EMPTY_TEST);
  const toastTimer = useRef<number>();
  const testAbort = useRef<AbortController | null>(null);

  useEffect(() => {
    const onHash = () => setScreen(screenFromHash());
    window.addEventListener("hashchange", onHash);
    return () => window.removeEventListener("hashchange", onHash);
  }, []);
  const goTo = useCallback((next: Screen) => {
    const hash = hashOf(next);
    if (window.location.hash !== hash) window.location.hash = hash;
    setScreen(next);
  }, []);

  const say = useCallback((message: string) => {
    setToast(message);
    window.clearTimeout(toastTimer.current);
    toastTimer.current = window.setTimeout(() => setToast(null), 6000);
  }, []);

  const graph: GraphPayload = useMemo(() => ({
    projectName,
    training,
    trainingOverride,
    nodes: nodes.map((node) => ({ id: node.id, type: node.data.brickId, params: node.data.params })),
    edges: edges.map((edge) => ({ source: edge.source, target: edge.target, targetHandle: edge.targetHandle ?? "in" })),
  }), [projectName, training, trainingOverride, nodes, edges]);
  const graphKey = JSON.stringify(graph);
  const latestGraph = useRef(graph);
  latestGraph.current = graph;

  // A test result only describes the exact build it ran on, so any change clears it.
  const firstGraphKey = useRef(graphKey);
  useEffect(() => {
    if (firstGraphKey.current === graphKey) return;
    firstGraphKey.current = graphKey;
    testAbort.current?.abort();
    testAbort.current = null;
    setTest(EMPTY_TEST);
  }, [graphKey]);

  const refresh = useCallback(async (payload: GraphPayload, signal?: AbortSignal) => {
    try {
      const result = await api.analyse(payload, signal);
      if (!signal?.aborted) {
        setAnalysis(result);
        setOffline(false);
      }
    } catch (error) {
      if (!(error instanceof DOMException && error.name === "AbortError")) setOffline(true);
    }
  }, []);

  useEffect(() => {
    const controller = new AbortController();
    const timer = window.setTimeout(() => refresh(JSON.parse(graphKey), controller.signal), 200);
    return () => { window.clearTimeout(timer); controller.abort(); };
  }, [graphKey, refresh]);

  useEffect(() => {
    const data: Saved = { projectName, training, trainingOverride, nodes, edges, kitId, projectId, datasetId };
    try { localStorage.setItem(STORAGE_KEY, JSON.stringify(data)); } catch { /* the build still works without saving */ }
  }, [projectName, training, trainingOverride, nodes, edges, kitId, projectId, datasetId]);

  // Keeps the open project on disk. A blank canvas isn't worth a place in the list.
  const projectData: ProjectData = useMemo(
    () => ({ projectName, training, trainingOverride, nodes, edges, kitId, datasetId }),
    [projectName, training, trainingOverride, nodes, edges, kitId, datasetId],
  );
  const unsaved = useRef<{ id: string; data: ProjectData } | null>(null);
  const writeProject = useCallback(async (id: string, data: ProjectData) => {
    setSaveState("saving");
    try {
      await api.saveProject(id, data);
      setSaveState("saved");
    } catch {
      setSaveState("failed");
    }
  }, []);
  const flushProject = useCallback(() => {
    const pending = unsaved.current;
    unsaved.current = null;
    if (pending) void writeProject(pending.id, pending.data);
  }, [writeProject]);
  useEffect(() => {
    if (!projectData.nodes.length) return;
    unsaved.current = { id: projectId, data: projectData };
    const timer = window.setTimeout(flushProject, AUTOSAVE_DELAY);
    return () => window.clearTimeout(timer);
  }, [projectData, projectId, flushProject]);

  const portName = useCallback((type: string) => catalogue.portTypes[type]?.name ?? type, [catalogue]);

  const converterHint = useCallback((from: string, accepts: string[]) => {
    const converter = catalogue.bricks.find((brick) =>
      brick.size === "small" && brick.inputs.length === 1 && brick.inputs[0].accepts.includes(from) && brick.output && accepts.includes(brick.output));
    return converter
      ? `Add ${converter.name} between them.`
      : "No connector exists for these two types yet.";
  }, [catalogue]);

  const wouldLoop = useCallback((source: string, target: string, currentEdges: Edge[]) => {
    const seen = new Set<string>();
    const walk = (id: string): boolean => {
      if (id === source) return true;
      if (seen.has(id)) return false;
      seen.add(id);
      return currentEdges.filter((edge) => edge.source === id).some((edge) => walk(edge.target));
    };
    return walk(target);
  }, []);

  const typeForNode = useCallback((nodeId: string) => bricks[flow.getNode(nodeId)?.data.brickId ?? ""], [bricks, flow]);

  const isValidConnection = useCallback((connection: Connection | Edge) => {
    const { source, target, targetHandle } = connection;
    if (!source || !target || source === target) return false;
    const from = typeForNode(source);
    const to = typeForNode(target);
    if (!from?.output || !to) return false;
    const port = to.inputs.find((candidate) => candidate.id === (targetHandle ?? "in"));
    if (!port || !port.accepts.includes(from.output)) return false;
    const current = flow.getEdges();
    if (current.some((edge) => edge.target === target && (edge.targetHandle ?? "in") === port.id)) return false;
    return !wouldLoop(source, target, current);
  }, [flow, typeForNode, wouldLoop]);

  const onConnect = useCallback((connection: Connection) => {
    setEdges((current) => addEdge({ ...connection, animated: false }, current));
  }, [setEdges]);

  const onConnectStart = useCallback((_event: unknown, params: OnConnectStartParams) => {
    const brick = params.nodeId ? typeForNode(params.nodeId) : undefined;
    if (!brick || !params.nodeId) return;
    if (params.handleType === "source") {
      setConnecting({ nodeId: params.nodeId, side: "source", types: brick.output ? [brick.output] : [] });
    } else {
      const port = brick.inputs.find((candidate) => candidate.id === params.handleId);
      setConnecting({ nodeId: params.nodeId, side: "target", types: port?.accepts ?? [] });
    }
  }, [typeForNode]);

  const onConnectEnd = useCallback((_event: MouseEvent | TouchEvent, state: FinalConnectionState) => {
    setConnecting(null);
    if (state.isValid || !state.fromNode || !state.toNode) return;
    const start = bricks[state.fromNode.data.brickId as string];
    const end = bricks[state.toNode.data.brickId as string];
    const [upper, lower] = state.fromHandle?.type === "source" ? [start, end] : [end, start];
    if (!upper?.output) return say(`${upper?.name ?? "That brick"} has no output to connect.`);
    if (!lower?.inputs.length) return say(`${lower?.name ?? "That brick"} has nothing to plug into.`);
    const accepted = lower.inputs.flatMap((port) => port.accepts);
    if (!accepted.includes(upper.output)) {
      say(`${portName(upper.output)} can't connect to ${accepted.map(portName).join(" or ")}. ${converterHint(upper.output, accepted)}`);
    } else {
      say(`That input is already connected, or the connection would create a cycle.`);
    }
  }, [bricks, converterHint, portName, say]);

  /** Finds an open output close above this brick that it could snap onto. */
  const findSnapTarget = useCallback((brick: Brick, position: { x: number; y: number }, excludeId: string | null, allNodes: BrickNodeType[], allEdges: Edge[]) => {
    const port = brick.inputs[0];
    if (!port || brick.inputs.filter((input) => !input.optional).length > 1) return null;
    let best: BrickNodeType | null = null;
    let bestDistance = SNAP_RADIUS;
    for (const candidate of allNodes) {
      if (candidate.id === excludeId) continue;
      const output = bricks[candidate.data.brickId].output;
      if (!output || !port.accepts.includes(output)) continue;
      if (allEdges.some((edge) => edge.source === candidate.id)) continue;
      if (excludeId && wouldLoop(candidate.id, excludeId, allEdges)) continue;
      const distance = Math.hypot(candidate.position.x - position.x, candidate.position.y + heightOf(candidate) - position.y);
      if (distance < bestDistance) { best = candidate; bestDistance = distance; }
    }
    return best;
  }, [bricks, wouldLoop]);

  const placeBrick = useCallback((brick: Brick, position: { x: number; y: number }, parent: BrickNodeType | null) => {
    const id = makeId();
    const placed = parent ? { x: parent.position.x, y: parent.position.y + heightOf(parent) + SNAP_GAP } : { ...position };
    // Branching from an output that is already connected: step sideways so nothing overlaps.
    while (parent && nodes.some((node) => Math.abs(node.position.x - placed.x) < BRICK_WIDTH && Math.abs(node.position.y - placed.y) < FALLBACK_HEIGHT)) {
      placed.x += BRICK_WIDTH + 60;
    }
    setNodes((current) => [
      ...current.map((node) => ({ ...node, selected: false })),
      { id, type: "brick", position: placed, selected: true, data: { brickId: brick.id, params: defaultParams(brick) } },
    ]);
    if (parent) {
      setEdges((current) => addEdge({ id: `e-${parent.id}-${id}`, source: parent.id, target: id, sourceHandle: "out", targetHandle: brick.inputs[0].id }, current));
    }
    setShowWelcome(false);
  }, [nodes, setEdges, setNodes]);

  const addFromPalette = useCallback((brick: Brick) => {
    const open = nodes.filter((node) => bricks[node.data.brickId].output && !edges.some((edge) => edge.source === node.id));
    const selected = nodes.find((node) => node.selected && bricks[node.data.brickId].output);
    const lowestFirst = [...open].sort((a, b) => b.position.y - a.position.y);
    const candidates = selected ? [selected, ...lowestFirst] : lowestFirst;
    const port = brick.inputs[0];
    const parent = port
      ? candidates.find((node) => port.accepts.includes(bricks[node.data.brickId].output as string)) ?? null
      : null;
    if (parent) return placeBrick(brick, parent.position, parent);

    const rightmost = nodes.reduce((max, node) => Math.max(max, node.position.x), -BRICK_WIDTH - 60);
    placeBrick(brick, { x: rightmost + BRICK_WIDTH + 60, y: nodes.length ? nodes[0].position.y : 0 }, null);
    if (port && open.length) {
      const blocked = bricks[(selected ?? candidates[0]).data.brickId].output as string;
      say(`${brick.name} takes ${port.accepts.map(portName).join(" or ")}, which doesn't fit ${portName(blocked)}. ${converterHint(blocked, port.accepts)}`);
    }
  }, [bricks, converterHint, edges, nodes, placeBrick, portName, say]);

  const canvasRef = useRef<HTMLElement>(null);
  const fitToCanvas = useCallback(() => {
    if (!canvasRef.current?.clientWidth) return; // the Design page is hidden
    flow.fitView({ padding: { top: "56px", bottom: "40px", right: "40px", left: "40px" }, maxZoom: 1, duration: 200 });
  }, [flow]);

  // Refit when bricks are added or removed, not on every drag.
  useEffect(() => {
    if (!nodes.length) return;
    window.requestAnimationFrame(fitToCanvas);
    // Refit once the check has run, because problem messages can make components taller.
    const settled = window.setTimeout(fitToCanvas, 450);
    return () => window.clearTimeout(settled);
  }, [nodes.length, fitToCanvas]);

  useEffect(() => {
    if (!canvasRef.current) return;
    const observer = new ResizeObserver(() => fitToCanvas());
    observer.observe(canvasRef.current);
    return () => observer.disconnect();
  }, [fitToCanvas]);

  const onDrop = useCallback((event: DragEvent) => {
    event.preventDefault();
    const brick = bricks[event.dataTransfer.getData("application/ai-brick")];
    if (!brick) return;
    const position = flow.screenToFlowPosition({ x: event.clientX - BRICK_WIDTH / 2, y: event.clientY - 20 });
    placeBrick(brick, position, findSnapTarget(brick, position, null, nodes, edges));
  }, [bricks, edges, findSnapTarget, flow, nodes, placeBrick]);

  const onNodeDragStop = useCallback((_event: unknown, dragged: BrickNodeType) => {
    const brick = bricks[dragged.data.brickId];
    if (!brick.inputs.length || edges.some((edge) => edge.target === dragged.id)) return;
    const parent = findSnapTarget(brick, dragged.position, dragged.id, nodes, edges);
    if (!parent) return;
    setNodes((current) => current.map((node) => node.id === dragged.id
      ? { ...node, position: { x: parent.position.x, y: parent.position.y + heightOf(parent) + SNAP_GAP } }
      : node));
    setEdges((current) => addEdge({ id: `e-${parent.id}-${dragged.id}`, source: parent.id, target: dragged.id, sourceHandle: "out", targetHandle: brick.inputs[0].id }, current));
  }, [bricks, edges, findSnapTarget, nodes, setEdges, setNodes]);

  const changeParam = useCallback((nodeId: string, key: string, value: ParamValue) => {
    setNodes((current) => current.map((node) => node.id === nodeId
      ? { ...node, data: { ...node.data, params: { ...node.data.params, [key]: value } } }
      : node));
  }, [setNodes]);

  const deleteNode = useCallback((nodeId: string) => {
    setNodes((current) => current.filter((node) => node.id !== nodeId));
    setEdges((current) => current.filter((edge) => edge.source !== nodeId && edge.target !== nodeId));
  }, [setEdges, setNodes]);

  const openBigBrick = useCallback((nodeId: string) => {
    const node = flow.getNode(nodeId);
    if (!node) return;
    const brick = bricks[node.data.brickId];
    const spacing = FALLBACK_HEIGHT + SNAP_GAP;
    const parts: BrickNodeType[] = brick.parts.map((part, index) => {
      const partBrick = bricks[part.brick];
      const params = { ...defaultParams(partBrick), ...part.params };
      for (const key of Object.keys(params)) {
        if (key in node.data.params) params[key] = node.data.params[key];
      }
      return {
        id: makeId(),
        type: "brick",
        position: { x: node.position.x, y: node.position.y + index * spacing },
        data: { brickId: part.brick, params },
      };
    });
    const shift = (parts.length - 1) * spacing;
    setNodes((current) => [
      ...current.filter((other) => other.id !== nodeId).map((other) =>
        other.position.y > node.position.y ? { ...other, position: { ...other.position, y: other.position.y + shift } } : other),
      ...parts,
    ]);
    setEdges((current) => {
      const rewired = current
        .map((edge) => {
          if (edge.target === nodeId) return { ...edge, id: makeId(), target: parts[0].id };
          if (edge.source === nodeId) return { ...edge, id: makeId(), source: parts[parts.length - 1].id };
          return edge;
        });
      const inner = parts.slice(1).map((part, index) => ({
        id: makeId(), source: parts[index].id, target: part.id, sourceHandle: "out", targetHandle: "in",
      }));
      return [...rewired, ...inner];
    });
  }, [bricks, flow, setEdges, setNodes]);

  const loadKit = useCallback((kit: Kit) => {
    flushProject();
    setProjectId(makeId());
    setNodes(kit.nodes.map((node) => ({
      id: node.id,
      type: "brick",
      position: { x: node.x, y: node.y },
      data: { brickId: node.type, params: { ...defaultParams(bricks[node.type]), ...node.params } },
    })));
    setEdges(kit.edges.map((edge) => ({
      id: `e-${edge.source}-${edge.target}`, source: edge.source, target: edge.target, sourceHandle: "out", targetHandle: edge.targetHandle,
    })));
    setProjectName(kit.projectName);
    setTraining(kit.training);
    setTrainingOverride(null);
    setCodeDraft(null);
    setKitId(kit.id);
    setDatasetId(pendingDataset?.id ?? null);
    if (pendingDataset && pendingDataset.categories.length >= 2) {
      const classes = pendingDataset.categories.join(", ");
      setNodes((current) => current.map((node) => (
        "classes" in node.data.params ? { ...node, data: { ...node.data, params: { ...node.data.params, classes } } } : node
      )));
    }
    setPendingDataset(null);
    setTest(EMPTY_TEST);
    setShowWelcome(false);
    setPage("design");
  }, [bricks, flushProject, pendingDataset, setEdges, setNodes]);

  const startBlank = useCallback(() => {
    flushProject();
    setProjectId(makeId());
    testAbort.current?.abort();
    setNodes([]);
    setEdges([]);
    setKitId(null);
    setDatasetId(pendingDataset?.id ?? null);
    setPendingDataset(null);
    setProjectName("Untitled model");
    setTraining("balanced");
    setTrainingOverride(null);
    setCodeDraft(null);
    setTest(EMPTY_TEST);
    setShowWelcome(false);
    setPage("design");
  }, [flushProject, pendingDataset, setEdges, setNodes]);

  const newModel = useCallback(() => {
    goTo({ name: "build" });
    setShowWelcome(true);
  }, [goTo]);

  /** Applies a training check's fix to the open design, then goes to the test, since the design changed. */
  const applyFix = useCallback((fix: FixKind) => {
    if (fix === "augment") {
      setNodes((current) => current.map((node) => node.data.brickId === "image_folder"
        ? { ...node, data: { ...node.data, params: { ...node.data.params, augment: true } } } : node));
    } else if (fix === "slower" || fix === "longer") {
      const dial = { ...catalogue.training[training], ...(trainingOverride ?? {}) };
      const epochs = Number(dial.epochs), rate = Number(dial.learning_rate), batch = Number(dial.batch_size);
      setTrainingOverride(fix === "slower"
        ? { epochs, batch_size: batch, learning_rate: Number((rate / 3).toPrecision(2)) }
        : { epochs: Math.min(10000, epochs * 2), batch_size: batch, learning_rate: rate });
    } else {
      return;
    }
    setPage("test");
  }, [catalogue, training, trainingOverride]);

  const buildWithDataset = useCallback((dataset: DatasetSummary) => {
    setPendingDataset(dataset);
    newModel();
  }, [newModel]);

  const openProject = useCallback((id: string, data: ProjectData) => {
    flushProject();
    testAbort.current?.abort();
    setProjectId(id);
    setNodes(data.nodes.filter((node) => bricks[node.data?.brickId]).map((node) => ({ ...node, selected: false })));
    setEdges(data.edges);
    setProjectName(data.projectName);
    setTraining(catalogue.training[data.training] ? data.training : "balanced");
    setTrainingOverride(data.trainingOverride ?? null);
    setKitId(data.kitId ?? null);
    setDatasetId(data.datasetId ?? null);
    goTo({ name: "build" });
    setCodeDraft(null);
    setTest(EMPTY_TEST);
    setShowProjects(false);
    setShowWelcome(false);
    setPage("design");
  }, [bricks, catalogue, flushProject, goTo, setEdges, setNodes]);

  const saveProjectFile = useCallback(() => {
    const body = JSON.stringify({ format: PROJECT_FILE_FORMAT, version: 1, ...projectData }, null, 2);
    const url = URL.createObjectURL(new Blob([body], { type: "application/json" }));
    const link = document.createElement("a");
    link.href = url;
    link.download = `${projectName.toLowerCase().match(/[a-z0-9]+/g)?.join("_") ?? "my_model"}.aibricks.json`;
    link.click();
    URL.revokeObjectURL(url);
  }, [projectData, projectName]);

  /** Opens a saved file as a new project. Returns why it couldn't, if it couldn't. */
  /** Opens a design made elsewhere, such as an experiment's winner, as a new project. */
  const openDesign = useCallback((graph: GraphPayload, dataset: string) => {
    openProject(makeId(), {
      projectName: graph.projectName, training: graph.training, trainingOverride: graph.trainingOverride, kitId: null, datasetId: dataset,
      nodes: graph.nodes.map((node, index) => ({ id: node.id, type: "brick" as const, position: { x: 0, y: index * 140 }, data: { brickId: node.type, params: node.params } })),
      edges: graph.edges.map((edge, index) => ({ id: `e-${edge.source}-${edge.target}-${index}`, source: edge.source, target: edge.target, targetHandle: edge.targetHandle })),
    });
    setPage("design");
    setTidyTick((value) => value + 1);
  }, [openProject]);

  const openProjectFile = useCallback((file: unknown): string | null => {
    const data = file as Partial<ProjectData> & { format?: string };
    if (data?.format !== PROJECT_FILE_FORMAT || !Array.isArray(data.nodes) || !Array.isArray(data.edges)) {
      return "This file isn't an AI Bricks project.";
    }
    const nodes = data.nodes.filter((node) => (
      typeof node?.id === "string" && bricks[node.data?.brickId] && Number.isFinite(node.position?.x) && Number.isFinite(node.position?.y)
    )).map((node) => ({
      id: node.id,
      type: "brick" as const,
      position: node.position,
      data: { brickId: node.data.brickId, params: { ...defaultParams(bricks[node.data.brickId]), ...node.data.params } },
    }));
    if (!nodes.length) return "This project file has no components this version of AI Bricks can read.";
    const ids = new Set(nodes.map((node) => node.id));
    const edges = data.edges.filter((edge) => ids.has(edge?.source) && ids.has(edge?.target));
    openProject(makeId(), {
      projectName: typeof data.projectName === "string" ? data.projectName.slice(0, 60) : "Untitled model",
      training: typeof data.training === "string" ? data.training : "balanced",
      trainingOverride: data.trainingOverride ?? null,
      nodes,
      edges,
      kitId: typeof data.kitId === "string" ? data.kitId : null,
      datasetId: null,
    });
    return null;
  }, [bricks, openProject]);

  /** Gives the model these categories, such as the ones from an open dataset. */
  const useCategories = useCallback((names: string[]) => {
    setNodes((current) => current.map((node) => (
      "classes" in node.data.params
        ? { ...node, data: { ...node.data, params: { ...node.data.params, classes: names.join(", ") } } }
        : node
    )));
    say("Your model's categories changed, so it needs a new test run before training.");
  }, [say, setNodes]);

  const selectNode = useCallback((id: string) => {
    setPage("design");
    setNodes((current) => current.map((node) => ({ ...node, selected: node.id === id })));
    const node = flow.getNode(id);
    if (node) {
      const { zoom } = flow.getViewport();
      flow.setCenter(node.position.x + BRICK_WIDTH / 2, node.position.y + heightOf(node) / 2, { zoom, duration: 250 });
    }
  }, [flow, setNodes]);

  const tidyLayout = useCallback(() => {
    const depth = depths(nodes, edges);
    const levels = new Map<number, BrickNodeType[]>();
    for (const node of [...nodes].sort((a, b) => a.position.x - b.position.x)) {
      const level = depth.get(node.id) ?? 0;
      levels.set(level, [...(levels.get(level) ?? []), node]);
    }
    const placed = new Map<string, { x: number; y: number }>();
    let y = 0;
    for (const level of [...levels.keys()].sort((a, b) => a - b)) {
      const row = levels.get(level)!;
      const spacing = BRICK_WIDTH + 60;
      row.forEach((node, index) => placed.set(node.id, { x: (index - (row.length - 1) / 2) * spacing, y }));
      y += Math.max(...row.map(heightOf)) + SNAP_GAP + 20;
    }
    setNodes((current) => current.map((node) => ({ ...node, position: placed.get(node.id) ?? node.position })));
    window.setTimeout(fitToCanvas, 30);
  }, [edges, fitToCanvas, nodes, setNodes]);

  const setMode = useCallback((next: Mode) => {
    setModeState(next);
    try { localStorage.setItem(MODE_KEY, next); } catch { /* the choice just won't be remembered */ }
  }, []);

  /** Rebuilds the canvas from code, keeping components that still exist where they were. */
  const applyCode = useCallback((reply: ParseReply) => {
    const unused = [...nodes];
    let reusedAll = true;
    const idMap = new Map<string, string>();
    const next: BrickNodeType[] = reply.nodes.map((node) => {
      const matchIndex = unused.findIndex((candidate) => candidate.data.brickId === node.type);
      const match = matchIndex >= 0 ? unused.splice(matchIndex, 1)[0] : undefined;
      if (!match) reusedAll = false;
      const id = match?.id ?? makeId();
      idMap.set(node.id, id);
      const defaults = defaultParams(bricks[node.type]);
      return {
        id,
        type: "brick",
        position: match?.position ?? { x: 0, y: 0 },
        data: { brickId: node.type, params: { ...defaults, ...node.params } },
      };
    });
    setNodes(next);
    setEdges(reply.edges.map((edge) => ({
      id: makeId(),
      source: idMap.get(edge.source) as string,
      target: idMap.get(edge.target) as string,
      sourceHandle: "out",
      targetHandle: edge.targetHandle,
    })));
    setKitId(null);
    setPage("design");
    if (!reusedAll || unused.length) setTidyTick((tick) => tick + 1);
    say(`Converted your code to ${next.length} blocks.`);
  }, [bricks, nodes, say, setEdges, setNodes]);

  const runTest = useCallback(async () => {
    testAbort.current?.abort();
    const controller = new AbortController();
    testAbort.current = controller;
    setPage("test");
    setTest({ ...EMPTY_TEST, running: true });
    const payload = latestGraph.current;
    try {
      await api.runTest(payload, (event) => {
        setTest((state) => {
          if (event.event === "model") return { ...state, parameters: { total: event.parameters, trainable: event.trainable } };
          if (event.event === "check") return { ...state, checks: { ...state.checks, [event.check]: event.passed } };
          if (event.event === "loss") return { ...state, points: [...state.points, { step: event.step, loss: event.loss }] };
          return { ...state, result: { passed: event.passed, detail: event.detail } };
        });
      }, controller.signal);
      await refresh(latestGraph.current);
    } catch (error) {
      if (error instanceof DOMException && error.name === "AbortError") return;
      setTest((state) => ({
        ...state,
        result: { passed: false, detail: error instanceof Error ? error.message : "The test run could not start." },
      }));
    } finally {
      if (testAbort.current === controller) setTest((state) => ({ ...state, running: false }));
    }
  }, [refresh]);

  const download = useCallback(async () => {
    try {
      await api.downloadProject(latestGraph.current);
    } catch (error) {
      say(error instanceof Error ? error.message : "The download failed.");
    }
  }, [say]);

  // The output the library matches: the selected component's, or else the lowest unconnected one.
  const nextNode = useMemo(() => {
    const picked = nodes.find((node) => node.selected && bricks[node.data.brickId].output);
    if (picked) return picked;
    const open = nodes.filter((node) => bricks[node.data.brickId].output && !edges.some((edge) => edge.source === node.id));
    return [...open].sort((a, b) => b.position.y - a.position.y)[0] ?? null;
  }, [bricks, edges, nodes]);
  const libraryFocus: LibraryFocus = !nodes.length
    ? { state: "empty" }
    : nextNode
      ? { state: "open", name: bricks[nextNode.data.brickId].name, type: bricks[nextNode.data.brickId].output as string }
      : { state: "complete" };
  const typedEdges = useMemo(() => edges.map((edge) => {
    const source = nodes.find((node) => node.id === edge.source);
    return { ...edge, className: source ? `edge-${bricks[source.data.brickId].output}` : "" };
  }), [bricks, edges, nodes]);

  const context = useMemo(
    () => ({ catalogue, bricks, analysis, mode, openBigBrick, connecting, nextNodeId: nextNode?.id ?? null, datasetId }),
    [analysis, bricks, catalogue, connecting, mode, nextNode, openBigBrick, datasetId],
  );

  useEffect(() => {
    if (tidyTick) tidyLayout();
  // Runs once after code is converted, when the new components are in place.
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [tidyTick]);
  const selected = nodes.find((node) => node.selected);
  const kit = catalogue.kits.find((candidate) => candidate.id === kitId);
  const issueCount = analysis?.issues.length ?? 0;
  const tested = analysis?.tested ?? false;
  const inputNode = nodes.find((node) => bricks[node.data.brickId].group === "socket");
  const inputShape = inputNode ? analysis?.shapes[inputNode.id] : undefined;
  const testStatus = test.running ? "Running" : tested ? "Passed" : test.result ? "Failed" : "Not run";
  const pages = [
    { id: "design", label: "Design", status: issueCount ? `${issueCount} ${issueCount === 1 ? "problem" : "problems"}` : nodes.length ? "Valid" : "Empty", tone: issueCount ? "bad" : nodes.length ? "good" : "" },
    { id: "code", label: "Code", status: codeDraft && codeDraft.text !== codeDraft.base ? "Unconverted edits" : analysis?.ok ? "Generated" : "Waiting", tone: codeDraft && codeDraft.text !== codeDraft.base ? "bad" : analysis?.ok ? "good" : "" },
    { id: "test", label: "Test", status: testStatus, tone: tested ? "good" : testStatus === "Failed" ? "bad" : "" },
    { id: "train", label: "Train", status: !analysis?.ok ? "Waiting" : !analysis.categories ? "Use export" : tested ? "Ready" : "Locked", tone: tested && analysis?.categories ? "good" : "" },
    { id: "export", label: "Export", status: tested ? "Ready" : "Locked", tone: tested ? "good" : "" },
  ] as const;

  return (
    <EditorContext.Provider value={context}>
      <div className="app">
        <nav className="appnav" aria-label="Main">
          <button className="logo" onClick={() => goTo({ name: "home" })}>
            <svg className="logo-mark" viewBox="0 0 20 20" aria-hidden="true"><rect x="1" y="1" width="8" height="8" rx="1.5" /><rect x="11" y="1" width="8" height="8" rx="1.5" opacity=".55" /><rect x="1" y="11" width="8" height="8" rx="1.5" opacity=".55" /><rect x="11" y="11" width="8" height="8" rx="1.5" /></svg>
            AI Bricks
          </button>
          {([["home", "Home"], ["data", "Data labeller"], ["build", "Model builder"]] as const).map(([name, label]) => (
            <button
              key={name}
              className={`appnav-link${screen.name === name ? " active" : ""}`}
              aria-current={screen.name === name ? "page" : undefined}
              onClick={() => {
                if (name === "build") {
                  goTo({ name: "build" });
                  if (!nodes.length) setShowWelcome(true);
                } else {
                  goTo(name === "data" ? { name: "data", dataset: null } : { name: "home" });
                }
              }}
            >
              {label}
            </button>
          ))}
          <div className="spacer" />
          <div className="mode-switch" role="radiogroup" aria-label="Experience level">
            {(["beginner", "advanced"] as const).map((value) => (
              <button key={value} role="radio" aria-checked={mode === value} className={mode === value ? "active" : ""} onClick={() => setMode(value)}>
                {value === "beginner" ? "Beginner" : "Advanced"}
              </button>
            ))}
          </div>
        </nav>
        {offline && <div className="offline" role="alert">The AI Bricks server isn't responding. Your work is saved in this browser and will be checked again when it returns.</div>}

        {screen.name === "home" && (
          <div className="page">
            <Home
              mode={mode}
              currentProjectId={projectId}
              onNewModel={newModel}
              onContinueModel={() => goTo({ name: "build" })}
              onOpenModel={(id) => { api.fetchProject(id).then((data) => openProject(id, data)).catch(() => say("That model could not open.")); }}
              onOpenData={() => goTo({ name: "data", dataset: null })}
              onOpenDataset={(id, tab) => goTo({ name: "data", dataset: id, tab })}
            />
          </div>
        )}
        {screen.name === "data" && (
          <div className="page">
            <DataScreen mode={mode} datasetId={screen.dataset} tab={screen.tab} onOpen={(id) => goTo({ name: "data", dataset: id })} onBuild={buildWithDataset} onOpenDesign={openDesign} />
          </div>
        )}
        {screen.name === "build" && (<>
        <header className="topbar">
          <label className="project-name">
            <span className="visually-hidden">Model name</span>
            <input value={projectName} maxLength={60} onChange={(event) => setProjectName(event.target.value)} />
          </label>
          <button className="button button-small" onClick={() => setShowWelcome(true)}>New model</button>
          <button className="button button-small" onClick={() => setShowProjects(true)}>Projects</button>
          <nav className="workflow" aria-label="Workflow">
            {pages.map((item, index) => (
              <button key={item.id} className={page === item.id ? "active" : ""} aria-current={page === item.id ? "page" : undefined} onClick={() => setPage(item.id)}>
                <span className="wf-index">{index + 1}</span>
                <span className="wf-text">
                  <span className="wf-label">{item.label}</span>
                  <span className={`wf-status tone-${item.tone || "none"}`}>{item.status}</span>
                </span>
              </button>
            ))}
          </nav>
          <div className="spacer" />
          {tested
            ? analysis?.categories && page !== "train" && page !== "export"
              ? <button className="button button-primary" onClick={() => setPage("train")}>Continue to training</button>
              : page !== "export" && <button className="button button-primary" onClick={() => setPage("export")}>Continue to export</button>
            : <button className="button button-primary" disabled={!analysis?.ok || test.running} onClick={runTest}>{test.running ? "Testing…" : "Run test"}</button>}
        </header>

        <div className={`design-page${page === "design" ? "" : " is-hidden"}`}>
          <aside className="left-pane" aria-label="Components and outline">
            <div className="pane-tabs" role="tablist">
              <button role="tab" aria-selected={leftTab === "library"} className={leftTab === "library" ? "active" : ""} onClick={() => setLeftTab("library")}>Library</button>
              <button role="tab" aria-selected={leftTab === "outline"} className={leftTab === "outline" ? "active" : ""} onClick={() => setLeftTab("outline")}>Outline</button>
            </div>
            {leftTab === "library"
              ? <Library onAdd={addFromPalette} focus={libraryFocus} />
              : <div className="pane-scroll"><Outline nodes={nodes} edges={edges} onSelect={selectNode} /></div>}
          </aside>
          <div className="center-pane">
            <main ref={canvasRef} className="canvas" onDragOver={(event) => { event.preventDefault(); event.dataTransfer.dropEffect = "copy"; }} onDrop={onDrop}>
              <ReactFlow<BrickNodeType, Edge>
                nodes={nodes}
                edges={typedEdges}
                nodeTypes={nodeTypes}
                onNodesChange={onNodesChange}
                onEdgesChange={onEdgesChange}
                onConnect={onConnect}
                onConnectStart={onConnectStart}
                onConnectEnd={onConnectEnd}
                onNodeDragStop={onNodeDragStop}
                isValidConnection={isValidConnection}
                snapToGrid
                snapGrid={[10, 10]}
                minZoom={0.3}
                deleteKeyCode={["Backspace", "Delete"]}
                proOptions={{ hideAttribution: true }}
              >
                <Background variant={BackgroundVariant.Dots} gap={20} size={1.2} color="var(--grid-dot)" />
                <Controls showInteractive={false} showFitView={false} />
              </ReactFlow>
              <div className="canvas-toolbar">
                <button className="button button-small" onClick={fitToCanvas} disabled={!nodes.length}>Fit to view</button>
                <button className="button button-small" onClick={tidyLayout} disabled={!nodes.length}>Tidy layout</button>
              </div>
              <div className="legend" aria-label="Connector types">
                {Object.entries(catalogue.portTypes).map(([key, value]) => (
                  <span key={key} title={value.description}><PortShape type={key} size={15} /> {value.name}</span>
                ))}
              </div>
              {!nodes.length && !showWelcome && (
                <div className="canvas-empty">
                  <strong>Empty canvas</strong>
                  <p className="muted">Add a data input from the library to begin.</p>
                </div>
              )}
              {toast && <div className="toast" role="status">{toast}</div>}
            </main>
            <Dock kit={kit} nodes={nodes} edges={edges} onSelect={selectNode} />
          </div>
          <aside className="right-pane" aria-label="Properties">
            <Inspector
              node={selected}
              training={training}
              override={trainingOverride}
              onOverride={setTrainingOverride}
              nodeCount={nodes.length}
              edgeCount={edges.length}
              onTraining={setTraining}
              onChange={changeParam}
              onDelete={deleteNode}
            />
          </aside>
        </div>
        {page === "code" && <div className="page"><CodePage
              graph={graph}
              draft={codeDraft}
              onDraft={setCodeDraft}
              onApply={applyCode}
              onDesign={() => setPage("design")}
              onAdvanced={() => setMode("advanced")}
            /></div>}
        {page === "test" && (
          <div className="page">
            <TestPage test={test} nodes={nodes} edges={edges} onRun={runTest} onDesign={() => setPage("design")} />
          </div>
        )}
        {page === "train" && (
          <div className="page">
            <TrainPage
              graph={graph}
              onDesign={() => setPage("design")}
              onTest={() => setPage("test")}
              onExport={() => setPage("export")}
              datasetId={datasetId}
              onDatasetChange={setDatasetId}
              onOpenData={(id) => goTo({ name: "data", dataset: id })}
              onOpenDatasetTab={(id, tab) => goTo({ name: "data", dataset: id, tab })}
              onApplyFix={applyFix}
              onUseCategories={useCategories}
            />
          </div>
        )}
        {page === "export" && (
          <div className="page">
            <ExportPage
              projectFolder={projectName.toLowerCase().match(/[a-z0-9]+/g)?.join("_") ?? "my_model"}
              onDownload={download}
              onTest={() => setPage("test")}
              onCode={() => setPage("code")}
            />
          </div>
        )}

        <footer className="statusbar">
          <span>{nodes.length} components</span>
          <span>{edges.length} connections</span>
          {inputShape && <span>Input: {inputShape}</span>}
          <span>Training: {trainingOverride && Object.keys(trainingOverride).length ? "Custom" : catalogue.training[training]?.name}</span>
          <span className="spacer" />
          {saveState !== "idle" && <span className={saveState === "failed" ? "tone-bad" : ""}>{saveState === "saving" ? "Saving…" : saveState === "saved" ? "Saved" : "Couldn't save to disk"}</span>}
          <span className={offline ? "tone-bad" : ""}>{offline ? "Server offline" : "Server connected"}</span>
        </footer>
        </>)}
        {showProjects && (
          <ProjectsDialog currentId={projectId} onOpen={openProject} onOpenFile={openProjectFile} onSaveFile={saveProjectFile} onClose={() => setShowProjects(false)} />
        )}
        {showWelcome && screen.name === "build" && (
          <NewModelDialog
            hasModel={nodes.length > 0}
            forDataset={pendingDataset}
            onKit={loadKit}
            onBlank={startBlank}
            onCancel={() => { setShowWelcome(false); setPendingDataset(null); }}
          />
        )}
      </div>
    </EditorContext.Provider>
  );
}

export function App() {
  const [catalogue, setCatalogue] = useState<Catalogue | null>(null);
  const [failed, setFailed] = useState(false);
  const load = useCallback(() => {
    setFailed(false);
    api.fetchCatalogue().then(setCatalogue).catch(() => setFailed(true));
  }, []);
  useEffect(load, [load]);
  if (failed) {
    return (
      <div className="splash" role="alert">
        <p>The AI Bricks server isn't responding.</p>
        <button className="button button-primary" onClick={load}>Try again</button>
      </div>
    );
  }
  if (!catalogue) return <div className="splash">Loading bricks…</div>;
  return <ReactFlowProvider><Editor catalogue={catalogue} /></ReactFlowProvider>;
}
