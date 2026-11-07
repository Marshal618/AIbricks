import { useRef, useState } from "react";

const WIDTH = 620;
const HEIGHT = 240;
const MARGIN = { top: 12, right: 12, bottom: 28, left: 52 };

export interface LossPoint {
  step: number;
  loss: number;
}

function niceNumber(value: number) {
  if (value === 0) return "0";
  if (Math.abs(value) >= 100) return value.toFixed(0);
  if (Math.abs(value) >= 1) return value.toFixed(2);
  return value.toPrecision(2);
}

/** Loss over training steps. Hover or tap for exact values. */
export function LossChart({ points, totalSteps, caption = "Training loss by step", unit = "Step" }: { points: LossPoint[]; totalSteps: number; caption?: string; unit?: string }) {
  const svg = useRef<SVGSVGElement>(null);
  const [hover, setHover] = useState<LossPoint | null>(null);
  const innerWidth = WIDTH - MARGIN.left - MARGIN.right;
  const innerHeight = HEIGHT - MARGIN.top - MARGIN.bottom;
  const finite = points.filter((point) => Number.isFinite(point.loss));
  const top = Math.max(...finite.map((point) => point.loss), 1e-9);
  const x = (step: number) => MARGIN.left + ((step - 1) / Math.max(totalSteps - 1, 1)) * innerWidth;
  const y = (loss: number) => MARGIN.top + (1 - loss / top) * innerHeight;
  const path = finite.map((point, index) => `${index ? "L" : "M"}${x(point.step).toFixed(1)} ${y(point.loss).toFixed(1)}`).join(" ");

  const onMove = (event: React.PointerEvent<SVGSVGElement>) => {
    const box = svg.current?.getBoundingClientRect();
    if (!box || !finite.length) return;
    const px = ((event.clientX - box.left) / box.width) * WIDTH;
    const step = Math.round(((px - MARGIN.left) / innerWidth) * (totalSteps - 1)) + 1;
    setHover(finite.reduce((best, point) => (Math.abs(point.step - step) < Math.abs(best.step - step) ? point : best), finite[0]));
  };

  return (
    <figure className="loss-chart">
      <figcaption>{caption}</figcaption>
      <div className="chart-wrap">
        <svg
          ref={svg}
          viewBox={`0 0 ${WIDTH} ${HEIGHT}`}
          role="img"
          aria-label={finite.length ? `Loss fell from ${niceNumber(finite[0].loss)} to ${niceNumber(finite[finite.length - 1].loss)} over ${finite.length} ${unit.toLowerCase()}s.` : "Waiting for the first result."}
          onPointerMove={onMove}
          onPointerLeave={() => setHover(null)}
        >
          {[0, 0.5, 1].map((fraction) => {
            const value = top * fraction;
            return (
              <g key={fraction}>
                <line className="grid" x1={MARGIN.left} x2={WIDTH - MARGIN.right} y1={y(value)} y2={y(value)} />
                <text className="tick" x={MARGIN.left - 6} y={y(value) + 4} textAnchor="end">{niceNumber(value)}</text>
              </g>
            );
          })}
          <text className="tick" x={MARGIN.left} y={HEIGHT - 8}>{unit} 1</text>
          <text className="tick" x={WIDTH - MARGIN.right} y={HEIGHT - 8} textAnchor="end">{unit} {totalSteps}</text>
          <path className="loss-line" d={path} />
          {hover && (
            <>
              <line className="crosshair" x1={x(hover.step)} x2={x(hover.step)} y1={MARGIN.top} y2={MARGIN.top + innerHeight} />
              <circle className="loss-dot" cx={x(hover.step)} cy={y(hover.loss)} r={4} />
            </>
          )}
        </svg>
        {hover && (
          <div className="chart-tip" style={{ left: `${Math.min((x(hover.step) / WIDTH) * 100, 80)}%` }}>
            {unit} {hover.step}: <strong>{niceNumber(hover.loss)}</strong>
          </div>
        )}
      </div>
      {finite.length > 0 && (
        <details className="chart-table">
          <summary>Show values as a table</summary>
          <table>
            <thead><tr><th scope="col">{unit}</th><th scope="col">Loss</th></tr></thead>
            <tbody>
              {finite.map((point) => <tr key={point.step}><td>{point.step}</td><td>{point.loss.toFixed(4)}</td></tr>)}
            </tbody>
          </table>
        </details>
      )}
    </figure>
  );
}
