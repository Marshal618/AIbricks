const SHAPES: Record<string, JSX.Element> = {
  image: <rect x="2.5" y="2.5" width="15" height="15" rx="1.5" />,
  series: <circle cx="10" cy="10" r="8" />,
  summary: <polygon points="10,1 19,10 10,19 1,10" />,
  text: <polygon points="10,1.8 19,17.8 1,17.8" />,
};

/** The one shape used everywhere for a data type: on components, in the library and in the legend. */
export function PortShape({ type, size = 13 }: { type: string; size?: number }) {
  return (
    <svg className={`shape shape-${type}`} width={size} height={size} viewBox="0 0 20 20" aria-hidden="true">
      {SHAPES[type]}
    </svg>
  );
}
