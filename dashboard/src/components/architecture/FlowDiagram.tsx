import { motion } from 'framer-motion';

export function DiagramBox({
  label,
  sublabel,
  color,
  plastic = false,
  highlight = false,
}: {
  label: string;
  sublabel?: string;
  color: string;
  plastic?: boolean;
  highlight?: boolean;
}) {
  return (
    <motion.div
      initial={{ opacity: 0, scale: 0.98 }}
      animate={{ opacity: 1, scale: 1 }}
      className={`
        px-4 py-3 rounded-lg text-center min-w-[90px]
        ${plastic ? 'border-2 border-dashed' : 'bg-surface-100 border border-surface-200'}
        ${highlight ? 'ring-2 ring-accent-green ring-offset-2 ring-offset-surface-50' : ''}
      `}
      style={{
        borderColor: plastic ? color : undefined,
        backgroundColor: plastic ? `${color}10` : undefined,
      }}
    >
      <div className="font-mono text-xs font-bold" style={{ color }}>
        {label}
      </div>
      {sublabel ? <div className="text-[11px] text-text-muted mt-0.5">{sublabel}</div> : null}
    </motion.div>
  );
}

export function Arrow() {
  return (
    <svg width="24" height="24" viewBox="0 0 24 24" className="text-text-muted flex-shrink-0">
      <path
        d="M5 12h14m-4-4l4 4-4 4"
        stroke="currentColor"
        strokeWidth="2"
        fill="none"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
    </svg>
  );
}

