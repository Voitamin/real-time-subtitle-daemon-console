type Props = {
  countdownSec: number;
  progress: number;
};

export function CountdownBar({ countdownSec, progress }: Props) {
  const pct = Math.max(0, Math.min(1, progress));
  const sec = Math.round(countdownSec);
  const label = sec >= 0 ? `${sec}s` : `过期 ${Math.abs(sec)}s`;

  return (
    <div className="countdown">
      <div className="countdown-label">{label}</div>
      <div className="countdown-track">
        <div className="countdown-fill" style={{ width: `${pct * 100}%` }} />
      </div>
    </div>
  );
}
