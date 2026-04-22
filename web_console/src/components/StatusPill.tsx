type Props = {
  displayed: boolean;
  deleted: boolean;
  late: boolean;
  suppressed: boolean;
};

export function StatusPill({ displayed, deleted, late, suppressed }: Props) {
  if (deleted) {
    return <span className="status-pill status-deleted">已删除</span>;
  }
  if (suppressed) {
    return <span className="status-pill status-pending">并入下一句</span>;
  }
  if (displayed) {
    return <span className="status-pill status-displayed">已展示</span>;
  }
  if (late) {
    return <span className="status-pill status-late">晚到</span>;
  }
  return <span className="status-pill status-pending">未展示</span>;
}
