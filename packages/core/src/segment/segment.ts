export type SegmentKind = "prose" | "code" | "kv";

export interface Segment {
  start: number;
  end: number;
  kind: SegmentKind;
  text: string;
}

const KV_LINE = /^\s*[A-Za-z_][A-Za-z0-9_.-]*\s*[=:]\s*\S/;

/**
 * Split text into contiguous, gap-free segments with absolute offsets.
 * Fenced blocks (```...```) are "code"; runs of key=value / key: value
 * lines are "kv"; everything else is "prose".
 */
export function segmentText(text: string): Segment[] {
  if (text.length === 0) return [];

  // Pass 1: fence boundaries. The closing fence consumes its trailing newline
  // ONLY for LF ("\n") endings, so the next region starts cleanly on the
  // following line. With CRLF the "\r" is not consumed: the code segment ends
  // at the backticks and the leading "\r\n" opens the next region, which
  // surfaces as a short prose segment between the code block and the text
  // after it. The tiling invariants still hold (segments stay contiguous and
  // offsets stay absolute); only the kind boundary lands one line-ending
  // earlier than an LF input would give. Pinned by the CRLF tests in
  // test/segment/segment.test.ts.
  //
  // Deliberately NOT normalized upstream: spans must stay faithful to the
  // original string, so rewriting CRLF before segmenting would shift every
  // downstream finding offset. Per review, CRLF-awareness belongs here in the
  // fence matcher when it is tuned (Task 10 owns that tuning).
  const regions: Array<{ start: number; end: number; kind: "code" | "other" }> = [];
  const fence = /```[\s\S]*?(?:```\n?|$)/g;
  let last = 0;
  for (let m = fence.exec(text); m !== null; m = fence.exec(text)) {
    if (m.index > last) regions.push({ start: last, end: m.index, kind: "other" });
    regions.push({ start: m.index, end: m.index + m[0].length, kind: "code" });
    last = m.index + m[0].length;
  }
  if (last < text.length) regions.push({ start: last, end: text.length, kind: "other" });

  // Pass 2: split "other" regions into kv / prose line runs.
  const out: Segment[] = [];
  for (const region of regions) {
    if (region.kind === "code") {
      out.push({ ...region, kind: "code", text: text.slice(region.start, region.end) });
      continue;
    }
    let runStart = region.start;
    let runKind: SegmentKind | undefined;
    let cursor = region.start;
    while (cursor < region.end) {
      const nl = text.indexOf("\n", cursor);
      const lineEnd = nl === -1 || nl >= region.end ? region.end : nl + 1;
      const line = text.slice(cursor, lineEnd);
      const kind: SegmentKind = KV_LINE.test(line) ? "kv" : "prose";
      if (runKind === undefined) {
        runKind = kind;
      } else if (kind !== runKind) {
        out.push({ start: runStart, end: cursor, kind: runKind, text: text.slice(runStart, cursor) });
        runStart = cursor;
        runKind = kind;
      }
      cursor = lineEnd;
    }
    if (runKind !== undefined && runStart < region.end) {
      out.push({ start: runStart, end: region.end, kind: runKind, text: text.slice(runStart, region.end) });
    }
  }
  return out;
}
