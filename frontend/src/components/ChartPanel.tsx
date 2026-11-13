import type { ReactNode, RefObject } from "react";
import { colors, headingStyle, panelStyle } from "../theme";

interface Props {
  title: ReactNode;
  toolbar?: ReactNode;
  status?: string;
  header?: ReactNode;
  legend?: ReactNode;
  containerRef: RefObject<HTMLDivElement | null>;
  height: number;
}

/** Panel chrome shared by every chart: title, toolbar, status line, canvas, legend. */
export function ChartPanel({ title, toolbar, status, header, legend, containerRef, height }: Props) {
  return (
    <section style={panelStyle}>
      <div
        style={{
          display: "flex",
          justifyContent: "space-between",
          alignItems: "center",
          marginBottom: "0.75rem",
          flexWrap: "wrap",
          gap: "0.5rem",
        }}
      >
        <h2 style={headingStyle}>{title}</h2>
        {toolbar}
      </div>
      {status && <div style={{ color: colors.warn, marginBottom: "0.5rem" }}>{status}</div>}
      {header}
      <div ref={containerRef} style={{ width: "100%", height, borderRadius: "8px", overflow: "hidden" }} />
      {legend && (
        <div style={{ color: colors.textMuted, fontSize: "0.75rem", marginTop: "0.5rem" }}>
          {legend}
        </div>
      )}
    </section>
  );
}
