import maplibregl from "maplibre-gl";
import { useEffect, useMemo, useRef, useState } from "react";

export type GridData = {
  lons: number[];
  lats: number[];
  values: (number | null)[][];
};

const SRC = "hydro-grid";
const LAYER = "hydro-grid-layer";

function clamp01(t: number) {
  return Math.max(0, Math.min(1, t));
}

export function sampleColor(t: number): [number, number, number, number] {
  const x = clamp01(t);
  const r = Math.floor(40 + 180 * x);
  const g = Math.floor(20 + 200 * Math.sqrt(x));
  const b = Math.floor(180 * (1 - x));
  return [r, g, b, 215];
}

function gradientCss(): string {
  return [0, 0.25, 0.5, 0.75, 1]
    .map((t) => {
      const [r, g, b] = sampleColor(t);
      return `rgb(${r},${g},${b}) ${t * 100}%`;
    })
    .join(", ");
}

function buildCanvas(
  data: GridData,
  scale?: { vmin: number; vmax: number },
): { url: string; coords: [[number, number], [number, number], [number, number], [number, number]] } {
  let rows = data.values;
  const lats = data.lats;
  const lons = data.lons;
  if (lats.length >= 2 && lats[0]! < lats[lats.length - 1]!) {
    rows = [...rows].reverse();
  }
  const H = rows.length;
  const W = rows[0]?.length ?? 0;
  let vmin = scale?.vmin ?? Infinity;
  let vmax = scale?.vmax ?? -Infinity;
  if (scale == null) {
    for (const row of rows) {
      for (const v of row) {
        if (v == null || !Number.isFinite(v)) continue;
        vmin = Math.min(vmin, v);
        vmax = Math.max(vmax, v);
      }
    }
  }
  if (!Number.isFinite(vmin) || !Number.isFinite(vmax) || vmin === vmax) {
    vmin = scale?.vmin ?? 0;
    vmax = scale?.vmax ?? 1;
    if (vmin === vmax) vmax = vmin + 1;
  }
  const canvas = document.createElement("canvas");
  canvas.width = W;
  canvas.height = H;
  const ctx = canvas.getContext("2d")!;
  const img = ctx.createImageData(W, H);
  let p = 0;
  for (let y = 0; y < H; y++) {
    for (let x = 0; x < W; x++) {
      const v = rows[y]![x];
      const t = v == null || !Number.isFinite(v) ? 0 : (v - vmin) / (vmax - vmin);
      const [r, g, b, a] = sampleColor(t);
      img.data[p++] = r;
      img.data[p++] = g;
      img.data[p++] = b;
      img.data[p++] = v == null || !Number.isFinite(v) ? 0 : a;
    }
  }
  ctx.putImageData(img, 0, 0);
  const lonMin = Math.min(...lons);
  const lonMax = Math.max(...lons);
  const latMin = Math.min(...lats);
  const latMax = Math.max(...lats);
  const coords: [[number, number], [number, number], [number, number], [number, number]] = [
    [lonMin, latMax],
    [lonMax, latMax],
    [lonMax, latMin],
    [lonMin, latMin],
  ];
  return { url: canvas.toDataURL("image/png"), coords };
}

function formatScaleValue(v: number, unit: string): string {
  const u = unit.toLowerCase();
  if (u === "°c") return v.toFixed(1);
  if (u === "psu") return v.toFixed(2);
  if (u === "m/s" || u === "σ") return v.toFixed(3);
  return v.toPrecision(3);
}

type Props = {
  data: GridData | null;
  insufficient?: boolean;
  mapHeight?: number;
  vmin?: number;
  vmax?: number;
  unit?: string;
  kind?: string;
};

export function HydroHeatmapMap({ data, insufficient, mapHeight = 480, vmin, vmax, unit = "", kind }: Props) {
  const wrap = useRef<HTMLDivElement>(null);
  const [mapInstance, setMapInstance] = useState<maplibregl.Map | null>(null);

  const scale = useMemo(() => {
    if (vmin == null || vmax == null || !Number.isFinite(vmin) || !Number.isFinite(vmax)) return undefined;
    return { vmin, vmax };
  }, [vmin, vmax]);

  const tickLabels = useMemo(() => {
    if (!scale) return null;
    const n = 5;
    const labels: string[] = [];
    for (let i = 0; i < n; i++) {
      const v = scale.vmin + ((scale.vmax - scale.vmin) * i) / (n - 1);
      labels.push(formatScaleValue(v, unit));
    }
    return labels;
  }, [scale, unit]);

  const legendTitle = useMemo(() => {
    if (kind === "abs_err") return `|误差| (${unit})`;
    if (kind === "gt") return `真值 (${unit})`;
    if (kind === "pred") return `预报 (${unit})`;
    return unit || "值";
  }, [kind, unit]);

  const gradCss = useMemo(() => gradientCss(), []);

  useEffect(() => {
    if (!wrap.current) return;
    const map = new maplibregl.Map({
      container: wrap.current,
      style: "https://demotiles.maplibre.org/style.json",
      center: [125, 20],
      zoom: 4,
    });
    map.addControl(new maplibregl.NavigationControl(), "top-right");
    const onLoad = () => setMapInstance(map);
    if (map.loaded()) onLoad();
    else map.once("load", onLoad);
    return () => {
      map.remove();
      setMapInstance(null);
    };
  }, []);

  useEffect(() => {
    if (!mapInstance) return;
    if (mapInstance.getLayer(LAYER)) mapInstance.removeLayer(LAYER);
    if (mapInstance.getSource(SRC)) mapInstance.removeSource(SRC);
    if (!data) return;
    const { url, coords } = buildCanvas(data, scale);
    mapInstance.addSource(SRC, { type: "image", url, coordinates: coords });
    mapInstance.addLayer({
      id: LAYER,
      type: "raster",
      source: SRC,
      paint: { "raster-opacity": 0.82, "raster-fade-duration": 0 },
    });
    const lonMin = Math.min(...data.lons);
    const lonMax = Math.max(...data.lons);
    const latMin = Math.min(...data.lats);
    const latMax = Math.max(...data.lats);
    mapInstance.fitBounds(
      [
        [lonMin, latMin],
        [lonMax, latMax],
      ],
      { padding: 48, maxZoom: 10, duration: 600 },
    );
  }, [mapInstance, data, scale]);

  return (
    <div
      style={{
        position: "relative",
        width: "100%",
        height: mapHeight,
        borderRadius: 8,
        overflow: "hidden",
      }}
    >
      <div ref={wrap} style={{ width: "100%", height: "100%" }} />
      {scale && tickLabels && (
        <div
          style={{
            position: "absolute",
            right: 10,
            top: 40,
            bottom: 16,
            width: 56,
            display: "flex",
            flexDirection: "column",
            alignItems: "center",
            pointerEvents: "none",
            zIndex: 2,
          }}
        >
          <div
            style={{
              fontSize: 10,
              fontWeight: 600,
              color: "#0f172a",
              textShadow: "0 0 4px #fff",
              marginBottom: 4,
              textAlign: "center",
              lineHeight: 1.2,
            }}
          >
            {legendTitle}
          </div>
          <div style={{ display: "flex", flex: 1, minHeight: 0, width: "100%", gap: 4 }}>
            <div
              style={{
                display: "flex",
                flexDirection: "column",
                justifyContent: "space-between",
                fontSize: 9,
                color: "#0f172a",
                textShadow: "0 0 4px #fff",
                flex: 1,
                textAlign: "right",
                paddingRight: 2,
              }}
            >
              {[...tickLabels].reverse().map((lab, i) => (
                <span key={i}>{lab}</span>
              ))}
            </div>
            <div
              style={{
                width: 14,
                flex: "0 0 14px",
                borderRadius: 3,
                border: "1px solid rgba(15,23,42,0.35)",
                background: `linear-gradient(to top, ${gradCss})`,
              }}
            />
          </div>
        </div>
      )}
      {insufficient && (
        <div
          style={{
            position: "absolute",
            inset: 0,
            background: "rgba(15,23,42,0.45)",
            color: "#fff",
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            fontSize: 16,
            pointerEvents: "none",
          }}
        >
          数据不足，等待中（时间步未达 T_need）
        </div>
      )}
    </div>
  );
}
