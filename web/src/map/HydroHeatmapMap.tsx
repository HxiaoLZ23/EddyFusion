import maplibregl from "maplibre-gl";
import { useEffect, useRef, useState } from "react";

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

function sampleColor(t: number): [number, number, number, number] {
  const x = clamp01(t);
  const r = Math.floor(40 + 180 * x);
  const g = Math.floor(20 + 200 * Math.sqrt(x));
  const b = Math.floor(180 * (1 - x));
  return [r, g, b, 215];
}

function buildCanvas(data: GridData): { url: string; coords: [[number, number], [number, number], [number, number], [number, number]] } {
  let rows = data.values;
  const lats = data.lats;
  const lons = data.lons;
  if (lats.length >= 2 && lats[0]! < lats[lats.length - 1]!) {
    rows = [...rows].reverse();
  }
  const H = rows.length;
  const W = rows[0]?.length ?? 0;
  let vmin = Infinity;
  let vmax = -Infinity;
  for (const row of rows) {
    for (const v of row) {
      if (v == null || !Number.isFinite(v)) continue;
      vmin = Math.min(vmin, v);
      vmax = Math.max(vmax, v);
    }
  }
  if (!Number.isFinite(vmin) || !Number.isFinite(vmax) || vmin === vmax) {
    vmin = 0;
    vmax = 1;
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

type Props = {
  data: GridData | null;
  insufficient?: boolean;
  /** 嵌入同屏时略减小高度，避免撑破视口 */
  mapHeight?: number;
};

export function HydroHeatmapMap({ data, insufficient, mapHeight = 480 }: Props) {
  const wrap = useRef<HTMLDivElement>(null);
  const [mapInstance, setMapInstance] = useState<maplibregl.Map | null>(null);

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
    const { url, coords } = buildCanvas(data);
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
  }, [mapInstance, data]);

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
