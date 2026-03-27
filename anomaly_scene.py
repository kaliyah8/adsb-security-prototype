from __future__ import annotations

import html
import json


def build_anomaly_scene_html(scenario: str = "normal") -> str:
    """
    Returns a self-contained HTML animation for one ADS-B attack scenario.

    Supported scenarios:
      - normal
      - teleportation
      - gps_spoofing
      - ghost_aircraft
    """

    scenario = (scenario or "normal").strip().lower()

    if scenario not in {
        "normal",
        "teleportation",
        "gps_spoofing",
        "ghost_aircraft",
    }:
        scenario = "normal"

    title_map = {
        "normal": "Normal Flight Path",
        "teleportation": "Teleportation Attack",
        "gps_spoofing": "GPS Spoofing Attack",
        "ghost_aircraft": "Ghost Aircraft Attack",
    }

    subtitle_map = {
        "normal": "Aircraft follows its expected route with smooth motion.",
        "teleportation": "Aircraft suddenly jumps to an impossible position.",
        "gps_spoofing": "Reported position drifts away from the true route.",
        "ghost_aircraft": "A false duplicate aircraft appears on the airspace view.",
    }

    config = {
        "scenario": scenario,
        "title": title_map[scenario],
        "subtitle": subtitle_map[scenario],
    }

    # Path is drawn in SVG coordinates.
    # This shape is intentionally smooth so the aircraft follows the line correctly.
    flight_path = "M 90 340 C 180 300, 260 235, 340 220 S 500 170, 640 115"

    html_doc = f"""
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8" />
<meta name="viewport" content="width=device-width, initial-scale=1.0" />
<title>Attack Visualization</title>
<style>
  :root {{
    --bg-0: #07111f;
    --bg-1: #0c1a2d;
    --panel: rgba(10, 20, 35, 0.76);
    --line: rgba(118, 169, 255, 0.30);
    --line-strong: rgba(118, 169, 255, 0.95);
    --good: #36d399;
    --warn: #fbbf24;
    --bad: #ff5d73;
    --text: #e9f1ff;
    --muted: #9eb1c8;
    --grid: rgba(255,255,255,0.06);
  }}

  * {{
    box-sizing: border-box;
  }}

  html, body {{
    margin: 0;
    padding: 0;
    width: 100%;
    height: 100%;
    overflow: hidden;
    font-family: Inter, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    background:
      radial-gradient(circle at 20% 20%, rgba(56, 189, 248, 0.08), transparent 28%),
      radial-gradient(circle at 80% 30%, rgba(99, 102, 241, 0.10), transparent 30%),
      linear-gradient(180deg, var(--bg-0), var(--bg-1));
    color: var(--text);
  }}

  .wrap {{
    width: 100%;
    height: 100%;
    padding: 16px;
  }}

  .panel {{
    position: relative;
    width: 100%;
    height: 100%;
    min-height: 500px;
    border-radius: 22px;
    background:
      linear-gradient(180deg, rgba(255,255,255,0.03), rgba(255,255,255,0.015)),
      var(--panel);
    border: 1px solid rgba(255,255,255,0.08);
    box-shadow:
      0 18px 40px rgba(0,0,0,0.35),
      inset 0 1px 0 rgba(255,255,255,0.05);
    overflow: hidden;
  }}

  .hud {{
    position: absolute;
    top: 18px;
    left: 18px;
    z-index: 5;
    padding: 14px 16px;
    border-radius: 16px;
    background: rgba(5, 10, 20, 0.45);
    border: 1px solid rgba(255,255,255,0.08);
    backdrop-filter: blur(10px);
    max-width: 360px;
  }}

  .eyebrow {{
    display: inline-flex;
    align-items: center;
    gap: 8px;
    font-size: 11px;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: var(--muted);
    margin-bottom: 8px;
  }}

  .pulse {{
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: var(--good);
    box-shadow: 0 0 10px var(--good);
    animation: pulse 1.4s infinite ease-in-out;
  }}

  .title {{
    font-size: 24px;
    font-weight: 700;
    margin: 0 0 6px 0;
    line-height: 1.15;
  }}

  .subtitle {{
    margin: 0;
    font-size: 13px;
    color: var(--muted);
    line-height: 1.45;
  }}

  .legend {{
    position: absolute;
    right: 18px;
    top: 18px;
    z-index: 5;
    display: flex;
    gap: 10px;
    flex-wrap: wrap;
    justify-content: flex-end;
    max-width: 45%;
  }}

  .chip {{
    padding: 8px 12px;
    border-radius: 999px;
    font-size: 12px;
    font-weight: 600;
    background: rgba(6, 12, 22, 0.54);
    border: 1px solid rgba(255,255,255,0.08);
    color: var(--text);
    backdrop-filter: blur(8px);
  }}

  .chip.good {{ color: var(--good); }}
  .chip.warn {{ color: var(--warn); }}
  .chip.bad {{ color: var(--bad); }}

  .scene {{
    position: absolute;
    inset: 0;
  }}

  svg {{
    width: 100%;
    height: 100%;
    display: block;
  }}

  .grid line {{
    stroke: var(--grid);
    stroke-width: 1;
  }}

  .route-base {{
    fill: none;
    stroke: var(--line);
    stroke-width: 5;
    stroke-linecap: round;
    stroke-linejoin: round;
  }}

  .route-glow {{
    fill: none;
    stroke: rgba(118, 169, 255, 0.18);
    stroke-width: 14;
    stroke-linecap: round;
    stroke-linejoin: round;
    filter: blur(3px);
  }}

  .route-dash {{
    fill: none;
    stroke: var(--line-strong);
    stroke-width: 2;
    stroke-linecap: round;
    stroke-dasharray: 5 10;
    opacity: 0.85;
  }}

  .spoof-route {{
    fill: none;
    stroke: rgba(251, 191, 36, 0.9);
    stroke-width: 3;
    stroke-linecap: round;
    stroke-dasharray: 6 10;
    opacity: 0;
  }}

  .teleport-line {{
    fill: none;
    stroke: rgba(255, 93, 115, 0.9);
    stroke-width: 2.5;
    stroke-dasharray: 4 8;
    opacity: 0;
  }}

  .plane-group {{
    filter: drop-shadow(0 0 10px rgba(84, 163, 255, 0.35));
  }}

  .ghost-group {{
    opacity: 0;
    filter: drop-shadow(0 0 12px rgba(255,255,255,0.18));
  }}

  .plane-body {{
    fill: #f8fbff;
    stroke: rgba(255,255,255,0.28);
    stroke-width: 1;
  }}

  .plane-core {{
    fill: #7dd3fc;
    opacity: 0.95;
  }}

  .ghost-body {{
    fill: rgba(255,255,255,0.26);
    stroke: rgba(255,255,255,0.22);
    stroke-width: 1;
  }}

  .trail {{
    fill: none;
    stroke: rgba(54, 211, 153, 0.95);
    stroke-width: 3;
    stroke-linecap: round;
    stroke-linejoin: round;
    filter: drop-shadow(0 0 5px rgba(54, 211, 153, 0.45));
  }}

  .trail.warn {{
    stroke: rgba(251, 191, 36, 0.95);
    filter: drop-shadow(0 0 5px rgba(251, 191, 36, 0.45));
  }}

  .trail.bad {{
    stroke: rgba(255, 93, 115, 0.95);
    filter: drop-shadow(0 0 5px rgba(255, 93, 115, 0.45));
  }}

  .scan-ring {{
    fill: none;
    stroke-width: 2;
    opacity: 0;
  }}

  .scan-ring.warn {{ stroke: rgba(251, 191, 36, 0.95); }}
  .scan-ring.bad {{ stroke: rgba(255, 93, 115, 0.95); }}

  .footer {{
    position: absolute;
    bottom: 16px;
    left: 18px;
    right: 18px;
    z-index: 5;
    display: flex;
    justify-content: space-between;
    align-items: center;
    gap: 16px;
    color: var(--muted);
    font-size: 12px;
  }}

  .footer-left, .footer-right {{
    display: flex;
    gap: 10px;
    align-items: center;
    flex-wrap: wrap;
  }}

  .footer-pill {{
    padding: 7px 10px;
    border-radius: 999px;
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08);
  }}

  @keyframes pulse {{
    0%, 100% {{ transform: scale(0.95); opacity: 0.8; }}
    50% {{ transform: scale(1.2); opacity: 1; }}
  }}
</style>
</head>
<body>
<div class="wrap">
  <div class="panel">
    <div class="hud">
      <div class="eyebrow"><span class="pulse"></span> Attack Visualization</div>
      <h2 class="title">{html.escape(config["title"])}</h2>
      <p class="subtitle">{html.escape(config["subtitle"])}</p>
    </div>

    <div class="legend">
      <div class="chip good">True path</div>
      <div class="chip warn">Reported anomaly</div>
      <div class="chip bad">Attack event</div>
    </div>

    <div class="scene">
      <svg viewBox="0 0 760 520" preserveAspectRatio="xMidYMid meet">
        <defs>
          <linearGradient id="bgFade" x1="0" x2="1" y1="0" y2="1">
            <stop offset="0%" stop-color="rgba(255,255,255,0.03)" />
            <stop offset="100%" stop-color="rgba(255,255,255,0.01)" />
          </linearGradient>
        </defs>

        <g class="grid">
          <line x1="0" y1="80" x2="760" y2="80"></line>
          <line x1="0" y1="160" x2="760" y2="160"></line>
          <line x1="0" y1="240" x2="760" y2="240"></line>
          <line x1="0" y1="320" x2="760" y2="320"></line>
          <line x1="0" y1="400" x2="760" y2="400"></line>
          <line x1="100" y1="0" x2="100" y2="520"></line>
          <line x1="220" y1="0" x2="220" y2="520"></line>
          <line x1="340" y1="0" x2="340" y2="520"></line>
          <line x1="460" y1="0" x2="460" y2="520"></line>
          <line x1="580" y1="0" x2="580" y2="520"></line>
          <line x1="700" y1="0" x2="700" y2="520"></line>
        </g>

        <path id="routeGlow" class="route-glow" d="{flight_path}"></path>
        <path id="routeBase" class="route-base" d="{flight_path}"></path>
        <path id="routeDash" class="route-dash" d="{flight_path}"></path>
        <path id="spoofRoute" class="spoof-route" d=""></path>

        <path id="trailPath" class="trail" d=""></path>
        <path id="ghostTrailPath" class="trail warn" d=""></path>
        <path id="teleportLine" class="teleport-line" d=""></path>

        <circle id="scanRing" class="scan-ring warn" cx="0" cy="0" r="10"></circle>

        <g id="planeGroup" class="plane-group">
          <g id="planeShape">
            <path class="plane-body" d="M 0 -14 L 9 6 L 3 6 L 3 16 L -3 16 L -3 6 L -9 6 Z"></path>
            <circle class="plane-core" cx="0" cy="2" r="3.4"></circle>
          </g>
        </g>

        <g id="ghostGroup" class="ghost-group">
          <g id="ghostShape">
            <path class="ghost-body" d="M 0 -14 L 9 6 L 3 6 L 3 16 L -3 16 L -3 6 L -9 6 Z"></path>
          </g>
        </g>
      </svg>
    </div>

    <div class="footer">
      <div class="footer-left">
        <div class="footer-pill">Loop: Continuous</div>
        <div class="footer-pill">Mode: {html.escape(config["title"])}</div>
      </div>
      <div class="footer-right">
        <div class="footer-pill">ADS-B anomaly demo</div>
      </div>
    </div>
  </div>
</div>

<script>
  const CONFIG = {json.dumps(config)};

  const route = document.getElementById("routeBase");
  const routeDash = document.getElementById("routeDash");
  const spoofRoute = document.getElementById("spoofRoute");
  const planeGroup = document.getElementById("planeGroup");
  const planeShape = document.getElementById("planeShape");
  const ghostGroup = document.getElementById("ghostGroup");
  const ghostShape = document.getElementById("ghostShape");
  const trailPath = document.getElementById("trailPath");
  const ghostTrailPath = document.getElementById("ghostTrailPath");
  const teleportLine = document.getElementById("teleportLine");
  const scanRing = document.getElementById("scanRing");

  const routeLength = route.getTotalLength();

  const POINTS = [];
  const SAMPLE_COUNT = 220;
  for (let i = 0; i <= SAMPLE_COUNT; i++) {{
    const pt = route.getPointAtLength((i / SAMPLE_COUNT) * routeLength);
    POINTS.push({{ x: pt.x, y: pt.y }});
  }}

  function lerp(a, b, t) {{
    return a + (b - a) * t;
  }}

  function clamp(v, min, max) {{
    return Math.max(min, Math.min(max, v));
  }}

  function easeInOut(t) {{
    return t < 0.5 ? 2 * t * t : 1 - Math.pow(-2 * t + 2, 2) / 2;
  }}

  function getPointByProgress(progress) {{
    const p = clamp(progress, 0, 1);
    const idx = p * (POINTS.length - 1);
    const i0 = Math.floor(idx);
    const i1 = Math.min(POINTS.length - 1, i0 + 1);
    const f = idx - i0;
    return {{
      x: lerp(POINTS[i0].x, POINTS[i1].x, f),
      y: lerp(POINTS[i0].y, POINTS[i1].y, f)
    }};
  }}

  function getAngleByProgress(progress) {{
    const p0 = getPointByProgress(clamp(progress - 0.003, 0, 1));
    const p1 = getPointByProgress(clamp(progress + 0.003, 0, 1));
    return Math.atan2(p1.y - p0.y, p1.x - p0.x) * 180 / Math.PI + 90;
  }}

  function pathFromProgress(progress) {{
    const count = Math.max(2, Math.floor(progress * SAMPLE_COUNT));
    const pts = POINTS.slice(0, count);
    if (!pts.length) return "";
    let d = `M ${{pts[0].x}} ${{pts[0].y}}`;
    for (let i = 1; i < pts.length; i++) {{
      d += ` L ${{pts[i].x}} ${{pts[i].y}}`;
    }}
    return d;
  }}

  function buildOffsetSpoofPath(progress, offsetMag) {{
    const count = Math.max(8, Math.floor(progress * SAMPLE_COUNT));
    const pts = [];
    for (let i = 0; i < count; i++) {{
      const base = POINTS[i];
      const t = i / Math.max(1, count - 1);
      const drift = Math.sin(t * Math.PI) * offsetMag;
      const wobble = Math.sin(t * 18) * 4;
      pts.push({{
        x: base.x + drift,
        y: base.y - drift * 0.42 + wobble
      }});
    }}
    if (!pts.length) return "";
    let d = `M ${{pts[0].x}} ${{pts[0].y}}`;
    for (let i = 1; i < pts.length; i++) {{
      d += ` L ${{pts[i].x}} ${{pts[i].y}}`;
    }}
    return d;
  }}

  function setPlaneTransform(group, x, y, angle, scale = 1) {{
    group.setAttribute(
      "transform",
      `translate(${{x}}, ${{y}}) rotate(${{angle}}) scale(${{scale}})`
    );
  }}

  function setRing(x, y, r, alpha, bad = false) {{
    scanRing.setAttribute("cx", x);
    scanRing.setAttribute("cy", y);
    scanRing.setAttribute("r", r);
    scanRing.style.opacity = alpha;
    scanRing.classList.remove("warn", "bad");
    scanRing.classList.add(bad ? "bad" : "warn");
  }}

  function resetDecorations() {{
    spoofRoute.style.opacity = 0;
    teleportLine.style.opacity = 0;
    ghostGroup.style.opacity = 0;
    scanRing.style.opacity = 0;
    ghostTrailPath.setAttribute("d", "");
    trailPath.classList.remove("warn", "bad");
  }}

  const LOOP_MS = 7000;
  let startTs = null;

  function animate(ts) {{
    if (startTs === null) startTs = ts;
    const elapsed = (ts - startTs) % LOOP_MS;
    const raw = elapsed / LOOP_MS;
    const p = easeInOut(raw);

    resetDecorations();

    if (CONFIG.scenario === "normal") {{
      const pos = getPointByProgress(p);
      const angle = getAngleByProgress(p);
      setPlaneTransform(planeGroup, pos.x, pos.y, angle, 1.22);
      trailPath.setAttribute("d", pathFromProgress(p));
      trailPath.setAttribute("class", "trail");
    }}

    if (CONFIG.scenario === "teleportation") {{
      const jumpAt = 0.56;
      const beforeP = clamp(p / jumpAt, 0, 1);
      const preJumpPos = getPointByProgress(beforeP * 0.44);
      const postJumpPos = {{
        x: preJumpPos.x + 155,
        y: preJumpPos.y - 95
      }};

      if (p < jumpAt) {{
        const angle = getAngleByProgress(beforeP * 0.44);
        setPlaneTransform(planeGroup, preJumpPos.x, preJumpPos.y, angle, 1.22);
        trailPath.setAttribute("d", pathFromProgress(beforeP * 0.44));
        trailPath.setAttribute("class", "trail");
      }} else {{
        const afterPhase = (p - jumpAt) / (1 - jumpAt);
        const resumedP = 0.70 + afterPhase * 0.30;
        const resumedPos = getPointByProgress(resumedP);
        const angle = getAngleByProgress(resumedP);

        teleportLine.setAttribute(
          "d",
          `M ${{preJumpPos.x}} ${{preJumpPos.y}} L ${{postJumpPos.x}} ${{postJumpPos.y}}`
        );
        teleportLine.style.opacity = 1;

        setPlaneTransform(planeGroup, resumedPos.x, resumedPos.y, angle, 1.22);
        trailPath.setAttribute("class", "trail bad");
        trailPath.setAttribute(
          "d",
          `M ${{POINTS[0].x}} ${{POINTS[0].y}} ` +
          POINTS.slice(1, Math.floor(0.44 * SAMPLE_COUNT))
            .map(pt => `L ${{pt.x}} ${{pt.y}}`).join(" ") +
          ` M ${{postJumpPos.x}} ${{postJumpPos.y}} L ${{resumedPos.x}} ${{resumedPos.y}}`
        );

        const pulse = 16 + 28 * Math.abs(Math.sin(afterPhase * Math.PI * 3));
        setRing(postJumpPos.x, postJumpPos.y, pulse, 0.95 - afterPhase * 0.5, true);
      }}
    }}

    if (CONFIG.scenario === "gps_spoofing") {{
      const truePos = getPointByProgress(p);
      const angle = getAngleByProgress(p);
      const drift = Math.sin(p * Math.PI) * 120;
      const wobble = Math.sin(p * 16) * 6;
      const spoofPos = {{
        x: truePos.x + drift,
        y: truePos.y - drift * 0.38 + wobble
      }};

      spoofRoute.setAttribute("d", buildOffsetSpoofPath(p, 120));
      spoofRoute.style.opacity = 1;

      setPlaneTransform(planeGroup, spoofPos.x, spoofPos.y, angle, 1.22);
      trailPath.setAttribute("d", buildOffsetSpoofPath(p, 120));
      trailPath.setAttribute("class", "trail warn");

      const ringR = 14 + Math.abs(Math.sin(p * Math.PI * 10)) * 18;
      setRing(spoofPos.x, spoofPos.y, ringR, 0.75, false);
    }}

    if (CONFIG.scenario === "ghost_aircraft") {{
      const pos = getPointByProgress(p);
      const angle = getAngleByProgress(p);
      setPlaneTransform(planeGroup, pos.x, pos.y, angle, 1.22);
      trailPath.setAttribute("d", pathFromProgress(p));
      trailPath.setAttribute("class", "trail");

      const ghostVisible = p > 0.28;
      if (ghostVisible) {{
        const ghostP = clamp(p - 0.10, 0, 1);
        const gBase = getPointByProgress(ghostP);
        const ghostPos = {{
          x: gBase.x + 64,
          y: gBase.y - 44
        }};
        const gAngle = getAngleByProgress(ghostP) + 6;

        ghostGroup.style.opacity = 0.78;
        setPlaneTransform(ghostGroup, ghostPos.x, ghostPos.y, gAngle, 1.12);
        ghostTrailPath.setAttribute(
          "d",
          `M ${{ghostPos.x - 5}} ${{ghostPos.y + 4}} L ${{ghostPos.x}} ${{ghostPos.y}}`
        );

        const ringR = 10 + Math.abs(Math.sin(p * Math.PI * 12)) * 14;
        setRing(ghostPos.x, ghostPos.y, ringR, 0.65, false);
      }}
    }}

    requestAnimationFrame(animate);
  }}

  routeDash.style.strokeDashoffset = "0";
  requestAnimationFrame(animate);
</script>
</body>
</html>
"""
    return html_doc