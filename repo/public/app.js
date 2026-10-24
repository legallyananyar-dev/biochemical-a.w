const $ = (id) => document.getElementById(id);
const LANES = [["pitch", "Pitch", "from your voice"], ["volume", "Volume", "from your voice"],
               ["bond_dissonance", "Bond dissonance", "from the molecule"], ["chirality", "Chirality", "from the molecule"]];
let cfg, mode = "protect", file = null, objectUrl = null;
let plotData = { expected: null, recovered: null };

const css = (n) => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
const say = (t, err) => { $("status").textContent = t; $("status").className = "status" + (err ? " err" : ""); };

async function init() {
  cfg = await (await fetch("/api/config")).json();
  $("maxMb").textContent = cfg.max_upload_mb;
  $("molecule").innerHTML = cfg.molecules.map(m => `<option value="${m.id}">${m.name}</option>`).join("") +
    `<option value="__smiles">Custom SMILES…</option>`;
  setMode("protect"); drawKey(); draw();
}

function setMode(m) {
  mode = m;
  for (const b of document.querySelectorAll("[role=tab]")) b.setAttribute("aria-selected", b.dataset.mode === m);
  const sel = $("molecule"), auto = sel.querySelector("[value=auto]");
  if (m === "verify" && !auto) sel.insertAdjacentHTML("afterbegin", `<option value="auto">Find the molecule automatically</option>`);
  if (m === "protect" && auto) auto.remove();
  if (m === "verify") sel.value = "auto"; else if (sel.value === "auto") sel.value = cfg.molecules[0].id;
  $("molLabel").textContent = m === "protect" ? "Molecule to lock into the voice" : "Molecule to check against";
  $("run").textContent = m === "protect" ? "Protect recording" : "Check recording";
  $("verdict").hidden = $("tracks").hidden = $("out").hidden = true; say("");
  plotData = { expected: null, recovered: null }; drawKey(); draw();
}

const molecule = () => ($("molecule").value === "__smiles" ? $("smiles").value.trim() : $("molecule").value);

async function drawKey() {  // preview the chemical half of the key as soon as a molecule is chosen
  const m = molecule();
  if (!m || m === "auto" || mode === "verify" && plotData.recovered) return;
  try {
    const r = await fetch("/api/key?molecule=" + encodeURIComponent(m));
    if (!r.ok) { plotData.expected = null; return draw(); }
    const k = await r.json();
    plotData.expected = k.dissonance.map((d, i) => [null, null, d, Math.abs(k.chirality[i])]);
  } catch { plotData.expected = null; }
  draw();
}

function setFile(f) {
  if (!f) return;
  if (f.size > cfg.max_upload_mb * 1024 * 1024) { say(`That file is ${(f.size / 1048576).toFixed(1)} MB. The limit is ${cfg.max_upload_mb} MB.`, true); return; }
  file = f; $("drop").classList.add("has"); $("dropText").textContent = f.name; say("");
}

async function run() {
  if (!file) return say("Choose a recording first.", true);
  const m = molecule();
  if (!m) return say("Enter a SMILES string or pick a molecule.", true);
  const fd = new FormData(); fd.append("audio", file); fd.append("molecule", m);
  $("run").disabled = true; say(mode === "protect" ? "Embedding the signature…" : "Reading the carriers…");
  try {
    const res = await fetch("/api/" + mode, { method: "POST", body: fd });
    if (!res.ok) {
      let msg = `Server returned ${res.status}.`;
      try { msg = (await res.json()).msg || msg; } catch {}
      if (res.status === 413) msg = `That file is too large for the hosted demo (limit ${cfg.max_upload_mb} MB).`;
      return say(msg, true);
    }
    if (mode === "protect") {
      const r = JSON.parse(res.headers.get("X-BAW-Result"));
      if (objectUrl) URL.revokeObjectURL(objectUrl);
      objectUrl = URL.createObjectURL(await res.blob());
      $("player").src = objectUrl; $("download").href = objectUrl;
      $("download").download = (res.headers.get("Content-Disposition") || "").match(/filename="([^"]+)"/)?.[1] || "protected.wav";
      $("out").hidden = false;
      show(r, r.score >= cfg.threshold ? "ok" : "bad", "Signature locked in",
        `The file was read back after saving and ${Math.round(r.score * 100)}% of the signature checked out. ${r.seconds}s at ${r.sample_rate / 1000} kHz.`);
      say("Done. Listen to the result, then download it.");
    } else {
      const r = await res.json(), mol = (cfg.molecules.find(x => x.id === r.molecule) || {}).name || r.molecule;
      const text = {
        intact: `Best match: ${mol}. The carriers are present and line up with this voice.`,
        damaged: `Some carrier energy is left (closest to ${mol}), but the pattern no longer lines up. The file has probably been filtered, compressed or edited.`,
        absent: "No watermark carriers found in this file. It was never protected, or the high band was removed.",
      }[r.verdict];
      show(r, r.verdict === "intact" ? "ok" : "bad", { intact: "Watermark intact", damaged: "Watermark damaged", absent: "No watermark found" }[r.verdict], text);
      say("");
    }
  } catch (e) { say("Could not reach the server. Check your connection and try again.", true); }
  finally { $("run").disabled = false; }
}

function show(r, kind, title, text) {
  $("verdict").hidden = false; $("verdict").className = "verdict" + (kind === "ok" ? "" : " bad");
  $("pct").textContent = Math.round(r.score * 100); $("vTitle").textContent = title; $("vText").textContent = text;
  $("tracks").hidden = false;
  $("tracks").innerHTML = LANES.map(([k, name]) => {
    const v = r.tracks[k];
    return `<li><b>${name}</b>${v === null ? "No variation to match" : Math.round(v * 100) + "% match"}<div class="bar"><span style="width:${(v || 0) * 100}%"></span></div></li>`;
  }).join("");
  plotData = { expected: r.expected, recovered: r.recovered }; draw();
}

function draw() {
  const cv = $("plot"), dpr = window.devicePixelRatio || 1, w = cv.clientWidth, h = cv.clientHeight;
  cv.width = w * dpr; cv.height = h * dpr;
  const g = cv.getContext("2d"); g.scale(dpr, dpr); g.clearRect(0, 0, w, h);
  const padL = 112, laneH = h / 4, T = (plotData.expected || plotData.recovered || [[]]).length || cfg.steps;
  const ink = css("--ink"), mute = css("--mute"), line = css("--line"), uv = css("--uv"), soft = css("--uv-soft");
  g.font = `600 13px ${css("--sans")}`; g.textBaseline = "middle";
  LANES.forEach(([, name, src], i) => {
    const y0 = i * laneH + 8, hh = laneH - 16;
    g.fillStyle = ink; g.textAlign = "left"; g.fillText(name, 6, y0 + hh / 2 - 8);
    g.fillStyle = mute; g.font = `400 11px ${css("--sans")}`; g.fillText(src, 6, y0 + hh / 2 + 9); g.font = `600 13px ${css("--sans")}`;
    g.strokeStyle = line; g.beginPath(); g.moveTo(padL, y0 + hh + 4); g.lineTo(w, y0 + hh + 4); g.stroke();
    const bw = (w - padL) / T, ex = plotData.expected, rc = plotData.recovered;
    if (ex) for (let t = 0; t < T; t++) {
      const v = ex[t][i]; if (v === null || v === undefined) continue;
      g.fillStyle = soft; g.strokeStyle = uv; g.lineWidth = 1;
      g.fillRect(padL + t * bw + 1, y0 + hh * (1 - v), Math.max(1, bw - 2), hh * v);
      g.fillStyle = uv; g.fillRect(padL + t * bw + 1, y0 + hh * (1 - v), Math.max(1, bw - 2), 2);
    }
    if (rc) { g.strokeStyle = ink; g.lineWidth = 2; g.beginPath();
      for (let t = 0; t < T; t++) { const x = padL + (t + .5) * bw, y = y0 + hh * (1 - rc[t][i]); t ? g.lineTo(x, y) : g.moveTo(x, y); }
      g.stroke(); }
  });
  if (!plotData.expected && !plotData.recovered) { g.fillStyle = mute; g.textAlign = "center"; g.font = `400 14px ${css("--sans")}`;
    g.fillText("Pick a molecule to see its signature. Run a recording to see the full key.", padL + (w - padL) / 2, h - 6); }
}

document.querySelectorAll("[role=tab]").forEach(b => b.addEventListener("click", () => setMode(b.dataset.mode)));
$("molecule").addEventListener("change", () => { $("smiles").hidden = $("molecule").value !== "__smiles"; if (!$("smiles").hidden) $("smiles").focus(); plotData.recovered = null; drawKey(); });
$("smiles").addEventListener("change", () => { plotData.recovered = null; drawKey(); });
$("audio").addEventListener("change", e => setFile(e.target.files[0]));
const drop = $("drop");
["dragenter", "dragover"].forEach(ev => drop.addEventListener(ev, e => { e.preventDefault(); drop.classList.add("over"); }));
["dragleave", "drop"].forEach(ev => drop.addEventListener(ev, e => { e.preventDefault(); drop.classList.remove("over"); }));
drop.addEventListener("drop", e => setFile(e.dataTransfer.files[0]));
$("run").addEventListener("click", run);
addEventListener("resize", draw);
matchMedia("(prefers-color-scheme: dark)").addEventListener("change", draw);
init();
