(() => {
  'use strict';

  const state = {
    project: null,
    scenarioIndex: 0,
    selectedActorIds: new Set(),
    activeRouteId: null,
    mode: 'actor', // actor | waypoint
    selectedWaypointIndices: new Set(),
    view: {
      centerX: 0,
      centerY: 0,
      scale: 4.0, // px / meter
      minScale: 0.2,
      maxScale: 240.0,
    },
    mapIndex: null,
    history: [],
    historyIndex: -1,
    interaction: null,
    keyState: {
      space: false,
    },
  };

  const els = {};

  function byId(id) {
    return document.getElementById(id);
  }

  function clamp(v, lo, hi) {
    return Math.max(lo, Math.min(hi, v));
  }

  function toFloat(value, fallback = 0.0) {
    const parsed = Number.parseFloat(value);
    return Number.isFinite(parsed) ? parsed : fallback;
  }

  function normalizeAngleDeg(v) {
    let a = Number(v) || 0;
    while (a > 180) a -= 360;
    while (a <= -180) a += 360;
    return a;
  }

  function angleDiffDeg(a, b) {
    return Math.abs(normalizeAngleDeg((a || 0) - (b || 0)));
  }

  function interpAngleDeg(a, b, t) {
    const aa = normalizeAngleDeg(a || 0);
    const bb = normalizeAngleDeg(b || 0);
    const delta = normalizeAngleDeg(bb - aa);
    return normalizeAngleDeg(aa + delta * t);
  }

  function hashColor(seedText) {
    let h = 0;
    const text = String(seedText || 'route');
    for (let i = 0; i < text.length; i += 1) {
      h = (h * 31 + text.charCodeAt(i)) >>> 0;
    }
    const hue = h % 360;
    return `hsl(${hue} 74% 62%)`;
  }

  function roleClass(role) {
    const r = String(role || 'npc').toLowerCase();
    if (r.includes('ego')) return 'role-ego';
    if (r.includes('walker')) return 'role-walker';
    if (r.includes('pedestrian')) return 'role-pedestrian';
    if (r.includes('bicycle')) return 'role-bicycle';
    if (r.includes('cyclist')) return 'role-cyclist';
    if (r.includes('static')) return 'role-static';
    return 'role-npc';
  }

  function currentScenario() {
    if (!state.project || !Array.isArray(state.project.scenarios) || state.project.scenarios.length === 0) {
      return null;
    }
    return state.project.scenarios[clamp(state.scenarioIndex, 0, state.project.scenarios.length - 1)];
  }

  function currentRoutes() {
    const scenario = currentScenario();
    return scenario && Array.isArray(scenario.routes) ? scenario.routes : [];
  }

  function routeId(route) {
    if (!route || typeof route !== 'object') {
      return 'route';
    }
    return String(route.uid || route.relpath || route.name || 'route');
  }

  function routeById(id) {
    if (!id) return null;
    const routes = currentRoutes();
    for (const route of routes) {
      if (routeId(route) === String(id)) {
        return route;
      }
    }
    return null;
  }

  function getCanvasCssSize() {
    const rect = els.mapCanvas.getBoundingClientRect();
    return { width: rect.width, height: rect.height };
  }

  function resizeCanvas() {
    const dpr = window.devicePixelRatio || 1;
    const { width, height } = getCanvasCssSize();
    const wantW = Math.max(2, Math.round(width * dpr));
    const wantH = Math.max(2, Math.round(height * dpr));
    if (els.mapCanvas.width !== wantW || els.mapCanvas.height !== wantH) {
      els.mapCanvas.width = wantW;
      els.mapCanvas.height = wantH;
    }
  }

  function worldToScreen(x, y) {
    const { width, height } = getCanvasCssSize();
    return {
      x: width * 0.5 + (x - state.view.centerX) * state.view.scale,
      y: height * 0.5 - (y - state.view.centerY) * state.view.scale,
    };
  }

  function screenToWorld(sx, sy) {
    const { width, height } = getCanvasCssSize();
    return {
      x: state.view.centerX + (sx - width * 0.5) / state.view.scale,
      y: state.view.centerY - (sy - height * 0.5) / state.view.scale,
    };
  }

  function eventScreenPos(evt) {
    const rect = els.mapCanvas.getBoundingClientRect();
    return {
      x: evt.clientX - rect.left,
      y: evt.clientY - rect.top,
    };
  }

  function setStatus(message, kind = 'ok') {
    els.statusBar.textContent = message;
    els.statusBar.classList.remove('status-ok', 'status-warn', 'status-error');
    if (kind === 'warn') {
      els.statusBar.classList.add('status-warn');
    } else if (kind === 'error') {
      els.statusBar.classList.add('status-error');
    } else {
      els.statusBar.classList.add('status-ok');
    }
  }

  function collectBoundsFromRoutes(routes) {
    let minX = Number.POSITIVE_INFINITY;
    let maxX = Number.NEGATIVE_INFINITY;
    let minY = Number.POSITIVE_INFINITY;
    let maxY = Number.NEGATIVE_INFINITY;

    const add = (x, y) => {
      if (!Number.isFinite(x) || !Number.isFinite(y)) return;
      minX = Math.min(minX, x);
      maxX = Math.max(maxX, x);
      minY = Math.min(minY, y);
      maxY = Math.max(maxY, y);
    };

    for (const route of routes) {
      for (const wp of route.waypoints || []) {
        add(Number(wp.x), Number(wp.y));
      }
    }

    if (!Number.isFinite(minX)) {
      return null;
    }

    return { minX, maxX, minY, maxY };
  }

  function fitViewToBounds(bounds) {
    if (!bounds) return;
    const { width, height } = getCanvasCssSize();
    const rangeX = Math.max(1.0, bounds.maxX - bounds.minX);
    const rangeY = Math.max(1.0, bounds.maxY - bounds.minY);
    const margin = 0.88;
    const sx = (width * margin) / rangeX;
    const sy = (height * margin) / rangeY;
    state.view.scale = clamp(Math.min(sx, sy), state.view.minScale, state.view.maxScale);
    state.view.centerX = 0.5 * (bounds.minX + bounds.maxX);
    state.view.centerY = 0.5 * (bounds.minY + bounds.maxY);
  }

  function fitViewToCurrentScenario() {
    const routes = currentRoutes();
    let bounds = collectBoundsFromRoutes(routes);
    if (!bounds && state.project && Array.isArray(state.project.map_lines) && state.project.map_lines.length > 0) {
      let minX = Number.POSITIVE_INFINITY;
      let maxX = Number.NEGATIVE_INFINITY;
      let minY = Number.POSITIVE_INFINITY;
      let maxY = Number.NEGATIVE_INFINITY;
      for (const line of state.project.map_lines) {
        for (const point of line) {
          if (!Array.isArray(point) || point.length < 2) continue;
          const x = Number(point[0]);
          const y = Number(point[1]);
          if (!Number.isFinite(x) || !Number.isFinite(y)) continue;
          minX = Math.min(minX, x);
          maxX = Math.max(maxX, x);
          minY = Math.min(minY, y);
          maxY = Math.max(maxY, y);
        }
      }
      if (Number.isFinite(minX)) {
        bounds = { minX, maxX, minY, maxY };
      }
    }
    fitViewToBounds(bounds);
    render();
  }

  function focusSelectedActors() {
    const routes = currentRoutes().filter((route) => state.selectedActorIds.has(routeId(route)));
    if (routes.length === 0) {
      setStatus('No selected actors to focus.', 'warn');
      return;
    }
    fitViewToBounds(collectBoundsFromRoutes(routes));
    setStatus(`Focused view on ${routes.length} selected actor(s).`);
    render();
  }

  function cloneScenarios() {
    return JSON.parse(JSON.stringify(state.project.scenarios || []));
  }

  function restoreScenarios(snapshot) {
    state.project.scenarios = JSON.parse(JSON.stringify(snapshot || []));
    normalizeSelectionAfterDataChange();
  }

  function resetHistory() {
    state.history = [{ label: 'Initial', scenarios: cloneScenarios() }];
    state.historyIndex = 0;
    updateHistoryUi();
  }

  function pushHistory(label) {
    const snapshot = cloneScenarios();
    state.history = state.history.slice(0, state.historyIndex + 1);
    state.history.push({ label: String(label || 'Edit'), scenarios: snapshot });

    const maxDepth = 150;
    if (state.history.length > maxDepth) {
      const drop = state.history.length - maxDepth;
      state.history.splice(0, drop);
    }

    state.historyIndex = state.history.length - 1;
    updateHistoryUi();
  }

  function updateHistoryUi() {
    const canUndo = state.historyIndex > 0;
    const canRedo = state.historyIndex >= 0 && state.historyIndex < state.history.length - 1;
    els.undoBtn.disabled = !canUndo;
    els.redoBtn.disabled = !canRedo;

    if (state.historyIndex >= 0 && state.historyIndex < state.history.length) {
      const entry = state.history[state.historyIndex];
      els.historyLabel.textContent = `Step ${state.historyIndex + 1}/${state.history.length}: ${entry.label}`;
    } else {
      els.historyLabel.textContent = 'No edits yet.';
    }
  }

  function undo() {
    if (state.historyIndex <= 0) return;
    state.historyIndex -= 1;
    restoreScenarios(state.history[state.historyIndex].scenarios);
    updateUiFromState();
    setStatus('Undo applied.');
  }

  function redo() {
    if (state.historyIndex >= state.history.length - 1) return;
    state.historyIndex += 1;
    restoreScenarios(state.history[state.historyIndex].scenarios);
    updateUiFromState();
    setStatus('Redo applied.');
  }

  function normalizeSelectionAfterDataChange() {
    const routes = currentRoutes();
    const ids = new Set(routes.map((route) => routeId(route)));

    const nextSelected = new Set();
    for (const id of state.selectedActorIds) {
      if (ids.has(id)) {
        nextSelected.add(id);
      }
    }
    state.selectedActorIds = nextSelected;

    if (!state.activeRouteId || !ids.has(String(state.activeRouteId))) {
      if (state.selectedActorIds.size > 0) {
        state.activeRouteId = Array.from(state.selectedActorIds)[0];
      } else if (routes.length > 0) {
        state.activeRouteId = routeId(routes[0]);
      } else {
        state.activeRouteId = null;
      }
    }

    const activeRoute = routeById(state.activeRouteId);
    if (!activeRoute) {
      state.selectedWaypointIndices.clear();
      return;
    }

    const maxIndex = (activeRoute.waypoints || []).length - 1;
    const nextWp = new Set();
    for (const idx of state.selectedWaypointIndices) {
      if (idx >= 0 && idx <= maxIndex) {
        nextWp.add(idx);
      }
    }
    state.selectedWaypointIndices = nextWp;
  }

  function setMode(mode) {
    state.mode = mode === 'waypoint' ? 'waypoint' : 'actor';
    els.modeActor.className = state.mode === 'actor' ? 'btn-primary' : 'btn-secondary';
    els.modeWaypoint.className = state.mode === 'waypoint' ? 'btn-primary' : 'btn-secondary';
    updateHud();
    render();
  }

  function selectSingleActor(id) {
    state.selectedActorIds.clear();
    if (id) {
      state.selectedActorIds.add(String(id));
      state.activeRouteId = String(id);
    }
    state.selectedWaypointIndices.clear();
    updateActorList();
    updateHud();
    render();
  }

  function toggleActor(id) {
    const rid = String(id);
    if (state.selectedActorIds.has(rid)) {
      state.selectedActorIds.delete(rid);
      if (state.activeRouteId === rid) {
        state.activeRouteId = state.selectedActorIds.size > 0 ? Array.from(state.selectedActorIds)[0] : null;
        state.selectedWaypointIndices.clear();
      }
    } else {
      state.selectedActorIds.add(rid);
      state.activeRouteId = rid;
    }
    updateActorList();
    updateHud();
    render();
  }

  function selectAllVisibleActors() {
    const rows = Array.from(els.actorList.querySelectorAll('.actor-row'));
    state.selectedActorIds.clear();
    for (const row of rows) {
      const rid = row.getAttribute('data-route-id');
      if (rid) state.selectedActorIds.add(rid);
    }
    if (state.selectedActorIds.size > 0) {
      state.activeRouteId = Array.from(state.selectedActorIds)[0];
    }
    updateActorList();
    updateHud();
    render();
  }

  function clearActorSelection() {
    state.selectedActorIds.clear();
    state.selectedWaypointIndices.clear();
    updateActorList();
    updateHud();
    render();
  }

  function switchScenario(newIndex, { fit = false } = {}) {
    if (!state.project || !Array.isArray(state.project.scenarios) || state.project.scenarios.length === 0) {
      return;
    }
    state.scenarioIndex = clamp(Number(newIndex) || 0, 0, state.project.scenarios.length - 1);
    const scenario = currentScenario();
    state.selectedActorIds.clear();
    state.selectedWaypointIndices.clear();
    state.activeRouteId = scenario && scenario.routes && scenario.routes.length > 0 ? routeId(scenario.routes[0]) : null;

    els.scenarioSelect.value = String(state.scenarioIndex);
    updateActorList();
    updateHud();
    if (fit) {
      fitViewToCurrentScenario();
    } else {
      render();
    }
  }

  function updateScenarioSelector() {
    els.scenarioSelect.innerHTML = '';
    const scenarios = (state.project && Array.isArray(state.project.scenarios))
      ? state.project.scenarios
      : [];
    scenarios.forEach((scenario, idx) => {
      const opt = document.createElement('option');
      opt.value = String(idx);
      const count = Array.isArray(scenario.routes) ? scenario.routes.length : 0;
      opt.textContent = `${scenario.name} (${count} route${count === 1 ? '' : 's'})`;
      els.scenarioSelect.appendChild(opt);
    });
    els.scenarioSelect.value = String(clamp(state.scenarioIndex, 0, Math.max(0, scenarios.length - 1)));
  }

  function actorMatchesSearch(route, query) {
    if (!query) return true;
    const hay = [route.name, route.role, route.relpath, route.route_id, route.town]
      .map((value) => String(value || '').toLowerCase())
      .join(' ');
    return hay.includes(query);
  }

  function updateActorList() {
    els.actorList.innerHTML = '';
    const routes = currentRoutes();
    const query = String(els.actorSearch.value || '').trim().toLowerCase();

    const visible = routes.filter((route) => actorMatchesSearch(route, query));
    for (const route of visible) {
      const rid = routeId(route);
      const row = document.createElement('div');
      row.className = 'actor-row';
      row.setAttribute('data-route-id', rid);
      if (state.selectedActorIds.has(rid)) row.classList.add('selected');
      if (state.activeRouteId === rid) row.classList.add('active');

      const checkbox = document.createElement('input');
      checkbox.type = 'checkbox';
      checkbox.checked = state.selectedActorIds.has(rid);
      checkbox.addEventListener('click', (evt) => {
        evt.stopPropagation();
        toggleActor(rid);
      });
      row.appendChild(checkbox);

      const middle = document.createElement('div');
      middle.style.minWidth = '0';
      const name = document.createElement('div');
      name.className = 'actor-name';
      name.textContent = route.name || route.relpath || rid;
      middle.appendChild(name);
      const meta = document.createElement('div');
      meta.className = 'actor-meta';
      const wpCount = Array.isArray(route.waypoints) ? route.waypoints.length : 0;
      meta.textContent = `${route.town || '-'} | wp=${wpCount}`;
      middle.appendChild(meta);
      row.appendChild(middle);

      const role = document.createElement('span');
      role.className = `badge ${roleClass(route.role)}`;
      role.textContent = String(route.role || 'npc');
      row.appendChild(role);

      row.addEventListener('click', (evt) => {
        if (evt.shiftKey || evt.metaKey || evt.ctrlKey) {
          toggleActor(rid);
        } else {
          selectSingleActor(rid);
        }
      });

      els.actorList.appendChild(row);
    }

    if (visible.length === 0) {
      const empty = document.createElement('div');
      empty.className = 'small';
      empty.style.padding = '10px';
      empty.textContent = query ? 'No actors match the current filter.' : 'No routes in this scenario.';
      els.actorList.appendChild(empty);
    }
  }

  function updateHud() {
    const scenario = currentScenario();
    const scenarioName = scenario ? scenario.name : '-';
    els.hudScenario.textContent = `Scenario: ${scenarioName}`;
    els.hudMode.textContent = `Mode: ${state.mode === 'actor' ? 'Actor Transform' : 'Waypoint Edit'}`;
    els.hudSelection.textContent = `Selected actors: ${state.selectedActorIds.size}`;
    els.hudWaypoints.textContent = `Selected waypoints: ${state.selectedWaypointIndices.size}`;

    const active = routeById(state.activeRouteId);
    els.activeRouteReadout.textContent = active ? `${active.name || active.relpath} (${active.role})` : '-';
    els.viewReadout.textContent = `scale=${state.view.scale.toFixed(3)} center=(${state.view.centerX.toFixed(2)}, ${state.view.centerY.toFixed(2)})`;

    const projectStats = (state.project && state.project.stats) ? state.project.stats : {};
    const scenarioCount = projectStats.scenario_count || (((state.project && state.project.scenarios) || []).length) || 0;
    const routeCount = projectStats.route_count || 0;
    const wpCount = projectStats.waypoint_count || 0;
    const mapLineCount = ((state.project && state.project.map_lines) || []).length;

    els.projectMeta.innerHTML =
      `<div>Source: <code>${(state.project && state.project.source_path) || '-'}</code></div>` +
      `<div>Scenarios: <b>${scenarioCount}</b> | Routes: <b>${routeCount}</b> | Waypoints: <b>${wpCount}</b></div>` +
      `<div>Map lines loaded: <b>${mapLineCount}</b></div>`;
  }

  function updateUiFromState() {
    updateScenarioSelector();
    updateActorList();
    updateHud();
    updateHistoryUi();
    render();
  }

  function pointToSegmentDistance(px, py, ax, ay, bx, by) {
    const vx = bx - ax;
    const vy = by - ay;
    const wx = px - ax;
    const wy = py - ay;
    const segLen2 = vx * vx + vy * vy;
    let t = 0;
    if (segLen2 > 1e-9) {
      t = clamp((wx * vx + wy * vy) / segLen2, 0, 1);
    }
    const qx = ax + t * vx;
    const qy = ay + t * vy;
    const dx = px - qx;
    const dy = py - qy;
    return {
      dist2: dx * dx + dy * dy,
      t,
      qx,
      qy,
    };
  }

  function pickRouteAt(worldX, worldY) {
    const routes = currentRoutes();
    if (routes.length === 0) return null;

    const thresholdPx = 10;
    const threshold = thresholdPx / state.view.scale;
    const threshold2 = threshold * threshold;
    let best = null;

    for (const route of routes) {
      const waypoints = route.waypoints || [];
      if (waypoints.length === 0) continue;

      let localBest = Number.POSITIVE_INFINITY;
      for (let i = 0; i < waypoints.length - 1; i += 1) {
        const a = waypoints[i];
        const b = waypoints[i + 1];
        const res = pointToSegmentDistance(worldX, worldY, Number(a.x), Number(a.y), Number(b.x), Number(b.y));
        localBest = Math.min(localBest, res.dist2);
      }
      if (waypoints.length === 1) {
        const dx = worldX - Number(waypoints[0].x);
        const dy = worldY - Number(waypoints[0].y);
        localBest = Math.min(localBest, dx * dx + dy * dy);
      }

      if (localBest <= threshold2 && (!best || localBest < best.dist2)) {
        best = { route, dist2: localBest };
      }
    }

    return best ? best.route : null;
  }

  function pickWaypointIndex(route, worldX, worldY) {
    const waypoints = (route && route.waypoints) ? route.waypoints : [];
    if (waypoints.length === 0) return null;

    const thresholdPx = 9;
    const threshold = thresholdPx / state.view.scale;
    const threshold2 = threshold * threshold;

    let bestIdx = null;
    let bestDist2 = Number.POSITIVE_INFINITY;
    for (let i = 0; i < waypoints.length; i += 1) {
      const wp = waypoints[i];
      const dx = worldX - Number(wp.x);
      const dy = worldY - Number(wp.y);
      const d2 = dx * dx + dy * dy;
      if (d2 <= threshold2 && d2 < bestDist2) {
        bestDist2 = d2;
        bestIdx = i;
      }
    }
    return bestIdx;
  }

  function buildMapIndex() {
    const lines = (state.project && state.project.map_lines) ? state.project.map_lines : [];
    const segments = [];
    const cellSize = 20.0;

    for (const line of lines) {
      if (!Array.isArray(line) || line.length < 2) continue;
      for (let i = 0; i < line.length - 1; i += 1) {
        const a = line[i];
        const b = line[i + 1];
        if (!Array.isArray(a) || !Array.isArray(b) || a.length < 2 || b.length < 2) continue;
        const ax = Number(a[0]);
        const ay = Number(a[1]);
        const bx = Number(b[0]);
        const by = Number(b[1]);
        const dx = bx - ax;
        const dy = by - ay;
        const len2 = dx * dx + dy * dy;
        if (!Number.isFinite(len2) || len2 < 1e-9) continue;
        const len = Math.sqrt(len2);
        const heading = (Math.atan2(dy, dx) * 180) / Math.PI;
        const nx = -dy / len;
        const ny = dx / len;
        segments.push({
          ax,
          ay,
          bx,
          by,
          dx,
          dy,
          len2,
          heading,
          nx,
          ny,
          minX: Math.min(ax, bx),
          maxX: Math.max(ax, bx),
          minY: Math.min(ay, by),
          maxY: Math.max(ay, by),
        });
      }
    }

    const grid = new Map();
    function key(ix, iy) {
      return `${ix},${iy}`;
    }

    segments.forEach((seg, idx) => {
      const minIx = Math.floor(seg.minX / cellSize);
      const maxIx = Math.floor(seg.maxX / cellSize);
      const minIy = Math.floor(seg.minY / cellSize);
      const maxIy = Math.floor(seg.maxY / cellSize);
      for (let ix = minIx; ix <= maxIx; ix += 1) {
        for (let iy = minIy; iy <= maxIy; iy += 1) {
          const k = key(ix, iy);
          if (!grid.has(k)) {
            grid.set(k, []);
          }
          grid.get(k).push(idx);
        }
      }
    });

    state.mapIndex = { segments, grid, cellSize };
  }

  function nearestRoadSegment(worldX, worldY) {
    if (!state.mapIndex || !Array.isArray(state.mapIndex.segments) || state.mapIndex.segments.length === 0) {
      return null;
    }

    const { segments, grid, cellSize } = state.mapIndex;
    const cx = Math.floor(worldX / cellSize);
    const cy = Math.floor(worldY / cellSize);

    let best = null;
    function evalIndices(indices) {
      for (const idx of indices) {
        const seg = segments[idx];
        const res = pointToSegmentDistance(worldX, worldY, seg.ax, seg.ay, seg.bx, seg.by);
        if (!best || res.dist2 < best.dist2) {
          best = {
            seg,
            dist2: res.dist2,
            projX: res.qx,
            projY: res.qy,
            t: res.t,
          };
        }
      }
    }

    for (let radius = 0; radius <= 2; radius += 1) {
      const bucket = new Set();
      for (let ix = cx - radius; ix <= cx + radius; ix += 1) {
        for (let iy = cy - radius; iy <= cy + radius; iy += 1) {
          const items = grid.get(`${ix},${iy}`);
          if (!items) continue;
          for (const idx of items) bucket.add(idx);
        }
      }
      if (bucket.size > 0) {
        evalIndices(Array.from(bucket));
        if (best) {
          break;
        }
      }
    }

    if (!best) {
      evalIndices(segments.map((_, idx) => idx));
    }

    return best;
  }

  function setYawFromRouteShape(route, selectedIndices = null) {
    const waypoints = (route && route.waypoints) ? route.waypoints : [];
    if (waypoints.length < 2) return false;

    const indices = selectedIndices && selectedIndices.size > 0
      ? Array.from(selectedIndices).sort((a, b) => a - b)
      : [...Array(waypoints.length).keys()];

    let changed = false;
    for (const idx of indices) {
      const i = Number(idx);
      if (i < 0 || i >= waypoints.length) continue;
      const prev = waypoints[Math.max(0, i - 1)];
      const next = waypoints[Math.min(waypoints.length - 1, i + 1)];
      const dx = Number(next.x) - Number(prev.x);
      const dy = Number(next.y) - Number(prev.y);
      if (Math.hypot(dx, dy) < 1e-6) continue;
      const yaw = (Math.atan2(dy, dx) * 180) / Math.PI;
      if (Math.abs(normalizeAngleDeg(yaw - Number(waypoints[i].yaw))) > 1e-4) {
        waypoints[i].yaw = normalizeAngleDeg(yaw);
        changed = true;
      }
    }
    return changed;
  }

  function snapWaypointToRoad(wp, opts) {
    const nearest = nearestRoadSegment(Number(wp.x), Number(wp.y));
    if (!nearest) return false;

    const offset = Number(opts.offset) || 0;
    const alignedX = nearest.projX + offset * nearest.seg.nx;
    const alignedY = nearest.projY + offset * nearest.seg.ny;

    const moved = Math.hypot(alignedX - Number(wp.x), alignedY - Number(wp.y)) > 1e-5;
    wp.x = alignedX;
    wp.y = alignedY;

    let rotated = false;
    if (opts.alignYaw) {
      let yaw = normalizeAngleDeg(nearest.seg.heading);
      if (opts.keepDirection) {
        const flipped = normalizeAngleDeg(yaw + 180);
        const currentYaw = Number(wp.yaw) || 0;
        yaw = angleDiffDeg(currentYaw, yaw) <= angleDiffDeg(currentYaw, flipped) ? yaw : flipped;
      }
      if (Math.abs(normalizeAngleDeg(yaw - Number(wp.yaw))) > 1e-4) {
        wp.yaw = yaw;
        rotated = true;
      }
    }

    return moved || rotated;
  }

  function applyToSelectedRoutes(mutator) {
    let changed = false;
    for (const route of currentRoutes()) {
      const rid = routeId(route);
      if (!state.selectedActorIds.has(rid)) continue;
      changed = mutator(route) || changed;
    }
    return changed;
  }

  function applyTranslation(dx, dy) {
    const moveX = Number(dx) || 0;
    const moveY = Number(dy) || 0;
    if (Math.hypot(moveX, moveY) < 1e-9) {
      setStatus('Translation is zero. Nothing changed.', 'warn');
      return;
    }
    const changed = applyToSelectedRoutes((route) => {
      let routeChanged = false;
      for (const wp of route.waypoints || []) {
        wp.x = Number(wp.x) + moveX;
        wp.y = Number(wp.y) + moveY;
        routeChanged = true;
      }
      return routeChanged;
    });
    if (!changed) {
      setStatus('No selected actors to translate.', 'warn');
      return;
    }
    pushHistory(`Translate actors dx=${moveX.toFixed(3)} dy=${moveY.toFixed(3)}`);
    updateUiFromState();
    setStatus(`Translated selected actors by (${moveX.toFixed(3)}, ${moveY.toFixed(3)}).`);
  }

  function applyRotation(deg) {
    const angleDeg = Number(deg) || 0;
    if (Math.abs(angleDeg) < 1e-9) {
      setStatus('Rotation is zero. Nothing changed.', 'warn');
      return;
    }

    const selectedRoutes = currentRoutes().filter((route) => state.selectedActorIds.has(routeId(route)));
    if (selectedRoutes.length === 0) {
      setStatus('No selected actors to rotate.', 'warn');
      return;
    }

    let count = 0;
    let sumX = 0;
    let sumY = 0;
    for (const route of selectedRoutes) {
      for (const wp of route.waypoints || []) {
        sumX += Number(wp.x);
        sumY += Number(wp.y);
        count += 1;
      }
    }
    if (count === 0) {
      setStatus('Selected actors have no waypoints.', 'warn');
      return;
    }

    const cx = sumX / count;
    const cy = sumY / count;
    const rad = (angleDeg * Math.PI) / 180;
    const cs = Math.cos(rad);
    const sn = Math.sin(rad);

    for (const route of selectedRoutes) {
      for (const wp of route.waypoints || []) {
        const x = Number(wp.x) - cx;
        const y = Number(wp.y) - cy;
        wp.x = cx + x * cs - y * sn;
        wp.y = cy + x * sn + y * cs;
        wp.yaw = normalizeAngleDeg(Number(wp.yaw) + angleDeg);
      }
    }

    pushHistory(`Rotate selected actors by ${angleDeg.toFixed(2)} deg`);
    updateUiFromState();
    setStatus(`Rotated selected actors by ${angleDeg.toFixed(2)}° around centroid.`);
  }

  function smoothSelectedWaypoints() {
    const route = routeById(state.activeRouteId);
    if (!route) {
      setStatus('No active actor route selected.', 'warn');
      return;
    }

    const waypoints = route.waypoints || [];
    if (waypoints.length < 3) {
      setStatus('Need at least 3 waypoints for smoothing.', 'warn');
      return;
    }

    const windowRadius = Math.max(1, Math.floor(toFloat(els.smoothWindow.value, 3)));
    const iterations = Math.max(1, Math.floor(toFloat(els.smoothIters.value, 1)));

    const selected = state.selectedWaypointIndices.size > 0
      ? Array.from(state.selectedWaypointIndices).filter((idx) => idx >= 0 && idx < waypoints.length)
      : [...Array(waypoints.length).keys()];

    if (selected.length === 0) {
      setStatus('No waypoints selected for smoothing.', 'warn');
      return;
    }

    for (let iter = 0; iter < iterations; iter += 1) {
      const clone = waypoints.map((wp) => ({ ...wp }));
      for (const idx of selected) {
        if (idx <= 0 || idx >= waypoints.length - 1) {
          continue; // keep endpoints stable by default
        }
        const start = Math.max(0, idx - windowRadius);
        const end = Math.min(waypoints.length - 1, idx + windowRadius);
        let accX = 0;
        let accY = 0;
        let accZ = 0;
        let n = 0;
        for (let j = start; j <= end; j += 1) {
          accX += Number(waypoints[j].x);
          accY += Number(waypoints[j].y);
          accZ += Number(waypoints[j].z);
          n += 1;
        }
        if (n > 0) {
          clone[idx].x = accX / n;
          clone[idx].y = accY / n;
          clone[idx].z = accZ / n;
        }
      }
      for (let i = 0; i < waypoints.length; i += 1) {
        waypoints[i].x = clone[i].x;
        waypoints[i].y = clone[i].y;
        waypoints[i].z = clone[i].z;
      }
    }

    setYawFromRouteShape(route, state.selectedWaypointIndices);
    pushHistory(`Smooth waypoints (window=${windowRadius}, iterations=${iterations})`);
    updateUiFromState();
    setStatus(`Smoothed ${selected.length} waypoint(s) on active route.`);
  }

  function resampleRoute(waypoints, spacing) {
    if (!Array.isArray(waypoints) || waypoints.length < 2) {
      return null;
    }
    const pts = waypoints;
    const cumulative = [0];
    for (let i = 1; i < pts.length; i += 1) {
      const dx = Number(pts[i].x) - Number(pts[i - 1].x);
      const dy = Number(pts[i].y) - Number(pts[i - 1].y);
      const dz = Number(pts[i].z) - Number(pts[i - 1].z);
      cumulative.push(cumulative[i - 1] + Math.sqrt(dx * dx + dy * dy + dz * dz));
    }

    const total = cumulative[cumulative.length - 1];
    if (total < 1e-6 || spacing <= 0) {
      return null;
    }

    const targets = [];
    for (let s = 0; s < total; s += spacing) {
      targets.push(s);
    }
    if (targets.length === 0 || targets[targets.length - 1] < total) {
      targets.push(total);
    }

    const result = [];
    let seg = 0;
    for (const tDist of targets) {
      while (seg < cumulative.length - 2 && cumulative[seg + 1] < tDist) {
        seg += 1;
      }
      const d0 = cumulative[seg];
      const d1 = cumulative[seg + 1];
      const alpha = d1 > d0 ? (tDist - d0) / (d1 - d0) : 0;
      const a = pts[seg];
      const b = pts[seg + 1];

      const wp = {
        index: result.length,
        x: Number(a.x) + (Number(b.x) - Number(a.x)) * alpha,
        y: Number(a.y) + (Number(b.y) - Number(a.y)) * alpha,
        z: Number(a.z) + (Number(b.z) - Number(a.z)) * alpha,
        yaw: interpAngleDeg(Number(a.yaw), Number(b.yaw), alpha),
        pitch: Number(a.pitch) + (Number(b.pitch) - Number(a.pitch)) * alpha,
        roll: Number(a.roll) + (Number(b.roll) - Number(a.roll)) * alpha,
        time: null,
        extra_attrs: {},
      };

      const aTime = a.time;
      const bTime = b.time;
      const at = Number(aTime);
      const bt = Number(bTime);
      if (Number.isFinite(at) && Number.isFinite(bt)) {
        wp.time = at + (bt - at) * alpha;
      }

      result.push(wp);
    }

    return result;
  }

  function resampleActiveRoute() {
    const route = routeById(state.activeRouteId);
    if (!route) {
      setStatus('No active actor route selected.', 'warn');
      return;
    }
    const spacing = Math.max(0.1, toFloat(els.resampleSpacing.value, 0.8));
    const resampled = resampleRoute(route.waypoints || [], spacing);
    if (!resampled || resampled.length < 2) {
      setStatus('Could not resample route. Check waypoint geometry.', 'warn');
      return;
    }

    route.waypoints = resampled;
    state.selectedWaypointIndices.clear();
    pushHistory(`Resample active route spacing=${spacing.toFixed(2)}m`);
    updateUiFromState();
    setStatus(`Resampled active route to ${resampled.length} waypoints (spacing ${spacing.toFixed(2)}m).`);
  }

  function snapSelectedActors() {
    if (!state.mapIndex || !state.mapIndex.segments || state.mapIndex.segments.length === 0) {
      setStatus('No map lines loaded. Provide --map-pkl or --map-lines-json.', 'error');
      return;
    }

    const opts = {
      offset: toFloat(els.snapOffset.value, 0),
      alignYaw: !!els.snapAlignYaw.checked,
      keepDirection: !!els.snapKeepDirection.checked,
    };

    let touchedRoutes = 0;
    let changedPoints = 0;
    const changed = applyToSelectedRoutes((route) => {
      let routeChanged = false;
      for (const wp of route.waypoints || []) {
        if (snapWaypointToRoad(wp, opts)) {
          routeChanged = true;
          changedPoints += 1;
        }
      }
      if (routeChanged) touchedRoutes += 1;
      return routeChanged;
    });

    if (!changed) {
      setStatus('No selected actors snapped (selection empty or already aligned).', 'warn');
      return;
    }

    pushHistory(`Snap selected actors to road (offset=${opts.offset.toFixed(2)}m)`);
    updateUiFromState();
    setStatus(`Snapped ${changedPoints} waypoint(s) across ${touchedRoutes} actor(s).`);
  }

  function snapSelectedWaypoints() {
    if (!state.mapIndex || !state.mapIndex.segments || state.mapIndex.segments.length === 0) {
      setStatus('No map lines loaded. Provide --map-pkl or --map-lines-json.', 'error');
      return;
    }

    const route = routeById(state.activeRouteId);
    if (!route) {
      setStatus('No active route selected for waypoint snap.', 'warn');
      return;
    }

    const opts = {
      offset: toFloat(els.snapOffset.value, 0),
      alignYaw: !!els.snapAlignYaw.checked,
      keepDirection: !!els.snapKeepDirection.checked,
    };

    const indices = state.selectedWaypointIndices.size > 0
      ? Array.from(state.selectedWaypointIndices)
      : [...Array((route.waypoints || []).length).keys()];

    let changedCount = 0;
    for (const idx of indices) {
      const wp = route.waypoints[idx];
      if (!wp) continue;
      if (snapWaypointToRoad(wp, opts)) {
        changedCount += 1;
      }
    }

    if (changedCount === 0) {
      setStatus('No waypoint changed during snap.', 'warn');
      return;
    }

    pushHistory(`Snap selected waypoints (offset=${opts.offset.toFixed(2)}m)`);
    updateUiFromState();
    setStatus(`Snapped ${changedCount} waypoint(s) on active route.`);
  }

  function exportBundle(scope) {
    const filenameBaseRaw = String(els.exportName.value || 'aligned_routes').trim();
    const filenameBase = filenameBaseRaw.length > 0 ? filenameBaseRaw : 'aligned_routes';
    const scenario = currentScenario();

    const projectForExport = {
      project_name: state.project.project_name,
      source_type: state.project.source_type,
      source_path: state.project.source_path,
      scenarios: state.project.scenarios,
    };

    const payload = {
      project: projectForExport,
      scope,
      scenario_names: scope === 'scenario' && scenario ? [scenario.name] : [],
      filename: scope === 'scenario' && scenario
        ? `${filenameBase}_${scenario.name}.zip`
        : `${filenameBase}.zip`,
    };

    setStatus('Preparing ZIP export...', 'ok');

    fetch('/api/export', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    })
      .then(async (res) => {
        if (!res.ok) {
          const err = await res.json().catch(() => ({ error: 'Export failed.' }));
          throw new Error(err.error || 'Export failed.');
        }

        const blob = await res.blob();
        const fallbackName = payload.filename;
        const contentDisp = res.headers.get('Content-Disposition') || '';
        const match = /filename="?([^\";]+)"?/.exec(contentDisp);
        const filename = match ? match[1] : fallbackName;

        const url = URL.createObjectURL(blob);
        const anchor = document.createElement('a');
        anchor.href = url;
        anchor.download = filename;
        document.body.appendChild(anchor);
        anchor.click();
        anchor.remove();
        URL.revokeObjectURL(url);
        setStatus(`Export ready: ${filename}`);
      })
      .catch((err) => {
        setStatus(`Export failed: ${err.message}`, 'error');
      });
  }

  function beginInteraction(interaction) {
    state.interaction = interaction;
  }

  function endInteraction() {
    const inter = state.interaction;
    if (inter && inter.changed && inter.historyLabel) {
      pushHistory(inter.historyLabel);
    }
    state.interaction = null;
    updateUiFromState();
  }

  function applyDragFromInteraction(worldX, worldY) {
    const inter = state.interaction;
    if (!inter) return;

    if (inter.type === 'pan') {
      const dx = worldX - inter.startWorld.x;
      const dy = worldY - inter.startWorld.y;
      state.view.centerX = inter.baseCenter.x - dx;
      state.view.centerY = inter.baseCenter.y - dy;
      render();
      return;
    }

    if (inter.type === 'moveActors') {
      const dx = worldX - inter.startWorld.x;
      const dy = worldY - inter.startWorld.y;
      const moved = Math.hypot(dx, dy) > 1e-8;
      for (const item of inter.base) {
        const route = routeById(item.routeId);
        if (!route) continue;
        for (let i = 0; i < item.points.length; i += 1) {
          const baseWp = item.points[i];
          const wp = route.waypoints[i];
          if (!wp) continue;
          wp.x = baseWp.x + dx;
          wp.y = baseWp.y + dy;
        }
      }
      inter.changed = moved;
      render();
      return;
    }

    if (inter.type === 'moveWaypoints') {
      const route = routeById(state.activeRouteId);
      if (!route) return;
      const dx = worldX - inter.startWorld.x;
      const dy = worldY - inter.startWorld.y;
      const moved = Math.hypot(dx, dy) > 1e-8;
      for (const item of inter.base) {
        const wp = route.waypoints[item.index];
        if (!wp) continue;
        wp.x = item.x + dx;
        wp.y = item.y + dy;
      }
      inter.changed = moved;
      render();
      return;
    }

    if (inter.type === 'boxWaypoints') {
      inter.currentScreen = { ...inter.currentScreen, ...worldToScreen(worldX, worldY) };
      render();
    }
  }

  function finalizeBoxWaypointSelection() {
    const inter = state.interaction;
    if (!inter || inter.type !== 'boxWaypoints') return;

    const route = routeById(state.activeRouteId);
    if (!route) return;

    const sx0 = inter.startScreen.x;
    const sy0 = inter.startScreen.y;
    const sx1 = inter.currentScreen.x;
    const sy1 = inter.currentScreen.y;
    const minSX = Math.min(sx0, sx1);
    const maxSX = Math.max(sx0, sx1);
    const minSY = Math.min(sy0, sy1);
    const maxSY = Math.max(sy0, sy1);

    if (!inter.additive) {
      state.selectedWaypointIndices.clear();
    }

    let added = 0;
    for (let i = 0; i < (route.waypoints || []).length; i += 1) {
      const wp = route.waypoints[i];
      const sp = worldToScreen(Number(wp.x), Number(wp.y));
      if (sp.x >= minSX && sp.x <= maxSX && sp.y >= minSY && sp.y <= maxSY) {
        if (!state.selectedWaypointIndices.has(i)) {
          state.selectedWaypointIndices.add(i);
          added += 1;
        }
      }
    }

    if (added > 0) {
      setStatus(`Selected ${added} waypoint(s) by box.`);
    }
  }

  function onMouseDown(evt) {
    const screen = eventScreenPos(evt);
    const world = screenToWorld(screen.x, screen.y);

    const wantsPan = evt.button === 1 || state.keyState.space || evt.altKey;
    if (wantsPan) {
      beginInteraction({
        type: 'pan',
        startWorld: world,
        baseCenter: { x: state.view.centerX, y: state.view.centerY },
        changed: false,
      });
      evt.preventDefault();
      return;
    }

    if (evt.button !== 0) {
      return;
    }

    if (state.mode === 'actor') {
      const picked = pickRouteAt(world.x, world.y);
      if (picked) {
        const rid = routeId(picked);
        if (evt.shiftKey || evt.metaKey || evt.ctrlKey) {
          toggleActor(rid);
        } else if (!state.selectedActorIds.has(rid) || state.selectedActorIds.size !== 1) {
          selectSingleActor(rid);
        }

        if (state.selectedActorIds.has(rid)) {
          const base = [];
          for (const route of currentRoutes()) {
            const id = routeId(route);
            if (!state.selectedActorIds.has(id)) continue;
            base.push({
              routeId: id,
              points: (route.waypoints || []).map((wp) => ({ x: Number(wp.x), y: Number(wp.y) })),
            });
          }
          beginInteraction({
            type: 'moveActors',
            startWorld: world,
            base,
            changed: false,
            historyLabel: `Drag ${base.length} actor(s)`,
          });
        }
      } else if (!(evt.shiftKey || evt.metaKey || evt.ctrlKey)) {
        clearActorSelection();
      }
      return;
    }

    // waypoint mode
    const route = routeById(state.activeRouteId) || pickRouteAt(world.x, world.y);
    if (!route) {
      setStatus('No active route. Select an actor first.', 'warn');
      return;
    }

    state.activeRouteId = routeId(route);
    if (!state.selectedActorIds.has(state.activeRouteId)) {
      state.selectedActorIds.clear();
      state.selectedActorIds.add(state.activeRouteId);
    }

    const wpIdx = pickWaypointIndex(route, world.x, world.y);
    if (wpIdx != null) {
      if (evt.shiftKey || evt.metaKey || evt.ctrlKey) {
        if (state.selectedWaypointIndices.has(wpIdx)) {
          state.selectedWaypointIndices.delete(wpIdx);
        } else {
          state.selectedWaypointIndices.add(wpIdx);
        }
      } else {
        state.selectedWaypointIndices.clear();
        state.selectedWaypointIndices.add(wpIdx);
      }

      const selected = Array.from(state.selectedWaypointIndices).sort((a, b) => a - b);
      const base = selected
        .map((index) => {
          const wp = route.waypoints[index];
          return wp ? { index, x: Number(wp.x), y: Number(wp.y) } : null;
        })
        .filter(Boolean);

      beginInteraction({
        type: 'moveWaypoints',
        startWorld: world,
        base,
        changed: false,
        historyLabel: `Drag ${base.length} waypoint(s)`,
      });
      updateHud();
      render();
      return;
    }

    if (evt.shiftKey) {
      beginInteraction({
        type: 'boxWaypoints',
        startScreen: screen,
        currentScreen: screen,
        additive: true,
        changed: false,
      });
      render();
      return;
    }

    state.selectedWaypointIndices.clear();
    updateHud();
    render();
  }

  function onMouseMove(evt) {
    if (!state.interaction) return;
    const screen = eventScreenPos(evt);
    const world = screenToWorld(screen.x, screen.y);
    applyDragFromInteraction(world.x, world.y);
  }

  function onMouseUp() {
    if (!state.interaction) return;
    if (state.interaction.type === 'boxWaypoints') {
      finalizeBoxWaypointSelection();
    }
    endInteraction();
  }

  function onWheel(evt) {
    evt.preventDefault();
    const screen = eventScreenPos(evt);
    const worldBefore = screenToWorld(screen.x, screen.y);

    const factor = evt.deltaY < 0 ? 1.1 : 0.9;
    state.view.scale = clamp(state.view.scale * factor, state.view.minScale, state.view.maxScale);

    const worldAfter = screenToWorld(screen.x, screen.y);
    state.view.centerX += worldBefore.x - worldAfter.x;
    state.view.centerY += worldBefore.y - worldAfter.y;

    updateHud();
    render();
  }

  function drawMap(ctx, width, height) {
    const lines = (state.project && state.project.map_lines) ? state.project.map_lines : [];
    if (!Array.isArray(lines) || lines.length === 0) {
      return;
    }
    ctx.strokeStyle = 'rgba(196, 217, 230, 0.23)';
    ctx.lineWidth = 1;

    for (const line of lines) {
      if (!Array.isArray(line) || line.length < 2) continue;
      ctx.beginPath();
      for (let i = 0; i < line.length; i += 1) {
        const p = line[i];
        if (!Array.isArray(p) || p.length < 2) continue;
        const sp = worldToScreen(Number(p[0]), Number(p[1]));
        if (i === 0) ctx.moveTo(sp.x, sp.y);
        else ctx.lineTo(sp.x, sp.y);
      }
      ctx.stroke();
    }
  }

  function drawRoutes(ctx) {
    const routes = currentRoutes();

    for (const route of routes) {
      const rid = routeId(route);
      const selected = state.selectedActorIds.has(rid);
      const active = state.activeRouteId === rid;
      const waypoints = route.waypoints || [];
      if (waypoints.length === 0) continue;

      const color = hashColor(rid);
      ctx.strokeStyle = selected ? color : `${color}88`;
      ctx.lineWidth = selected ? 2.6 : 1.25;
      ctx.beginPath();
      for (let i = 0; i < waypoints.length; i += 1) {
        const wp = waypoints[i];
        const sp = worldToScreen(Number(wp.x), Number(wp.y));
        if (i === 0) ctx.moveTo(sp.x, sp.y);
        else ctx.lineTo(sp.x, sp.y);
      }
      ctx.stroke();

      const shouldDrawPoints = selected || (state.mode === 'waypoint' && active);
      if (shouldDrawPoints) {
        for (let i = 0; i < waypoints.length; i += 1) {
          const wp = waypoints[i];
          const sp = worldToScreen(Number(wp.x), Number(wp.y));
          const wpSelected = active && state.selectedWaypointIndices.has(i);
          const radius = wpSelected ? 4.8 : (active ? 3.3 : 2.6);
          ctx.fillStyle = wpSelected ? '#ffe06b' : (active ? '#b8f3ff' : '#d4e9f6');
          ctx.beginPath();
          ctx.arc(sp.x, sp.y, radius, 0, Math.PI * 2);
          ctx.fill();
        }
      }

      // draw start marker
      const start = waypoints[0];
      const ss = worldToScreen(Number(start.x), Number(start.y));
      ctx.fillStyle = selected ? '#ffffff' : '#bdd1df';
      ctx.beginPath();
      ctx.arc(ss.x, ss.y, 2.2, 0, Math.PI * 2);
      ctx.fill();
    }
  }

  function drawInteractionOverlay(ctx) {
    const inter = state.interaction;
    if (!inter || inter.type !== 'boxWaypoints') return;

    const sx0 = inter.startScreen.x;
    const sy0 = inter.startScreen.y;
    const sx1 = inter.currentScreen.x;
    const sy1 = inter.currentScreen.y;
    const x = Math.min(sx0, sx1);
    const y = Math.min(sy0, sy1);
    const w = Math.abs(sx1 - sx0);
    const h = Math.abs(sy1 - sy0);

    ctx.strokeStyle = 'rgba(126, 220, 255, 0.95)';
    ctx.fillStyle = 'rgba(126, 220, 255, 0.17)';
    ctx.lineWidth = 1.2;
    ctx.beginPath();
    ctx.rect(x, y, w, h);
    ctx.fill();
    ctx.stroke();
  }

  function render() {
    resizeCanvas();
    const dpr = window.devicePixelRatio || 1;
    const { width, height } = getCanvasCssSize();
    const ctx = els.mapCanvas.getContext('2d');
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, width, height);

    // layered background
    const grad = ctx.createLinearGradient(0, 0, 0, height);
    grad.addColorStop(0, '#0f1d28');
    grad.addColorStop(1, '#0a141d');
    ctx.fillStyle = grad;
    ctx.fillRect(0, 0, width, height);

    // subtle crosshair at view center
    const center = worldToScreen(state.view.centerX, state.view.centerY);
    ctx.strokeStyle = 'rgba(118, 172, 202, 0.22)';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(center.x - 8, center.y);
    ctx.lineTo(center.x + 8, center.y);
    ctx.moveTo(center.x, center.y - 8);
    ctx.lineTo(center.x, center.y + 8);
    ctx.stroke();

    drawMap(ctx, width, height);
    drawRoutes(ctx);
    drawInteractionOverlay(ctx);

    updateHud();
  }

  function bindCanvasEvents() {
    els.mapCanvas.addEventListener('mousedown', onMouseDown);
    window.addEventListener('mousemove', onMouseMove);
    window.addEventListener('mouseup', onMouseUp);
    els.mapCanvas.addEventListener('wheel', onWheel, { passive: false });

    window.addEventListener('keydown', (evt) => {
      if (evt.key === ' ') {
        state.keyState.space = true;
      }

      const isMac = navigator.platform.toUpperCase().includes('MAC');
      const cmd = isMac ? evt.metaKey : evt.ctrlKey;

      if (cmd && evt.key.toLowerCase() === 'z') {
        evt.preventDefault();
        if (evt.shiftKey) redo();
        else undo();
        return;
      }

      if (!cmd && !evt.altKey) {
        const k = evt.key.toLowerCase();
        if (k === 'a') {
          setMode('actor');
        } else if (k === 'w') {
          setMode('waypoint');
        }
      }
    });

    window.addEventListener('keyup', (evt) => {
      if (evt.key === ' ') {
        state.keyState.space = false;
      }
    });

    window.addEventListener('resize', () => {
      render();
    });
  }

  function bindUiEvents() {
    els.scenarioSelect.addEventListener('change', (evt) => {
      switchScenario(toFloat(evt.target.value, 0), { fit: true });
    });
    els.prevScenario.addEventListener('click', () => {
      switchScenario(state.scenarioIndex - 1, { fit: true });
    });
    els.nextScenario.addEventListener('click', () => {
      switchScenario(state.scenarioIndex + 1, { fit: true });
    });

    els.actorSearch.addEventListener('input', () => {
      updateActorList();
    });
    els.selectAllActors.addEventListener('click', selectAllVisibleActors);
    els.clearActorSelection.addEventListener('click', clearActorSelection);
    els.focusSelection.addEventListener('click', focusSelectedActors);

    els.modeActor.addEventListener('click', () => setMode('actor'));
    els.modeWaypoint.addEventListener('click', () => setMode('waypoint'));

    els.undoBtn.addEventListener('click', undo);
    els.redoBtn.addEventListener('click', redo);

    els.applyTranslate.addEventListener('click', () => {
      applyTranslation(toFloat(els.translateDx.value, 0), toFloat(els.translateDy.value, 0));
    });

    const nudgeStep = 0.15;
    els.nudgeLeft.addEventListener('click', () => applyTranslation(-nudgeStep, 0));
    els.nudgeRight.addEventListener('click', () => applyTranslation(nudgeStep, 0));
    els.nudgeUp.addEventListener('click', () => applyTranslation(0, nudgeStep));
    els.nudgeDown.addEventListener('click', () => applyTranslation(0, -nudgeStep));

    els.applyRotate.addEventListener('click', () => {
      applyRotation(toFloat(els.rotateDeg.value, 0));
    });

    els.setYawFromPathActors.addEventListener('click', () => {
      let changed = false;
      changed = applyToSelectedRoutes((route) => setYawFromRouteShape(route, null)) || changed;
      if (!changed) {
        setStatus('No selected actors to update yaw.', 'warn');
        return;
      }
      pushHistory('Set selected actor yaw from path shape');
      updateUiFromState();
      setStatus('Updated yaw from path geometry for selected actors.');
    });

    els.smoothWaypoints.addEventListener('click', smoothSelectedWaypoints);
    els.resampleActiveRoute.addEventListener('click', resampleActiveRoute);

    els.clearWaypointSelection.addEventListener('click', () => {
      state.selectedWaypointIndices.clear();
      updateHud();
      render();
    });

    els.setYawFromPathWaypoints.addEventListener('click', () => {
      const route = routeById(state.activeRouteId);
      if (!route) {
        setStatus('No active route selected.', 'warn');
        return;
      }
      const changed = setYawFromRouteShape(route, state.selectedWaypointIndices.size > 0 ? state.selectedWaypointIndices : null);
      if (!changed) {
        setStatus('No waypoint yaw update needed.', 'warn');
        return;
      }
      pushHistory('Set active waypoint yaw from path shape');
      updateUiFromState();
      setStatus('Updated waypoint yaw from local path direction.');
    });

    els.snapSelectedActors.addEventListener('click', snapSelectedActors);
    els.snapSelectedWaypoints.addEventListener('click', snapSelectedWaypoints);

    els.exportScenario.addEventListener('click', () => exportBundle('scenario'));
    els.exportAll.addEventListener('click', () => exportBundle('all'));
  }

  function cacheElements() {
    els.mapCanvas = byId('mapCanvas');

    els.projectMeta = byId('projectMeta');
    els.prevScenario = byId('prevScenario');
    els.nextScenario = byId('nextScenario');
    els.scenarioSelect = byId('scenarioSelect');

    els.actorSearch = byId('actorSearch');
    els.selectAllActors = byId('selectAllActors');
    els.clearActorSelection = byId('clearActorSelection');
    els.focusSelection = byId('focusSelection');
    els.actorList = byId('actorList');

    els.modeActor = byId('modeActor');
    els.modeWaypoint = byId('modeWaypoint');

    els.undoBtn = byId('undoBtn');
    els.redoBtn = byId('redoBtn');
    els.historyLabel = byId('historyLabel');

    els.translateDx = byId('translateDx');
    els.translateDy = byId('translateDy');
    els.applyTranslate = byId('applyTranslate');
    els.nudgeLeft = byId('nudgeLeft');
    els.nudgeRight = byId('nudgeRight');
    els.nudgeUp = byId('nudgeUp');
    els.nudgeDown = byId('nudgeDown');

    els.rotateDeg = byId('rotateDeg');
    els.applyRotate = byId('applyRotate');
    els.setYawFromPathActors = byId('setYawFromPathActors');

    els.smoothWindow = byId('smoothWindow');
    els.smoothIters = byId('smoothIters');
    els.smoothWaypoints = byId('smoothWaypoints');
    els.setYawFromPathWaypoints = byId('setYawFromPathWaypoints');
    els.resampleSpacing = byId('resampleSpacing');
    els.resampleActiveRoute = byId('resampleActiveRoute');
    els.clearWaypointSelection = byId('clearWaypointSelection');

    els.snapOffset = byId('snapOffset');
    els.snapAlignYaw = byId('snapAlignYaw');
    els.snapKeepDirection = byId('snapKeepDirection');
    els.snapSelectedActors = byId('snapSelectedActors');
    els.snapSelectedWaypoints = byId('snapSelectedWaypoints');

    els.exportName = byId('exportName');
    els.exportScenario = byId('exportScenario');
    els.exportAll = byId('exportAll');

    els.viewReadout = byId('viewReadout');
    els.activeRouteReadout = byId('activeRouteReadout');

    els.hudScenario = byId('hudScenario');
    els.hudMode = byId('hudMode');
    els.hudSelection = byId('hudSelection');
    els.hudWaypoints = byId('hudWaypoints');

    els.statusBar = byId('statusBar');
  }

  async function loadProject() {
    const response = await fetch('/api/project');
    if (!response.ok) {
      throw new Error(`Failed to load project: HTTP ${response.status}`);
    }
    const payload = await response.json();
    if (!payload || typeof payload !== 'object' || !payload.project) {
      throw new Error('Invalid project payload from server.');
    }
    state.project = payload.project;
    if (!Array.isArray(state.project.scenarios)) {
      state.project.scenarios = [];
    }
  }

  async function init() {
    cacheElements();
    bindCanvasEvents();
    bindUiEvents();

    try {
      await loadProject();
    } catch (err) {
      setStatus(`Failed to load project: ${err.message}`, 'error');
      return;
    }

    buildMapIndex();
    updateScenarioSelector();
    switchScenario(0, { fit: true });
    resetHistory();
    setMode('actor');
    setStatus('Studio ready. Start by selecting one or more actors, then drag on canvas to align.');
  }

  init();
})();
