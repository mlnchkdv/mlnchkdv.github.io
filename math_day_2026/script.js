const prefersReducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
const scrollProgress = document.getElementById("scrollProgress");

function clamp(value, min, max) {
  return Math.min(Math.max(value, min), max);
}

function updateScrollProgress() {
  const scrollable = document.documentElement.scrollHeight - window.innerHeight;
  const progress = scrollable > 0 ? (window.scrollY / scrollable) * 100 : 0;
  scrollProgress.style.width = `${progress}%`;
}

const reveals = document.querySelectorAll(".reveal");
if (!prefersReducedMotion) {
  const observer = new IntersectionObserver(
    (entries) => {
      entries.forEach((entry) => {
        if (entry.isIntersecting) {
          entry.target.classList.add("is-visible");
        }
      });
    },
    { threshold: 0.14 }
  );

  reveals.forEach((element) => observer.observe(element));

  let ticking = false;
  window.addEventListener(
    "scroll",
    () => {
      if (!ticking) {
        ticking = true;
        requestAnimationFrame(() => {
          updateScrollProgress();
          ticking = false;
        });
      }
    },
    { passive: true }
  );
} else {
  reveals.forEach((element) => element.classList.add("is-visible"));
}

updateScrollProgress();

const batteryLoad = document.getElementById("batteryLoad");
const batteryPath = document.getElementById("batteryPath");
const batteryArea = document.getElementById("batteryArea");
const batteryMode = document.getElementById("batteryMode");
const batteryHours = document.getElementById("batteryHours");
const batterySlope = document.getElementById("batterySlope");

function buildBatteryGraph(load) {
  const points = [];
  const hoursLeft = 14 - load * 0.7;
  const mode =
    load <= 3 ? "энергосбережение" :
    load <= 6 ? "сбалансированный" :
    load <= 8 ? "активный" :
    "максимальная нагрузка";

  for (let i = 0; i <= 7; i += 1) {
    const x = 30 + i * 57;
    const base = 18 + i * (14 + load * 1.2);
    const wave = Math.sin(i * 0.8 + load * 0.45) * (10 + load);
    const y = Math.min(190, Math.max(22, base + wave));
    points.push([x, y]);
  }

  const line = points.map((point, index) => `${index === 0 ? "M" : "L"} ${point[0]} ${point[1]}`).join(" ");
  const avgSlope = -(100 / hoursLeft);

  batteryPath.setAttribute("d", line);
  batteryArea.setAttribute("d", `${line} L 429 190 L 30 190 Z`);
  batteryMode.textContent = mode;
  batteryHours.textContent = `${hoursLeft.toFixed(1)} ч`;
  batterySlope.textContent = `${avgSlope.toFixed(1)} %/ч`;
}

batteryLoad.addEventListener("input", (event) => {
  buildBatteryGraph(Number(event.target.value));
});
buildBatteryGraph(Number(batteryLoad.value));

const cameraStage = document.getElementById("cameraStage");
const focusPoint = document.getElementById("focusPoint");
const compositionScore = document.getElementById("compositionScore");
const compositionHint = document.getElementById("compositionHint");
let draggingFocus = false;

function updateComposition(clientX, clientY) {
  const rect = cameraStage.getBoundingClientRect();
  const x = clamp(clientX - rect.left, 0, rect.width);
  const y = clamp(clientY - rect.top, 0, rect.height);
  const left = (x / rect.width) * 100;
  const top = (y / rect.height) * 100;

  focusPoint.style.left = `${left}%`;
  focusPoint.style.top = `${top}%`;

  const targets = [
    [33.33, 33.33],
    [66.66, 33.33],
    [33.33, 66.66],
    [66.66, 66.66],
  ];
  const bestDistance = Math.min(...targets.map(([tx, ty]) => Math.hypot(left - tx, top - ty)));
  const score = Math.max(0, Math.round(100 - bestDistance * 2.2));

  compositionScore.textContent = `${score} / 100`;
  compositionHint.textContent =
    score > 86 ? "Отличная геометрия кадра" :
    score > 68 ? "Почти в сильной точке" :
    "Смести объект ближе к пересечению";
}

focusPoint.addEventListener("pointerdown", (event) => {
  draggingFocus = true;
  focusPoint.setPointerCapture(event.pointerId);
});

focusPoint.addEventListener("pointermove", (event) => {
  if (draggingFocus) {
    updateComposition(event.clientX, event.clientY);
  }
});

["pointerup", "pointercancel", "lostpointercapture"].forEach((eventName) => {
  focusPoint.addEventListener(eventName, () => {
    draggingFocus = false;
  });
});

focusPoint.addEventListener("keydown", (event) => {
  const rect = cameraStage.getBoundingClientRect();
  const currentLeft = parseFloat(focusPoint.style.left || "68");
  const currentTop = parseFloat(focusPoint.style.top || "34");
  const step = 3;
  let left = currentLeft;
  let top = currentTop;

  if (event.key === "ArrowLeft") left -= step;
  if (event.key === "ArrowRight") left += step;
  if (event.key === "ArrowUp") top -= step;
  if (event.key === "ArrowDown") top += step;

  if (left !== currentLeft || top !== currentTop) {
    event.preventDefault();
    updateComposition(
      rect.left + (clamp(left, 0, 100) / 100) * rect.width,
      rect.top + (clamp(top, 0, 100) / 100) * rect.height
    );
  }
});

{
  const rect = cameraStage.getBoundingClientRect();
  updateComposition(rect.left + rect.width * 0.68, rect.top + rect.height * 0.34);
}

const scenePicker = document.getElementById("scenePicker");
const kernelPicker = document.getElementById("kernelPicker");
const patchCaption = document.getElementById("patchCaption");
const patchGrid = document.getElementById("patchGrid");
const kernelGrid = document.getElementById("kernelGrid");
const kernelProduct = document.getElementById("kernelProduct");
const kernelLabel = document.getElementById("kernelLabel");
const kernelEffect = document.getElementById("kernelEffect");
const kernelInterpretation = document.getElementById("kernelInterpretation");
const kernelValue = document.getElementById("kernelValue");
const kernelMath = document.getElementById("kernelMath");
const kernelExplain = document.getElementById("kernelExplain");

const kernels = {
  blur: {
    title: "blur",
    values: [1, 1, 1, 1, 1, 1, 1, 1, 1],
    effect: "подавляет шум и локально усредняет освещенность",
    interpretation: "Интерпретация: среднее по окрестности"
  },
  edge: {
    title: "edge",
    values: [-1, -1, -1, -1, 8, -1, -1, -1, -1],
    effect: "выделяет переходы яркости и контуры объектов",
    interpretation: "Интерпретация: высокая локальная неоднородность"
  },
  sharp: {
    title: "sharp",
    values: [0, -1, 0, -1, 5, -1, 0, -1, 0],
    effect: "поднимает локальный контраст вокруг деталей",
    interpretation: "Интерпретация: усиление центральной структуры"
  }
};

const patchPresets = [
  {
    caption: "Фрагмент с яркой границей справа",
    values: [18, 22, 84, 20, 58, 90, 19, 60, 92]
  },
  {
    caption: "Почти однородный участок с небольшим шумом",
    values: [46, 48, 50, 47, 49, 52, 45, 48, 51]
  },
  {
    caption: "Яркая точка в центре кадра",
    values: [15, 17, 14, 18, 95, 16, 13, 18, 15]
  }
];

let currentPatchValues = [...patchPresets[0].values];
let currentKernelName = "blur";

function renderPatch() {
  patchGrid.innerHTML = "";
  currentPatchValues.forEach((value) => {
    const cell = document.createElement("div");
    cell.className = "patch-cell";
    cell.textContent = value;
    cell.style.background = `rgb(${255 - value * 0.7}, ${248 - value * 0.45}, ${240 - value * 0.25})`;
    patchGrid.appendChild(cell);
  });
}

function renderKernel(name) {
  currentKernelName = name;
  const kernel = kernels[name];
  kernelGrid.innerHTML = "";
  kernelProduct.innerHTML = "";

  kernel.values.forEach((value) => {
    const cell = document.createElement("div");
    cell.className = "kernel-cell";
    cell.textContent = value;
    kernelGrid.appendChild(cell);
  });

  const products = kernel.values.map((value, index) => value * currentPatchValues[index]);
  products.forEach((value) => {
    const cell = document.createElement("div");
    cell.className = "product-cell";
    cell.textContent = value;
    kernelProduct.appendChild(cell);
  });

  const rawResponse = kernel.values.reduce((sum, value, index) => sum + value * currentPatchValues[index], 0);
  const response = name === "blur" ? rawResponse / 9 : rawResponse;

  kernelLabel.textContent = `Оператор: ${kernel.title}`;
  kernelEffect.textContent = kernel.effect;
  kernelInterpretation.textContent = kernel.interpretation;
  kernelValue.textContent = `Отклик центра: ${response.toFixed(1)}`;

  if (name === "blur") {
    const patchSum = currentPatchValues.reduce((sum, value) => sum + value, 0);
    kernelMath.innerHTML = `\\[ y = \\frac{${patchSum}}{9} = ${response.toFixed(1)} \\]`;
    kernelExplain.textContent = "Blur полезен, когда соседние значения примерно описывают один и тот же объект, а резкие колебания вызваны шумом сенсора.";
  } else if (name === "edge") {
    kernelMath.innerHTML = `\\[ y = ${products.join(" + ")} = ${response.toFixed(0)} \\]`;
    kernelExplain.textContent = "Edge-ядро почти зануляется на однородных областях, но дает сильный отклик там, где свет слева и справа различается.";
  } else {
    kernelMath.innerHTML = `\\[ y = ${products.join(" + ")} = ${response.toFixed(0)} \\]`;
    kernelExplain.textContent = "Sharpen сочетает сохранение центра с вычитанием соседей, поэтому текстуры и границы становятся визуально выразительнее.";
  }

  kernelPicker.querySelectorAll(".vision-chip[data-kernel]").forEach((button) => {
    button.classList.toggle("active", button.dataset.kernel === name);
  });
  typesetMath(kernelMath);
}

kernelPicker.querySelectorAll(".vision-chip[data-kernel]").forEach((button) => {
  button.addEventListener("click", () => renderKernel(button.dataset.kernel));
});

scenePicker.querySelectorAll(".vision-chip[data-scene]").forEach((button) => {
  button.addEventListener("click", () => {
    const preset = patchPresets[Number(button.dataset.scene)];
    currentPatchValues = [...preset.values];
    patchCaption.textContent = preset.caption;
    scenePicker.querySelectorAll(".vision-chip[data-scene]").forEach((chip) => {
      chip.classList.toggle("active", chip === button);
    });
    renderPatch();
    renderKernel(currentKernelName);
  });
});

renderPatch();
renderKernel("blur");

const mapStage = document.getElementById("mapStage");
const userDot = document.getElementById("userDot");
const distA = document.getElementById("distA");
const distB = document.getElementById("distB");
const distC = document.getElementById("distC");
const circleA = document.getElementById("circleA");
const circleB = document.getElementById("circleB");
const circleC = document.getElementById("circleC");

const towers = [
  { x: 18, y: 20, element: circleA, label: distA },
  { x: 78, y: 32, element: circleB, label: distB },
  { x: 56, y: 78, element: circleC, label: distC },
];

function updateMap(clientX, clientY) {
  const rect = mapStage.getBoundingClientRect();
  const x = clamp(((clientX - rect.left) / rect.width) * 100, 4, 96);
  const y = clamp(((clientY - rect.top) / rect.height) * 100, 4, 96);

  userDot.style.left = `${x}%`;
  userDot.style.top = `${y}%`;

  towers.forEach((tower) => {
    const distance = Math.hypot(x - tower.x, y - tower.y);
    tower.element.setAttribute("r", distance.toFixed(2));
    tower.label.textContent = `${(distance / 7.8).toFixed(1)} км`;
  });
}

mapStage.addEventListener("pointerdown", (event) => updateMap(event.clientX, event.clientY));
mapStage.addEventListener("pointermove", (event) => {
  if (event.buttons > 0 || event.pointerType === "mouse") {
    updateMap(event.clientX, event.clientY);
  }
});

{
  const rect = mapStage.getBoundingClientRect();
  updateMap(rect.left + rect.width * 0.52, rect.top + rect.height * 0.54);
}

const routePath = document.getElementById("routePath");
const routeBest = document.getElementById("routeBest");
const routeLength = document.getElementById("routeLength");
const routeOption1 = document.getElementById("routeOption1");
const routeOption2 = document.getElementById("routeOption2");
const routeStartLabel = document.getElementById("routeStartLabel");
const routeEndLabel = document.getElementById("routeEndLabel");
const routeMode = document.getElementById("routeMode");
const routeNodes = Array.from(document.querySelectorAll(".graph-node[data-node]"));

const routePoints = {
  A: { x: 10, y: 58 },
  B: { x: 30, y: 24 },
  C: { x: 34, y: 76 },
  D: { x: 58, y: 28 },
  E: { x: 62, y: 72 },
  T: { x: 88, y: 48 }
};

const graphEdges = [
  ["A", "B", 4],
  ["A", "C", 5],
  ["B", "D", 5],
  ["C", "E", 7],
  ["D", "T", 5],
  ["E", "T", 6]
];

const graphAdjacency = {};
Object.keys(routePoints).forEach((node) => {
  graphAdjacency[node] = [];
});

graphEdges.forEach(([from, to, weight]) => {
  graphAdjacency[from].push({ node: to, weight });
  graphAdjacency[to].push({ node: from, weight });
});

const routeState = {
  start: null,
  end: null,
  selecting: "start"
};

function routeToPath(route) {
  return route.map((node, index) => `${index === 0 ? "M" : "L"}${routePoints[node].x} ${routePoints[node].y}`).join(" ");
}

function getEdgeWeight(from, to) {
  const edge = graphAdjacency[from].find((item) => item.node === to);
  return edge ? edge.weight : null;
}

function dijkstra(start, end) {
  const distances = {};
  const previous = {};
  const unvisited = new Set(Object.keys(graphAdjacency));

  Object.keys(graphAdjacency).forEach((node) => {
    distances[node] = Number.POSITIVE_INFINITY;
    previous[node] = null;
  });
  distances[start] = 0;

  while (unvisited.size > 0) {
    let current = null;
    let bestDistance = Number.POSITIVE_INFINITY;

    unvisited.forEach((node) => {
      if (distances[node] < bestDistance) {
        bestDistance = distances[node];
        current = node;
      }
    });

    if (!current || current === end) {
      break;
    }

    unvisited.delete(current);

    graphAdjacency[current].forEach(({ node, weight }) => {
      if (!unvisited.has(node)) {
        return;
      }

      const candidate = distances[current] + weight;
      if (candidate < distances[node]) {
        distances[node] = candidate;
        previous[node] = current;
      }
    });
  }

  const route = [];
  let cursor = end;

  while (cursor) {
    route.unshift(cursor);
    cursor = previous[cursor];
  }

  if (route[0] !== start) {
    return null;
  }

  return {
    route,
    length: distances[end]
  };
}

function renderRoute() {
  const hasPair = Boolean(routeState.start && routeState.end);
  const result = hasPair ? dijkstra(routeState.start, routeState.end) : null;

  routeNodes.forEach((node) => {
    const name = node.dataset.node;
    node.classList.toggle("route-start", name === routeState.start);
    node.classList.toggle("route-end", name === routeState.end);
    node.classList.toggle("active", result ? result.route.includes(name) : false);
  });

  routeStartLabel.textContent = routeState.start ?? "не выбран";
  routeEndLabel.textContent = routeState.end ?? "не выбран";
  routeMode.textContent = routeState.selecting === "start"
    ? "Выбери на карте стартовую вершину"
    : "Теперь выбери конечную вершину";

  if (!hasPair) {
    routePath.setAttribute("d", "");
    routePath.classList.remove("visible");
    routeBest.textContent = "Кратчайший путь: выбери две вершины";
    routeLength.textContent = "Вес: —";
    routeOption1.textContent = "Первый клик задает стартовую вершину.";
    routeOption2.textContent = "Второй клик задает финиш, после чего строится маршрут.";
    routeOption1.classList.add("active");
    routeOption2.classList.remove("active");
    return;
  }

  if (!result) {
    routePath.setAttribute("d", "");
    routePath.classList.remove("visible");
    routeBest.textContent = "Маршрут не найден";
    routeLength.textContent = "Вес: —";
    routeOption1.textContent = "В этом графе между выбранными вершинами нет пути.";
    routeOption2.textContent = "Выбери другую пару вершин.";
    routeOption1.classList.add("active");
    routeOption2.classList.remove("active");
    return;
  }

  const weights = result.route
    .slice(0, -1)
    .map((node, index) => getEdgeWeight(node, result.route[index + 1]))
    .join(" + ");

  routePath.setAttribute("d", routeToPath(result.route));
  routePath.classList.add("visible");
  routeBest.textContent = `Кратчайший путь: ${result.route.join("-")}`;
  routeLength.textContent = `Вес: ${result.length}`;
  routeOption1.innerHTML = `<strong>Маршрут</strong><br>${result.route.join(" → ")}`;
  routeOption2.innerHTML = `<strong>Сумма ребер</strong><br>${weights} = ${result.length}`;
  routeOption1.classList.add("active");
  routeOption2.classList.remove("active");
}

routeNodes.forEach((node) => {
  node.addEventListener("click", () => {
    const selected = node.dataset.node;

    if (routeState.selecting === "start") {
      routeState.start = selected;
      if (routeState.end === selected) {
        routeState.end = null;
      }
      routeState.selecting = "end";
    } else {
      if (selected === routeState.start) {
        routeState.start = null;
        routeState.end = null;
        routeState.selecting = "start";
      } else {
        routeState.end = selected;
        routeState.selecting = "start";
      }
    }

    if (!routeState.start && !routeState.end) {
      routeMode.textContent = "Выбор сброшен. Укажи старт заново.";
    } else {
      routeMode.textContent = routeState.selecting === "start"
        ? "Маршрут построен. Можешь выбрать новую пару."
        : "Теперь выбери конечную вершину";
    }

    renderRoute();
  });
  node.addEventListener("pointerdown", () => {
    node.focus();
  });
});

renderRoute();

const signalScenePicker = document.getElementById("signalScenePicker");
const signalFilterPicker = document.getElementById("signalFilterPicker");
const signalCanvas = document.getElementById("signalCanvas");
const recoveredCanvas = document.getElementById("recoveredCanvas");
const signalContext = signalCanvas.getContext("2d");
const recoveredContext = recoveredCanvas.getContext("2d");
const noiseRange = document.getElementById("noiseRange");
const noiseFill = document.getElementById("noiseFill");
const noiseInfo = document.getElementById("noiseInfo");
const snrInfo = document.getElementById("snrInfo");
const spectrumPeak = document.getElementById("spectrumPeak");
const filterInfo = document.getElementById("filterInfo");
const dspMath = document.getElementById("dspMath");
const dspExplain = document.getElementById("dspExplain");
const spectrumBars = document.getElementById("spectrumBars");

const dspState = {
  signal: "voice",
  filter: "none"
};

const dspScenes = {
  voice: {
    harmonics: [
      { amp: 1.0, freq: 2, phase: 0.2 },
      { amp: 0.55, freq: 5, phase: 0.8 },
      { amp: 0.25, freq: 8, phase: 1.6 }
    ],
    hint: "Речь обычно сосредоточена в нескольких низких и средних гармониках."
  },
  street: {
    harmonics: [
      { amp: 0.75, freq: 2, phase: 0.1 },
      { amp: 0.45, freq: 9, phase: 1.4 },
      { amp: 0.35, freq: 13, phase: 2.0 }
    ],
    hint: "Улица дает более широкий и неровный спектр: часть энергии уходит в высокие частоты."
  },
  music: {
    harmonics: [
      { amp: 0.9, freq: 3, phase: 0.3 },
      { amp: 0.7, freq: 6, phase: 1.1 },
      { amp: 0.5, freq: 10, phase: 2.2 }
    ],
    hint: "Музыка обычно имеет несколько выраженных пиков и более сложную структуру, чем речь."
  }
};

for (let k = 0; k < 16; k += 1) {
  const bar = document.createElement("div");
  bar.className = "spectrum-bar";
  bar.dataset.k = k;
  spectrumBars.appendChild(bar);
}

function buildSignalSamples() {
  const samples = [];
  const pure = [];
  const noiseLevel = Number(noiseRange.value) / 100;
  const harmonics = dspScenes[dspState.signal].harmonics;
  const N = 64;

  for (let n = 0; n < N; n += 1) {
    const t = n / N;
    const pureValue = harmonics.reduce(
      (sum, item) => sum + item.amp * Math.sin(2 * Math.PI * item.freq * t + item.phase),
      0
    );
    const noiseValue =
      noiseLevel *
      (0.55 * Math.sin(2 * Math.PI * 11 * t + 0.4) +
        0.35 * Math.sin(2 * Math.PI * 14 * t + 1.2) +
        0.25 * Math.sin(2 * Math.PI * 17 * t + 2.1));
    pure.push(pureValue);
    samples.push(pureValue + noiseValue);
  }

  return { pure, samples };
}

function dft(signal) {
  const N = signal.length;
  const spectrum = [];

  for (let k = 0; k < N; k += 1) {
    let re = 0;
    let im = 0;
    for (let n = 0; n < N; n += 1) {
      const angle = (-2 * Math.PI * k * n) / N;
      re += signal[n] * Math.cos(angle);
      im += signal[n] * Math.sin(angle);
    }
    spectrum.push({ re, im, mag: Math.sqrt(re ** 2 + im ** 2) });
  }

  return spectrum;
}

function idft(spectrum) {
  const N = spectrum.length;
  const signal = [];

  for (let n = 0; n < N; n += 1) {
    let value = 0;
    for (let k = 0; k < N; k += 1) {
      const angle = (2 * Math.PI * k * n) / N;
      value += spectrum[k].re * Math.cos(angle) - spectrum[k].im * Math.sin(angle);
    }
    signal.push(value / N);
  }

  return signal;
}

function buildMask(index) {
  if (dspState.filter === "none") {
    return 1;
  }
  if (dspState.filter === "low") {
    return index <= 5 || index >= 59 ? 1 : 0;
  }
  const mirrored = 64 - index;
  return (index >= 3 && index <= 8) || (mirrored >= 3 && mirrored <= 8) ? 1 : 0;
}

function drawSignal(context, canvas, values, color) {
  const { width, height } = canvas;
  context.clearRect(0, 0, width, height);
  context.lineWidth = 2;
  context.strokeStyle = "rgba(24, 34, 47, 0.08)";

  for (let y = 22; y < height; y += 32) {
    context.beginPath();
    context.moveTo(0, y);
    context.lineTo(width, y);
    context.stroke();
  }

  const maxAbs = Math.max(...values.map((value) => Math.abs(value)), 0.01);
  context.beginPath();
  context.lineWidth = 3.5;
  context.strokeStyle = color;

  values.forEach((value, index) => {
    const x = (index / (values.length - 1)) * width;
    const y = height / 2 - (value / maxAbs) * (height * 0.34);
    if (index === 0) {
      context.moveTo(x, y);
    } else {
      context.lineTo(x, y);
    }
  });

  context.stroke();
}

function updateNoise(value) {
  noiseFill.style.width = `${value}%`;
  const { pure, samples } = buildSignalSamples();
  const spectrum = dft(samples);
  const filteredSpectrum = spectrum.map((item, index) => {
    const mask = buildMask(index);
    return {
      re: item.re * mask,
      im: item.im * mask,
      mag: item.mag * mask
    };
  });
  const recovered = idft(filteredSpectrum);

  drawSignal(signalContext, signalCanvas, samples, "#f27347");
  drawSignal(recoveredContext, recoveredCanvas, recovered, "#5b8def");

  const bars = Array.from(spectrumBars.children);
  const magnitudes = spectrum.slice(0, 16).map((item) => item.mag);
  const maxMagnitude = Math.max(...magnitudes, 0.001);
  let peakIndex = 0;
  let peakValue = 0;

  bars.forEach((bar, index) => {
    const magnitude = magnitudes[index];
    const height = Math.max(8, (magnitude / maxMagnitude) * 138);
    bar.style.height = `${height}px`;
    bar.classList.toggle("filtered", buildMask(index) === 1);
    bar.classList.toggle("active", magnitude === Math.max(...magnitudes));
    if (magnitude > peakValue) {
      peakValue = magnitude;
      peakIndex = index;
    }
  });

  const signalPower = pure.reduce((sum, item) => sum + item ** 2, 0) / pure.length;
  const noisePower = samples.reduce((sum, item, index) => sum + (item - pure[index]) ** 2, 0) / samples.length;
  const snr = (10 * Math.log10(signalPower / Math.max(noisePower, 1e-6))).toFixed(1);

  noiseInfo.textContent = `Шум: ${value}%`;
  snrInfo.textContent = `SNR: ${snr} dB`;
  spectrumPeak.textContent = `Пик спектра: k = ${peakIndex}`;
  filterInfo.textContent =
    dspState.filter === "none"
      ? "Фильтр: пропускает все частоты"
      : dspState.filter === "low"
        ? "Фильтр: оставляет низкие гармоники"
        : "Фильтр: оставляет средний диапазон";

  dspMath.innerHTML =
    dspState.filter === "none"
      ? "\\[ M_k = 1 \\text{ для всех } k,\\quad \\hat X_k = X_k \\]"
      : dspState.filter === "low"
        ? "\\[ M_k = 1 \\text{ для низких } k,\\; M_k = 0 \\text{ иначе} \\]"
        : "\\[ M_k = 1 \\text{ в выбранной полосе},\\; M_k = 0 \\text{ вне ее} \\]";
  dspExplain.textContent = `${dspScenes[dspState.signal].hint} ${dspState.filter === "none" ? "Без фильтра спектр сохраняется целиком." : dspState.filter === "low" ? "Низкочастотная маска подавляет быстрые колебания и часть шума." : "Полосовой фильтр отсекает все вне выбранного диапазона, сохраняя только часть структуры сигнала."}`;
  typesetMath(dspMath);
}

noiseRange.addEventListener("input", (event) => updateNoise(Number(event.target.value)));
signalScenePicker.querySelectorAll(".vision-chip[data-signal]").forEach((button) => {
  button.addEventListener("click", () => {
    dspState.signal = button.dataset.signal;
    signalScenePicker.querySelectorAll(".vision-chip[data-signal]").forEach((chip) => {
      chip.classList.toggle("active", chip === button);
    });
    updateNoise(Number(noiseRange.value));
  });
});

signalFilterPicker.querySelectorAll(".vision-chip[data-filter]").forEach((button) => {
  button.addEventListener("click", () => {
    dspState.filter = button.dataset.filter;
    signalFilterPicker.querySelectorAll(".vision-chip[data-filter]").forEach((chip) => {
      chip.classList.toggle("active", chip === button);
    });
    updateNoise(Number(noiseRange.value));
  });
});

updateNoise(Number(noiseRange.value));

const faceRange = document.getElementById("faceRange");
const faceScenarioPicker = document.getElementById("faceScenarioPicker");
const referenceVector = document.getElementById("referenceVector");
const probeVector = document.getElementById("probeVector");
const faceDistance = document.getElementById("faceDistance");
const faceVerdict = document.getElementById("faceVerdict");
const faceScenarioInfo = document.getElementById("faceScenarioInfo");
const faceMath = document.getElementById("faceMath");
const faceExplain = document.getElementById("faceExplain");

const faceReference = [0.82, 0.41, 0.67, 0.29];
const faceScenarios = {
  direct: {
    label: "прямой взгляд",
    vector: [0.79, 0.44, 0.61, 0.26],
    explain: "Небольшое отклонение в embedding допустимо: это тот же человек при близких условиях съемки."
  },
  glasses: {
    label: "очки",
    vector: [0.75, 0.48, 0.56, 0.22],
    explain: "Очки и блики искажают часть локальных признаков, но embedding все еще должен оставаться рядом с эталоном."
  },
  angle: {
    label: "поворот головы",
    vector: [0.68, 0.51, 0.49, 0.18],
    explain: "Поворот увеличивает расстояние сильнее: системе нужно решить, считать ли это тем же человеком."
  },
  impostor: {
    label: "посторонний",
    vector: [0.31, 0.79, 0.22, 0.74],
    explain: "Embedding постороннего должен уходить далеко от эталона, иначе возрастает риск ложного допуска."
  }
};

const faceState = {
  scenario: "direct"
};

function renderFeatureVector(container, values) {
  container.innerHTML = values
    .map((value, index) => `
      <div class="feature-row">
        <span>f${index + 1}</span>
        <div class="feature-track" style="--feature-width:${(value * 100).toFixed(0)}%;"></div>
        <strong>${value.toFixed(2)}</strong>
      </div>
    `)
    .join("");
}

function updateFace(threshold) {
  const scenario = faceScenarios[faceState.scenario];
  const probe = scenario.vector;
  const distance = Math.sqrt(faceReference.reduce((sum, value, index) => sum + (value - probe[index]) ** 2, 0));
  const limit = threshold / 100;
  renderFeatureVector(referenceVector, faceReference);
  renderFeatureVector(probeVector, probe);
  faceDistance.textContent = `Расстояние: ${distance.toFixed(3)}`;
  faceVerdict.textContent = distance <= limit ? "Доступ разрешен" : "Доступ отклонен";
  faceScenarioInfo.textContent = `Сценарий: ${scenario.label}`;
  faceMath.innerHTML = `\\[ \\lVert z-z_0 \\rVert_2 = ${distance.toFixed(3)},\\quad \\tau = ${limit.toFixed(2)} \\]`;
  faceExplain.textContent = scenario.explain;
  typesetMath(faceMath);
}

faceRange.addEventListener("input", (event) => updateFace(Number(event.target.value)));
faceScenarioPicker.querySelectorAll(".vision-chip[data-face]").forEach((button) => {
  button.addEventListener("click", () => {
    faceState.scenario = button.dataset.face;
    faceScenarioPicker.querySelectorAll(".vision-chip[data-face]").forEach((chip) => {
      chip.classList.toggle("active", chip === button);
    });
    updateFace(Number(faceRange.value));
    updateEmbedding(Number(faceRange.value));
  });
});
updateFace(Number(faceRange.value));

const embedRing = document.getElementById("embedRing");
const embedRange = document.getElementById("embedRange");
const embedRadius = document.getElementById("embedRadius");
const embedCount = document.getElementById("embedCount");
const embedRule = document.getElementById("embedRule");
const embedPoints = Array.from(document.querySelectorAll(".embed-point"));
const faceFar = document.getElementById("faceFar");
const faceFrr = document.getElementById("faceFrr");
const embedPointInfo = document.getElementById("embedPointInfo");

function updateEmbedding(threshold) {
  const tau = threshold / 100;
  const radius = Math.round(22 + threshold * 0.55);
  embedRange.value = String(threshold);
  embedRing.style.width = `${radius * 2}px`;
  embedRing.style.height = `${radius * 2}px`;
  let count = 0;
  let genuineAccepted = 0;
  let genuineTotal = 0;
  let impostorAccepted = 0;
  let impostorTotal = 0;
  embedPoints.forEach((point) => {
    const accepted = Number(point.dataset.dist) <= tau;
    const kind = point.dataset.kind;
    point.classList.toggle("accepted", accepted);
    point.classList.toggle("rejected", !accepted);
    point.style.transform = accepted ? "translate(-50%, -50%) scale(1.08)" : "translate(-50%, -50%) scale(1)";
    if (accepted) {
      count += 1;
    }
    if (kind === "genuine") {
      genuineTotal += 1;
      if (accepted) genuineAccepted += 1;
    } else {
      impostorTotal += 1;
      if (accepted) impostorAccepted += 1;
    }
  });
  embedRadius.textContent = `Радиус принятия: ${radius}`;
  embedCount.textContent = `Принято: ${count} точек`;
  faceFar.textContent = `FAR: ${((impostorAccepted / impostorTotal) * 100).toFixed(0)}%`;
  faceFrr.textContent = `FRR: ${(((genuineTotal - genuineAccepted) / genuineTotal) * 100).toFixed(0)}%`;
  embedRule.innerHTML = `\\[ \\tau = ${tau.toFixed(2)},\\quad \\lVert z - z_0 \\rVert_2 < \\tau \\]`;
  typesetMath(embedRule);
}

faceRange.addEventListener("input", (event) => updateEmbedding(Number(event.target.value)));
embedRange.addEventListener("input", (event) => {
  const value = Number(event.target.value);
  faceRange.value = String(value);
  updateFace(value);
  updateEmbedding(value);
});

embedPoints.forEach((point) => {
  point.addEventListener("click", () => {
    const threshold = Number(embedRange.value) / 100;
    const dist = Number(point.dataset.dist);
    const accepted = dist <= threshold;
    embedPoints.forEach((item) => item.classList.remove("active-point"));
    point.classList.add("active-point");
    embedPointInfo.textContent = `${point.dataset.label}: dist = ${dist.toFixed(2)}. ${accepted ? "Точка попадает внутрь области принятия." : "Точка лежит вне области принятия."}`;
  });
});

updateEmbedding(Number(faceRange.value));

const recommendationPresetPicker = document.getElementById("recommendationPresetPicker");
const recommendationFeatures = document.getElementById("recommendationFeatures");
const recommendationScore = document.getElementById("recommendationScore");
const probabilityFill = document.getElementById("probabilityFill");
const probabilityText = document.getElementById("probabilityText");
const probabilityHint = document.getElementById("probabilityHint");
const recommendationTopFeature = document.getElementById("recommendationTopFeature");
const recommendationMath = document.getElementById("recommendationMath");
const recommendationExplain = document.getElementById("recommendationExplain");

const recommendationModel = {
  names: ["История", "Свежесть", "Контекст", "Вовлеченность"],
  weights: [1.2, 0.9, 0.65, 1.4],
  bias: -1.75,
  presets: {
    study: { label: "Учеба", values: [0.62, 0.74, 0.58, 0.43] },
    commute: { label: "Дорога", values: [0.46, 0.52, 0.88, 0.35] },
    night: { label: "Вечер", values: [0.71, 0.81, 0.42, 0.76] }
  }
};

let recommendationPreset = "study";

function sigmoid(x) {
  return 1 / (1 + Math.exp(-x));
}

function renderRecommendation() {
  const preset = recommendationModel.presets[recommendationPreset];
  const contributions = preset.values.map((value, index) => ({
    name: recommendationModel.names[index],
    value,
    contribution: value * recommendationModel.weights[index]
  }));
  const score = contributions.reduce((sum, item) => sum + item.contribution, recommendationModel.bias);
  const probability = sigmoid(score);
  const topContribution = contributions.reduce((best, item) =>
    Math.abs(item.contribution) > Math.abs(best.contribution) ? item : best
  );

  recommendationFeatures.innerHTML = contributions
    .map((item) => `
      <div class="feature-row">
        <span>${item.name}</span>
        <div class="feature-track" style="--feature-width:${(item.value * 100).toFixed(0)}%;"></div>
        <strong>${item.value.toFixed(2)}</strong>
      </div>
    `)
    .join("");

  recommendationScore.innerHTML = `\\(w^Tx+b = ${score.toFixed(2)}\\)`;
  probabilityFill.style.width = `${(probability * 100).toFixed(0)}%`;
  probabilityText.textContent = `Шанс рекомендации: ${(probability * 100).toFixed(0)}%`;
  probabilityHint.textContent =
    probability > 0.78 ? "Модель уверенно повышает приоритет рекомендации" :
    probability > 0.52 ? "Сигналов достаточно, но решение еще не жесткое" :
    "Признаки пока не тянут на высокий приоритет";
  recommendationTopFeature.textContent = `Главный вклад: ${topContribution.name.toLowerCase()}`;
  recommendationMath.innerHTML = `\\[ p = \\sigma(${contributions.map((item) => item.contribution.toFixed(2)).join(" + ")} ${recommendationModel.bias < 0 ? "-" : "+"} ${Math.abs(recommendationModel.bias).toFixed(2)}) = ${(probability).toFixed(2)} \\]`;
  recommendationExplain.textContent = `Сценарий "${preset.label.toLowerCase()}" дает разные значения признаков. После линейной комбинации модель пропускает скор через сигмоиду и получает вероятность показа или клика.`;
  recommendationPresetPicker.querySelectorAll(".vision-chip[data-preset]").forEach((button) => {
    button.classList.toggle("active", button.dataset.preset === recommendationPreset);
  });
  typesetMath(recommendationScore);
  typesetMath(recommendationMath);
}

recommendationPresetPicker.querySelectorAll(".vision-chip[data-preset]").forEach((button) => {
  button.addEventListener("click", () => {
    recommendationPreset = button.dataset.preset;
    renderRecommendation();
  });
});

renderRecommendation();

const gradientRange = document.getElementById("gradientRange");
const lossLine = document.getElementById("lossLine");
const gradientInfo = document.getElementById("gradientInfo");
const lossInfo = document.getElementById("lossInfo");
const gradientSteps = document.getElementById("gradientSteps");
const gradientMode = document.getElementById("gradientMode");

function updateGradient(value) {
  const alpha = value / 100;
  let weight = 4.2;
  const losses = [];

  for (let step = 0; step < 6; step += 1) {
    const loss = (weight - 1) ** 2 + 0.12;
    losses.push({ step, weight, loss });
    weight -= alpha * 2 * (weight - 1);
  }

  const maxLoss = Math.max(...losses.map((item) => item.loss));
  const minLoss = Math.min(...losses.map((item) => item.loss));
  const span = Math.max(maxLoss - minLoss, 0.001);
  const polygon = losses
    .map((item, index) => {
      const x = (index / (losses.length - 1)) * 100;
      const y = 12 + ((item.loss - minLoss) / span) * 68;
      return `${x}% ${y}%`;
    })
    .join(", ");

  lossLine.style.clipPath = `polygon(${polygon}, 100% 100%, 0% 100%)`;
  gradientInfo.textContent = `α = ${alpha.toFixed(2)}`;
  lossInfo.textContent = `Loss после 5 шагов: ${losses[5].loss.toFixed(3)}`;
  gradientMode.textContent =
    alpha < 0.18 ? "Режим: шаг слишком мал, обучение медленное" :
    alpha < 0.55 ? "Режим: устойчивое обучение" :
    "Режим: шаг велик, возможны колебания";
  gradientSteps.innerHTML = losses
    .slice(1)
    .map((item) => {
      const width = `${(item.loss / maxLoss) * 100}%`;
      return `
        <div class="gradient-step">
          <span>Шаг ${item.step}</span>
          <div class="gradient-bar" style="--loss-width:${width};"></div>
          <strong>${item.loss.toFixed(3)}</strong>
        </div>
      `;
    })
    .join("");
}

gradientRange.addEventListener("input", (event) => updateGradient(Number(event.target.value)));
updateGradient(Number(gradientRange.value));

const entropyRange = document.getElementById("entropyRange");
const charsetPicker = document.getElementById("charsetPicker");
const attackPicker = document.getElementById("attackPicker");
const entropyFill = document.getElementById("entropyFill");
const entropySpace = document.getElementById("entropySpace");
const entropyBits = document.getElementById("entropyBits");
const entropyLabel = document.getElementById("entropyLabel");
const entropyTime = document.getElementById("entropyTime");
const entropyMath = document.getElementById("entropyMath");
const entropyExplain = document.getElementById("entropyExplain");

const entropyCharsets = {
  digits: { size: 10, label: "цифры" },
  latin: { size: 52, label: "латиница" },
  full: { size: 94, label: "полный набор" }
};
const attackRates = {
  online: { rate: 1e3, label: "online-атака с ограничением запросов" },
  offline: { rate: 1e9, label: "offline-перебор по утекшему хешу" }
};

let entropyCharset = "digits";
let attackMode = "online";

function formatScientific(value) {
  const exponent = Math.floor(Math.log10(value));
  const mantissa = value / 10 ** exponent;
  return `${mantissa.toFixed(2)} \\cdot 10^{${exponent}}`;
}

function formatBruteforce(seconds) {
  if (seconds < 60) return `${seconds.toFixed(0)} с`;
  if (seconds < 3600) return `${(seconds / 60).toFixed(1)} мин`;
  if (seconds < 86400) return `${(seconds / 3600).toFixed(1)} ч`;
  if (seconds < 31536000) return `${(seconds / 86400).toFixed(1)} д`;
  if (seconds < 31536000 * 100) return `${(seconds / 31536000).toFixed(1)} лет`;
  return "астрономически долго";
}

function updateEntropy(length) {
  const charset = entropyCharsets[entropyCharset];
  const bits = Math.round(length * Math.log2(charset.size));
  const combinations = charset.size ** length;
  const bruteForceSeconds = combinations / attackRates[attackMode].rate;
  const percent = clamp(bits, 8, 100);
  entropyFill.style.width = `${percent}%`;
  entropySpace.innerHTML = `\\( ${formatScientific(combinations)} \\)`;
  entropyBits.textContent = `Энтропия: ${bits} бит`;
  entropyLabel.textContent =
    bits > 100 ? "Очень высокая стойкость" :
    bits > 72 ? "Высокая стойкость" :
    bits > 52 ? "Хорошая стойкость" :
    "Средняя стойкость";
  entropyTime.textContent = `Перебор: ${formatBruteforce(bruteForceSeconds)}`;
  entropyMath.innerHTML = `\\[ H \\approx ${length} \\cdot \\log_2 ${charset.size} = ${bits} \\]`;
  entropyExplain.textContent = `Алфавит "${charset.label}" дает ${charset.size} вариантов на позицию. При длине ${length} пространство перебора равно ${charset.size}^{${length}}, а время атаки дополнительно зависит от сценария: ${attackRates[attackMode].label}.`;
  typesetMath(entropySpace);
  typesetMath(entropyMath);
}

entropyRange.addEventListener("input", (event) => updateEntropy(Number(event.target.value)));
charsetPicker.querySelectorAll(".vision-chip[data-charset]").forEach((button) => {
  button.addEventListener("click", () => {
    entropyCharset = button.dataset.charset;
    charsetPicker.querySelectorAll(".vision-chip[data-charset]").forEach((chip) => {
      chip.classList.toggle("active", chip === button);
    });
    updateEntropy(Number(entropyRange.value));
  });
});
attackPicker.querySelectorAll(".vision-chip[data-attack]").forEach((button) => {
  button.addEventListener("click", () => {
    attackMode = button.dataset.attack;
    attackPicker.querySelectorAll(".vision-chip[data-attack]").forEach((chip) => {
      chip.classList.toggle("active", chip === button);
    });
    updateEntropy(Number(entropyRange.value));
  });
});
updateEntropy(Number(entropyRange.value));

const modValue = document.getElementById("modValue");
const rsaPublic = document.getElementById("rsaPublic");
const rsaPrivate = document.getElementById("rsaPrivate");
const rsaMessageInfo = document.getElementById("rsaMessageInfo");
const rsaStepMessage = document.getElementById("rsaStepMessage");
const rsaStepCipher = document.getElementById("rsaStepCipher");
const rsaStepDecode = document.getElementById("rsaStepDecode");
const modResult = document.getElementById("modResult");
const modInfo = document.getElementById("modInfo");
const modCycles = document.getElementById("modCycles");
const modRing = document.getElementById("modRing");
const rsaCheck = document.getElementById("rsaCheck");
const rsaMath = document.getElementById("rsaMath");
const rsaExplain = document.getElementById("rsaExplain");

const rsaMessages = [2, 4, 5, 7, 8, 10, 13, 14, 16, 17, 19, 20, 23, 25, 26, 28, 29, 31];
const rsaParams = { e: 3, d: 7, n: 33 };

function modPow(base, exponent, mod) {
  let result = 1;
  let value = base % mod;
  let power = exponent;
  while (power > 0) {
    if (power % 2 === 1) {
      result = (result * value) % mod;
    }
    value = (value * value) % mod;
    power = Math.floor(power / 2);
  }
  return result;
}

function updateModulo() {
  const message = rsaMessages[Number(modValue.value)];
  const cipher = modPow(message, rsaParams.e, rsaParams.n);
  const decoded = modPow(cipher, rsaParams.d, rsaParams.n);
  rsaPublic.innerHTML = `\\((e,n)=(${rsaParams.e},${rsaParams.n})\\)`;
  rsaPrivate.innerHTML = `\\((d,n)=(${rsaParams.d},${rsaParams.n})\\)`;
  rsaMessageInfo.innerHTML = `Текущее сообщение: \\(m = ${message}\\)`;
  rsaStepMessage.innerHTML = `Сообщение: \\(m = ${message}\\)`;
  rsaStepCipher.innerHTML = `Шифрование: \\(c = ${cipher}\\)`;
  rsaStepDecode.innerHTML = `Расшифровка: \\(m' = ${decoded}\\)`;
  modResult.innerHTML = `\\[ c = ${message}^{${rsaParams.e}} \\bmod ${rsaParams.n} = ${cipher} \\]`;
  modInfo.textContent = `Шифртекст: ${cipher}`;
  modCycles.textContent = `Расшифровка: ${decoded}`;
  rsaCheck.innerHTML = `\\( m' = ${decoded} ${decoded === message ? "=" : "\\neq"} m \\)`;
  rsaMath.innerHTML = `\\[ c = m^e \\bmod n,\\quad m' = c^d \\bmod n \\]`;
  rsaExplain.textContent = `Для сообщения m = ${message} шифрование переносит число в другой класс вычетов. Закрытый ключ возвращает исходное значение, потому что степени e и d согласованы с арифметикой по модулю ${rsaParams.n}.`;
  modRing.innerHTML = `<div class="mod-core">mod ${rsaParams.n}</div>`;

  for (let index = 0; index < rsaParams.n; index += 1) {
    const angle = (Math.PI * 2 * index) / rsaParams.n - Math.PI / 2;
    const x = 50 + Math.cos(angle) * 42;
    const y = 50 + Math.sin(angle) * 42;
    const node = document.createElement("div");
    const classes = ["mod-node"];
    if (index === message) classes.push("source");
    if (index === cipher) classes.push("cipher");
    if (index === message || index === cipher) classes.push("active");
    node.className = classes.join(" ");
    node.textContent = index;
    node.style.left = `${x}%`;
    node.style.top = `${y}%`;
    modRing.appendChild(node);
  }

  typesetMath(rsaPublic);
  typesetMath(rsaPrivate);
  typesetMath(rsaMessageInfo);
  typesetMath(rsaStepMessage);
  typesetMath(rsaStepCipher);
  typesetMath(rsaStepDecode);
  typesetMath(rsaCheck);
  typesetMath(rsaMath);
  typesetMath(modResult);
}

modValue.addEventListener("input", updateModulo);
updateModulo();

const quizCard = document.getElementById("quizCard");
const quizStep = document.getElementById("quizStep");
const quizQuestion = document.getElementById("quizQuestion");
const quizFormula = document.getElementById("quizFormula");
const quizOptions = document.getElementById("quizOptions");
const quizFeedback = document.getElementById("quizFeedback");
const nextQuestion = document.getElementById("nextQuestion");
const quizResult = document.getElementById("quizResult");

const quizData = [
  {
    question: "Какое выражение соответствует геометрической модели расстояния до базовой станции?",
    formula: "\\[ d_i = \\sqrt{(x-x_i)^2 + (y-y_i)^2} \\]",
    options: [
      { id: "a", text: "x = y + 1" },
      { id: "b", text: "\\( d_i = \\sqrt{(x-x_i)^2 + (y-y_i)^2} \\)" },
      { id: "c", text: "sin x + cos y = 0" }
    ],
    answer: "b",
    explanation: "Это стандартная евклидова метрика на плоскости."
  },
  {
    question: "Какая операция особенно важна для обработки изображения и выделения деталей?",
    formula: "\\[ (I * K)(u,v) = \\sum_i \\sum_j I(u-i,v-j)K(i,j) \\]",
    options: [
      { id: "a", text: "Свертка с ядром фильтра" },
      { id: "b", text: "Случайная перестановка пикселей" },
      { id: "c", text: "Простая нумерация кадров" }
    ],
    answer: "a",
    explanation: "Свертка лежит в основе фильтрации, размытия, резкости и многих CNN-операций."
  },
  {
    question: "Что дает переход сигнала в частотную область?",
    formula: "\\[ X_k = \\sum_{n=0}^{N-1} x_n e^{-2\\pi i k n / N} \\]",
    options: [
      { id: "a", text: "Удаляет информацию о звуке" },
      { id: "b", text: "Делает изображение резче" },
      { id: "c", text: "Выделяет гармоники и частотный состав" }
    ],
    answer: "c",
    explanation: "Именно частотное представление позволяет анализировать спектр сигнала."
  },
  {
    question: "Почему Face ID удобно описывать через расстояние между векторами признаков?",
    formula: "\\[ \\operatorname{dist}(z_1, z_2) = \\lVert z_1 - z_2 \\rVert_2 \\]",
    options: [
      { id: "a", text: "Потому что телефон хранит только фотографию без обработки" },
      { id: "b", text: "Потому что изображение переводится в пространство признаков" },
      { id: "c", text: "Потому что система измеряет только яркость экрана" }
    ],
    answer: "b",
    explanation: "Сравниваются не сами пиксели, а компактные векторы признаков."
  },
  {
    question: "Что означает высокая энтропия пароля?",
    formula: "\\[ H = \\log_2 N \\]",
    options: [
      { id: "a", text: "Большое пространство комбинаций и более сложный перебор" },
      { id: "b", text: "Пароль вводится быстрее" },
      { id: "c", text: "Экран становится ярче" }
    ],
    answer: "a",
    explanation: "Энтропия измеряет информационную неопределенность и размер пространства перебора."
  },
  {
    question: "Что означает шаг градиентного спуска в модели рекомендаций?",
    formula: "\\[ w_{t+1} = w_t - \\alpha \\nabla L \\]",
    options: [
      { id: "a", text: "Случайно менять все коэффициенты" },
      { id: "b", text: "Всегда увеличивать веса" },
      { id: "c", text: "Двигать параметры в сторону уменьшения ошибки" }
    ],
    answer: "c",
    explanation: "Градиент указывает направление наибольшего роста, поэтому идем в противоположную сторону."
  }
];

let currentQuestionIndex = 0;
let currentScore = 0;
let questionAnswered = false;

function pulseCard(card) {
  card.classList.remove("answered");
  void card.offsetWidth;
  card.classList.add("answered");
}

function typesetMath(target) {
  if (window.MathJax && window.MathJax.typesetPromise) {
    window.MathJax.typesetClear?.([target]);
    window.MathJax.typesetPromise([target]).catch(() => {});
  } else {
    window.setTimeout(() => {
      if (window.MathJax && window.MathJax.typesetPromise) {
        window.MathJax.typesetClear?.([target]);
        window.MathJax.typesetPromise([target]).catch(() => {});
      }
    }, 300);
  }
}

function renderQuizQuestion() {
  const item = quizData[currentQuestionIndex];
  questionAnswered = false;
  nextQuestion.disabled = true;
  quizFeedback.className = "quiz-feedback";
  quizFeedback.textContent = "Выбери вариант ответа.";
  quizStep.textContent = String(currentQuestionIndex + 1).padStart(2, "0");
  quizQuestion.textContent = item.question;
  quizFormula.textContent = "";
  quizFormula.innerHTML = item.formula;
  quizOptions.innerHTML = "";

  item.options.forEach((option) => {
    const button = document.createElement("button");
    button.className = "quiz-option";
    button.innerHTML = option.text;
    button.type = "button";
    button.dataset.choice = option.id;
    button.addEventListener("click", () => handleQuizAnswer(option.id));
    quizOptions.appendChild(button);
  });

  quizResult.textContent = `Текущий счет: ${currentScore} / ${currentQuestionIndex}`;
  pulseCard(quizCard);
  typesetMath(quizFormula);
  typesetMath(quizOptions);
}

function handleQuizAnswer(choice) {
  if (questionAnswered) {
    return;
  }

  questionAnswered = true;
  const item = quizData[currentQuestionIndex];
  const buttons = Array.from(quizOptions.querySelectorAll(".quiz-option"));

  buttons.forEach((button) => {
    const isCorrect = button.dataset.choice === item.answer;
    const isSelected = button.dataset.choice === choice;

    if (isCorrect) {
      button.classList.add("correct");
    }
    if (isSelected && !isCorrect) {
      button.classList.add("wrong");
    }
    if (isSelected) {
      button.classList.add("selected");
    }
    button.disabled = true;
  });

  if (choice === item.answer) {
    currentScore += 1;
    quizFeedback.classList.add("correct");
    quizFeedback.textContent = `Верно. ${item.explanation}`;
  } else {
    quizFeedback.classList.add("wrong");
    quizFeedback.textContent = `Неверно. ${item.explanation}`;
  }

  quizResult.textContent = `Текущий счет: ${currentScore} / ${currentQuestionIndex + 1}`;
  nextQuestion.disabled = false;
  nextQuestion.textContent =
    currentQuestionIndex === quizData.length - 1 ? "Завершить" : "Следующий вопрос";
  pulseCard(quizCard);
}

nextQuestion.addEventListener("click", () => {
  if (currentQuestionIndex < quizData.length - 1) {
    currentQuestionIndex += 1;
    renderQuizQuestion();
  } else {
    quizStep.textContent = "OK";
    quizQuestion.textContent = "Квиз завершен";
    quizFormula.innerHTML = "";
    quizOptions.innerHTML = "";
    quizFeedback.className = "quiz-feedback correct";
    quizFeedback.textContent =
      `Финальный результат: ${currentScore} из ${quizData.length}. ` +
      (currentScore === quizData.length
        ? "Отличный уровень понимания."
        : currentScore >= 4
          ? "Хороший результат, основные идеи усвоены."
          : "Стоит еще раз пройти сцены и формулы.");
    nextQuestion.disabled = true;
    quizResult.textContent = `Итог: ${currentScore} / ${quizData.length}`;
    pulseCard(quizCard);
  }
});

renderQuizQuestion();
