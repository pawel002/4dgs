// Shared procedural "synthetic room" point generator used by the demos.
// Mirrors the spirit of the thesis' Blender room: floor, three walls,
// rug, sofa, table, wall pictures, plant. Y is up (three.js convention).
export function buildRoomPoints(density = 520) {
  const pos = [];
  const col = [];

  const rnd = mulberry32(1337);

  function push(x, y, z, r, g, b) {
    pos.push(x, y, z);
    col.push(r, g, b);
  }

  function jitter(scale) {
    return (rnd() - 0.5) * scale;
  }

  // Sample a quad: origin o, edge vectors u, v. shade(px, py) -> [r,g,b]
  function quad(o, u, v, shade, dens = density) {
    const area =
      Math.hypot(u[0], u[1], u[2]) * Math.hypot(v[0], v[1], v[2]);
    const n = Math.floor(area * dens);
    for (let i = 0; i < n; i++) {
      const a = rnd(), b = rnd();
      const x = o[0] + a * u[0] + b * v[0];
      const y = o[1] + a * u[1] + b * v[1];
      const z = o[2] + a * u[2] + b * v[2];
      const c = shade(a, b, x, y, z);
      push(x, y, z, c[0], c[1], c[2]);
    }
  }

  function flat(r, g, b, noise = 0.04) {
    return () => [r + jitter(noise), g + jitter(noise), b + jitter(noise)];
  }

  // Axis-aligned box sampled on its faces.
  function box(min, max, shade, dens = density) {
    const [x0, y0, z0] = min, [x1, y1, z1] = max;
    const dx = x1 - x0, dy = y1 - y0, dz = z1 - z0;
    quad([x0, y1, z0], [dx, 0, 0], [0, 0, dz], shade, dens);          // top
    quad([x0, y0, z1], [dx, 0, 0], [0, dy, 0], shade, dens);          // front
    quad([x0, y0, z0], [dx, 0, 0], [0, dy, 0], shade, dens * 0.6);    // back
    quad([x0, y0, z0], [0, 0, dz], [0, dy, 0], shade, dens * 0.6);    // left
    quad([x1, y0, z0], [0, 0, dz], [0, dy, 0], shade, dens * 0.6);    // right
  }

  const W = 4, D = 3, H = 3; // half-width x, half-depth z, height

  // Floor: wood planks
  quad([-W, 0, -D], [2 * W, 0, 0], [0, 0, 2 * D], (a, b, x, _, z) => {
    const plank = Math.floor((x + W) / 0.65) % 2 === 0 ? 0.0 : 0.05;
    const t = 0.55 + plank + jitter(0.05);
    return [t, t * 0.78, t * 0.55];
  });

  // Walls (back z=-D, left x=-W, right x=+W)
  quad([-W, 0, -D], [2 * W, 0, 0], [0, H, 0], flat(0.82, 0.78, 0.70));
  quad([-W, 0, -D], [0, 0, 2 * D], [0, H, 0], flat(0.76, 0.72, 0.64));
  quad([W, 0, -D], [0, 0, 2 * D], [0, H, 0], flat(0.76, 0.72, 0.64));

  // Skirting boards
  quad([-W, 0, -D + 0.001], [2 * W, 0, 0], [0, 0.12, 0], flat(0.95, 0.95, 0.93), density * 2);

  // Rug: two-tone disc
  {
    const cx = 0.5, cz = 0.7, R = 1.55;
    const n = Math.floor(Math.PI * R * R * density * 1.2);
    for (let i = 0; i < n; i++) {
      const r = R * Math.sqrt(rnd());
      const th = rnd() * Math.PI * 2;
      const ring = r > R * 0.82 || r < R * 0.3 ? 0.12 : 0;
      push(
        cx + r * Math.cos(th), 0.015, cz + r * Math.sin(th),
        0.45 + ring + jitter(0.04), 0.16 + ring + jitter(0.03), 0.20 + ring
      );
    }
  }

  // Sofa: seat + backrest + armrests
  box([-3.5, 0.12, -2.85], [-1.5, 0.62, -1.95], flat(0.24, 0.38, 0.58));
  box([-3.5, 0.62, -2.85], [-1.5, 1.25, -2.55], flat(0.21, 0.34, 0.53));
  box([-3.62, 0.12, -2.85], [-3.5, 0.85, -1.95], flat(0.19, 0.31, 0.49));
  box([-1.5, 0.12, -2.85], [-1.38, 0.85, -1.95], flat(0.19, 0.31, 0.49));

  // Coffee table: top + 4 legs
  box([1.2, 0.46, 0.2], [2.5, 0.52, 1.05], flat(0.42, 0.27, 0.15));
  for (const [lx, lz] of [[1.27, 0.27], [2.43, 0.27], [1.27, 0.98], [2.43, 0.98]]) {
    box([lx - 0.035, 0, lz - 0.035], [lx + 0.035, 0.46, lz + 0.035], flat(0.32, 0.20, 0.11), density * 1.5);
  }

  // Pictures on the back wall (the thesis room hangs many photos for texture detail)
  const pics = [
    { x: -3.1, w: 1.0, y: 1.45, h: 0.75, c: [0.75, 0.40, 0.22] },
    { x: -1.6, w: 0.8, y: 1.65, h: 1.0, c: [0.25, 0.55, 0.45] },
    { x: -0.3, w: 1.2, y: 1.40, h: 0.8, c: [0.55, 0.32, 0.62] },
    { x: 1.3, w: 0.7, y: 1.75, h: 0.55, c: [0.85, 0.70, 0.30] },
    { x: 2.4, w: 1.1, y: 1.35, h: 0.9, c: [0.30, 0.45, 0.70] },
  ];
  for (const p of pics) {
    // frame
    quad([p.x - 0.04, p.y - 0.04, -D + 0.02], [p.w + 0.08, 0, 0], [0, p.h + 0.08, 0],
      flat(0.12, 0.10, 0.09), density * 1.6);
    // canvas with simple gradient pattern
    quad([p.x, p.y, -D + 0.03], [p.w, 0, 0], [0, p.h, 0], (a, b) => [
      p.c[0] * (0.55 + 0.45 * a) + jitter(0.06),
      p.c[1] * (0.55 + 0.45 * b) + jitter(0.06),
      p.c[2] * (0.55 + 0.45 * (1 - a)) + jitter(0.06),
    ], density * 2.5);
  }

  // One picture on each side wall
  quad([-W + 0.02, 1.3, -1.2], [0, 0, 1.4], [0, 0.9, 0], (a, b) =>
    [0.2 + 0.5 * a + jitter(0.05), 0.3 + jitter(0.05), 0.45 + 0.3 * b], density * 2);
  quad([W - 0.02, 1.45, -0.2], [0, 0, 1.1], [0, 0.7, 0], (a, b) =>
    [0.7 - 0.3 * b + jitter(0.05), 0.5 + jitter(0.05), 0.25 + 0.3 * a], density * 2);

  // Plant in the corner: pot + foliage cone
  box([3.15, 0, -2.75], [3.65, 0.4, -2.25], flat(0.55, 0.30, 0.18));
  {
    const n = 2600;
    for (let i = 0; i < n; i++) {
      const h = rnd();
      const r = 0.55 * (1 - h * 0.75) * Math.sqrt(rnd());
      const th = rnd() * Math.PI * 2;
      push(3.4 + r * Math.cos(th), 0.4 + h * 1.15, -2.5 + r * Math.sin(th),
        0.12 + jitter(0.06), 0.42 + h * 0.25 + jitter(0.08), 0.15 + jitter(0.05));
    }
  }

  return {
    positions: new Float32Array(pos),
    colors: new Float32Array(col),
    count: pos.length / 3,
  };
}

// Camera positions along the thesis' parametric trajectory:
// x(t) = rx sin(2*pi*t/T), y(t) = ry cos(2*pi*t/T)  (ground plane)
// z(t) = rz sin(w*t)                                  (height oscillation)
// Mapped to three.js: ground = xz plane, height = y.
export function trajectoryPoint(t, { rx = 3.1, rz = 2.1, ry = 0.55, omega = 9, base = 1.5 } = {}) {
  return [
    rx * Math.sin(2 * Math.PI * t),
    base + ry * Math.sin(omega * 2 * Math.PI * t),
    rz * Math.cos(2 * Math.PI * t),
  ];
}

// Deterministic PRNG so every reload yields the identical room.
export function mulberry32(seed) {
  let a = seed >>> 0;
  return function () {
    a |= 0; a = (a + 0x6D2B79F5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

// Build merged line-segment geometry for K red camera frusta (COLMAP style).
export function frustaSegments(THREE, K = 40, opts = {}) {
  const verts = [];
  const w = 0.16, h = 0.11, d = 0.22;
  const corners = [
    [-w, -h, d], [w, -h, d], [w, h, d], [-w, h, d],
  ];
  const m = new THREE.Matrix4();
  const eye = new THREE.Vector3(), target = new THREE.Vector3(0, 1.0, 0), up = new THREE.Vector3(0, 1, 0);
  for (let k = 0; k < K; k++) {
    const t = k / K;
    const [x, y, z] = trajectoryPoint(t, opts);
    eye.set(x, y, z);
    m.lookAt(eye, target, up);
    const tip = [x, y, z];
    const cs = corners.map((c) => {
      const v = new THREE.Vector3(c[0], c[1], -c[2]).applyMatrix4(m);
      return [v.x + x, v.y + y, v.z + z];
    });
    for (let i = 0; i < 4; i++) {
      verts.push(...tip, ...cs[i]);                 // apex -> corner
      verts.push(...cs[i], ...cs[(i + 1) % 4]);     // base edge
    }
  }
  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.BufferAttribute(new Float32Array(verts), 3));
  return g;
}
