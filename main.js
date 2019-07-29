(window["webpackJsonp"] = window["webpackJsonp"] || []).push([["main"],{

/***/ "./src/$$_lazy_route_resource lazy recursive":
/*!**********************************************************!*\
  !*** ./src/$$_lazy_route_resource lazy namespace object ***!
  \**********************************************************/
/*! no static exports found */
/***/ (function(module, exports) {

function webpackEmptyAsyncContext(req) {
	// Here Promise.resolve().then() is used instead of new Promise() to prevent
	// uncaught exception popping up in devtools
	return Promise.resolve().then(function() {
		var e = new Error("Cannot find module '" + req + "'");
		e.code = 'MODULE_NOT_FOUND';
		throw e;
	});
}
webpackEmptyAsyncContext.keys = function() { return []; };
webpackEmptyAsyncContext.resolve = webpackEmptyAsyncContext;
module.exports = webpackEmptyAsyncContext;
webpackEmptyAsyncContext.id = "./src/$$_lazy_route_resource lazy recursive";

/***/ }),

/***/ "./src/app/animation/wind/windy.js":
/*!*****************************************!*\
  !*** ./src/app/animation/wind/windy.js ***!
  \*****************************************/
/*! no static exports found */
/***/ (function(module, exports) {

function createShader(gl, type, source) {
  var shader = gl.createShader(type);
  gl.shaderSource(shader, source);
  gl.compileShader(shader);
  if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
    throw new Error(gl.getShaderInfoLog(shader));
  }
  return shader;
}

function createProgram(gl, vertexSource, fragmentSource) {
  var program = gl.createProgram();
  var vertexShader = createShader(gl, gl.VERTEX_SHADER, vertexSource);
  var fragmentShader = createShader(gl, gl.FRAGMENT_SHADER, fragmentSource);
  gl.attachShader(program, vertexShader);
  gl.attachShader(program, fragmentShader);
  gl.linkProgram(program);
  if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
    throw new Error(gl.getProgramInfoLog(program));
  }
  var wrapper = {program: program};
  var numAttributes = gl.getProgramParameter(program, gl.ACTIVE_ATTRIBUTES);
  for (var i = 0; i < numAttributes; i++) {
    var attribute = gl.getActiveAttrib(program, i);
    wrapper[attribute.name] = gl.getAttribLocation(program, attribute.name);
  }
  var numUniforms = gl.getProgramParameter(program, gl.ACTIVE_UNIFORMS);
  for (var i$1 = 0; i$1 < numUniforms; i$1++) {
    var uniform = gl.getActiveUniform(program, i$1);
    wrapper[uniform.name] = gl.getUniformLocation(program, uniform.name);
  }

  return wrapper;
}

function createTexture(gl, filter, data, width, height) {
  var texture = gl.createTexture();
  gl.bindTexture(gl.TEXTURE_2D, texture);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, filter);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, filter);
  if (data instanceof Uint8Array) {
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, width, height, 0, gl.RGBA, gl.UNSIGNED_BYTE, data);
  } else {
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, data);
  }
  gl.bindTexture(gl.TEXTURE_2D, null);
  return texture;
}

function bindTexture(gl, texture, unit) {
  gl.activeTexture(gl.TEXTURE0 + unit);
  gl.bindTexture(gl.TEXTURE_2D, texture);
}

function createBuffer(gl, data) {
  var buffer = gl.createBuffer();
  gl.bindBuffer(gl.ARRAY_BUFFER, buffer);
  gl.bufferData(gl.ARRAY_BUFFER, data, gl.STATIC_DRAW);
  return buffer;
}

function bindAttribute(gl, buffer, attribute, numComponents) {
  gl.bindBuffer(gl.ARRAY_BUFFER, buffer);
  gl.enableVertexAttribArray(attribute);
  gl.vertexAttribPointer(attribute, numComponents, gl.FLOAT, false, 0, 0);
}

function bindFramebuffer(gl, framebuffer, texture) {
  gl.bindFramebuffer(gl.FRAMEBUFFER, framebuffer);
  if (texture) {
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, texture, 0);
  }
}

var drawVert = "precision mediump float;\n\n" +
  "attribute float a_index;\n\n" +
  "uniform sampler2D u_particles;\n" +
  "uniform float u_particles_res;\n\n" +
  "varying vec2 v_particle_pos;\n\n" +
  "void main() {\n" +
  "    vec4 color = texture2D(u_particles, vec2(\n" +
  "        fract(a_index / u_particles_res),\n" +
  "        floor(a_index / u_particles_res) / u_particles_res));\n\n" +
  "    // decode current particle position from the pixel's RGBA value\n" +
  "    v_particle_pos = vec2(\n" +
  "        color.r / 255.0 + color.b,\n" +
  "        color.g / 255.0 + color.a);\n\n" +
  "    gl_PointSize = 1.5;\n" +
  "    gl_Position = vec4(2.0 * v_particle_pos.x - 1.0, 1.0 - 2.0 * v_particle_pos.y, 0, 1);\n" +
  "}\n";

// for (let vi = 0; vi < nbVortices; ++vi) {
//   let vp = vortices[vi];
//   // Distance to current vortex
//   let delta0 = vp[0] - xp;
//   let delta1 = vp[1] - yp;
//   let d2 = delta0 * delta0 + delta1 * delta1;
//   // Extinction factor
//   let extinction = Math.exp(-d2 / (vp[2] * gridScale));
//   mean[0] += extinction * delta1 * vp[3];
//   mean[1] += extinction * (-delta0 * vp[3]);
// }

var drawFrag = "precision mediump float;\n\n" +
  "uniform mat4 luc;\n" +
  "uniform vec4 lucQ;\n" +
  "uniform vec4 lucL;\n" +
  "uniform sampler2D u_wind;\n" +
  "uniform sampler2D u_conv_texture;\n" +
  "uniform vec2 u_wind_res;\n" +
  "uniform vec2 u_wind_min;\n" +
  "uniform vec2 u_wind_max;\n" +
  "uniform sampler2D u_color_ramp;\n\n" +
  "varying vec2 v_particle_pos;\n\n" +
  "void main() {\n" +
  "    float facX = u_wind_res.x > u_wind_res.y ? u_wind_res.x / u_wind_res.y : 1.0;\n" +
  "    float facY = u_wind_res.x > u_wind_res.y ? 1.0 : u_wind_res.y / u_wind_res.x;\n" +
  "    \n" +
  "    // compute for 8 tracers\n" +
  "    vec2 vel = vec2(0.0, 0.0);\n" +
  "    float gamma = 0.01;\n" +
  "    float epsilon = 2.0;\n" +
  "    for (int i = 0; i < 4; ++i) {\n" +
  "        float w1 = lucQ[i];\n" +
  "        float delta0 = facX * (luc[i][0] - v_particle_pos[0]);\n" +
  "        float delta1 = facY * (luc[i][1] - v_particle_pos[1]);\n" +
  "        float d2 = delta0 * delta0 + delta1 * delta1;\n" +
  "        float extinction = exp(-d2 / gamma);\n" +
  "        vel[0] += extinction * delta1 * epsilon * w1;\n" +
  "        vel[1] += extinction * (delta0 * epsilon * w1);\n" +
  "\n" +
  "        w1 = lucL[i];\n" +
  "        delta0 = facX * (luc[i][2] - v_particle_pos[0]);\n" +
  "        delta1 = facY * (luc[i][3] - v_particle_pos[1]);\n" +
  "        d2 = delta0 * delta0 + delta1 * delta1;\n" +
  "        extinction = exp(-d2 / gamma);\n" +
  "        vel[0] += extinction * delta1 * epsilon * w1;\n" +
  "        vel[1] += extinction * (delta0 * epsilon * w1);\n" +
  "    \n" +
  "    }\n" +
  "    vel[0] += 0.05;\n" +
  "    vec2 velocity = mix(u_wind_min, u_wind_max, vel);\n" +
  // "    vec2 velocity = mix(u_wind_min, u_wind_max, texture2D(u_wind, v_particle_pos).rg);\n" +
  "    float speed_t = length(velocity) / length(u_wind_max);\n\n" +
  "    vec3 s2 = mix(vec3(0.0), vec3(1.0), texture2D(u_conv_texture, v_particle_pos).rgb);\n\n" +
  "    float speed2 = s2.r + s2.g + s2.b;\n\n" +
  "    // color ramp is encoded in a 16x16 texture\n" +
  "    vec2 ramp_pos = vec2(\n" +
  "        fract(16.0 * speed2),\n" +
  "        floor(16.0 * speed2) / 16.0);\n\n" +
  "    vec4 outputColor = texture2D(u_color_ramp, ramp_pos);\n" +
  "    float powSpeedT = 10.0 * speed_t;\n" +
  "    outputColor[0] = max(powSpeedT, outputColor[0]);\n" +
  // "    outputColor[1] = powSpeedT;\n" +
  "    gl_FragColor = outputColor;\n" +
  "}\n";

var quadVert = "precision mediump float;\n\n" +
  "attribute vec2 a_pos;\n\n" +
  "varying vec2 v_tex_pos;\n\n" +
  "void main() {\n" +
  "    v_tex_pos = a_pos;\n" +
  "    gl_Position = vec4(1.0 - 2.0 * a_pos, 0, 1);\n" +
  "}\n";

var screenFrag = "precision mediump float;\n\n" +
  "uniform sampler2D u_screen;\n" +
  "uniform float u_opacity;\n\n" +
  "varying vec2 v_tex_pos;\n\n" +
  "void main() {\n" +
  "    vec4 color = texture2D(u_screen, 1.0 - v_tex_pos);\n" +
  "    // a hack to guarantee opacity fade out even with a value close to 1.0\n" +
  "    gl_FragColor = vec4(floor(255.0 * color * u_opacity) / 255.0);\n" +
  "}\n";

var updateFrag = "precision highp float;\n\n" +
  "uniform sampler2D u_particles;\n" +
  "uniform sampler2D u_wind;\n" +
  "uniform sampler2D u_conv_texture;\n" +
  "uniform vec4 lucQ;\n" +
  "uniform vec4 lucL;\n" +
  "uniform mat4 luc;\n" +
  "uniform vec2 u_wind_res;\n" +
  "uniform vec2 u_wind_min;\n" +
  "uniform vec2 u_wind_max;\n" +
  "uniform float u_rand_seed;\n" +
  "uniform float u_speed_factor;\n" +
  "uniform float u_drop_rate;\n" +
  "uniform float u_drop_rate_bump;\n\n" +
  "varying vec2 v_tex_pos;\n\n" +
  "// pseudo-random generator\n" +
  "const vec3 rand_constants = vec3(12.9898, 78.233, 4375.85453);\n" +
  "float rand(const vec2 co) {\n" +
  "    float t = dot(rand_constants.xy, co);\n" +
  "    return fract(sin(t) * (rand_constants.z + t));\n" +
  "}\n" +
  "\n" +
  "// wind speed lookup; use manual bilinear filtering based on 4 adjacent pixels for smooth interpolation\n" +
  "vec2 lookup_wind(const vec2 uv) {\n" +
  "    // return texture2D(u_wind, uv).rg; // lower-res hardware filtering\n" +
  "    vec2 px = 1.0 / u_wind_res;\n" +
  "    vec2 vc = (floor(uv * u_wind_res)) * px;\n" +
  // "    vec2 f = fract(uv * u_wind_res);\n" +
  // "    vec2 tl = texture2D(u_wind, vc).rg;\n" +
  // "    vec2 tr = texture2D(u_wind, vc + vec2(px.x, 0)).rg;\n" +
  // "    vec2 bl = texture2D(u_wind, vc + vec2(0, px.y)).rg;\n" +
  // "    vec2 br = texture2D(u_wind, vc + px).rg;\n" +
  // "    return mix(mix(tl, tr, f.x), mix(bl, br, f.x), f.y);\n" +
  "    float facX = u_wind_res.x > u_wind_res.y ? u_wind_res.x / u_wind_res.y : 1.0;\n" +
  "    float facY = u_wind_res.x > u_wind_res.y ? 1.0 : u_wind_res.y / u_wind_res.x;\n" +
  "    vec2 n = normalize(u_wind_res);\n" +
  "    float gamma = 0.02 * max(n.x, n.y);\n" +
  "    float epsilon = -2.0;\n" +
  "    vec2 vel = vec2(0.0, 0.0);\n" +
  "    for (int i = 0; i < 4; ++i) {\n" +
  "        float w1 = lucQ[i];\n" +
  "        float delta0 = facX * (luc[i][0] - vc[0]);\n" +
  "        float delta1 = facY * (luc[i][1] - vc[1]);\n" +
  "        float d2 = delta0 * delta0 + delta1 * delta1;\n" +
  "        float extinction = exp(-d2 / gamma);\n" +
  "        vel[0] += extinction * delta1 * epsilon * w1;\n" +
  "        vel[1] += extinction * (delta0 * epsilon * w1);\n" +
  "\n" +
  "        w1 = lucL[i];\n" +
  "        delta0 = facX * (luc[i][2] - vc[0]);\n" +
  "        delta1 = facY * (luc[i][3] - vc[1]);\n" +
  "        d2 = delta0 * delta0 + delta1 * delta1;\n" +
  "        extinction = exp(-d2 / gamma);\n" +
  "        vel[0] += extinction * delta1 * epsilon * w1;\n" +
  "        vel[1] += extinction * (delta0 * epsilon * w1);\n" +
  "    \n" +
  "    }\n" +
  "    vel[0] += 0.05;\n" +
  "    vec2 velocity = mix(u_wind_min, u_wind_max, vel);\n" +
  "    return velocity;\n" +
  "}\n" +
  "\n" +
  "void main() {\n" +
  "    vec4 color = texture2D(u_particles, v_tex_pos);\n" +
  "    vec2 pos = vec2(\n" +
  "        color.r / 255.0 + color.b,\n" +
  "        color.g / 255.0 + color.a); // decode particle position from pixel RGBA\n\n" +
  "    vec2 velocity = mix(u_wind_min, u_wind_max, lookup_wind(pos));\n" +
  "    float speed_t = length(velocity) / length(u_wind_max);\n\n" +
  "    // take EPSG:4236 distortion into account for calculating where the particle moved\n" +
  "    float distortion = cos(radians(pos.y * 180.0 - 90.0)) + 0.1;\n" +
  // "    vec2 offset = vec2(velocity.x / 1.0, -velocity.y) * 0.0001 * u_speed_factor;\n\n" +
  "    vec2 offset = vec2(velocity.x * distortion, -velocity.y) * 0.0001 * u_speed_factor;\n\n" +
  // No distortion
  "    // update particle position, wrapping around the date line\n" +
  "    pos = fract(1.0 + pos + offset);\n\n" +
  "    // a random seed to use for the particle drop\n" +
  "    vec2 seed = (pos + v_tex_pos) * u_rand_seed;\n\n" +
  "    // drop rate is a chance a particle will restart at random position, to avoid degeneration\n" +
  "    float drop_rate = u_drop_rate + speed_t * u_drop_rate_bump;\n" +
  "    float drop = step(1.0 - drop_rate, rand(seed));\n\n" +
  "    vec2 random_pos = vec2(\n" +
  "        rand(seed + 1.3),\n" +
  "        rand(seed + 2.1));\n" +
  "    pos = mix(pos, random_pos, drop);\n\n" +
  // "    if (distance(mix(u_wind_min, u_wind_max, lookup_wind(pos)), vec2(0.0)) < 1.0) {\n" +
  // "        gl_FragColor = vec4(1.0, 0.0, 0.0, 1.0);" +
  // "    } else {\n" +
  "        // encode the new particle position back into RGBA\n" +
  "        gl_FragColor = vec4(\n" +
  "            fract(pos * 255.0),\n" +
  "            floor(pos * 255.0) / 255.0);\n" +
  // "    }\n" +
  "}\n";

var palette2 = {
  0.0: '#6a5b4e',
  0.1: '#826f69',
  0.2: '#9c897f',
  0.3: '#c4a898',
  0.4: '#d8c7a6',
  0.5: '#f1d6bb',
  0.6: '#f6e3ce',
  1.0: '#fff0d4'
};

var palette3 = {
  0.0: '#3288bd',
  0.1: '#66c2a5',
  0.2: '#abdda4',
  0.3: '#e6f598',
  0.4: '#fee08b',
  0.5: '#fdae61',
  0.6: '#f46d43',
  1.0: '#d53e4f'
};

var defaultRampColors = {
  // 1.0: "#d73027",
  // 0.4: "#f46d43",
  // 0.3: "#fdae61",
  // 0.2: "#fee090",
  1.0: "#4575b4",
  0.4: "#4575b4",
  0.3: "#4575b4",
  0.2: "#4575b4",
  // 0.4: "#ffffbf",
  // 0.3: "#e0f3f8",
  // 0.2: "#abd9e9",
  0.1: "#4575b4",
  0.05: "#6694d1",
  0.01: "#6694d1",
  0.001: "#6694d1",
  0.0001: "#6694d1",
  0.0: "#4575b4"
};


var WindGL = function WindGL(gl) {
  this.gl = gl;
  this.fadeOpacity = 0.996; // how fast the particle trails fade on each frame
  this.speedFactor = 0.08; // 0.25; // how fast the particles move
  this.dropRate = 0.003; // how often the particles move to a random place
  this.dropRateBump = 0.01; // drop rate increase relative to individual particle speed
  this.eightVortices = [ // LUC
    0.3, 0.1,    0.1, 0.4,
    0.5, 0.5,    0.1, 0.7,
    0.7, 0.8,    0.3, 0.5,
    0.5, 0.2,    0.8, 0.3
  ];
  this.eightWeights1 = [
    2, 2, -2, 2
  ];
  this.eightWeights2 = [
    2, -2, -2, 2
  ];
  this.eightSpeeds = [
    0.0, 0.0,    0.0, 0.0,
    0.0, 0.0,    0.0, 0.0,
    0.0, 0.0,    0.0, 0.0,
    0.0, 0.0,    0.0, 0.0
  ];
  this.drawProgram = createProgram(gl, drawVert, drawFrag);
  this.screenProgram = createProgram(gl, quadVert, screenFrag);
  this.updateProgram = createProgram(gl, quadVert, updateFrag);
  this.quadBuffer = createBuffer(gl, new Float32Array([0, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 1]));
  this.framebuffer = gl.createFramebuffer();
  this.setColorRamp(defaultRampColors);
  this.resize();
};

var prototypeAccessors = { numParticles: {} };

WindGL.prototype.resize = function resize () {
  var gl = this.gl;
  var emptyPixels = new Uint8Array(gl.canvas.width * gl.canvas.height * 4);
  // screen textures to hold the drawn screen for the previous and the current frame
  this.backgroundTexture = createTexture(gl, gl.NEAREST, emptyPixels, gl.canvas.width, gl.canvas.height);
  this.screenTexture = createTexture(gl, gl.NEAREST, emptyPixels, gl.canvas.width, gl.canvas.height);
};

WindGL.prototype.setColorRamp = function setColorRamp (colors) {
  // lookup texture for colorizing the particles according to their speed
  this.colorRampTexture = createTexture(this.gl, this.gl.LINEAR, getColorRamp(
    // colors
    palette2
  ), 16, 16);
};

prototypeAccessors.numParticles.set = function (numParticles) {
  var gl = this.gl;
  // we create a square texture where each pixel will hold a particle position encoded as RGBA
  var particleRes = this.particleStateResolution = Math.ceil(Math.sqrt(numParticles));
  this._numParticles = particleRes * particleRes;
  var particleState = new Uint8Array(this._numParticles * 4);
  for (var i = 0; i < particleState.length; i++) {
    particleState[i] = Math.floor(Math.random() * 256); // randomize the initial particle positions
  }
  // textures to hold the particle state for the current and the next frame
  this.particleStateTexture0 = createTexture(gl, gl.NEAREST, particleState, particleRes, particleRes);
  this.particleStateTexture1 = createTexture(gl, gl.NEAREST, particleState, particleRes, particleRes);

  var particleIndices = new Float32Array(this._numParticles);
  for (var i$1 = 0; i$1 < this._numParticles; i$1++) { particleIndices[i$1] = i$1; }
  this.particleIndexBuffer = createBuffer(gl, particleIndices);
};
prototypeAccessors.numParticles.get = function () {
  return this._numParticles;
};

// TODO cleanup / get this out to another project.
// WindGL.prototype.updateLuc = function() {
//   let v = this.eightVortices;
//   let s = this.eightSpeeds;
//   let w1 = this.eightWeights1;
//   let w2 = this.eightWeights2;
//   let w = [...w1, ...w2];
//   let mix = (min, max) => (x) => Math.max(min, Math.min(max, x));
//   let maxSpeed = 0.0005;
//
//   for (let i = 0; i < 8; ++i) {
//     let v1x = v[2 * i]; let v1y = v[2 * i + 1];
//     let s1x = s[2 * i]; let s1y = s[2 * i + 1];
//     let w1 = Math.abs(w[i]);
//     let sign1 = Math.sign(w[i]);
//     let ax = 0; let ay = 0;
//
//     for (let j = 0; j < 8; ++j) {
//       if (j === i) continue;
//       let v2x = v[2 * j]; let v2y = v[2 * j + 1];
//       // let s2x = s[2 * j]; let s2y = s[2 * j + 1];
//       let w2 = Math.abs(w[j]);
//       let sign2 = Math.sign(w[j]);
//
//       let d = Math.sqrt(Math.pow(v1x - v2x, 2) + Math.pow(v1y - v2y, 2));
//       d = Math.max(d, 0.001);
//       let G = 0.00000001;
//       let repulsion = -1.0; //sign1 === sign2 ? -1 : 1;
//       let f = repulsion * G * w1 * w2 / (d * d);
//       ax += f * (v2x - v1x) / d;
//       ay += f * (v2y - v1y) / d;
//     }
//
//     this.eightVortices[2 * i] = mix(0, 1)(v1x + s1x);
//     this.eightVortices[2 * i + 1] = mix(0, 1)(v1y + s1y);
//     this.eightSpeeds[2 * i] += mix(-maxSpeed, maxSpeed)(ax);
//     this.eightSpeeds[2 * i + 1] += mix(-maxSpeed, maxSpeed)(ay);
//
//     v1x = this.eightVortices[2 * i];
//     v1y = this.eightVortices[2 * i + 1];
//     //  let c = 10;
//     if (v1x <= 0 || v1x >= 1) this.eightSpeeds[2 * i] = -s1x;
//     if (v1y <= 0 || v1y >= 1) this.eightSpeeds[2 * i + 1] = -s1y;
//   }
// };

WindGL.prototype.setConvolutionImage = function setConvolutionImage(windData, width, height) {
  let buffer = new ArrayBuffer(4 * width * height);
  let ubuf = new Uint8Array(buffer);
  const length = ubuf.length;
  let wd = windData.data;
  for (var i = 0; i < length; ++i) { // r, g, b, a
    ubuf[i] = wd[i] * 0.2;
    // (
    //   windData.data[4 * i] +
    //   windData.data[4 * i + 1] +
    //   windData.data[4 * i + 2]
    // ) / 3;
    // windData.data[4 * i + 3]
  }

  this.convData = ubuf;
  this.convTexture = createTexture(this.gl, this.gl.LINEAR, ubuf, width, height);
};

WindGL.prototype.setWind = function setWind (windData) {
  this.windData = windData;
  this.windTexture = createTexture(this.gl, this.gl.LINEAR, windData.image);
};

WindGL.prototype.draw = function draw () {
  var gl = this.gl;
  gl.disable(gl.DEPTH_TEST);
  gl.disable(gl.STENCIL_TEST);
  bindTexture(gl, this.windTexture, 0);
  bindTexture(gl, this.particleStateTexture0, 1);
  bindTexture(gl, this.convTexture, 3);
  this.drawScreen();
  this.updateParticles();
};

WindGL.prototype.drawScreen = function drawScreen () {
  var gl = this.gl;
  // draw the screen into a temporary framebuffer to retain it as the background on the next frame
  bindFramebuffer(gl, this.framebuffer, this.screenTexture);
  gl.viewport(0, 0, gl.canvas.width, gl.canvas.height);
  this.drawTexture(this.backgroundTexture, this.fadeOpacity);
  this.drawParticles();
  bindFramebuffer(gl, null);
  // enable blending to support drawing on top of an existing background (e.g. a map)
  gl.enable(gl.BLEND);
  gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);
  this.drawTexture(this.screenTexture, 1.0);
  gl.disable(gl.BLEND);
  // save the current screen as the background for the next frame
  var temp = this.backgroundTexture;
  this.backgroundTexture = this.screenTexture;
  this.screenTexture = temp;
};

WindGL.prototype.drawTexture = function drawTexture (texture, opacity) {
  var gl = this.gl;
  var program = this.screenProgram;
  gl.useProgram(program.program);
  bindAttribute(gl, this.quadBuffer, program.a_pos, 2);
  bindTexture(gl, texture, 2);
  gl.uniform1i(program.u_screen, 2);
  gl.uniform1f(program.u_opacity, opacity);
  gl.drawArrays(gl.TRIANGLES, 0, 6);
};

WindGL.prototype.drawParticles = function drawParticles () {
  var gl = this.gl;
  var program = this.drawProgram;
  gl.useProgram(program.program);
  bindAttribute(gl, this.particleIndexBuffer, program.a_index, 1);
  bindTexture(gl, this.colorRampTexture, 2);
  gl.uniform1i(program.u_wind, 0);
  gl.uniform1i(program.u_particles, 1);
  gl.uniform1i(program.u_color_ramp, 2);
  gl.uniform1i(program.u_conv_texture, 3);
  gl.uniform1f(program.u_particles_res, this.particleStateResolution);
  gl.uniform2f(program.u_wind_res, this.windData.width, this.windData.height);
  gl.uniform2f(program.u_wind_min, this.windData.uMin, this.windData.vMin);
  gl.uniform2f(program.u_wind_max, this.windData.uMax, this.windData.vMax);
  gl.uniformMatrix4fv(program.luc, false, this.eightVortices);
  gl.uniform4fv(program.lucQ, this.eightWeights1);
  gl.uniform4fv(program.lucL, this.eightWeights2);
  gl.drawArrays(gl.POINTS, 0, this._numParticles);
};

WindGL.prototype.updateParticles = function updateParticles () {
  var gl = this.gl;
  bindFramebuffer(gl, this.framebuffer, this.particleStateTexture1);
  gl.viewport(0, 0, this.particleStateResolution, this.particleStateResolution);
  var program = this.updateProgram;
  gl.useProgram(program.program);
  bindAttribute(gl, this.quadBuffer, program.a_pos, 2);
  gl.uniform1i(program.u_wind, 0);
  gl.uniform1i(program.u_particles, 1);
  gl.uniform1i(program.u_conv_texture, 3);
  gl.uniform1f(program.u_rand_seed, Math.random());
  gl.uniform2f(program.u_wind_res, this.windData.width, this.windData.height);
  gl.uniform2f(program.u_wind_min, this.windData.uMin, this.windData.vMin);
  gl.uniform2f(program.u_wind_max, this.windData.uMax, this.windData.vMax);
  gl.uniform1f(program.u_speed_factor, this.speedFactor);
  gl.uniform1f(program.u_drop_rate, this.dropRate);
  gl.uniform1f(program.u_drop_rate_bump, this.dropRateBump);
  gl.uniformMatrix4fv(program.luc, false, this.eightVortices);
  gl.uniform4fv(program.lucQ, this.eightWeights1);
  gl.uniform4fv(program.lucL, this.eightWeights2);
  gl.drawArrays(gl.TRIANGLES, 0, 6);
  // swap the particle state textures so the new one becomes the current one
  var temp = this.particleStateTexture0;
  this.particleStateTexture0 = this.particleStateTexture1;
  this.particleStateTexture1 = temp;
};

Object.defineProperties( WindGL.prototype, prototypeAccessors );

function getColorRamp(colors) {
  var canvas = document.createElement('canvas');
  var ctx = canvas.getContext('2d');
  canvas.width = 256;
  canvas.height = 1;
  var gradient = ctx.createLinearGradient(0, 0, 256, 0);
  for (var stop in colors) {
    gradient.addColorStop(+stop, colors[stop]);
  }
  ctx.fillStyle = gradient;
  ctx.fillRect(0, 0, 256, 1);
  return new Uint8Array(ctx.getImageData(0, 0, 256, 1).data);
}


// TODO [REFACTOR] Make this a clean module.
var PARTICLE_LINE_WIDTH = 1;
var MAX_PARTICLE_AGE = 10000;
var FADE_FILL_STYLE = 'rgba(0, 0, 0, 0.97)';

// The palette can be easily tuned by adding colors.
var palette = [
  "#d73027",
  "#d73027",
  "#f46d43",
  "#f46d43",
  "#fdae61",
  "#fee090",
  "#ffffbf",
  "#e0f3f8",
  "#abd9e9",
  "#74add1",
  "#6694d1",
  "#4575b4"
];

// Draw objects
var af;
var buckets = [];
var NUMBER_BUCKETS = palette.length;
var particles = [];
var DOMElement;

// Simulation vars
var xPixels = 100;
var yPixels = 100;
var gridSize = xPixels * yPixels;
var gridScale = 100 * Math.sqrt(2);
var nbSamples;
var simulationType = 'gaussian';

// Simulation objects
var vortices = [];
var vortexSpeeds = [];
var vortexRanges = [];
var nbVortices = 100;
var MAX_VORTEX_NUMBER = 150;
var maxVectorFieldNorm = 5;

// Interaction objects
var isRightMouseDown = false;
var isLeftMouseDown = false;
var vortexAugmentationTimeout;
var REFRESH_RATE = 16; // 60 fps
var mouseRepulsionActive = false;
var mousePosition = [0, 0];

var g = null;

// var stamp1 = 0;
// var maxNumberParticles = 30000;
// var currentNumberParticles = 100;
var gl = null;
var windGL = null;
var heightVF = 180;
var widthVF = 360;
var staticVF = null;

var Windy =
{
  start: function(
    gl, element, screenWidth, screenHeight, nbParticles, type, imageDataSource, width, height)
  {
    this.end();
    DOMElement = element;
    // g = DOMElement.getContext("2d");

    // Internet Explorer

    windGL = new WindGL(gl);
    windGL.setConvolutionImage(imageDataSource, width, height);
    windGL.numParticles = nbParticles;
    staticVF = new Uint8Array(heightVF * widthVF);

    xPixels = screenWidth;
    yPixels = screenHeight;
    gridSize = xPixels * yPixels;
    gridScale = Math.sqrt(Math.pow(xPixels, 2) + Math.pow(yPixels, 2));
    nbSamples = nbParticles;
    // nbSamples = currentNumberParticles;
    // maxNumberParticles = nbParticles;

    if (type) simulationType = type;
    vortices = [];
    vortexSpeeds = [];
    vortexRanges = [];
    particles = [];
    buckets = [];

    this.makeVectorField();
    this.makeBuckets();
    this.makeParticles();
    this.animate();

    let windy = document.getElementById('windy');
    windy.addEventListener('contextmenu', function(e) {e.preventDefault()});
    windy.addEventListener('mousedown', this.mouseDownCallback.bind(this));
    windy.addEventListener('mouseup', this.mouseUpCallback.bind(this));
    windy.addEventListener('mousemove', this.mouseMoveCallback.bind(this));
    windy.addEventListener('mouseout', function() {mouseRepulsionActive = false}.bind(this));
  },

  end: function() {
    cancelAnimationFrame(af);
  },

  updateVF: function(domElement) {
    let c = 0;
    staticVF = new Uint8Array(heightVF * widthVF);
    // for (let i = 0; i < heightVF; ++i) {
    //   let px = (i / heightVF) * xPixels;
    //   for (let j = 0; j < widthVF; ++j) {
    //     let py = (j / widthVF) * yPixels;
    //     let vec2 = this.computeVectorFieldAt(px, py);
    //     if (!vec2) vec2 = [0, 0];
    //     staticVF[c++] = vec2[0];
    //     staticVF[c++] = vec2[1];
    //   }
    // }

    let fac = 4.0;
    var data = {
      width: DOMElement.offsetWidth ,
      height: DOMElement.offsetHeight ,
      uMin: 0, // -21.32,
      uMax: 30, // 26.8,
      vMin: 0, // -21.57,
      vMax: 30, // 21.42,
      image: staticVF
    };

    // if (!window.done) console.log(staticVF);
    // window.done = true;

    return data;
  },

  animate: function() {
    af = requestAnimationFrame(this.animate.bind(this));
    // let deltaT = Date.now() - stamp1;
    // let fps = 30;
    // let deltaThreshold = 1000 / fps;
    // if (deltaT < deltaThreshold) return;
    // stamp1 = Date.now();

    // this.update();
    // this.draw();

    if (!windGL.windData) {
      var wd = this.updateVF();
      windGL.windData = wd;
    }
    if (windGL.windData) {
      // windGL.updateLuc();
      windGL.draw();
    }
    // let stamp2 = ;
    // let deltaTime = Date.now() - stamp1;
    // if (deltaTime < 1000) {
    //   let numberOfParticlesToAdd = Math.min(100, maxNumberParticles - currentNumberParticles);
    //   if (numberOfParticlesToAdd > 0) {
    //     currentNumberParticles = currentNumberParticles + numberOfParticlesToAdd;
    //     nbSamples = currentNumberParticles;
    //     for (let i = 0; i < numberOfParticlesToAdd; ++i)
    //       particles.push(this.newParticle(i));
    //   }
    // } else {
    //   let numberOfParticlesToRemove = Math.min(0, currentNumberParticles - 100);
    //   if (numberOfParticlesToRemove > 0) {
    //     currentNumberParticles = currentNumberParticles - numberOfParticlesToRemove;
    //     nbSamples = currentNumberParticles;
    //     for (let i = 0; i < numberOfParticlesToRemove; ++i)
    //       particles.pop();
    //   }
    // }
  },

  makeBuckets: function() {
    // 1 bucket per color, NUMBER_BUCKETS colors.
    buckets = Array.from(Array(NUMBER_BUCKETS).keys()).map(function(){return []});
  },

  addParticleToDrawBucket: function(particle, vector) {
    let maxVectorNorm = maxVectorFieldNorm;
    let thisVectorNorm = this.computeNorm(vector);
    let nbBuckets = buckets.length;

    let bucketIndex =
      thisVectorNorm < 0.001 ? 0 :
      thisVectorNorm >= maxVectorNorm ? nbBuckets - 1 :
      Math.ceil(nbBuckets * thisVectorNorm / maxVectorNorm);

    bucketIndex = bucketIndex >= buckets.length ? bucketIndex - 1 : bucketIndex;
    buckets[bucketIndex].push(particle);
  },

  makeParticles: function() {
    particles = [];
    for (let i = 0; i < nbSamples; ++i)
      particles.push(this.newParticle(i));
  },

  newParticle: function(particleRank) {
    let x0 = Math.floor(Math.random() * xPixels);
    let y0 = Math.floor(Math.random() * yPixels);
    return {
      x: x0,
      y: y0,
      xt: x0 + 0.01 * Math.random(),
      yt: y0 + 0.01 * Math.random(),
      age: Math.floor(Math.random() * MAX_PARTICLE_AGE),
      rank: particleRank
    };
  },

  evolveVectorField: function() {
    for (let vortex1Id = 0; vortex1Id < nbVortices; ++vortex1Id) {
      let vortex1 = vortices[vortex1Id];
      let o1 = vortex1[3] > 0; // orientation
      let mass1 = Math.abs(vortex1[3]);
      let charge1 = vortex1[2];
      let acceleration = [0, 0];
      // repulsion
      let coeff = 1 / gridScale; // 0.1;

      for (let vortex2Id = 0; vortex2Id < nbVortices; ++vortex2Id) {
        if (vortex2Id === vortex1Id) continue;

        let vortex2 = vortices[vortex2Id];
        let o2 = vortex2[3] > 0;

        let delta0 = coeff * (vortex1[0] - vortex2[0]);
        let delta1 = coeff * (vortex1[1] - vortex2[1]);
        let d2 =
          Math.pow(delta0, 2) +
          Math.pow(delta1, 2);

        // Everything is repulsive
        let sign = 1;
        // Same sign vortices are attracted, opposite sign are repulsed
        // o1 === o2 ? 1 : -1;

        // !! Eulerian physics
        // !! Charge could also be vortexI[3]
        // !! Mass could also be vortexI[2]
        let charge2 = vortex2[2];
        let mass2 = Math.abs(vortex2[3]);
        if (Math.abs(delta0) > 0.0001)
          acceleration[0] += sign * Math.abs(charge1 * charge2 * mass1 * mass2) * delta0 /
            (d2 * d2 * Math.abs(delta0));
        if (Math.abs(delta1) > 0.0001)
          acceleration[1] += sign * Math.abs(charge1 * charge2 * mass1 * mass2) * delta1 /
            (d2 * d2 * Math.abs(delta1));
      }

      // Add four walls
      // coeff = 0.5;
      let v0x = coeff * vortex1[0]; let v0y = coeff * vortex1[1];
      let d0x = - coeff * xPixels + v0x; let d0y = - coeff * yPixels + v0y;
      let da = 0;
      if (Math.abs(v0x) > 0.001) {
        da = (v0x) / (v0x * v0x * Math.abs(v0x));
        acceleration[0] += da;
        acceleration[1] += da * Math.sign(vortex1[3]);
      }
      if (Math.abs(d0x) > 0.001) {
        da = (d0x) / (d0x * d0x * Math.abs(d0x));
        acceleration[0] += da;
        acceleration[1] += da * Math.sign(vortex1[3]);
      }
      if (Math.abs(v0y) > 0.001) {
        da = (v0y) / (v0y * v0y * Math.abs(v0y));
        acceleration[1] += da;
        acceleration[0] -= da * Math.sign(vortex1[3]);
      }
      if (Math.abs(d0y) > 0.001) {
        da = (d0y) / (d0y * d0y * Math.abs(d0y));
        acceleration[1] += da;
        acceleration[0] -= da * Math.sign(vortex1[3]);
      }

      // Add mouse
      if (mouseRepulsionActive) {
        coeff *= 0.4;
        let deltaX = coeff * (vortex1[0] - mousePosition[0]);
        let deltaY = coeff * (vortex1[1] - mousePosition[1]);
        let dist = deltaX * deltaX + deltaY * deltaY;
        // Doesn't seem to matter after all...
        if (Math.abs(deltaX) > 0.001) acceleration[0] += deltaX / (dist * dist * Math.abs(deltaX));
        if (Math.abs(deltaY) > 0.001) acceleration[1] += deltaY / (dist * dist * Math.abs(deltaY));
      }

      let speedX = vortexSpeeds[vortex1Id][0] + 0.000001 * acceleration[0];
      let speedY = vortexSpeeds[vortex1Id][1] + 0.000001 * acceleration[1];

      vortexSpeeds[vortex1Id][0] = Math.sign(speedX) * Math.min(Math.abs(speedX), 0.3);
      vortexSpeeds[vortex1Id][1] = Math.sign(speedY) * Math.min(Math.abs(speedY), 0.3);

      let np0 = vortex1[0] + vortexSpeeds[vortex1Id][0];
      let np1 = vortex1[1] + vortexSpeeds[vortex1Id][1];
      vortex1[0] = Math.min(Math.max(np0, 0), xPixels);
      vortex1[1] = Math.min(Math.max(np1, 0), yPixels);

      // Update swiper.
      vortexRanges[vortex1Id] = this.computeVortexRange(vortex1);
    }
  },

  computeVortexRange: function(vortex) {
    let fadeCoefficient = 100;
    return [
      vortex[0] - vortex[2] * fadeCoefficient,
      vortex[0] + vortex[2] * fadeCoefficient,
      vortex[1] - vortex[2] * fadeCoefficient,
      vortex[1] + vortex[2] * fadeCoefficient
    ]
  },

  computeVectorFieldAt: function(xp, yp)
  {
    if (xp <= 1 || xp >= xPixels - 1 || yp <= 1 || yp >= yPixels - 1)
      return null;

    let mean = [0, 0];
    for (let vi = 0; vi < nbVortices; ++vi) {
      let vp = vortices[vi];
      let bounds = vortexRanges[vi];
      if (xp < bounds[0] || xp > bounds[1] || yp < bounds[2] || yp > bounds[3])
        continue;

      // Distance to current vortex
      let delta0 = vp[0] - xp;
      let delta1 = vp[1] - yp;
      let d2 = delta0 * delta0 + delta1 * delta1;

      // To be clear with what we do here:
      // let gamma = vp[2] * gridScale;
      // let delta = [vp[0] - xp, vp[1] - yp, 0];
      // let up = [0, 0, vp[3]];

      // Cross product (the one used there)
      // let cross = [delta[1] * up[2], -delta[0] * up[2]];

      // Cute but odd (mangled cross product, interesting visual)
      // let cross = [delta[0] * up[2], -delta[1] * up[2]];

      let extinction = Math.exp(-d2 / (vp[2] * gridScale));
      mean[0] += extinction * delta1 * vp[3];    // cross[0];
      mean[1] += extinction * (-delta0 * vp[3]); // cross[1];
    }

    return mean;
  },

  makeVectorField: function() {
    vortices.length = 0;
    for (let v = 0; v < nbVortices; ++v) {
      let sg = Math.random() > 0.5 ? 1 : -1;
      let newVortex = [
        Math.min(Math.random() * xPixels + 20, xPixels - 20), // x position
        Math.min(Math.random() * yPixels + 20, yPixels - 20), // y position
        5.0 * Math.max(0.25, Math.random()), // gaussian range
        0.2 * sg * Math.max(Math.min(Math.random(), 0.5), 0.4) // gaussian intensity and clockwiseness
      ];

      vortices.push(newVortex);

      // Initial speeds
      vortexSpeeds.push([
        0, // Math.random() - 0.5,
        0  // Math.random() - 0.5
      ]);

      vortexRanges.push(this.computeVortexRange(newVortex));
    }
  },

  isNullVectorFieldAt: function(fx, fy)
  {
    return (fx <= 1 || fx >= xPixels - 1 || fy <= 1 || fy >= yPixels - 1);
  },

  computeNorm: function(vector) {
    return Math.sqrt(Math.pow(vector[0], 2) + Math.pow(vector[1], 2));
  },

  update: function() {
    // Empty buckets.
    for (let b = 0; b < buckets.length; ++b) buckets[b].length = 0;

    // Move particles and add them to buckets.
    for (let p = 0; p < particles.length; ++p) {
      let particle = particles[p];

      if (particle.age > MAX_PARTICLE_AGE) {
        particles[particle.rank] = this.newParticle(particle.rank);
      }

      let x = particle.x;
      let y = particle.y;
      let v = this.computeVectorFieldAt(x, y);  // vector at current position

      if (v === null) {
        // This particle is outside the grid
        particle.age = MAX_PARTICLE_AGE;
      } else {
        let xt = x + 0.1 * v[0];
        let yt = y + 0.1 * v[1];

        if (!this.isNullVectorFieldAt(xt, yt)) {
          // The path of this particle is visible
          particle.xt = xt;
          particle.yt = yt;

          if (Math.abs(x - xt) > 0.05 || Math.abs(y - yt) > 0.05) {
            this.addParticleToDrawBucket(particle, v);
          }
        } else {
          // This particle isn't visible, but still moves through the field.
          particle.x = xt;
          particle.y = yt;
        }
      }

      particle.age += 1;
    }

    this.evolveVectorField();
  },

  // Enhancement: try out twojs
  // (Not fan of the loading overhead)
  draw: function() {
    g.lineWidth = PARTICLE_LINE_WIDTH;
    g.fillStyle = FADE_FILL_STYLE;
    g.mozImageSmoothingEnabled = false;
    g.webkitImageSmoothingEnabled = false;
    g.msImageSmoothingEnabled = false;
    g.imageSmoothingEnabled = false;

    // Fade existing particle trails.
    let prev = g.globalCompositeOperation;
    g.globalCompositeOperation = "destination-in";
    // g.fillStyle = "#ffffff";
    // g.fillStyle = "#000000";
    g.fillRect(0, 0, xPixels, yPixels);
    g.globalCompositeOperation = prev;

    // Draw new particle trails.
    let nbBuckets = buckets.length;
    for (let b = 0; b < nbBuckets; ++b) {
      let bucket = buckets[b];
      if (bucket.length > 0) {
        g.beginPath();
        g.strokeStyle = palette[b];
        for (let p = 0; p < bucket.length; ++p) {
          let particle = bucket[p];
          let x = particle.x;
          let xt = particle.xt;
          let y = particle.y;
          let yt = particle.yt;
          // (This was for better extremal sampling:)
          // g.moveTo(x - (xt - x) * 1.1, y - (yt - y) * 1.1);
          // g.lineTo(xt + (xt - x) * 1.1, yt + (yt - y) * 1.1);
          g.moveTo(x, y);
          g.lineTo(xt, yt);
          particle.x = xt;
          particle.y = yt;
        }
        g.stroke();
      }
    }
  },

  getEventPositionInCanvas: function(event) {
    // YES, this is quick and dirty, please <i>please</i> be indulgent.
    // jQuery would have been a loading overhead
    // (Hyphenator is an overhead as well, but it is mandatory for Fr support).
    let windyElement = document.getElementById('windy');
    let rect = windyElement.getBoundingClientRect();
    let top = rect.top;
    let left = rect.left;
    return [event.clientX - left, event.clientY - top];
  },

  mouseDownCallback: function(event) {
    if (isLeftMouseDown) {
      // This should be possible with alt-tab, maybe.
      console.log('[MouseDownCallBack]: multiple mousedown events ' +
        'without a mouseup.');
      return;
    }
    isLeftMouseDown = true;

    // Get coordinates for the click.
    let positionInCanvas = this.getEventPositionInCanvas(event);
    let sx = positionInCanvas[0];
    let sy = positionInCanvas[1];

    // Kind of a polyfill for detecting a right-click,
    // No jQuery should be involved.
    let rightclick =
      event.which ? (event.which === 3) :
      event.button ? event.button === 2 : false;

    // We make it so the added vortex is always the last.
    let newVortex = [sx, sy, 1, rightclick ? -0.1 : 0.1];
    let newRange = this.computeVortexRange(newVortex);
    if (nbVortices < MAX_VORTEX_NUMBER) {
      nbVortices += 1;
    } else {
      vortices.shift();
      vortexRanges.shift();
      vortexSpeeds.shift();
    }
    vortices.push(newVortex);
    vortexRanges.push(newRange);
    vortexSpeeds.push([0, 0]);

    // Then we can progressively augment the size and speed of the created vortex.
    vortexAugmentationTimeout = setTimeout(
      this.augmentCreatedVortex.bind(this), REFRESH_RATE
    );
  },

  augmentCreatedVortex: function() {
    let lastVortexIndex = vortices.length - 1;
    let lastVortex = vortices[lastVortexIndex];

    if (mouseRepulsionActive) {
      lastVortex[0] = mousePosition[0];
      lastVortex[1] = mousePosition[1];
    }

    // Augment vortex.
    lastVortex[2] = Math.min(lastVortex[2] + 0.02, 5);
    if (lastVortex[3] > 0)
      lastVortex[3] = Math.min(lastVortex[3] + 0.01, 0.2);
    else
      lastVortex[3] = Math.max(lastVortex[3] - 0.01, -0.2);

    // Recompute vortex range.
    // Not strictly necessary: this is done at every vortex field evolution.
    vortexRanges[lastVortexIndex] = this.computeVortexRange(lastVortex);

    // Call again.
    vortexAugmentationTimeout = setTimeout(
      this.augmentCreatedVortex.bind(this), REFRESH_RATE
    );
  },

  mouseUpCallback: function(event) {
    mouseRepulsionActive = false;

    event.preventDefault();
    clearTimeout(vortexAugmentationTimeout);

    isLeftMouseDown = false;
  },

  mouseMoveCallback: function(event) {
    // Prevent dragging the canvas
    event.preventDefault();

    // Get new pointer position.
    let positionInCanvas = this.getEventPositionInCanvas(event);
    let sx = positionInCanvas[0];
    let sy = positionInCanvas[1];
    mousePosition = [sx, sy];

    // Check mouse status
    if (!isLeftMouseDown && !isRightMouseDown) {
      mouseRepulsionActive = true;
      return;
    }

    let lastVortexIndex = vortices.length - 1;
    let lastVortex = vortices[lastVortexIndex];

    let oldX = lastVortex[0];
    let oldY = lastVortex[1];
    let lastSpeed = vortexSpeeds[lastVortexIndex];
    let deltaX = sx - oldX;
    let deltaY = sy - oldY;

    lastSpeed[0] = Math.sign(deltaX) * Math.sqrt(Math.pow(deltaX / 500, 2));
    lastSpeed[1] = Math.sign(deltaY) * Math.sqrt(Math.pow(deltaY / 500, 2));

    lastVortex[0] = sx;
    lastVortex[1] = sy;
  }
};

// 'Polyfill'
if (!window.requestAnimationFrame) {
  module.exports = { Windy: { start: function(){}, end: function(){} } };
} else {
  module.exports = { Windy: Windy };
}


/***/ }),

/***/ "./src/app/app-routing.module.ts":
/*!***************************************!*\
  !*** ./src/app/app-routing.module.ts ***!
  \***************************************/
/*! exports provided: AppRoutingModule */
/***/ (function(module, __webpack_exports__, __webpack_require__) {

"use strict";
__webpack_require__.r(__webpack_exports__);
/* harmony export (binding) */ __webpack_require__.d(__webpack_exports__, "AppRoutingModule", function() { return AppRoutingModule; });
/* harmony import */ var tslib__WEBPACK_IMPORTED_MODULE_0__ = __webpack_require__(/*! tslib */ "./node_modules/tslib/tslib.es6.js");
/* harmony import */ var _angular_core__WEBPACK_IMPORTED_MODULE_1__ = __webpack_require__(/*! @angular/core */ "./node_modules/@angular/core/fesm5/core.js");
/* harmony import */ var _angular_router__WEBPACK_IMPORTED_MODULE_2__ = __webpack_require__(/*! @angular/router */ "./node_modules/@angular/router/fesm5/router.js");
/* harmony import */ var _articles_articles_component__WEBPACK_IMPORTED_MODULE_3__ = __webpack_require__(/*! ./articles/articles.component */ "./src/app/articles/articles.component.ts");
/* harmony import */ var _dashboard_dashboard_component__WEBPACK_IMPORTED_MODULE_4__ = __webpack_require__(/*! ./dashboard/dashboard.component */ "./src/app/dashboard/dashboard.component.ts");
/* harmony import */ var _article_detail_article_detail_component__WEBPACK_IMPORTED_MODULE_5__ = __webpack_require__(/*! ./article-detail/article-detail.component */ "./src/app/article-detail/article-detail.component.ts");






var routes = [
    { path: '', redirectTo: '/dashboard', pathMatch: 'full' },
    { path: 'articles/:cat', component: _articles_articles_component__WEBPACK_IMPORTED_MODULE_3__["ArticlesComponent"] },
    { path: 'dashboard', component: _dashboard_dashboard_component__WEBPACK_IMPORTED_MODULE_4__["DashboardComponent"] },
    { path: 'detail/:id', component: _article_detail_article_detail_component__WEBPACK_IMPORTED_MODULE_5__["ArticleDetailComponent"] },
    { path: '', redirectTo: '/dashboard', pathMatch: 'full' },
    { path: '**', redirectTo: '/dashboard', pathMatch: 'full' }
];
var AppRoutingModule = /** @class */ (function () {
    function AppRoutingModule() {
    }
    AppRoutingModule = tslib__WEBPACK_IMPORTED_MODULE_0__["__decorate"]([
        Object(_angular_core__WEBPACK_IMPORTED_MODULE_1__["NgModule"])({
            imports: [_angular_router__WEBPACK_IMPORTED_MODULE_2__["RouterModule"].forRoot(routes)],
            exports: [_angular_router__WEBPACK_IMPORTED_MODULE_2__["RouterModule"]]
        })
    ], AppRoutingModule);
    return AppRoutingModule;
}());



/***/ }),

/***/ "./src/app/app.component.css":
/*!***********************************!*\
  !*** ./src/app/app.component.css ***!
  \***********************************/
/*! no static exports found */
/***/ (function(module, exports) {

module.exports = "/* AppComponent's private CSS styles */\r\nh1 {\r\n  font-size: 1.2em;\r\n  color: #999;\r\n  margin-bottom: 0;\r\n}\r\nh2 {\r\n  font-size: 2em;\r\n  margin-top: 0;\r\n  padding-top: 0;\r\n}\r\nnav a {\r\n  color: #454545;\r\n  padding: 5px 10px;\r\n  margin: 10px 2px;\r\n  text-decoration: none;\r\n  display: inline-block;\r\n  background-color: #eee;\r\n  border-radius: 4px;\r\n}\r\nnav a:hover {\r\n  color: #000;\r\n  background-color: #d6d6d6;\r\n}\r\nnav a.active {\r\n}\r\n@media (max-width: 576px) {\r\n  #submain-div {\r\n    margin-right: 0;\r\n  }\r\n  #main-div {\r\n    padding-right: 0;\r\n  }\r\n}\r\n\r\n/*# sourceMappingURL=data:application/json;base64,eyJ2ZXJzaW9uIjozLCJzb3VyY2VzIjpbInNyYy9hcHAvYXBwLmNvbXBvbmVudC5jc3MiXSwibmFtZXMiOltdLCJtYXBwaW5ncyI6IkFBQUEsc0NBQXNDO0FBQ3RDO0VBQ0UsZ0JBQWdCO0VBQ2hCLFdBQVc7RUFDWCxnQkFBZ0I7QUFDbEI7QUFFQTtFQUNFLGNBQWM7RUFDZCxhQUFhO0VBQ2IsY0FBYztBQUNoQjtBQUVBO0VBQ0UsY0FBYztFQUNkLGlCQUFpQjtFQUNqQixnQkFBZ0I7RUFDaEIscUJBQXFCO0VBQ3JCLHFCQUFxQjtFQUNyQixzQkFBc0I7RUFDdEIsa0JBQWtCO0FBQ3BCO0FBRUE7RUFDRSxXQUFXO0VBQ1gseUJBQXlCO0FBQzNCO0FBRUE7QUFDQTtBQUVBO0VBQ0U7SUFDRSxlQUFlO0VBQ2pCO0VBQ0E7SUFDRSxnQkFBZ0I7RUFDbEI7QUFDRiIsImZpbGUiOiJzcmMvYXBwL2FwcC5jb21wb25lbnQuY3NzIiwic291cmNlc0NvbnRlbnQiOlsiLyogQXBwQ29tcG9uZW50J3MgcHJpdmF0ZSBDU1Mgc3R5bGVzICovXHJcbmgxIHtcclxuICBmb250LXNpemU6IDEuMmVtO1xyXG4gIGNvbG9yOiAjOTk5O1xyXG4gIG1hcmdpbi1ib3R0b206IDA7XHJcbn1cclxuXHJcbmgyIHtcclxuICBmb250LXNpemU6IDJlbTtcclxuICBtYXJnaW4tdG9wOiAwO1xyXG4gIHBhZGRpbmctdG9wOiAwO1xyXG59XHJcblxyXG5uYXYgYSB7XHJcbiAgY29sb3I6ICM0NTQ1NDU7XHJcbiAgcGFkZGluZzogNXB4IDEwcHg7XHJcbiAgbWFyZ2luOiAxMHB4IDJweDtcclxuICB0ZXh0LWRlY29yYXRpb246IG5vbmU7XHJcbiAgZGlzcGxheTogaW5saW5lLWJsb2NrO1xyXG4gIGJhY2tncm91bmQtY29sb3I6ICNlZWU7XHJcbiAgYm9yZGVyLXJhZGl1czogNHB4O1xyXG59XHJcblxyXG5uYXYgYTpob3ZlciB7XHJcbiAgY29sb3I6ICMwMDA7XHJcbiAgYmFja2dyb3VuZC1jb2xvcjogI2Q2ZDZkNjtcclxufVxyXG5cclxubmF2IGEuYWN0aXZlIHtcclxufVxyXG5cclxuQG1lZGlhIChtYXgtd2lkdGg6IDU3NnB4KSB7XHJcbiAgI3N1Ym1haW4tZGl2IHtcclxuICAgIG1hcmdpbi1yaWdodDogMDtcclxuICB9XHJcbiAgI21haW4tZGl2IHtcclxuICAgIHBhZGRpbmctcmlnaHQ6IDA7XHJcbiAgfVxyXG59XHJcbiJdfQ== */"

/***/ }),

/***/ "./src/app/app.component.html":
/*!************************************!*\
  !*** ./src/app/app.component.html ***!
  \************************************/
/*! no static exports found */
/***/ (function(module, exports) {

module.exports = "<div id=\"main-div\" class=\"col-12\">\r\n  <!--<hr id=\"dashboard-hr\">-->\r\n\r\n  <div id=\"submain-div\" class=\"row col-12\">\r\n    <div class=\"col-1 display-if-sm\"></div>\r\n\r\n    <div class=\"col-sm-10 col-12\">\r\n      <div id=\"dashboard-hr\"></div>\r\n      <div id=\"outlet-wrapper\" class=\"col-12\">\r\n        <router-outlet></router-outlet>\r\n      </div>\r\n\r\n    </div>\r\n  </div>\r\n</div>\r\n\r\n<!-- DEV/INCLUDE -->\r\n<!-- put that in spix -->\r\n<!--<div class=\"smartphones\">-->\r\n  <!--<p class=\"hea1\">mad. has encountered a problem :'(</p>-->\r\n  <!--<p>The display screen is too small [LT450px]-->\r\n    <!--and this website will not support smartphone-->\r\n    <!--or small-screen browsing in the near future.-->\r\n    <!--Please find a larger display and use-->\r\n    <!--a modern browser [Firefox, Chrome] for-->\r\n    <!--maximum compatibility.-->\r\n  <!--</p>-->\r\n  <!--<p>-->\r\n    <!--If the problem persists then it means that you-->\r\n    <!--still haven't found a proper display. Here is-->\r\n    <!--what you should do:-->\r\n    <!--while the width of your display is smaller than-->\r\n    <!--450 pixels, then-->\r\n    <!--try connecting a <i>strictly</i> wider display.-->\r\n  <!--</p>-->\r\n<!--</div>-->\r\n"

/***/ }),

/***/ "./src/app/app.component.ts":
/*!**********************************!*\
  !*** ./src/app/app.component.ts ***!
  \**********************************/
/*! exports provided: AppComponent */
/***/ (function(module, __webpack_exports__, __webpack_require__) {

"use strict";
__webpack_require__.r(__webpack_exports__);
/* harmony export (binding) */ __webpack_require__.d(__webpack_exports__, "AppComponent", function() { return AppComponent; });
/* harmony import */ var tslib__WEBPACK_IMPORTED_MODULE_0__ = __webpack_require__(/*! tslib */ "./node_modules/tslib/tslib.es6.js");
/* harmony import */ var _angular_core__WEBPACK_IMPORTED_MODULE_1__ = __webpack_require__(/*! @angular/core */ "./node_modules/@angular/core/fesm5/core.js");


var AppComponent = /** @class */ (function () {
    function AppComponent() {
        this.title = 'mad.';
    }
    AppComponent = tslib__WEBPACK_IMPORTED_MODULE_0__["__decorate"]([
        Object(_angular_core__WEBPACK_IMPORTED_MODULE_1__["Component"])({
            selector: 'app-root',
            template: __webpack_require__(/*! ./app.component.html */ "./src/app/app.component.html"),
            styles: [__webpack_require__(/*! ./app.component.css */ "./src/app/app.component.css")]
        })
    ], AppComponent);
    return AppComponent;
}());



/***/ }),

/***/ "./src/app/app.hyphenation.js":
/*!************************************!*\
  !*** ./src/app/app.hyphenation.js ***!
  \************************************/
/*! no static exports found */
/***/ (function(module, exports) {

/*
 *  Hyphenator 5.3.0 - client side hyphenation for webbrowsers
 *  Copyright (C) 2017  Mathias Nater, Zürich (mathiasnater at gmail dot com)
 *  https://github.com/mnater/Hyphenator
 *
 *  Released under the MIT license
 *  http://mnater.github.io/Hyphenator/LICENSE.txt
 */

var Hyphenator;Hyphenator=(function(window){"use strict";var contextWindow=window;var supportedLangs=(function(){var r={},o=function(code,file,script,prompt){r[code]={"file":file,"script":script,"prompt":prompt};};o("be","be.js",1,"Мова гэтага сайта не можа быць вызначаны аўтаматычна. Калі ласка пакажыце мову:");o("ca","ca.js",0,"");o("cs","cs.js",0,"Jazyk této internetové stránky nebyl automaticky rozpoznán. Určete prosím její jazyk:");o("cu","cu.js",1,"Ꙗ҆зы́къ сегѡ̀ са́йта не мо́жетъ ѡ҆предѣле́нъ бы́ти. Прошꙋ́ тѧ ᲂу҆каза́ти ꙗ҆зы́къ:");o("da","da.js",0,"Denne websides sprog kunne ikke bestemmes. Angiv venligst sprog:");o("bn","bn.js",4,"");o("de","de.js",0,"Die Sprache dieser Webseite konnte nicht automatisch bestimmt werden. Bitte Sprache angeben:");o("el","el-monoton.js",6,"");o("el-monoton","el-monoton.js",6,"");o("el-polyton","el-polyton.js",6,"");o("en","en-us.js",0,"The language of this website could not be determined automatically. Please indicate the main language:");o("en-gb","en-gb.js",0,"The language of this website could not be determined automatically. Please indicate the main language:");o("en-us","en-us.js",0,"The language of this website could not be determined automatically. Please indicate the main language:");o("eo","eo.js",0,"La lingvo de ĉi tiu retpaĝo ne rekoneblas aŭtomate. Bonvolu indiki ĝian ĉeflingvon:");o("es","es.js",0,"El idioma del sitio no pudo determinarse autom%E1ticamente. Por favor, indique el idioma principal:");o("et","et.js",0,"Veebilehe keele tuvastamine ebaõnnestus, palun valige kasutatud keel:");o("fi","fi.js",0,"Sivun kielt%E4 ei tunnistettu automaattisesti. M%E4%E4rit%E4 sivun p%E4%E4kieli:");o("fr","fr.js",0,"La langue de ce site n%u2019a pas pu %EAtre d%E9termin%E9e automatiquement. Veuillez indiquer une langue, s.v.p.%A0:");o("ga","ga.js",0,"Níorbh fhéidir teanga an tsuímh a fháil go huathoibríoch. Cuir isteach príomhtheanga an tsuímh:");o("grc","grc.js",6,"");o("gu","gu.js",7,"");o("hi","hi.js",5,"");o("hu","hu.js",0,"A weboldal nyelvét nem sikerült automatikusan megállapítani. Kérem adja meg a nyelvet:");o("hy","hy.js",3,"Չհաջողվեց հայտնաբերել այս կայքի լեզուն։ Խնդրում ենք նշեք հիմնական լեզուն՝");o("it","it.js",0,"Lingua del sito sconosciuta. Indicare una lingua, per favore:");o("ka","ka.js",16,"");o("kn","kn.js",8,"ಜಾಲ ತಾಣದ ಭಾಷೆಯನ್ನು ನಿರ್ಧರಿಸಲು ಸಾಧ್ಯವಾಗುತ್ತಿಲ್ಲ. ದಯವಿಟ್ಟು ಮುಖ್ಯ ಭಾಷೆಯನ್ನು ಸೂಚಿಸಿ:");o("la","la.js",0,"");o("lt","lt.js",0,"Nepavyko automatiškai nustatyti šios svetainės kalbos. Prašome įvesti kalbą:");o("lv","lv.js",0,"Šīs lapas valodu nevarēja noteikt automātiski. Lūdzu norādiet pamata valodu:");o("ml","ml.js",10,"ഈ വെ%u0D2C%u0D4D%u200Cസൈറ്റിന്റെ ഭാഷ കണ്ടുപിടിയ്ക്കാ%u0D28%u0D4D%u200D കഴിഞ്ഞില്ല. ഭാഷ ഏതാണെന്നു തിരഞ്ഞെടുക്കുക:");o("nb","nb-no.js",0,"Nettstedets språk kunne ikke finnes automatisk. Vennligst oppgi språk:");o("no","nb-no.js",0,"Nettstedets språk kunne ikke finnes automatisk. Vennligst oppgi språk:");o("nb-no","nb-no.js",0,"Nettstedets språk kunne ikke finnes automatisk. Vennligst oppgi språk:");o("nl","nl.js",0,"De taal van deze website kan niet automatisch worden bepaald. Geef de hoofdtaal op:");o("or","or.js",11,"");o("pa","pa.js",13,"");o("pl","pl.js",0,"Języka tej strony nie można ustalić automatycznie. Proszę wskazać język:");o("pt","pt.js",0,"A língua deste site não pôde ser determinada automaticamente. Por favor indique a língua principal:");o("ru","ru.js",1,"Язык этого сайта не может быть определен автоматически. Пожалуйста укажите язык:");o("sk","sk.js",0,"");o("sl","sl.js",0,"Jezika te spletne strani ni bilo mogoče samodejno določiti. Prosim navedite jezik:");o("sr-cyrl","sr-cyrl.js",1,"Језик овог сајта није детектован аутоматски. Молим вас наведите језик:");o("sr-latn","sr-latn.js",0,"Jezika te spletne strani ni bilo mogoče samodejno določiti. Prosim navedite jezik:");o("sv","sv.js",0,"Spr%E5ket p%E5 den h%E4r webbplatsen kunde inte avg%F6ras automatiskt. V%E4nligen ange:");o("ta","ta.js",14,"");o("te","te.js",15,"");o("tr","tr.js",0,"Bu web sitesinin dili otomatik olarak tespit edilememiştir. Lütfen dökümanın dilini seçiniz%A0:");o("uk","uk.js",1,"Мова цього веб-сайту не може бути визначена автоматично. Будь ласка, вкажіть головну мову:");o("ro","ro.js",0,"Limba acestui sit nu a putut fi determinată automat. Alege limba principală:");return r;}());var locality=(function getLocality(){var r={isBookmarklet:false,basePath:"//mnater.github.io/Hyphenator/",isLocal:false};var fullPath;function getBasePath(path){if(!path){return r.basePath;}return path.substring(0,path.lastIndexOf("/")+1);}function findCurrentScript(){var scripts=contextWindow.document.getElementsByTagName("script");var num=scripts.length-1;var currScript;var src;while(num>=0){currScript=scripts[num];if((currScript.src||currScript.hasAttribute("src"))&&currScript.src.indexOf("Hyphenator")!==-1){src=currScript.src;break;}num-=1;}return src;}if(!!document.currentScript){fullPath=document.currentScript.src;}else{fullPath=findCurrentScript();}r.basePath=getBasePath(fullPath);if(fullPath&&fullPath.indexOf("bm=true")!==-1){r.isBookmarklet=true;}if(window.location.href.indexOf(r.basePath)!==-1){r.isLocal=true;}return r;}());var basePath=locality.basePath;var isLocal=locality.isLocal;var documentLoaded=false;var persistentConfig=false;var doFrames=false;var dontHyphenate={"video":true,"audio":true,"script":true,"code":true,"pre":true,"img":true,"br":true,"samp":true,"kbd":true,"var":true,"abbr":true,"acronym":true,"sub":true,"sup":true,"button":true,"option":true,"label":true,"textarea":true,"input":true,"math":true,"svg":true,"style":true};var enableCache=true;var storageType="local";var storage;var enableReducedPatternSet=false;var enableRemoteLoading=true;var displayToggleBox=false;var onError=function(e){window.alert("Hyphenator.js says:\n\nAn Error occurred:\n"+e.message);};var onWarning=function(e){window.console.log(e.message);};function createElem(tagname,context){context=context||contextWindow;var el;if(window.document.createElementNS){el=context.document.createElementNS("http://www.w3.org/1999/xhtml",tagname);}else if(window.document.createElement){el=context.document.createElement(tagname);}return el;}function forEachKey(o,f){var k;if(Object.hasOwnProperty("keys")){Object.keys(o).forEach(f);}else{for(k in o){if(o.hasOwnProperty(k)){f(k);}}}}var css3=false;function css3_gethsupport(){var support=false,supportedBrowserLangs={},property="",checkLangSupport,createLangSupportChecker=function(prefix){var testStrings=["aabbccddeeffgghhiijjkkllmmnnooppqqrrssttuuvvwwxxyyzz","абвгдеёжзийклмнопрстуфхцчшщъыьэюя","أبتثجحخدذرزسشصضطظعغفقكلمنهوي","աբգդեզէըթժիլխծկհձղճմյնշոչպջռսվտրցւփքօֆ","ঁংঃঅআইঈউঊঋঌএঐওঔকখগঘঙচছজঝঞটঠডঢণতথদধনপফবভমযরলশষসহ়ঽািীুূৃৄেৈোৌ্ৎৗড়ঢ়য়ৠৡৢৣ","ँंःअआइईउऊऋऌएऐओऔकखगघङचछजझञटठडढणतथदधनपफबभमयरलळवशषसहऽािीुूृॄेैोौ्॒॑ॠॡॢॣ","αβγδεζηθικλμνξοπρσςτυφχψω","બહઅઆઇઈઉઊઋૠએઐઓઔાિીુૂૃૄૢૣેૈોૌકખગઘઙચછજઝઞટઠડઢણતથદધનપફસભમયરલળવશષ","ಂಃಅಆಇಈಉಊಋಌಎಏಐಒಓಔಕಖಗಘಙಚಛಜಝಞಟಠಡಢಣತಥದಧನಪಫಬಭಮಯರಱಲಳವಶಷಸಹಽಾಿೀುೂೃೄೆೇೈೊೋೌ್ೕೖೞೠೡ","ກຂຄງຈຊຍດຕຖທນບປຜຝພຟມຢຣລວສຫອຮະັາິີຶືຸູົຼເແໂໃໄ່້໊໋ໜໝ","ംഃഅആഇഈഉഊഋഌഎഏഐഒഓഔകഖഗഘങചഛജഝഞടഠഡഢണതഥദധനപഫബഭമയരറലളഴവശഷസഹാിീുൂൃെേൈൊോൌ്ൗൠൡൺൻർൽൾൿ","ଁଂଃଅଆଇଈଉଊଋଌଏଐଓଔକଖଗଘଙଚଛଜଝଞଟଠଡଢଣତଥଦଧନପଫବଭମଯରଲଳଵଶଷସହାିୀୁୂୃେୈୋୌ୍ୗୠୡ","أبتثجحخدذرزسشصضطظعغفقكلمنهوي","ਁਂਃਅਆਇਈਉਊਏਐਓਔਕਖਗਘਙਚਛਜਝਞਟਠਡਢਣਤਥਦਧਨਪਫਬਭਮਯਰਲਲ਼ਵਸ਼ਸਹਾਿੀੁੂੇੈੋੌ੍ੰੱ","ஃஅஆஇஈஉஊஎஏஐஒஓஔகஙசஜஞடணதநனபமயரறலளழவஷஸஹாிீுூெேைொோௌ்ௗ","ఁంఃఅఆఇఈఉఊఋఌఎఏఐఒఓఔకఖగఘఙచఛజఝఞటఠడఢణతథదధనపఫబభమయరఱలళవశషసహాిీుూృౄెేైొోౌ్ౕౖౠౡ","აიერთხტუფბლდნვკწსგზმქყშჩცძჭჯოღპჟჰ"],f=function(lang){var shadow,computedHeight,bdy,r=false;if(supportedBrowserLangs.hasOwnProperty(lang)){r=supportedBrowserLangs[lang];}else if(supportedLangs.hasOwnProperty(lang)){bdy=window.document.getElementsByTagName("body")[0];shadow=createElem("div",window);shadow.id="Hyphenator_LanguageChecker";shadow.style.width="5em";shadow.style.padding="0";shadow.style.border="none";shadow.style[prefix]="auto";shadow.style.hyphens="auto";shadow.style.fontSize="12px";shadow.style.lineHeight="12px";shadow.style.wordWrap="normal";shadow.style.wordBreak="normal";shadow.style.visibility="hidden";shadow.lang=lang;shadow.style["-webkit-locale"]="\""+lang+"\"";shadow.innerHTML=testStrings[supportedLangs[lang].script];bdy.appendChild(shadow);computedHeight=shadow.offsetHeight;bdy.removeChild(shadow);r=!!(computedHeight>12);supportedBrowserLangs[lang]=r;}else{r=false;}return r;};return f;},s;if(window.getComputedStyle){s=window.getComputedStyle(window.document.getElementsByTagName("body")[0],null);if(s.hyphens!==undefined){support=true;property="hyphens";checkLangSupport=createLangSupportChecker("hyphens");}else if(s["-webkit-hyphens"]!==undefined){support=true;property="-webkit-hyphens";checkLangSupport=createLangSupportChecker("-webkit-hyphens");}else if(s.MozHyphens!==undefined){support=true;property="-moz-hyphens";checkLangSupport=createLangSupportChecker("MozHyphens");}else if(s["-ms-hyphens"]!==undefined){support=true;property="-ms-hyphens";checkLangSupport=createLangSupportChecker("-ms-hyphens");}}return{support:support,property:property,supportedBrowserLangs:supportedBrowserLangs,checkLangSupport:checkLangSupport};}var css3_h9n;var hyphenateClass="hyphenate";var urlHyphenateClass="urlhyphenate";var classPrefix="Hyphenator"+Math.round(Math.random()*1000);var hideClass=classPrefix+"hide";var hideClassRegExp=new RegExp("\\s?\\b"+hideClass+"\\b","g");var unhideClass=classPrefix+"unhide";var unhideClassRegExp=new RegExp("\\s?\\b"+unhideClass+"\\b","g");var css3hyphenateClass=classPrefix+"css3hyphenate";var css3hyphenateClassHandle;var dontHyphenateClass="donthyphenate";var min=6;var leftmin=0;var rightmin=0;var compound="auto";var orphanControl=1;var isBookmarklet=locality.isBookmarklet;var mainLanguage=null;var defaultLanguage="";var elements=(function(){var makeElement=function(element){return{element:element,hyphenated:false,treated:false};},makeElementCollection=function(){var counters=[0,0],list={},add=function(el,lang){var elo=makeElement(el);if(!list.hasOwnProperty(lang)){list[lang]=[];}list[lang].push(elo);counters[0]+=1;return elo;},each=function(fn){forEachKey(list,function(k){if(fn.length===2){fn(k,list[k]);}else{fn(list[k]);}});};return{counters:counters,list:list,add:add,each:each};};return makeElementCollection();}());var exceptions={};var docLanguages={};var url="(?:\\w*:\/\/)?(?:(?:\\w*:)?(?:\\w*)@)?(?:(?:(?:[\\d]{1,3}\\.){3}(?:[\\d]{1,3}))|(?:(?:www\\.|[a-zA-Z]\\.)?[a-zA-Z0-9\\-]+(?:\\.[a-z]{2,})+))(?::\\d*)?(?:\/[\\w#!:\\.?\\+=&%@!\\-]*)*";var mail="[\\w-\\.]+@[\\w\\.]+";var zeroWidthSpace=(function(){var zws,ua=window.navigator.userAgent.toLowerCase();zws=String.fromCharCode(8203);if(ua.indexOf("msie 6")!==-1){zws="";}if(ua.indexOf("opera")!==-1&&ua.indexOf("version/10.00")!==-1){zws="";}return zws;}());var onBeforeWordHyphenation=function(word){return word;};var onAfterWordHyphenation=function(word){return word;};var onHyphenationDone=function(context){return context;};var selectorFunction=false;function flattenNodeList(nl){var parentElements=[],i=1,j=0,isParent=true;parentElements.push(nl[0]);while(i<nl.length){while(j<parentElements.length){if(parentElements[j].contains(nl[i])){isParent=false;break;}j+=1;}if(isParent){parentElements.push(nl[i]);}isParent=true;i+=1;}return parentElements;}function mySelectorFunction(hyphenateClass){var tmp,el=[],i=0;if(window.document.getElementsByClassName){el=contextWindow.document.getElementsByClassName(hyphenateClass);}else if(window.document.querySelectorAll){el=contextWindow.document.querySelectorAll("."+hyphenateClass);}else{tmp=contextWindow.document.getElementsByTagName("*");while(i<tmp.length){if(tmp[i].className.indexOf(hyphenateClass)!==-1&&tmp[i].className.indexOf(dontHyphenateClass)===-1){el.push(tmp[i]);}i+=1;}}return el;}function selectElements(){var elems;if(selectorFunction){elems=selectorFunction();}else{elems=mySelectorFunction(hyphenateClass);}if(elems.length!==0){elems=flattenNodeList(elems);}return elems;}var intermediateState="hidden";var unhide="wait";var CSSEditors=[];function makeCSSEdit(w){w=w||window;var doc=w.document,sheet=(function(){var i=0,l=doc.styleSheets.length,s,element,r=false;while(i<l){s=doc.styleSheets[i];try{if(!!s.cssRules){r=s;break;}}catch(ignore){}i+=1;}if(r===false){element=doc.createElement("style");element.type="text/css";doc.getElementsByTagName("head")[0].appendChild(element);r=doc.styleSheets[doc.styleSheets.length-1];}return r;}()),changes=[],findRule=function(sel){var s,rule,sheets=w.document.styleSheets,rules,i=0,j=0,r=false;while(i<sheets.length){s=sheets[i];try{if(!!s.cssRules){rules=s.cssRules;}else if(!!s.rules){rules=s.rules;}}catch(ignore){}if(!!rules&&!!rules.length){while(j<rules.length){rule=rules[j];if(rule.selectorText===sel){r={index:j,rule:rule};}j+=1;}}i+=1;}return r;},addRule=function(sel,rulesStr){var i,r;if(!!sheet.insertRule){if(!!sheet.cssRules){i=sheet.cssRules.length;}else{i=0;}r=sheet.insertRule(sel+"{"+rulesStr+"}",i);}else if(!!sheet.addRule){if(!!sheet.rules){i=sheet.rules.length;}else{i=0;}sheet.addRule(sel,rulesStr,i);r=i;}return r;},removeRule=function(sheet,index){if(sheet.deleteRule){sheet.deleteRule(index);}else{sheet.removeRule(index);}};return{setRule:function(sel,rulesString){var i,existingRule,cssText;existingRule=findRule(sel);if(!!existingRule){if(!!existingRule.rule.cssText){cssText=existingRule.rule.cssText;}else{cssText=existingRule.rule.style.cssText.toLowerCase();}if(cssText!==sel+" { "+rulesString+" }"){if(cssText.indexOf(rulesString)!==-1){existingRule.rule.style.visibility="";}i=addRule(sel,rulesString);changes.push({sheet:sheet,index:i});}}else{i=addRule(sel,rulesString);changes.push({sheet:sheet,index:i});}},clearChanges:function(){var change=changes.pop();while(!!change){removeRule(change.sheet,change.index);change=changes.pop();}}};}var hyphen=String.fromCharCode(173);var urlhyphen=zeroWidthSpace;function hyphenateURL(url){var tmp=url.replace(/([:\/.?#&\-_,;!@]+)/gi,"$&"+urlhyphen),parts=tmp.split(urlhyphen),i=0;while(i<parts.length){if(parts[i].length>(2*min)){parts[i]=parts[i].replace(/(\w{3})(\w)/gi,"$1"+urlhyphen+"$2");}i+=1;}if(parts[parts.length-1]===""){parts.pop();}return parts.join(urlhyphen);}var safeCopy=true;var zeroTimeOut=(function(){if(window.postMessage&&window.addEventListener){return(function(){var timeouts=[],msg="Hyphenator_zeroTimeOut_message",setZeroTimeOut=function(fn){timeouts.push(fn);window.postMessage(msg,"*");},handleMessage=function(event){if(event.source===window&&event.data===msg){event.stopPropagation();if(timeouts.length>0){timeouts.shift()();}}};window.addEventListener("message",handleMessage,true);return setZeroTimeOut;}());}return function(fn){window.setTimeout(fn,0);};}());var hyphRunFor={};function removeHyphenationFromElement(el){var h,u,i=0,n;if(".\\+*?[^]$(){}=!<>|:-".indexOf(hyphen)!==-1){h="\\"+hyphen;}else{h=hyphen;}if(".\\+*?[^]$(){}=!<>|:-".indexOf(urlhyphen)!==-1){u="\\"+urlhyphen;}else{u=urlhyphen;}n=el.childNodes[i];while(!!n){if(n.nodeType===3){n.data=n.data.replace(new RegExp(h,"g"),"");n.data=n.data.replace(new RegExp(u,"g"),"");}else if(n.nodeType===1){removeHyphenationFromElement(n);}i+=1;n=el.childNodes[i];}}var copy=(function(){var factory=function(){var registeredElements=[];var oncopyHandler=function(e){e=e||window.event;var shadow,selection,range,rangeShadow,restore,target=e.target||e.srcElement,currDoc=target.ownerDocument,bdy=currDoc.getElementsByTagName("body")[0],targetWindow=currDoc.defaultView||currDoc.parentWindow;if(target.tagName&&dontHyphenate[target.tagName.toLowerCase()]){return;}shadow=currDoc.createElement("div");shadow.style.color=window.getComputedStyle?targetWindow.getComputedStyle(bdy,null).backgroundColor:"#FFFFFF";shadow.style.fontSize="0px";bdy.appendChild(shadow);if(!!window.getSelection){selection=targetWindow.getSelection();range=selection.getRangeAt(0);shadow.appendChild(range.cloneContents());removeHyphenationFromElement(shadow);selection.selectAllChildren(shadow);restore=function(){shadow.parentNode.removeChild(shadow);selection.removeAllRanges();selection.addRange(range);};}else{selection=targetWindow.document.selection;range=selection.createRange();shadow.innerHTML=range.htmlText;removeHyphenationFromElement(shadow);rangeShadow=bdy.createTextRange();rangeShadow.moveToElementText(shadow);rangeShadow.select();restore=function(){shadow.parentNode.removeChild(shadow);if(range.text!==""){range.select();}};}zeroTimeOut(restore);};var removeOnCopy=function(){var i=registeredElements.length-1;while(i>=0){if(window.removeEventListener){registeredElements[i].removeEventListener("copy",oncopyHandler,true);}else{registeredElements[i].detachEvent("oncopy",oncopyHandler);}i-=1;}};var reactivateOnCopy=function(){var i=registeredElements.length-1;while(i>=0){if(window.addEventListener){registeredElements[i].addEventListener("copy",oncopyHandler,true);}else{registeredElements[i].attachEvent("oncopy",oncopyHandler);}i-=1;}};var registerOnCopy=function(el){registeredElements.push(el);if(window.addEventListener){el.addEventListener("copy",oncopyHandler,true);}else{el.attachEvent("oncopy",oncopyHandler);}};return{oncopyHandler:oncopyHandler,removeOnCopy:removeOnCopy,registerOnCopy:registerOnCopy,reactivateOnCopy:reactivateOnCopy};};return(safeCopy?factory():false);}());function runWhenLoaded(w,f){var toplevel,add=window.document.addEventListener?"addEventListener":"attachEvent",rem=window.document.addEventListener?"removeEventListener":"detachEvent",pre=window.document.addEventListener?"":"on";function init(context){if(hyphRunFor[context.location.href]){onWarning(new Error("Warning: multiple execution of Hyphenator.run() – This may slow down the script!"));}contextWindow=context||window;f();hyphRunFor[contextWindow.location.href]=true;}function doScrollCheck(){try{w.document.documentElement.doScroll("left");}catch(ignore){window.setTimeout(doScrollCheck,1);return;}if(!hyphRunFor[w.location.href]){documentLoaded=true;init(w);}}function doOnEvent(e){var i=0,fl,haveAccess;if(!!e&&e.type==="readystatechange"&&w.document.readyState!=="interactive"&&w.document.readyState!=="complete"){return;}w.document[rem](pre+"DOMContentLoaded",doOnEvent,false);w.document[rem](pre+"readystatechange",doOnEvent,false);fl=w.frames.length;if(fl===0||!doFrames){w[rem](pre+"load",doOnEvent,false);documentLoaded=true;init(w);}else if(doFrames&&fl>0){if(!!e&&e.type==="load"){w[rem](pre+"load",doOnEvent,false);while(i<fl){haveAccess=undefined;try{haveAccess=w.frames[i].document.toString();}catch(ignore){haveAccess=undefined;}if(!!haveAccess){runWhenLoaded(w.frames[i],f);}i+=1;}init(w);}}}if(documentLoaded||w.document.readyState==="complete"){documentLoaded=true;doOnEvent({type:"load"});}else{w.document[add](pre+"DOMContentLoaded",doOnEvent,false);w.document[add](pre+"readystatechange",doOnEvent,false);w[add](pre+"load",doOnEvent,false);toplevel=false;try{toplevel=!window.frameElement;}catch(ignore){}if(toplevel&&w.document.documentElement.doScroll){doScrollCheck();}}}function getLang(el,fallback){try{return!!el.getAttribute("lang")?el.getAttribute("lang").toLowerCase():!!el.getAttribute("xml:lang")?el.getAttribute("xml:lang").toLowerCase():el.tagName.toLowerCase()!=="html"?getLang(el.parentNode,fallback):fallback?mainLanguage:null;}catch(ignore){return fallback?mainLanguage:null;}}function autoSetMainLanguage(w){w=w||contextWindow;var el=w.document.getElementsByTagName("html")[0],m=w.document.getElementsByTagName("meta"),i=0,getLangFromUser=function(){var text="";var ul="";var languageHint=(function(){var r="";forEachKey(supportedLangs,function(k){r+=k+", ";});r=r.substring(0,r.length-2);return r;}());ul=window.navigator.language||window.navigator.userLanguage;ul=ul.substring(0,2);if(!!supportedLangs[ul]&&supportedLangs[ul].prompt!==""){text=supportedLangs[ul].prompt;}else{text=supportedLangs.en.prompt;}text+=" (ISO 639-1)\n\n"+languageHint;return window.prompt(window.unescape(text),ul).toLowerCase();};mainLanguage=getLang(el,false);if(!mainLanguage){while(i<m.length){if(!!m[i].getAttribute("http-equiv")&&(m[i].getAttribute("http-equiv").toLowerCase()==="content-language")){mainLanguage=m[i].getAttribute("content").toLowerCase();}if(!!m[i].getAttribute("name")&&(m[i].getAttribute("name").toLowerCase()==="dc.language")){mainLanguage=m[i].getAttribute("content").toLowerCase();}if(!!m[i].getAttribute("name")&&(m[i].getAttribute("name").toLowerCase()==="language")){mainLanguage=m[i].getAttribute("content").toLowerCase();}i+=1;}}if(!mainLanguage&&doFrames&&(!!contextWindow.frameElement)){autoSetMainLanguage(window.parent);}if(!mainLanguage&&defaultLanguage!==""){mainLanguage=defaultLanguage;}if(!mainLanguage){mainLanguage=getLangFromUser();}el.lang=mainLanguage;}function gatherDocumentInfos(){var elToProcess,urlhyphenEls,tmp,i=0;function process(el,pLang,isChild){var n,j=0,hyphenate=true,eLang,useCSS3=function(){css3hyphenateClassHandle=makeCSSEdit(contextWindow);css3hyphenateClassHandle.setRule("."+css3hyphenateClass,css3_h9n.property+": auto;");css3hyphenateClassHandle.setRule("."+dontHyphenateClass,css3_h9n.property+": manual;");if((eLang!==pLang)&&css3_h9n.property.indexOf("webkit")!==-1){css3hyphenateClassHandle.setRule("."+css3hyphenateClass,"-webkit-locale : "+eLang+";");}el.className=el.className+" "+css3hyphenateClass;},useHyphenator=function(){if(isBookmarklet&&eLang!==mainLanguage){return;}if(supportedLangs.hasOwnProperty(eLang)){docLanguages[eLang]=true;}else{if(supportedLangs.hasOwnProperty(eLang.split("-")[0])){eLang=eLang.split("-")[0];docLanguages[eLang]=true;}else if(!isBookmarklet){hyphenate=false;onError(new Error("Language \""+eLang+"\" is not yet supported."));}}if(hyphenate){if(intermediateState==="hidden"){el.className=el.className+" "+hideClass;}elements.add(el,eLang);}};isChild=isChild||false;if(el.lang&&typeof el.lang==="string"){eLang=el.lang.toLowerCase();}else if(!!pLang&&pLang!==""){eLang=pLang.toLowerCase();}else{eLang=getLang(el,true);}if(!isChild){if(css3&&css3_h9n.support&&!!css3_h9n.checkLangSupport(eLang)){useCSS3();}else{if(safeCopy){copy.registerOnCopy(el);}useHyphenator();}}else{if(eLang!==pLang){if(css3&&css3_h9n.support&&!!css3_h9n.checkLangSupport(eLang)){useCSS3();}else{useHyphenator();}}else{if(!css3||!css3_h9n.support||!css3_h9n.checkLangSupport(eLang)){useHyphenator();}}}n=el.childNodes[j];while(!!n){if(n.nodeType===1&&!dontHyphenate[n.nodeName.toLowerCase()]&&n.className.indexOf(dontHyphenateClass)===-1&&n.className.indexOf(urlHyphenateClass)===-1&&!elToProcess[n]){process(n,eLang,true);}j+=1;n=el.childNodes[j];}}function processUrlStyled(el){var n,j=0;n=el.childNodes[j];while(!!n){if(n.nodeType===1&&!dontHyphenate[n.nodeName.toLowerCase()]&&n.className.indexOf(dontHyphenateClass)===-1&&n.className.indexOf(hyphenateClass)===-1&&!urlhyphenEls[n]){processUrlStyled(n);}else if(n.nodeType===3){if(safeCopy){copy.registerOnCopy(n.parentNode);}elements.add(n.parentNode,"urlstyled");}j+=1;n=el.childNodes[j];}}if(css3){css3_h9n=css3_gethsupport();}if(isBookmarklet){elToProcess=contextWindow.document.getElementsByTagName("body")[0];process(elToProcess,mainLanguage,false);}else{if(!css3&&intermediateState==="hidden"){CSSEditors.push(makeCSSEdit(contextWindow));CSSEditors[CSSEditors.length-1].setRule("."+hyphenateClass,"visibility: hidden;");CSSEditors[CSSEditors.length-1].setRule("."+hideClass,"visibility: hidden;");CSSEditors[CSSEditors.length-1].setRule("."+unhideClass,"visibility: visible;");}elToProcess=selectElements();tmp=elToProcess[i];while(!!tmp){process(tmp,"",false);i+=1;tmp=elToProcess[i];}urlhyphenEls=mySelectorFunction(urlHyphenateClass);i=0;tmp=urlhyphenEls[i];while(!!tmp){processUrlStyled(tmp);i+=1;tmp=urlhyphenEls[i];}}if(elements.counters[0]===0){i=0;while(i<CSSEditors.length){CSSEditors[i].clearChanges();i+=1;}onHyphenationDone(contextWindow.location.href);}}function makeCharMap(){var int2code=[],code2int={},add=function(newValue){if(!code2int[newValue]){int2code.push(newValue);code2int[newValue]=int2code.length-1;}};return{int2code:int2code,code2int:code2int,add:add};}function makeValueStore(len){var indexes=(function(){var arr;if(Object.prototype.hasOwnProperty.call(window,"Uint32Array")){arr=new window.Uint32Array(3);arr[0]=1;arr[1]=1;arr[2]=1;}else{arr=[1,1,1];}return arr;}()),keys=(function(){var i,r;if(Object.prototype.hasOwnProperty.call(window,"Uint8Array")){return new window.Uint8Array(len);}r=[];r.length=len;i=r.length-1;while(i>=0){r[i]=0;i-=1;}return r;}()),add=function(p){keys[indexes[1]]=p;indexes[2]=indexes[1];indexes[1]+=1;},add0=function(){indexes[1]+=1;},finalize=function(){var start=indexes[0];keys[indexes[2]+1]=255;indexes[0]=indexes[2]+2;indexes[1]=indexes[0];return start;};return{keys:keys,add:add,add0:add0,finalize:finalize};}function convertPatternsToArray(lo){var trieNextEmptyRow=0,i,charMapc2i,valueStore,indexedTrie,trieRowLength,extract=function(patternSizeInt,patterns){var charPos=0,charCode=0,mappedCharCode=0,rowStart=0,nextRowStart=0,prevWasDigit=false;while(charPos<patterns.length){charCode=patterns.charCodeAt(charPos);if((charPos+1)%patternSizeInt!==0){if(charCode<=57&&charCode>=49){valueStore.add(charCode-48);prevWasDigit=true;}else{if(!prevWasDigit){valueStore.add0();}prevWasDigit=false;if(nextRowStart===-1){nextRowStart=trieNextEmptyRow+trieRowLength;trieNextEmptyRow=nextRowStart;indexedTrie[rowStart+mappedCharCode*2]=nextRowStart;}mappedCharCode=charMapc2i[charCode];rowStart=nextRowStart;nextRowStart=indexedTrie[rowStart+mappedCharCode*2];if(nextRowStart===0){indexedTrie[rowStart+mappedCharCode*2]=-1;nextRowStart=-1;}}}else{if(charCode<=57&&charCode>=49){valueStore.add(charCode-48);indexedTrie[rowStart+mappedCharCode*2+1]=valueStore.finalize();}else{if(!prevWasDigit){valueStore.add0();}valueStore.add0();if(nextRowStart===-1){nextRowStart=trieNextEmptyRow+trieRowLength;trieNextEmptyRow=nextRowStart;indexedTrie[rowStart+mappedCharCode*2]=nextRowStart;}mappedCharCode=charMapc2i[charCode];rowStart=nextRowStart;if(indexedTrie[rowStart+mappedCharCode*2]===0){indexedTrie[rowStart+mappedCharCode*2]=-1;}indexedTrie[rowStart+mappedCharCode*2+1]=valueStore.finalize();}rowStart=0;nextRowStart=0;prevWasDigit=false;}charPos+=1;}};lo.charMap=makeCharMap();i=0;while(i<lo.patternChars.length){lo.charMap.add(lo.patternChars.charCodeAt(i));i+=1;}charMapc2i=lo.charMap.code2int;valueStore=makeValueStore(lo.valueStoreLength);lo.valueStore=valueStore;if(Object.prototype.hasOwnProperty.call(window,"Int32Array")){lo.indexedTrie=new window.Int32Array(lo.patternArrayLength*2);}else{lo.indexedTrie=[];lo.indexedTrie.length=lo.patternArrayLength*2;i=lo.indexedTrie.length-1;while(i>=0){lo.indexedTrie[i]=0;i-=1;}}indexedTrie=lo.indexedTrie;trieRowLength=lo.charMap.int2code.length*2;forEachKey(lo.patterns,function(i){extract(parseInt(i,10),lo.patterns[i]);});}function recreatePattern(pattern,nodePoints){var r=[],c=pattern.split(""),i=0;while(i<=c.length){if(nodePoints[i]&&nodePoints[i]!==0){r.push(nodePoints[i]);}if(c[i]){r.push(c[i]);}i+=1;}return r.join("");}function convertExceptionsToObject(exc){var w=exc.split(", "),r={},i=0,l=w.length,key;while(i<l){key=w[i].replace(/-/g,"");if(!r.hasOwnProperty(key)){r[key]=w[i];}i+=1;}return r;}function loadPatterns(lang,cb){var location,xhr,head,script,done=false;function getXHRforIElt6(){try{xhr=new window.ActiveXObject("Msxml2.XMLHTTP");}catch(ignore){xhr=null;}}function getXHRforIElt10(){try{xhr=new window.ActiveXObject("Microsoft.XMLHTTP");}catch(ignore){getXHRforIElt6();}}if(supportedLangs.hasOwnProperty(lang)&&!Hyphenator.languages[lang]){location=basePath+"patterns/"+supportedLangs[lang].file;}else{return;}if(isLocal&&!isBookmarklet){xhr=null;try{xhr=new window.XMLHttpRequest();}catch(ignore){getXHRforIElt10();}if(xhr){xhr.open("HEAD",location,true);xhr.setRequestHeader("Cache-Control","no-cache");xhr.onreadystatechange=function(){if(xhr.readyState===2){if(xhr.status>=400){onError(new Error("Could not load\n"+location));delete docLanguages[lang];return;}xhr.abort();}};xhr.send(null);}}if(createElem){head=window.document.getElementsByTagName("head").item(0);script=createElem("script",window);script.src=location;script.type="text/javascript";script.charset="utf8";script.onreadystatechange=function(){if(!done&&(!script.readyState||script.readyState==="loaded"||script.readyState==="complete")){done=true;cb();script.onreadystatechange=null;script.onload=null;if(head&&script.parentNode){head.removeChild(script);}}};script.onload=script.onreadystatechange;head.appendChild(script);}}function createWordRegExp(lang){var lo=Hyphenator.languages[lang],wrd="";if(String.prototype.normalize){wrd="[\\w"+lo.specialChars+lo.specialChars.normalize("NFD")+hyphen+String.fromCharCode(8204)+"-]{"+min+",}(?!:\\/\\/)";}else{wrd="[\\w"+lo.specialChars+hyphen+String.fromCharCode(8204)+"-]{"+min+",}(?!:\\/\\/)";}return wrd;}function prepareLanguagesObj(lang){var lo=Hyphenator.languages[lang];if(!lo.prepared){if(enableCache){lo.cache={};}if(enableReducedPatternSet){lo.redPatSet={};}if(leftmin>lo.leftmin){lo.leftmin=leftmin;}if(rightmin>lo.rightmin){lo.rightmin=rightmin;}if(lo.hasOwnProperty("exceptions")){Hyphenator.addExceptions(lang,lo.exceptions);delete lo.exceptions;}if(exceptions.hasOwnProperty("global")){if(exceptions.hasOwnProperty(lang)){exceptions[lang]+=", "+exceptions.global;}else{exceptions[lang]=exceptions.global;}}if(exceptions.hasOwnProperty(lang)){lo.exceptions=convertExceptionsToObject(exceptions[lang]);delete exceptions[lang];}else{lo.exceptions={};}convertPatternsToArray(lo);lo.genRegExp=new RegExp("("+createWordRegExp(lang)+")|("+url+")|("+mail+")","gi");lo.prepared=true;}}function prepare(callback){var tmp1;function languagesLoaded(){forEachKey(docLanguages,function(l){if(Hyphenator.languages.hasOwnProperty(l)){delete docLanguages[l];if(!!storage){storage.setItem(l,window.JSON.stringify(Hyphenator.languages[l]));}prepareLanguagesObj(l);callback(l);}});}if(!enableRemoteLoading){forEachKey(Hyphenator.languages,function(lang){prepareLanguagesObj(lang);});callback("*");return;}callback("urlstyled");forEachKey(docLanguages,function(lang){if(!!storage&&storage.test(lang)){Hyphenator.languages[lang]=window.JSON.parse(storage.getItem(lang));prepareLanguagesObj(lang);if(exceptions.hasOwnProperty("global")){tmp1=convertExceptionsToObject(exceptions.global);forEachKey(tmp1,function(tmp2){Hyphenator.languages[lang].exceptions[tmp2]=tmp1[tmp2];});}if(exceptions.hasOwnProperty(lang)){tmp1=convertExceptionsToObject(exceptions[lang]);forEachKey(tmp1,function(tmp2){Hyphenator.languages[lang].exceptions[tmp2]=tmp1[tmp2];});delete exceptions[lang];}Hyphenator.languages[lang].genRegExp=new RegExp("("+createWordRegExp(lang)+")|("+url+")|("+mail+")","gi");if(enableCache){if(!Hyphenator.languages[lang].cache){Hyphenator.languages[lang].cache={};}}delete docLanguages[lang];callback(lang);}else{loadPatterns(lang,languagesLoaded);}});languagesLoaded();}var toggleBox=function(){var bdy,myTextNode,text=(Hyphenator.doHyphenation?"Hy-phen-a-tion":"Hyphenation"),myBox=contextWindow.document.getElementById("HyphenatorToggleBox");if(!!myBox){myBox.firstChild.data=text;}else{bdy=contextWindow.document.getElementsByTagName("body")[0];myBox=createElem("div",contextWindow);myBox.setAttribute("id","HyphenatorToggleBox");myBox.setAttribute("class",dontHyphenateClass);myTextNode=contextWindow.document.createTextNode(text);myBox.appendChild(myTextNode);myBox.onclick=Hyphenator.toggleHyphenation;myBox.style.position="absolute";myBox.style.top="0px";myBox.style.right="0px";myBox.style.zIndex="1000";myBox.style.margin="0";myBox.style.backgroundColor="#AAAAAA";myBox.style.color="#FFFFFF";myBox.style.font="6pt Arial";myBox.style.letterSpacing="0.2em";myBox.style.padding="3px";myBox.style.cursor="pointer";myBox.style.WebkitBorderBottomLeftRadius="4px";myBox.style.MozBorderRadiusBottomleft="4px";myBox.style.borderBottomLeftRadius="4px";bdy.appendChild(myBox);}};function doCharSubst(loCharSubst,w){var r=w;forEachKey(loCharSubst,function(subst){r=r.replace(new RegExp(subst,"g"),loCharSubst[subst]);});return r;}var wwAsMappedCharCodeStore=(function(){if(Object.prototype.hasOwnProperty.call(window,"Int32Array")){return new window.Int32Array(64);}return[];}());var wwhpStore=(function(){var r;if(Object.prototype.hasOwnProperty.call(window,"Uint8Array")){r=new window.Uint8Array(64);}else{r=[];}return r;}());function hyphenateCompound(lo,lang,word){var hw,parts,i=0;switch(compound){case"auto":parts=word.split("-");while(i<parts.length){if(parts[i].length>=min){parts[i]=hyphenateWord(lo,lang,parts[i]);}i+=1;}hw=parts.join("-");break;case"all":parts=word.split("-");while(i<parts.length){if(parts[i].length>=min){parts[i]=hyphenateWord(lo,lang,parts[i]);}i+=1;}hw=parts.join("-"+zeroWidthSpace);break;case"hyphen":hw=word.replace("-","-"+zeroWidthSpace);break;default:onError(new Error("Hyphenator.settings: compound setting \""+compound+"\" not known."));}return hw;}function hyphenateWord(lo,lang,word){var pattern="",ww,wwlen,wwhp=wwhpStore,pstart=0,plen,hp,hpc,wordLength=word.length,hw="",charMap=lo.charMap.code2int,charCode,mappedCharCode,row=0,link=0,value=0,values,indexedTrie=lo.indexedTrie,valueStore=lo.valueStore.keys,wwAsMappedCharCode=wwAsMappedCharCodeStore;word=onBeforeWordHyphenation(word,lang);if(word===""){hw="";}else if(enableCache&&lo.cache&&lo.cache.hasOwnProperty(word)){hw=lo.cache[word];}else if(word.indexOf(hyphen)!==-1){hw=word;}else if(lo.exceptions.hasOwnProperty(word)){hw=lo.exceptions[word].replace(/-/g,hyphen);}else if(word.indexOf("-")!==-1){hw=hyphenateCompound(lo,lang,word);}else{ww=word.toLowerCase();if(String.prototype.normalize){ww=ww.normalize("NFC");}if(lo.hasOwnProperty("charSubstitution")){ww=doCharSubst(lo.charSubstitution,ww);}if(word.indexOf("'")!==-1){ww=ww.replace(/'/g,"’");}ww="_"+ww+"_";wwlen=ww.length;while(pstart<wwlen){wwhp[pstart]=0;charCode=ww.charCodeAt(pstart);wwAsMappedCharCode[pstart]=charMap.hasOwnProperty(charCode)?charMap[charCode]:-1;pstart+=1;}pstart=0;while(pstart<wwlen){row=0;pattern="";plen=pstart;while(plen<wwlen){mappedCharCode=wwAsMappedCharCode[plen];if(mappedCharCode===-1){break;}if(enableReducedPatternSet){pattern+=ww.charAt(plen);}link=indexedTrie[row+mappedCharCode*2];value=indexedTrie[row+mappedCharCode*2+1];if(value>0){hpc=0;hp=valueStore[value+hpc];while(hp!==255){if(hp>wwhp[pstart+hpc]){wwhp[pstart+hpc]=hp;}hpc+=1;hp=valueStore[value+hpc];}if(enableReducedPatternSet){if(!lo.redPatSet){lo.redPatSet={};}if(valueStore.subarray){values=valueStore.subarray(value,value+hpc);}else{values=valueStore.slice(value,value+hpc);}lo.redPatSet[pattern]=recreatePattern(pattern,values);}}if(link>0){row=link;}else{break;}plen+=1;}pstart+=1;}hp=0;while(hp<wordLength){if(hp>=lo.leftmin&&hp<=(wordLength-lo.rightmin)&&(wwhp[hp+1]%2)!==0){hw+=hyphen+word.charAt(hp);}else{hw+=word.charAt(hp);}hp+=1;}}hw=onAfterWordHyphenation(hw,lang);if(enableCache){lo.cache[word]=hw;}return hw;}function checkIfAllDone(){var allDone=true,i=0,doclist={};elements.each(function(ellist){var j=0,l=ellist.length;while(j<l){allDone=allDone&&ellist[j].hyphenated;if(!doclist.hasOwnProperty(ellist[j].element.baseURI)){doclist[ellist[j].element.ownerDocument.location.href]=true;}doclist[ellist[j].element.ownerDocument.location.href]=doclist[ellist[j].element.ownerDocument.location.href]&&ellist[j].hyphenated;j+=1;}});if(allDone){if(intermediateState==="hidden"&&unhide==="progressive"){elements.each(function(ellist){var j=0,l=ellist.length,el;while(j<l){el=ellist[j].element;el.className=el.className.replace(unhideClassRegExp,"");if(el.className===""){el.removeAttribute("class");}j+=1;}});}while(i<CSSEditors.length){CSSEditors[i].clearChanges();i+=1;}forEachKey(doclist,function(doc){onHyphenationDone(doc);});if(!!storage&&storage.deferred.length>0){i=0;while(i<storage.deferred.length){storage.deferred[i].call();i+=1;}storage.deferred=[];}}}function controlOrphans(ignore,leadingWhiteSpace,lastWord,trailingWhiteSpace){var h=hyphen;if(".\\+*?[^]$(){}=!<>|:-".indexOf(hyphen)!==-1){h="\\"+hyphen;}else{h=hyphen;}if(orphanControl===3&&leadingWhiteSpace===" "){leadingWhiteSpace=String.fromCharCode(160);}return leadingWhiteSpace+lastWord.replace(new RegExp(h+"|"+zeroWidthSpace,"g"),"")+trailingWhiteSpace;}function hyphenateElement(lang,elo){var el=elo.element,hyphenate,n,i,lo;if(lang==="urlstyled"&&Hyphenator.doHyphenation){i=0;n=el.childNodes[i];while(!!n){if(n.nodeType===3&&(/\S/).test(n.data)){n.data=hyphenateURL(n.data);}i+=1;n=el.childNodes[i];}}else if(Hyphenator.languages.hasOwnProperty(lang)&&Hyphenator.doHyphenation){lo=Hyphenator.languages[lang];hyphenate=function(match,word,url,mail){var r;if(!!url||!!mail){r=hyphenateURL(match);}else{r=hyphenateWord(lo,lang,word);}return r;};i=0;n=el.childNodes[i];while(!!n){if(n.nodeType===3&&(/\S/).test(n.data)&&n.data.length>=min){n.data=n.data.replace(lo.genRegExp,hyphenate);if(orphanControl!==1){n.data=n.data.replace(/(\u0020*)(\S+)(\s*)$/,controlOrphans);}}i+=1;n=el.childNodes[i];}}if(intermediateState==="hidden"&&unhide==="wait"){el.className=el.className.replace(hideClassRegExp,"");if(el.className===""){el.removeAttribute("class");}}if(intermediateState==="hidden"&&unhide==="progressive"){el.className=el.className.replace(hideClassRegExp," "+unhideClass);}elo.hyphenated=true;elements.counters[1]+=1;if(elements.counters[0]<=elements.counters[1]){checkIfAllDone();}}function hyphenateLanguageElements(lang){var i=0,l;if(lang==="*"){elements.each(function(lang,ellist){var j=0,le=ellist.length;while(j<le){hyphenateElement(lang,ellist[j]);j+=1;}});}else{if(elements.list.hasOwnProperty(lang)){l=elements.list[lang].length;while(i<l){hyphenateElement(lang,elements.list[lang][i]);i+=1;}}}}function removeHyphenationFromDocument(){elements.each(function(ellist){var i=0,l=ellist.length;while(i<l){removeHyphenationFromElement(ellist[i].element);ellist[i].hyphenated=false;i+=1;}});}function createStorage(){var s;function makeStorage(s){var store=s,prefix="Hyphenator_"+Hyphenator.version+"_",deferred=[],test=function(name){var val=store.getItem(prefix+name);return!!val;},getItem=function(name){return store.getItem(prefix+name);},setItem=function(name,value){try{store.setItem(prefix+name,value);}catch(e){onError(e);}};return{deferred:deferred,test:test,getItem:getItem,setItem:setItem};}try{if(storageType!=="none"&&window.JSON!==undefined&&window.localStorage!==undefined&&window.sessionStorage!==undefined&&window.JSON.stringify!==undefined&&window.JSON.parse!==undefined){switch(storageType){case"session":s=window.sessionStorage;break;case"local":s=window.localStorage;break;default:s=undefined;}s.setItem("storageTest","1");s.removeItem("storageTest");}}catch(ignore){s=undefined;}if(s){storage=makeStorage(s);}else{storage=undefined;}}function storeConfiguration(){if(!storage){return;}var settings={"STORED":true,"classname":hyphenateClass,"urlclassname":urlHyphenateClass,"donthyphenateclassname":dontHyphenateClass,"minwordlength":min,"hyphenchar":hyphen,"urlhyphenchar":urlhyphen,"togglebox":toggleBox,"displaytogglebox":displayToggleBox,"remoteloading":enableRemoteLoading,"enablecache":enableCache,"enablereducedpatternset":enableReducedPatternSet,"onhyphenationdonecallback":onHyphenationDone,"onerrorhandler":onError,"onwarninghandler":onWarning,"intermediatestate":intermediateState,"selectorfunction":selectorFunction||mySelectorFunction,"safecopy":safeCopy,"doframes":doFrames,"storagetype":storageType,"orphancontrol":orphanControl,"dohyphenation":Hyphenator.doHyphenation,"persistentconfig":persistentConfig,"defaultlanguage":defaultLanguage,"useCSS3hyphenation":css3,"unhide":unhide,"onbeforewordhyphenation":onBeforeWordHyphenation,"onafterwordhyphenation":onAfterWordHyphenation,"leftmin":leftmin,"rightmin":rightmin,"compound":compound};storage.setItem("config",window.JSON.stringify(settings));}function restoreConfiguration(){var settings;if(storage.test("config")){settings=window.JSON.parse(storage.getItem("config"));Hyphenator.config(settings);}}var version='5.3.0';var doHyphenation=true;var languages={};function config(obj){var assert=function(name,type){var r,t;t=typeof obj[name];if(t===type){r=true;}else{onError(new Error("Config onError: "+name+" must be of type "+type));r=false;}return r;};if(obj.hasOwnProperty("storagetype")){if(assert("storagetype","string")){storageType=obj.storagetype;}if(!storage){createStorage();}}if(!obj.hasOwnProperty("STORED")&&storage&&obj.hasOwnProperty("persistentconfig")&&obj.persistentconfig===true){restoreConfiguration();}forEachKey(obj,function(key){switch(key){case"STORED":break;case"classname":if(assert("classname","string")){hyphenateClass=obj[key];}break;case"urlclassname":if(assert("urlclassname","string")){urlHyphenateClass=obj[key];}break;case"donthyphenateclassname":if(assert("donthyphenateclassname","string")){dontHyphenateClass=obj[key];}break;case"minwordlength":if(assert("minwordlength","number")){min=obj[key];}break;case"hyphenchar":if(assert("hyphenchar","string")){if(obj.hyphenchar==="&shy;"){obj.hyphenchar=String.fromCharCode(173);}hyphen=obj[key];}break;case"urlhyphenchar":if(obj.hasOwnProperty("urlhyphenchar")){if(assert("urlhyphenchar","string")){urlhyphen=obj[key];}}break;case"togglebox":if(assert("togglebox","function")){toggleBox=obj[key];}break;case"displaytogglebox":if(assert("displaytogglebox","boolean")){displayToggleBox=obj[key];}break;case"remoteloading":if(assert("remoteloading","boolean")){enableRemoteLoading=obj[key];}break;case"enablecache":if(assert("enablecache","boolean")){enableCache=obj[key];}break;case"enablereducedpatternset":if(assert("enablereducedpatternset","boolean")){enableReducedPatternSet=obj[key];}break;case"onhyphenationdonecallback":if(assert("onhyphenationdonecallback","function")){onHyphenationDone=obj[key];}break;case"onerrorhandler":if(assert("onerrorhandler","function")){onError=obj[key];}break;case"onwarninghandler":if(assert("onwarninghandler","function")){onWarning=obj[key];}break;case"intermediatestate":if(assert("intermediatestate","string")){intermediateState=obj[key];}break;case"selectorfunction":if(assert("selectorfunction","function")){selectorFunction=obj[key];}break;case"safecopy":if(assert("safecopy","boolean")){safeCopy=obj[key];}break;case"doframes":if(assert("doframes","boolean")){doFrames=obj[key];}break;case"storagetype":if(assert("storagetype","string")){storageType=obj[key];}break;case"orphancontrol":if(assert("orphancontrol","number")){orphanControl=obj[key];}break;case"dohyphenation":if(assert("dohyphenation","boolean")){Hyphenator.doHyphenation=obj[key];}break;case"persistentconfig":if(assert("persistentconfig","boolean")){persistentConfig=obj[key];}break;case"defaultlanguage":if(assert("defaultlanguage","string")){defaultLanguage=obj[key];}break;case"useCSS3hyphenation":if(assert("useCSS3hyphenation","boolean")){css3=obj[key];}break;case"unhide":if(assert("unhide","string")){unhide=obj[key];}break;case"onbeforewordhyphenation":if(assert("onbeforewordhyphenation","function")){onBeforeWordHyphenation=obj[key];}break;case"onafterwordhyphenation":if(assert("onafterwordhyphenation","function")){onAfterWordHyphenation=obj[key];}break;case"leftmin":if(assert("leftmin","number")){leftmin=obj[key];}break;case"rightmin":if(assert("rightmin","number")){rightmin=obj[key];}break;case"compound":if(assert("compound","string")){compound=obj[key];}break;default:onError(new Error("Hyphenator.config: property "+key+" not known."));}});if(storage&&persistentConfig){storeConfiguration();}}function run(){var process=function(){try{if(contextWindow.document.getElementsByTagName("frameset").length>0){return;}autoSetMainLanguage(undefined);gatherDocumentInfos();if(displayToggleBox){toggleBox();}prepare(hyphenateLanguageElements);}catch(e){onError(e);}};if(!storage){createStorage();}runWhenLoaded(window,process);}function addExceptions(lang,words){if(lang===""){lang="global";}if(exceptions.hasOwnProperty(lang)){exceptions[lang]+=", "+words;}else{exceptions[lang]=words;}}function hyphenate(target,lang){var turnout,n,i,lo;lo=Hyphenator.languages[lang];if(Hyphenator.languages.hasOwnProperty(lang)){if(!lo.prepared){prepareLanguagesObj(lang);}turnout=function(match,word,url,mail){var r;if(!!url||!!mail){r=hyphenateURL(match);}else{r=hyphenateWord(lo,lang,word);}return r;};if(typeof target==="object"&&!(typeof target==="string"||target.constructor===String)){i=0;n=target.childNodes[i];while(!!n){if(n.nodeType===3&&(/\S/).test(n.data)&&n.data.length>=min){n.data=n.data.replace(lo.genRegExp,turnout);}else if(n.nodeType===1){if(n.lang!==""){Hyphenator.hyphenate(n,n.lang);}else{Hyphenator.hyphenate(n,lang);}}i+=1;n=target.childNodes[i];}}else if(typeof target==="string"||target.constructor===String){return target.replace(lo.genRegExp,turnout);}}else{onError(new Error("Language \""+lang+"\" is not loaded."));}}function getRedPatternSet(lang){return Hyphenator.languages[lang].redPatSet;}function getConfigFromURI(){var loc=null,re={},jsArray=contextWindow.document.getElementsByTagName("script"),i=0,j=0,l=jsArray.length,s,gp,option;while(i<l){if(!!jsArray[i].getAttribute("src")){loc=jsArray[i].getAttribute("src");}if(loc&&(loc.indexOf("Hyphenator.js?")!==-1)){s=loc.indexOf("Hyphenator.js?");gp=loc.substring(s+14).split("&");while(j<gp.length){option=gp[j].split("=");if(option[0]!=="bm"){if(option[1]==="true"){option[1]=true;}else if(option[1]==="false"){option[1]=false;}else if(isFinite(option[1])){option[1]=parseInt(option[1],10);}if(option[0]==="togglebox"||option[0]==="onhyphenationdonecallback"||option[0]==="onerrorhandler"||option[0]==="selectorfunction"||option[0]==="onbeforewordhyphenation"||option[0]==="onafterwordhyphenation"){option[1]=new Function("",option[1]);}re[option[0]]=option[1];}j+=1;}break;}i+=1;}return re;}function toggleHyphenation(){if(Hyphenator.doHyphenation){if(!!css3hyphenateClassHandle){css3hyphenateClassHandle.setRule("."+css3hyphenateClass,css3_h9n.property+": none;");}removeHyphenationFromDocument();if(safeCopy){copy.removeOnCopy();}Hyphenator.doHyphenation=false;storeConfiguration();if(displayToggleBox){toggleBox();}}else{if(!!css3hyphenateClassHandle){css3hyphenateClassHandle.setRule("."+css3hyphenateClass,css3_h9n.property+": auto;");}Hyphenator.doHyphenation=true;hyphenateLanguageElements("*");if(safeCopy){copy.reactivateOnCopy();}storeConfiguration();if(displayToggleBox){toggleBox();}}}return{version:version,doHyphenation:doHyphenation,languages:languages,config:config,run:run,addExceptions:addExceptions,hyphenate:hyphenate,getRedPatternSet:getRedPatternSet,isBookmarklet:isBookmarklet,getConfigFromURI:getConfigFromURI,toggleHyphenation:toggleHyphenation};}(window));if(Hyphenator.isBookmarklet){Hyphenator.config({displaytogglebox:true,intermediatestate:"visible",storagetype:"local",doframes:true,useCSS3hyphenation:true});Hyphenator.config(Hyphenator.getConfigFromURI());Hyphenator.run();}Hyphenator.languages['en-gb']={leftmin:2,rightmin:3,specialChars:"",patterns:{3:"sw2s2ym1p2chck1cl2cn2st24sss1rzz21moc1qcr2m5q2ct2byb1vcz2z5sd3bs1jbr4m3rs2hd2gbo2t3gd1jb1j1dosc2d1pdr2dt4m1v1dum3w2myd1vea2r2zr1we1bb2e2edn1az1irt2e1fe1j4aya4xr1q2av2tlzd4r2kr1jer1m1frh2r1fr2er1bqu44qft3ptr22ffy3wyv4y3ufl21fo1po2pn2ft3fut1wg1ba2ra4q2gh4ucm2ep5gp1fm5d2ap2aom1cg3p2gyuf2ha2h1bh1ch1d4nda2nhe22oz2oyo4xh1fh5h4hl2ot2hrun1h1wh2y2yp2aki2d2upie22ah2oo2igu4r2ii2omo1j2oiyn1lz42ip2iq2ir1aba4a2ocn3fuu4uv22ix1iz1jay1iy1h2lylx4l3wn5w2ji4jr4ng4jsy1gk1ck1fkk4y5fk1mkn21vok1pvr44vsk1t4vyk5vk1wl2aw5cn2ul3bw5fwh2wi2w1m1wowt4wy2wz4x1an1in1rn1ql3hxe4x1hx1ill24lsn3mlm2n1jx1ox3plr4x5wxx4",4:"d3gr_fi2xy3ty1a2x5usy5acx1urxu4on2ielph2xti4ni2gx4thn2ilx1t2x1s25niql3rix4osxo4n1logn2ivx5om1locl3ro2lo_l3nel1n4_hi2l5rul1mexi4pl1max3io_ex1l1lu_ig3ll5tll3sll3p_in14n2kl1loll3mn3le_ew4n1n4nne4l1lixi4cll3fn3nil1lal5skls4p_eu14no_l4ivx3erx3enl1itx1eml1isx5eg3lirli1qxe2d3lik5lihx1ec1lig4y1bn1oun4ow4li_x3c4yb2il1g2l2fox2as1leyn3p42lev1letx2ag4ni_l1te_es1nhy2yc1l4n1sw3tow5tenho4ns2cwra42lerle5qn2si3womwol4l1try1d4lek42ledwl1in3suw3la4le_l3don1teldi2nth2lce4yda4l1c2l1tu4lu_l4by_od4lbe4lu1a4laz_oi4l4awnt2iwes4l4aul4asn2tjla4p_or1n1tr5wein1tun2tyn1h2w4ednu1awe4b5nuc_os13nudl4all4af_ov4w3drl4aey3eenu3iw1b45nukl4ac5laa4la_4lue3kyllu1in1gu4wabn1go_ph2v5vikur5_en12vv2ks4ty3enk3slv5rov5ri4k1sk3rung1n2vowy1erkol4ko5a4vonk2novo2l2vo_5lupn2gingh4k3lok3lik3lak2l2ng2aki4wvi2tkis4k1inki2l5kihk3holu1vke4g3kee4kedkdo4_sa2k5d2_eg4k1b4kav4kap4vim4ka3ovi4lk4ann3v2nve2vic2ka4lju1v4vi_ju5ljui4_sh2ygi2nfo4_st44jo_3jo2jil43jigl4vi2vel3veive3gjew3jeu42ve_4jesjeo2y3gljal43jac2ja__th44ly_2izz_ti22izo_do2i5yeix3oy3in2i1wn2x4i2vov4ad2ny25nyc5vacn1z24va_nzy4uy4aux2o2oa2o3ag2ivauve2u4vayle2i3um2ittly1c4obau3tu2itrob2i4obo_up12ithob5tuts2lym2ut2o_ve2oc2ait1a2isyo1clo1crut2ioct2is1pis1lo1cy4usto2doo2du4isblyp2n4ew2ab_2abai4saoe3a2abbus1pir2sir4qoe4do5eeir1ioep5o5eqo3er2usco1etir1a3lyr3lywipy43oeuo3evi3poab1ro3ex4ofo2o1gur1uo2ga2abyac2a3lyzi5oxo3gii3oti1orioe4ur2so2gui1od2io22acio1h2ur1o2inuo3hao3heohy44ma_oi4cins24inqoig4ac1r2ino2inn4inl4inkur1ioi4our2f4oisoi4t2iniynd4ok3lok5u2ind2inco1loyn2eo1mai2moom1iur2ca2doim1iil3v4iluon1co2nead1ril3f4onh2ik24iju4adyae5aija4i5in4aed2mahae5gihy4ae5pur1aae4s2i1h4igions2i1geyng42ont4af_4afe5maka4fui3fyu2pri3foon2zn1eru4po4agli2fe2i1foo1iu1ph4ieua2groo4moo2pyn4yi1er4iemie5ia1heah4n4iec2ai24ai_ai3aa1icne2p4idraig2oo2tu1peo1paop1iy1o2u1ouu3os4oplid1ayo3d2icuop1uor1a2ick4ich2a1ja4ju2mam4iceak5u4ibuunu44iboib1i2oreiav4i3aui3atun5ror1iun5o2alei5aii3ah2unniaf4i5ae2ormhy4thyr4hy3ohyn4hy2m2orthy2l1man2nedhuz4un2ihu4gh1th4alko1sch4skhsi42mapu1mu2h1shry4hri4hre41mar4h1pum2ph2ou4osp4osuy2ph4oth4ho_u1mi2h1mh1leh3la2ne_h4irhi2pu1mao4u2oub2h1in2a2mhi4l4oueu1lu2ulsoug4h1ic2hi_u1loul3mnde24ulln2daheu2ul2iou3mam1ihet12ounhep1ow1iows4ow5yyp1nox3ih4eiox5oypo1oy5aoys4u1la4ul_am2pu2izmav4h2ea4he_y2prhdu42m1ban2ao1zo_ch4mb4dy5pu4pa_ha4m1paru2ic5pau2ui2h4ac4ha_u4gon1cug5z2uft43gynu4fou3fl3ufa5gymmb2iue4tgy2b4anhnc1t2g1w5paw3gun2p1bu4edueb4p1c42guep5d2an1og5to2pe_gs4tgs4c2g1san2s2ped3grug4rou2dog4reud4g1gr2n1crgov12gou3gosud4e3goop4ee3goe5god3goc5goa2go_pe2fg2nog1niuc3lg1na2gn2an2y2pes3gluyr4r3pet5aowyr4s4ap_4apa3glo4pexyr5uu4ch2gl24y2s5gip2me_3gioap1i2ph_gi4g3gib4gi_uba41g2igh2tg3hoa2prphe44aps2medg2gegg4ame2g2g1gy3shu1alua5hu2ag2g1f3get2ua2ph2lge4o1pho2tz23gen4phs1gel1typ4gef2ge_g5d4me2m1phug1at4pi_p2iety4a4ty_p2ilt3wopim23gait2wi3gagn3b44ga_5piqar3har1i1tutfu4c4fu_1menp2l23tunna2vfs4p2f3s1pla1fr2tu1ifo3v4tufp4ly2p1myso53foo2arrme4par2stu1afo2n4tu_4po_t2tytt5s3pod2aru4poffo2e3foc4fo_ar5zas1ays1t3flu2asc3flo3flan2asas2et3ti2fin5poypph44f5hf3fr1pr2f1fif1fena5o3feufe4t4pry2ps22asotta4p3sh5fei3fecass2p1sits2its4ht2sc2fe_4t1s2f5d4f5b5faw5farp1st2pt2as1u2fa_1f2aeyl44ey_1expe1wre3whe1waevu4p4trp1tupub1puc4p4uneus44eumeuk5eue4p4uset5zyzy4z1a14p1wet2t2p4y4tovpy3e3pyg3pylpy5t2za__av44ra_r2adras2et2ae1su1namr2bat1orr2berb2ir1c2r2clrct4nak24re_rea4e2sc4es_2erza2to5tok2erurei4erk44erj1tog3toere1qre1vza2irf4lr1g2r2gez4as4ri_2ereto1b2erd2to_2erc4m3hri3ori5reph14mi_2au24au_m1ic4auc4t3me1paeo3mt1lieo2leof2eo3b4enur1lar1leaun2r1loen2sen1ot1laen3kzeb4r1mur2n24ene2end3tiurn5nrnt4ze4d4ro_r2od4roiroo4r2opelv4e1lur4owti4q1tip4roxrpe2r2ph1tior3puaw1i5nahaw5y4mijr3ri_as12eleay3mayn4ays2r5rurry5ek4l2az2m2ilaze4e2ize2iv4eis2ba_t1ineig24eifeid45bahba4ir2seehy21timeh5se5hoe1h2e2gr2efuef4lna2ceep1ee2mee1iee5gee2fr3su2na_rt3ced4g1basede23mytr1turu3ar2udr4ufe1clru2le1ceru2pb1c2ec2a2b1deb2te2bre4bl3myi4be_3beaeb2iebe4eb2b2bedzib5r1v2r2veeau3t1icmy3e5bee3bef2r2yry2tz2ie1bel2sa_2sabeap25saebe3meak1ea4gsa4g3sai4ti_5sak4beobe3q4eabmy4dd3zo3dyndyl25dyksa2l2d2y2d1wsa4mbe3w2b1fbfa44b1hb4ha2bi_1biazi5mdu3udu2ps3apb4ie3ducbif42ths2du_z4isb1ilmi3od4swds3m4bimd5sl1saumi3pz3li3dox4s3bd4osd2or3doosby3bip4bi5qbir44zo_s1cab2iss1cedo4jd4ob4do_5zoa2d1mmtu4d5lu2bl2d1losch2d1la2dl4tha42th_m5si4m1ss2co2t3f1diu2se_se2a4bly2b1m3texbmi44b1nm4ry4bo_3boa2sed5bobdil4bo5h3sei1didse2p1dia4di_d4hu3bon4d1hxys4dg4ami2t2d5f1boo3dexs2es1set3sev3sex3sey2s1fsfi4_an1d3eqde1ps4idsif4bow2si4g2sin5boyzo5p3sipde3gs1it3dec2de_d3di2tep3miute2od1d4d3c4zot23davs2k24sk_d1atske2d3ap4sksd1agb3sc2sl44da_5zumb5sicy4tbso2te2ltei4cys4cy4m2b1tcyl34bu_5bubte2g1cyc2cy_bun2cu5v5cuu1cuss2le1curt4edc4ufc1tyc1tu4te_c1trs1n2s2na2so_t1ca5mix4b3w4zy_4by_3byibys45byt2ca_2tc23soes2olc1te5cafsos45cai5cakc1al3sou4t3bt4axc2ta4m1lcry2sph2s1plc2res2pos4pym3pum3pocoz4cov14mo_sre22moc5cao1caps1sa3cooss3mcon11cars4sns1sos1su1takss3wmod13coe4st_1tai3tah3coc3coa4co_taf4c3nim2pist3cc1atste2mo1mc4kem4ons1th2cim3cau2tab2ta_3cayc1c44stl3cilc3ch3syn4cigci3f4ce_4ci_3chrs1tu1cho2ced4chm1sylch5k4stw4cefce5gs4tysy4d4su_sug3sy1c3sui4ch_m3pa2cem4sy_cew4ce2t1cepsu5zm4op2swo2s3vzzo3",5:"n5tau2cenn3centsves45swee5cencsu5sus4urg1cen2sur3csu5pe3cerasun4a3cerdsum3i5cern5cesss4u2m1s2ulce4mo3cemi4celysy4bi4chab3chae3chaisui5ccelo45cellchec44ched3chee3chemsuf3fch1ersu3etsud4asuct44chessubt2ch5eusu4b13chewch5ex5chi_3chiasu5ansy4ce1styl3ceiv3chio5chip3cedi3cedestu4m5cedace4cicho3a5choc4chois4tud3chor3ceas2st3sstre43chots2tou3stonchow5cean3chur43chut5chyd3chyl3chym1c2i24ceab4ciaccia4mci3ca4cids4cie_ci3ers4toeci5etccle3cifi4ccip4ci3gast3lisyn5esyr5icat4ucim3aci3mes5tizs4thu4cinds4thac4atss4tec4cintci3olci5omci4pocisi4cit3rt2abockar5cka5tt5adeck5ifck4scc2atcs4teb3clasc2le22cle_c5lecc4at_clev3cli1mtad4icli2qclo4q4stakclue4clyp55clystad2rtae5n1c2o2case5car4vco5ba3tagrco3cico5custab23tail4cody2tairco5etco3grcar5mt4ais4col_col3atal2css5poco5lyta3lyco4met4anecomp4cap3uta4pass5liss1ins1sifs1siccon3scon3ts3siacapt4coop4co3orcop4eco3phco5plco3pocop4t2corassev3s5seus1sel1tard3corn4corotar3n5cort3cos_sre4ssreg5co5ta3tarr5cotytas3it3asmco3vacow5a5tassco5zic4anotas4t5craftat4rc4ran5spomcam4is4plysple2ca3maca3lys2pins2pids3phacal4m4speocri3lcron4so3vi4crousov5et5awacrym3cryo34c5s4csim5tawn43calcc3tacc4alaso5thct1an4soseca3gos3orycad4rc4teasor3os2o2ps4onect5esct5etct2ics2onaso3mo1so2mc3timsol3acaco3c4acesody4sod3oc5tio2s3odc3tittcas4tch5u4t1d4smo4dsmi3gc1tomc3tons3mensmas4b3utec2tres3man3bustc2tumte3cr2s1m4buss2s5lucslov5c2ulislo3cs3lits5leycu4mi5cunacun4e5cuni5cuolcu5pacu3pic3upl4tedds3lets5leabur3ebunt4cus5a3slauc3utr4tedobun4a4teeicy4bib4ulit3egoteg1rcy5noteg3us1latbsin41tellbsen4d4abr1d2acdach43tels3dact4b1s2sky3ld4aled4alg4bry_dam5a3damed3amida5mu3dangs5keybrum4d3ard5darms3ketbros4tem3as5kardat4ub4roa4teme4tenet5enm4tenob2ridteo5l4bre_5sivad3dlid3dyite3pe4s1ivde5awde4bisi4teb2ranbram44sismde1cr4dectded3i4sishs1is24bralde4gude3iosi4prtep5i4sio_1sio45sinkde5lo1d4emsin3is2ine4boxy1silibow3ssif5f4demybous4den4d4dened3enh4sidssi4de4sid_3bourde3oddeo3ldeon2si4cu5terd3sicc4s1ibde2pu5botishys44shu4d4eres3hon5shipsh3io1derider3k3dermsh5etsh1er4shab1teri2s1g4der3s5deru4des_de3sa5descbor4nter5k3terrdes4isexo23borides1psewo4de3sq2t2es5seum1de1t4tes_de5thde2tise5sh4ses_bor3d3septsep3atesi4t3esqdfol4tes4tteti4dgel4d4genbon4ebon4cdhot4bol4tbol3itet1rdi2ad3diarbol4e4d1ibd1ic_3sensdi4cedi3chd5iclsen5g1dictsem4osem2i5self4sele4boke5selasei3gd4ifo2boid3seedbod5i5dilldilo4di3luse4dabo5amdi1mi2d1indin4ese2cosec4a3di1odio4csea3wdip5t3diredi3riseas4di4s1d4iscs4eamb3lis3dissbli2q2s1d22s1cud3itos4coi2ditybli3oscof44blikscid5dix4i3bler4the_b3lan5dlefblag43dlewdlin45blac4b5k4bi5ve4d1n24bity4thea4thed4sceidog4abis4od4ol_s4ced5bismscav3sca2pd4ols5dom_1thei3theobi3ousbe4sdo5mos4bei4donybio5mbio3l4dor_dor4mdort41bi2ot4hersavi2dot1asaur52dousd4own4thi_th5lo2thm25binad3ral3dramdran4d4rassat1u3dreldres4sa2tedri4ed4rifs2a1td4romsas3s3sas_4d1s2th4mi3thotds4mi1th2rb2iledt5hobigu3bi5gadu1at5thurduch5sar5sdu4cosap3rbid5idu5en2santdu5indul3cd3uledul4lsan3adun4asamp43b2iddu3pl5durod5usesam5o5thymbi4b1dver2be3trsa3lube3sl3sale2bes_be1s2dy5ar5dy4e3thyrber5sdyll35dymi5berrdys3pberl4thys42beree1actbe5nuea5cue5addbe1neead1i1ti2ati3abben4deal3abel4tsad5osad5is3actean5i2t3ibsac4qe3appear3a5sacks3abl2belebe3labe3gube5grryp5arym4bry4goeas4t5rygmry5erbe3gobe4durvi4tr3veyr3vetr3vene4atube4doeav5ibed2it3ic_eaz5ibe3daebar43becube3caru3tirus4pe2beneb5et4bease5bile4bine4bisbdi4ve4bosrur4ibde4beb1rat2icie4bucru3putic1ut3id_run4trun4ge5camrun2eec3atr4umib3blir4umeech3ie4cibeci4ft4ida2b1b2ru3in3tidirue4lt5idsru4cerub3rr4ube1tif2ec1ror4tusti3fert5sirto5lr1t4oec1ulrt3li4tiffr2tize2dat3tigie4dede5dehrt3ivr2tinrth2ir5teue3deve5dew5barsr5tetr1ted4tigmr3tarrta4grt3abed1itedi2v5tigued3liedor4e4doxed1ror4suse2dulbar4nrs5liee4cers3ivee4doti4kabar4d5barbr4sitba4p1r3sioeem3ib4ansee4par4sileesi4ee3tot4illr5sieefal4rs3ibr3shir3sha5bangr3setb4anee4fugrsel4egel3egi5ae4gibe3glaeg3leeg4mir3secr3seat4ilte5gurban4abam4abal5utim1abal3abag4a5eidobaen43backr4sare4in_e3ince2inee1ingein5ir2sanei4p4eir3oazz4leis3ir2saleith4azyg4r4sagaz5eeaz3ar2r1s2ek3enek5isayth4e4lace5ladr3rymelam4r3ryi3tinnay5sirro4trrog5rrob3ay5larric4ax2idrrhe3rre2lele3orrap4el1ere1lesrra4h4r1r44tinst4intrpre4el5exrp5ise1lierph5ee3limav1isti3ocrp3atav3ige3livavas3r4oute3loae3locroul35rouero3tue2logro1te4rossr4osa4roreel3soror5dav5arelu4melus42t1ise5lyi3elytr4opr4rop_emar4tis4c5root1roomem5bie1me4e4meee4mele3mem3tissro1noro3murom4pe4miee2migro3lyro3laroid3e3mioro3ictis2te4miuro3gnro1fero3doava4ge2moge4moiro3cuem5om4emon5roccro5bre2morro4beav4abr5nute5mozrnuc4au3thr5nogr3noc3titlem3ume5muten3ace4nalrn3izrni5vr1nisrn3inr3nicrn5ibr5niaenct42t1ivr3neyr3netr3nelaus5pene5den3eern5are5nepe2nerr5nadr3nacrn3abt3iveen1et4aus_rmol4e3newen3gien3icr3mocrmil5en5inr5migaur4o5tleben3oieno2mrm4ieenov3aun3dr2micen3sprme2arm4asr2malr5madr3mac3tlefen2tor4litau3marlat33tlem5tlenen3uaen3ufen3uren5ut5enwa5tlewe4oche4odaaul4taul3ir3keyr3ketrk1ere5olutlin4eon4ae3onteop4te1or1r5kaseor3eeor5oeo1s2eo4toauc3oep4alaub5iepa4t4a2tyr2i4vr2ispris4cep5extmet2eph4ie2pige5pla2t3n2ri5orri4oprio4gatu4mrin4sr4inorin4e4rimse1p4u4rimmr4imbri2ma4rim_at1ulr4ileri2esera4gera4lri3erri5elrid4e2ricur4icl2riceri3boer3be2r2ib2a2tuer3cher3cltoas4ri5apri3am4toccat1ri4ered3r2hyrhos4tod4irgu5frg5lier3enr3gerr3geor5geee3reqer3erere4sa4trergal4r4gagat3rarfu4meret42a2tra5tozatos4ere4ver3exreur4er3glre3unre3tur3esq2res_er2ider3ierere4rer4aer3into5dore5phre1pe3reos3reogre3oce3river5iza3too4atoner3mer4enirene2rena4r3empr5em_re1le4ero_re1lam5ordreit3re3isre1inre3if2atolre2fe3reerree3mre1drre1de2r4ed4atogeru4beru5dre3cure3ce3reavr5eautol4ltolu5es5ames5an4atiure3agre3afr4ea_to5lye3seatom4be5seeat1itese4lr4dolrd3lie1shie5shurdi3ord2inr5digr4dier4desr2dares3imes3inr5dame4sitrc5titon4er5clor4clees4od3tonnrcis2rcil4eso3pe1sorr2cesrca4ston3ses4plr4bumr2bosrbit1r2binrbic4top4er4beses2sor3belrbe5ca4timrbar3e2stirb1anr4baga2tif4toreest4rrawn4tor5pra3sor4asktor4qr2aseras3cati2crare2eta3p4rarcran2tet4asra3mur5amnet5ayra3lyra3grra4de3tos_eter2r2acurac4aetex4e2th1r2abo2etia5rabera3bae5timet3inath5re3tir5quireti4u1quet2que_e2ton4quar5quaktos4ttot5uath3ipyr3etou4fet1ri5tourt3ousath3aet1ro4a2that5etetud4pu3tre4tumet4wetra5q3tray4ater4tre_4trede3urgeur5itren4pur3cpur5beut3ipu3pipun2tpun3i3puncev3atpun4aeve4n4trewpum4op4u4mpu5ere4vese1viapuch4e2vict2rieevid3ev5igpu5be2trilt2rit4trixe4viuevoc3p5tomp3tilata3st4rode4wage5wayew1erata3pew5ieew1inp5tiee3witatam4ex5icpt4ictro5ft2rotey4as2a2taey3s2p5tetp1tedez5ieas5uras4unfab4ip2tarfact2p4tan2f3agp4tad5falopt3abtro1v3psyc3troypso3mt4rucfar3itru3i2t4rytrys42asta3feast4silfeb5ras3ph2fed1as5orfe1lifem3i2t1t4p3sacf5enias4loas4la3feropro1l4pro_3ferrfer3v2fes_priv24priopren3aski43prempre1dfet4ot3tabpreb3as5iva3sit4pre_f5feta5siof5fiaf3ficf5fieffil3prar4ff4lepra5dffoc3prac1as3int5tanppi4ct5tast3tedfib5u4fic_ppet33fici4ficsppar34p1p2fiel4asep4p5oxi1fi2l4asedfin2apo1tefind3fin2ef1ing3p4os3portpor3pf3itapo4paas2crt3tlifle2s2ponyflin4t5toip4o2nasan2pom4eas4afa5ryta3ryot5torar3umt3tospo3caar2thar3soar2rhar4pupnos4tu5bufor5bar3oxtu5en5formplu2m2plesaro4ntu4is3plen3plegfrar44ple_fre4sar3odfruc42tum_3tumi4tumsf1tedtun4aft5es2p3k2p2itutu4netur4dtur4npis2sfug4ap4iscfun2gp4is_fur3npir4tfus5oar3guar5ghpi4pegadi4pip4at3wa4ar3en3gale3pi1op4innpin4e3galot3wit5pilo3piletwon4pig3n5tychpict4g5arcg4arepi4crpi3co4picagar5p5garr1ga4sgas5igas3o3piarar4bl3phyltyl5ig4at_2phy_phu5ity5mig4attgat5ugaud5ga5zaar3baara3va3rau5geal3gean2ge4d3gedi5gednar1at3type4gelege4li1tyr13phrage4lu2gelygem3i5gemoara3mph3ou3phorgen3oa3rajt5ziat5zie4gereph1is2ges_5gessphi4nua3ciget3aara2ga5quia5punua5lu1philg3ger4phic3phibg3gligglu3g5glyph3etg4grouan4og5haiuar3auar2dg4hosuar3iap5lia5pirph2angi4atu1b2igi5coap3in4phaeub5loub3ragi4orgi4otaph3igi5pag4i4s5gis_gi2t15gituu1c2aa5peug3laru5chrglec43glerap3alpe4wag4leypet3rpe2tia1pacaol3iglom34glopa5nyian5yap4ery3glyp2g1m4a5nuta3nurg4nabper3vp4eri4pere5percpe5ongn5eegn3eru4comg4niapen5upel5v4pelean3uluco5tgno4suc2trant4ruc3ubuc5ulu5cumgo4etgo4geu5dacg5oidgo3isgo2me5gonnpe2duud1algoph44gor_5gorg4gorsg4oryud5epgos4t1anth3pedsg1ousan2teu4derudev4grab43gram3pedigra2pudi3ogril43pedeu5doigro4gg5rongrop4ud5onan3scgru5ipe4coan5otan2osanor3g4stiu5doran2oeg4u2agu5ab5guan4annyg5uatan5no5gueu4aniuuen4ogu2magu4mi4anigpawk4uer3agur4ngur4u4gurypau3pani3fan3icues4san3euan4eagyn5ouga4cug2niug3uluhem3ui3alp5atohae3opas1t1p4ashag5uha5ichais4par3luid5ouil4apa3pypap3uhan2gpa3pepa4pahan4tpan3iha4pehap3lhar1ahar5bhar4dpan1ep4alspa3lohar3opain2paes4pad4rhat5ouil4to3zygozo5ihav5oana5kuin4san3aeuint4amyl5am3ului5pruis4t1head3hearui3vou4laba3mon4ulacu5lathe3doheek4ul4bohe3isul3caul4ch4uleaow5slow5shu5leehem1aow5in3amidow5hahem4pow1elhe3orulet4h1er_owd3lher2bowd4io5wayow3anow3ago1vish5erho5varouv5ah1erlouss42ouseh1ersoun2dul4evami2cul2fahet3ioul4tul4iaheum3ou5gihe4v4hev5ihex5oa3men3ambuu5lomhi4aram1atou5gaul4poh4iclh5ie_h1ierou3eth1iesama4gh3ifyhig4ohi5kaa5madoud5iou5coou5caa5lynhin4dou5brul1v45ou3aalv5uh2ins4o1trh4ioral1vahip3lum3amhir4ro4touhit4ahiv5aumar4u5masalu3bh3leth1l2ihli4aum2bio1t2oot4iv2h1n2o5tiaal3phho3anho4cou4micho5duho5epo4tedhold1o3taxo3tapot3ama5lowh2o4nos1uru4mos4ostaos4saos1pihon1o1hoodhoo5rh4opea4louo5sono5skeh4orno4sisos1inos5ifhosi4o3siaalos4os5eual1ora3looo2seta3lomoser4hr5erhres4um4paos5eohrim4h5rith3rodose5ga5loeo3secumpt4un5abun4aeht5aght5eeo4scio2schos4ceos4caht5eoht5esun2ce4aliuosar5un3doos3alosa5iory5phun4chunk4hun4thur3ior4unu1nicun4ie4or1uun3inal1in5aligal3ifal1iduni5por4schy1pehy3phuni1vor1ouun3iz2i1a2ia4blo5rooorm1ii2achiac3oa2letork5a5origa1leoun3kni2ag4ia3gnor3ifia3graleg4a3lec4ori_al3chor5gn4ialnor4fria5lyi5ambia3me5orexi3anti5apeia3phi2ardore4va5lavor3eiore3giat4uore3fal3atun3s4un5shun2tiibio4or4duib5lia1laei4bonibor4or4chi5bouib1riun3usoram4ic3acor5ali4calic1an2icariccu4akel4i5ceoa5ismich4io5raiora4g4icini5cioais1iic4lo2i2coico3cair3sair5pi5copop2ta2i1cri4crii4crui4cry1op1top5soopre4air5aop2plic3umopon4i5cut2i1cyuo3deain5oi5dayide4mo4poiain3iu1pato1phyid3ifi5digi5dili3dimo4pheo1phaidir4op1ero5peco4pabidi4vid3liid3olail3oai5guid3owu5peeid5riid3ulaid4aa5hoo2ieg2ie3gauper3i5ellahar22i1enien2da1h2aoo4sei2erio3opt4iernier2oi4erti3escagru5oon3iag3ri2i1eti4et_oo4leag5otook3iiev3au5pidiev3o4ag1nagli4if4fau5pola5giao5nuson5urifi4difi4n4i2fla5gheifoc5ont4rupre4af5tai3gadaev3a3igaraeth4i3geraet4aono3saes3ton5oionk4si3gonig1orig3oto1nioo5nigon3ifig1urae5siae3on4ura_aeco34uraead3umura2gik5anike4bi2l3aila4gon4id4a2duil4axil5dril4dui3lenon4guuras5on1eto3neoon1ee4oned4oneaad1owon5dyon3dril1ina3dos4onauon3aiil5iqona4do2mouil4moi5lonil3ouilth4il2trad3olil5uli5lumo4moi4adoi4ilymima4cim2agomni3im1alim5amom2naomme4om2itomil44adoeomi2co3mia3adjuome4gurc3ai5mogi3monim5ooome4dom4beo3mato2malo2macim5primpu4im1ulim5umin3abo4mabur4duadi4p4olytina4lol1ouin5amin3anin3apo3losol1or4olocur3eain3auin4aw4adilol3mia5difolle2ol2itolis4o5lifoli2eo1lia4inea4inedin5eeo3leuol1erine4so3lepo3leo4ineuinev5ol5chol4an4infu4ingaola4c4ingeur5ee4ingiad4haur1er4ingo4inguoith44adeeada3v4inico3isma5daiur3faac2too3inguril4ur1m4ac3ry4ino_in3oioil5i4inos4acou4oideo2i4d4acosurn5soi5chinse2o3ic_aco3din3si5insk4aco_ac3lio3ho4ack5aohab34acitacif4in5ulin5umin3unin3ura4cicuro4do5gyrur5oturph4iod5our3shio3gr4i1olio3maog4shio3moi5opeio3phi5opoiop4sa5cato4gro4ioreo2grio4got4iorlior4nio3sci3osei3osii4osoog2naur5taiot4aio5tho4gioio5tri4otyur1teo5geyac3alurth2ip3alipap4ogen1o3gasip1ato3gamurti4ur4vaofun4iphi4i4phuip3idi5pilip3ino4fulipir4ip5isab1uloflu42abs_ip3lou3sadi4pogus3agi4pomipon3i4powip2plab3omip4reoet4rip1uli5putus3alabli4i3quaab3laus4apoet3iira4co4et_ir4agus3atoes3t4abio2abiniray4ird3iire3air3ecir5eeirel4a3bieires4oelo4ab1icoe5icir4ima3bet5irizush5aoe5cuir5olir3omusil52abe4ir5taoe4biabay4us4pais5ado5dytis1alis3amis1anis3aris5av_za5ri2s3cod3ul_xy3lod5ruo3drouss4eod3liis2er5odizod5it4iseuod4ilodes4o5degode4co5cyt2isiais5icis3ie4isim_vo1c4isisis4keus1troc5uo2ismais1onocum4iso5pu5teooc1to5ispr2is1soc2te_vi2socre3u3tieiss4o4istao2cleu3tioo5chuoch4e4istho4cea4istloc5ago3cadis1tro4cab4istyi5sulis3urut3leutli4it5abita4c4itaiit3am_vec5it4asit3at_ur4oit3eeo3busob3ul_ura4_up3lo3braith5io5botith3rithy52itiao5bolob3ocit1ieit3ig4itim_un5uob1lio3blaob3iti5tiqut5smit3ivit4liit5lo4ito_it5ol2itonit1ou_un5sobe4lu4tul_un3goat5aoap5ioan4t4itueit1ulit1urit3us2i1u2_un3eiur5euven3oal4iiv1ati4vedu5vinoad5io3acto5ace_ul4luy5er2v3abives4iv3eti4vieiv3ifnyth4va1cavacu1iv1itva4geivoc3vag5rv1al_1vale_tor1vali25valu4izahiz3i2_til4iz5oivam4i_tho4va5mo5vannnwom4jac3ujag5u_te4mja5lonwin44vasev4at_jeop34vatuvect4_ta4m4velev1ellve1nejill55jis_4venu5ve3ojoc5ojoc5ujol4e_sis35verbju1di4ves__ses1ju3ninvi4tjut3a_se1qk4abinvel3kach4k3a4gkais5vi1b4vi4ca5vicuvign3vil3i5vimekar4i1kas_kaur42v1invin2evint4kcom43vi1oviol3kdol5vi5omke5dak5ede_rit2_rin4ken4dkeno4kep5tker5ak4erenu1trker4jker5okes4iket5anu4to5vi3pkfur4_re3w_re5uvire4kilo3vir3uk2in_3kind3nunc5numik3ingkin4ik2inskir3mkir4rv3ism3kis_k1ishkit5cvit2avit1rk5kervi3tu_re5ok5leak3lerk3let_re1mv3ity_re1ivi5zovolv41know3vorc4voreko5miko5pe3vorok5ro4_po2pv5ra4vrot4ks2miv3ure_pi2ev5verwag3owais4w3al_w3alswar4fwass4nu1men3ult5labrwas4tla2can4ulowa1tela4chla2conu4isw4bonla3cula4del5admw5die_out1nug4anu3enlag3r5lah4nud5i_oth54lale_osi4_or2o_or4ilam1ol5amu_ore4lan2d_or3dn5turntub5n3tua3weedweir4n5topwel3ilapi4n3tomn1t2o_op2i_on4ent3izla4tenti3pn3tign1tient4ibwent45laur_ome2_ol4d_of5twest3_oed5l4bit_ob3lw5hidl2catwid4elcen4n1thelch4el3darl3dedl3dehwi5ern4teol5dew_no4cl3dien3teln4tecwim2pld5li_ni4cwin2ecen3int1atnt1aln3swale3cawl1ernsta4_na5kle5drleg1an3s2t3leggn5sonleg3ons3ivwl4iensi2tlel5olelu5n3sion3sien3sid5lemml3emnle2mon4sicns3ibwon2tn3sh2n5seule1nen2seslen3on5seclen5ule3onleo4swoun4wp5inn4scun2sco_mis1_mi4enre3mnre4ix4ach4les_x4adenpri4x3aggnpos4npla4npil4leur5x3amil3eva5levexan5dle4wil5exaxano4lf5id_lyo3lf3on_lub3l4gall4gemlgi4al4gidl4goixas5pxcav3now3llias4lib1rl1ic_5lich_lo2pnove2nou5v2nousli4cul3ida3nounn4oug3lieul4ifel4ifoxcor5_li4p3notenot1a_li3oxec3r1l4illil4ilim2bno3splim4pnos4on4os_lin4dl4inenor4tn4oronop5i5nood4noneno2mo1nomi3linqnol4i3liogli4ollio3mliot4li3ou5liphlipt5x5edlx5edn_le2pl4iskno3la_le4ml2it_n5ol_no4fa3lithnoe4c3litrlit4uxer4gn4odyno4dinob4ln5obilk5atxer3on5nyi_ki4ex3ia_nnov3x4iasl5lasl4lawl5lebl1lecl1legl3leil1lellle5ml1lenl3lepl3leul3lev_is4o_is4c_ir3rx5ige_in3tllic4nlet4_in3ol5lie4n1l2l2linnk5ilnk5ifn3keyl5liolli5v_in2ixim3ank5ar_in3dllo2ql4lovnjam2_im5b_il4i_ig1n_idi2llun4l5lyal3lycl3lygl3lyhl3lyil5lymx4ime_hov3_ho2ll4mer_hi3bl5mipni3vox4it__he4ilneo4x4its5loadniv4ax4ode_hab2ni4ten5iss2locynis4onis4l_gos3n4isk4loi_lo5milom4mn4is_lon4expel43nipuni1ou5nioln4inu5ninnnin4jn4imelop4en3im1l3opm1lo1qnil4ax4tednik5e3nignn3igml4os_lo1soloss4_ga4mnift4nif4flo5tu5louplp1atlp3erxtre4l5phe_fo3cl2phol3piel3pitxur4b1y2ar_eye3_ex3a3yardl5samls5an4nicllsi4mls4isyas4i_eur4l1s2tni3ba3niac_es3tl5tar_es3pl4teiyca5mlth3inhyd5y3choltin4lti3tycom4lt4ory2cosnhab3_er2al4tusyder4_epi1luch4_eos5n2gumlu4cu_ent2lu1enlu5er_en3slu4ityel5olu4mo5lumpn4gry_en5c5lune_emp4n5gic_em3by5ettlusk5luss4_el2in5geen4gae_ei5rlut5r_ei3dygi5a_ec3t_eco3l4vorygo4i_dys3_du4c_do4eyl3osly4calyc4lyl5ouy1me4news3_de4wly4pay3meny5metnet1ry5miaym5inymot4yn4cim4acanet3an1est1nessn1escmact44mad_4mada4madsma4ge5magn2nes_yn3erma5ho3ma4i4mai_maid3_der2ner2vner5oyni4c_de1mneon4m3algneo3ln3end4n1enne2moyoun4n4ely2neleyp5alneis4man3a5negune3goneg3a3nedi_dav5m4ansne2coyper3m3aphy4petne4cl5neckn3earyph4en3dyind2wemar3vn4dunndu4bn2doundor4n5docnd1lin3diem4at_n1dicnd4hin5deznde4snde4ln1dedn3deayph3in3damm4atsn3daly4p1iy4poxyp5riyp4siypt3am5becn4cuny3ragm4besyr3atm2bicnct2oyr3icm4bisy5rigncoc4n1c2lm3blimbru4mbu3lmbur4yr3is_can1ys5agys5atmea5gn4cifme4bame4biy3s2c4med_n4cicn3chun3chon3chan5ceyme4dom5edy_bre2n5cetn3cer4melen1c2anbit4nbet4mel4tnbe4n_bov4ys1icys3in3men_2menaysi4o3nautnaus3me1nenat4rnati45meogys4sonas3s4merenas5p2me2snas5iys4tomes5qyz5er1me2tnam4nmet1e3nameza4bina3lyn5algmet3o_aus5_au3b_at3t_at3rza4tena5ivmi3co5nailm4ictzen4an5agom4idina4ginag4ami5fimig5an2ae_mi2gr_as4qmi5kaz5engm3ilanadi4nach4zer5a3millmi5lomil4t3m2immim5iz3et4_ari4_ar4e_ar5d5zic4_ap4i5my3c_any5z3ing3zlemz3ler_an3smu4sem5uncm2is_m4iscmi4semuff4zo3anmsol43zoo2_and2zo3olzo3onzo5op4mity_am2i_al1k_air3_ag5nmlun42m1m2_ag4amp5trmp3tompov5mpo2tmmig3_af3tmmis3mmob3m5mocmmor3mp3is4m1n2mnif4m4ninmni5omnis4mno5l_af3f_ae5d_ad3o_ad3em3pirmp1inmo4gom5pigm5oirmok4imol3amp5idz3zarm4phlmo3lyz5zasm4phe_ach4mona4z3ziemon1gmo4no_ace45most_ab4imo3spmop4t3morpz5zot",6:"reit4i_ab3olmo5rel3moriam5orizmor5onm3orab3morse_acet3_aer3i_al5immo3sta2m1ous_al3le4monedm4pancm4pantmpath3_am5ar_am3pemper3izo5oti_am3phmo4mis_ana3b_ana3s_an5damog5rimp3ily_an4el_an4enmmut3ammin3u_an4glmmet4e_ant3am3medizing5imman4d_ar5abm5itanm3ists_ar5apmsel5fm3ist_5missimis3hamuck4e4misemmul1t2_ar4cimu5niomun3ismus5comirab4mus5kemu3til_at5ar1m4intmin3olm4initmin5ie_bas4i_be3di5myst4_be3lo_be5sm5min4d_bi4er_bo3lo_ca3de_cam5inac4te_cam3oyr5olona4d4amil4adnad4opyr3i4t_car4imid5onn4agen_ca4timid4inmi4cus_cer4imi3cul3micromi4cinmet3ri4naledyp5syfn4aliameti4cmeth4i4metedmeta3tna5nas_cit4anan4ta_co5itnan4to_co3pa4n4ard_co3ru_co3simes5enmer4iam5erannas5tenat5alna5tatn4ateena3thenath4l5mentsn4ati_nat5icn4ato_na3tomna4tosy4peroy4periy5peremend5oyoung5naut3imen4agna5vel4m5emeyo4gisnbeau4_de3linbene4mel3on_de3nomel5een4cal_yn4golncel4i_de3ra_de3rimega5tncer4en4ces_yn5ast3medityn5ap4nch4ie4medieynand5ynago43mediaym4phame5and_de3vem5blern4cles_dia3s_di4atmb5ist_din4anc4tin_dio5cm5bil5m4beryncu4lo_east5_ed5emncus4tmbat4t_elu5sn3da4c3m4attn4dalema3topnd3ancmat5omma3tognde3ciyes5tey3est__em5innd3enc_em5pyn3derlm4atit_en5tay4drouma3term4atenndic5undid5aydro5snd5ilynd4inend3ise_epi3d_er4i4nd5itynd3ler_er4o2_eros43mas1ty4collnd5ourndrag5ndram4n5dronmassi4y4colima3sonyclam4mar5rima3roone3aloma5ronne2b3umar5ol5maran_erot3_er4rima5nilych5isne4du4manic4man3dr_eth3e3m4an__eval3ne5lianeli4g_far4imal4limal3le_fen4dm3alismal3efmal5ed5male24nered_fin3gxtra3vner4r5mal3apxtra5d2mago4ma4cisne3sia5machy_fu5ganes3trmac3adnet3icne4toglys5erxtern3neut5rnev5erlypt5olymph5n4eys_lyc5osl5vet4xter3ixpoun4nfran3lv5atelu5tocxpo5n2_ge3ron3gerin5gerolut5an3lur3olu3oringio4gn5glemn3glien5gliol3unta_go3nolu2m5uxo4matluc5ralu2c5o_hama5l3t4ivltim4alti4ciltern3lt5antl4tangltan3en4icabni4cen_hem5anict5a_hy3loni4diol3phinni4ersximet4lot5atnif5ti_ico3s_in3e2loros4lo5rof_is4li_iso5ml4ored_ka5ro_kin3e5nimetn4inesl3onizl3onisloni4e3lonia_lab4olo5neyl5onellon4allo5gan3lo3drl3odis_la4me_lan5ixen4opnitch4loc5ulni3thon4itosni5tra_lep5rni3trinit4urloc3al5lob3al2m3odnivoc4niz5enlm3ing_lig3anjur5illoc5ulloc3an5kerol3linel3linal5lin__loc3anland5lli5col4liclllib4e_loph3_mac5ulli4anlli5amxa5met_math5llact4nni3killa4balk3erslk3er_lkal5ono5billiv5id_ment4_mi3gr_mirk4liv3erl5ivat5litia5liternois5il3it5a5lisselint5inom3al3lingu5lingtling3i3nonicw5sterws5ingnora4tnor5dinor4ianor4isnor3ma_mi5to_mo3bil4inasl4ina_wotch4word5ili5ger_mon3a5lidifl4idarlict4o_mu3ninova4l5licionov3el_mu3sili4cienow5erli4ani_myth3_nari4le5trenpoin4npo5lale5tra3les4sle3scon4quefler3otleros4ler3om_nast4le5rigl4eric3w4isens3cotle5recwin4tr_nec3tle5nielen4dolend4e_nom3ol5endalem5onn5sickl5emizlem3isns5ifins3ing_nos3tn3s2is4leledle3gransolu4le4ginn4soren4soryn3spirl3egan_obed5nstil4le5chansur4e_ob3elntab4unt3agew5est__oe5sont5and_om5el_on4cewel4liweliz4nt3ast_opt5ant5athnt3ati_or3eo3leaguld3ish_pal5in4tee_n4teesld4ine_pa5tald3estn4ter_n3terin5tern_pecu3war4tel5deral4cerenther5_ped3elav5atlat5usn4tic_ward5r_pend4n4tics_pep3tn3tid4_pi3la_plic4_plos4_po3lan5tillnt3ing_pop5lvo3tar_pur4rn4tis_nt3ismnt3istvo5raclat5al4laredlar5delar5anntoni4lan4tr_re3cantra3dnt3ralviv5orn3tratviv5alnt3rilv5itien5trymlan3etlan4er3landsvi5telland3i3land_lan3atlam4ievi3tal2v5istla4ic_la4gisla3gerlac5on5visiola5cerla5ceolabel4vi5ridlab5ar_re3ta5numerkin5et_rib5anu3tatn5utivkey4wok5erelkal4iska5limk2a5bunven4enven5o_ros3ajuscu4_sac5rjel5laja5panja2c5oi5vorevin5ta_sal4inym5itv5iniz5vinit3vinciiv3erii4ver_iv5elsoad5ervin4aciv5el_oak5ero3alesiv5ancoal5ino5alitit5uar_sanc5oar5eroar4se_sap5ait4titoat5eeoat5eri4tric_sa3vo4i5titob3ing2obi3o_sci3e4itio_it4insit4in_it5icuiti4coi5tholitha5lobrom4it3erait3entit3enci3tectit4ana3istry_sea3si4s1to5vider_sect4oc5ato4o3ce25vict2ocen5ovice3r_se3groch5ino3chon_sen3tvi4atroci3aboci4al5verseis4taliss4ivis5sanis4saliss5adi3s2phocu4luver4neislun4ocuss4ver3m4ocut5ris3incis5horocyt5ood3al_ish3op4ishioode4gao5dendo3dentish5eeod3icao4d1ieod3igais3harod1is2v5eriei2s3etis5ere4is3enis3ellod5olood5ousise5cr4i1secisci5cver3eiver5eaven4tris5chiis3agevent5oir5teeir5ochve5niair4is_ir2i4do3elecoelli4ir5essoe3o4pire5liven4doi5rasoven4alvel3liir4ae_ir4abiv4ellaip3plii4poliip3linip4itiip1i4tip4ine_su5daiphen3i1ph2ei3pendog5ar5v3eleripar3oi4oursi4our_iot5icio5staogoni45ioriz4ioritiora4mvel3atiod3i4ioact4_sul3tintu5m_tar5oin3til_tect45vateein4tee_tel5avast3av5a4sovar4isin3osiin5osei3nos_oi5ki5oil3eri5noleoin3de4vantlvanta4oin4tr_ter4pin3ionin4iciin5ia_oit4aling3um4ingliok4ine4ingleing5hain5galo4lacko5laliinfol4olan5dol5ast_thol45val4vole2c4ol5eciol5efiine5teole4onin3esi4in5eoo3lestin5egain5drool3icao3lice_ti5niol5ickol3icsol5id_va5lieo3lier_tri3dinde3tvager4oli5goo5linaol3ingoli5osol5ip4indes5inde5pin5darollim34vagedol4lyi3vag3ava5ceo4inataol3oido4lona_tro4vi3nas_in4ars_turb44ol1ubo3lumi_turi4ol3us_oly3phin3airin5aglin4ado4inaceimpot5im5pieo4maneomast4_tu5te_tu3toi3mos_im5mesomeg5aome3liom3enaomen4to3meriim5inoim4inei3m2ieomic5rom4ie_imat5uom4inyomiss4uv5eri_un5cei5m2asim3ageil5ureomoli3o2mo4nom5onyo4mos__un5chilit5uom5pil_un3d2il4iteil5ippo5nas__uni3c_uni3o4iliou_un3k4oncat3on4cho_un3t4u4t1raon3deru4to5sili4feili4eri5lienonec4ri3lici_ve5loon5ellil3iaron3essil3ia_ong3atilesi45u5tiz4o1niaon5iar2oni4conic5aut3istut5ismon3iesigu5iti4g5roi5gretigno5m4onneson5odiign5izono4miu5tiniut3ingo5nota_ver3nig3andu4tereon4ter_vis3ionton5if5teeon4treif5icsut5eniutch4eif3ic_u3taneoof3eriev3erook3eri5eutiiet3ieool5iei3est_i1es2ties3eloop4ieieri4ni3eresus5uri4idomioot3erooz5eridol3ausur4eo5paliopa5raopath5id4istopens4id1is43operaus4treidios4_vi5sooph4ieo5philop5holi3dicuus1to4iderm5op3iesop5ingo3p2itid3eraust3ilid3encopol3ii5cun4op5onyop5oriopoun4o2p5ovicu4luop5plioprac4op3ranict5icopro4lop5ropic4terust5igust4icicon3ous5tanic5olaor5adoich5olus3tacic5ado4oralsib3utaoran3eab5areorb3ini4boseorch3iibios4ib3eraor5eadore5arore5caab5beri5atomia5theoreo5lor3escore3shor3essusk5eru4s1inor5ett4iaritianch5i2a3loial5lii3alitab3erdor3ia_4orianori4cius5ianorien4ab3erria5demori5gaori4no4orio_or5ion4oriosia5crii2ac2rus4canor3n4a5ornisor3nitor3oneabi5onor5oseor5osohys3teorrel3orres3hyol5ior4seyor4stihyl5enort3anort3atort3erab3itaor3thior4thror4titort3izor4toror5traort3reh4warthu3siahu4minhu5merhu4matht4ineht4fooht3ensht3eniab4ituht3en_ab3otah3rym3osec3uhrom4ios5encosens43abouthre5maabu4loab3useho4tonosi4alosi4anos5ideo3sierhort5hho5roghorn5ihor5etab3usio3sophos3opoho2p5ro3specho5niohong3ioss5aros4sithon3eyur3theos4taros5teeos5tenac5ablur5tesos3tilac5ardost3orho5neuhon5emhom5inot3a4gurs3orho4magach5alho5lysurs5ero5ta5vurs5alhol3aroter4muroti4ho3donachro4ur5o4mach5urac5onro5thorurn3ero5tillurn3alh5micao3tivao5tiviur5lieo5toneo4tornhirr5ihio5looturi4oty3lehi5noph5inizhi5nieh2in2ehimos4hi5merhi5ma4h3ifi4url5erhi4cinur5ionur4iliur4ie_ac2t5roult5ih4et3ahes3trh5erwaound5aac5uatur3ettoun3troup5liour3erou5sanh4eron5ousiaher5omur1e2tur3ersova3lead5eni4ovatiad3icao4ver_over3bover3sov4eteadi4opadis4iovis5oo2v5oshere3ohere3aherb3iherb3aher4ashende5ur5diehe5mopa3ditihemis4he3menowi5neh3el3ohel4lihe5liuhe3lioh5elinhe5lat5admithe5delhec3t4adram4heast5ad3ulahdeac5ae4cithavel4ura4cipac4tepa5douhas4tehar4tipa3gan4pagataed5isu5quet4pairmpa5lanpal3inag4ariharge4pan5ac4agerihant3ah5anizh1ani4agi4asham5an4aginopara5sup3ingpa3rocpa3rolpar5onhagi3oag3onihaged5agor4apa3terpati4naha5raaid5erail3erhadi4epaul5egust5apa5vilg4uredg4uraspaw5kigui5ta5guit43guardaim5erai5neagrum4bpec4tugru3en5ped3agrim3a4grameped3isgour4igo5noma3ing_5gnorig4ni2ope5leogn4in_pen4at5p4encu5orospen5drpen4ic3p4ennal5ablg2n3ingn5edlalact4until4g5natial5ais5gnathala3map3eronalc3atald5riun4nagg5nateglu5tiglu5tepes4s3ale5ma4g5lodun5ketpet3eng5lis4gli5ong4letrg4letoal3ibrali4cigin5gigi5ganun3istph5al_gi4alluni3sogh5eniph5esiggrav3ggi4a5al5icsg5gedlun4ine3germ4phi5thgeo3logen5ti4phobla5linigen5italin5ophos3pgen4dugel5ligel4ing4atosg4ato_gat5ivgast3ral5ipegasol5ga5rotp5icalu3n2ergar3eeg5antsgan4trp4iestpi5etip5ifieg5ant_un4dus4ganed4alis_gan5atpi3lotgam4blun4diepin5et3pingegali4a5p4insga5lenga4dosga4ciefu5tilpir5acfu3sil4furedfu4minundi4cpiss5aunde4tpis4trft4inefti4etf4ter_un3dedpla5noun4dalalk5ieun4as_al4lab4pled_frant4frag5aunabu44plism4plistal4lagu4n3a4umu4lofore3tfor4difor5ayfo5ramfon4deallig4fo4liefo1l4ifoeti42p5oidpois5iump5tepo4ly1poly3spoman5flum4iump5lipon4acpon4ceump3er3ponifpon5taf3licaf5iteepo5pleal3ogrpor3ea4poredpori4ffir2m1fin4nial3ous5fininpos1s2fi3nalu4moraumi4fyu2m5iffight5fier4cfid3enfi5delal5penp4pene4ficalumen4tal3tiep4pledp5plerp5pletal5uedal3uesffor3effoni4ff3linf2f3isal5ver2a1ly4fet4inaman5dul3siffet4ala3mas_fest5ipres3aulph3op3reseulph3i5pricipri4es4pri4mam5atuam4binfest3ap5riolpri4osul4litfess3o4privafer5ompro3boul4lispro4chfe5rocpron4aul4latam5elopro3r2pros4iu5litypro3thfer3ee4feredu5litipsal5tfemin5fea3tup5sin_fant3iul5ishpsul3i4fan3aul3ingfa5lonu3linefa2c3ufa3cetpt5arcez5ersp5tenapt5enn5pteryez5er_ex4on_ew5ishamen4dp2t3inpt4inep3tisep5tisievol5eevis5oam3eraev5ishev4ileam5erle4viabpudi4ce4veriam5icapu4laramic5rpu5lisu5lentu1len4a3miliev5eliev3astpun5gieva2p3eval5eev4abieu3tereu5teneudio5am5ilypu3tat5ulcheet3udeet3tere4trima5mis_et4riaul5ardet4ranetra5mamor5aetra5getor3iet3onaamort3am5ose3quera4quere4ques_et5olo5quinauit5er3quito4quitueti4naeti4gie3ticuuisti4ethyl3ra3bolamp3liuis3erampo5luin4taet5enia5nadian3agerag5ouuinc5u3raillra5ist4raliaet3eeret3atiet3ater4andian3aliran4dura5neeui3libra3niara3noiet5aryan3arca5nastan4conrant5orapol5rap5toet3arieta5merar3efand5auug3uraan5delet3al_es4ur5e2s3ulrass5aan5difug5lifra5tapra5tatrat5eurath4erat3ifan5ditra5tocan5eeran3ellra4tosra5tuirat5umrat3urrav5aian3ganrav3itestud4ra3ziees5tooe3stocangov4rb3alian4gures5taue5starest3anesta4brbel5orb3entes4siless5eeessar5rbic5uan5ifor5binee5s2pres5potan5ionrbu5t4es5pitrcant54anityr4celean3omaan4scoans3ilrcha3irch3alan4suran2t2ar3cheor4cherud3iedr4chinrch3isr3chites3onaan3talan5tamrciz4ies3olae3s4mie3skinrcolo4rcrit5an4thies4itses4it_e5sion3anthrrd4an_es5iesr5de4lr3dens4anticrd5essrd5ianan4tiee5sickes5ic_rd3ingesi4anrd1is2rd5lere3sh4aes5encrd5ouse5seg5e3sectescut5esci5eant4ives5chees5canre5altre5ambre3anire5antre5ascreas3oeryth35erwauan4tusreb5ucre3calrec4ceer4vilan5tymre3chaan3um_an5umsap5aroerund5ert5izer4thire3disre4dolape5lireed5iu4cender4terer5tedre3finuccen5re5grare3grereg3rire3groreg3ulaph5emer4repaph5olaphyl3ero5stero5iser3oidern3it4reledre3liarel3icre5ligreli4qrel3liern3isrem5acap5icuub3linern3errem5ulu4bicuren5atr4endiap4ineren4eser4moirenic5ren4itub5blyre5num4eri2ta3planre5olare3olier4iscer3ioure4pereri4onrep5idre3pinre3plere4preeri4nauari4ner3iffre5reare3r2uapo3thre3scrre3selre3semre3serap5ronre5sitre3speapt5at4arabiara5bore5stu3retarre3tenar3agear5agire1t2ore5tonre3trare3trere5trier4ianer3ia_ergi3ver3ettrev3elrevi4ter3etser3et_ar3agoar3allaran4ger3esier5eseere5olr4geneeren4e5erende4remeer5elser5ellr5hel4rhe5oler5el_er3egrer3ealerdi4eerd5arerb5oser3batar5apaer5atuarb5etar4bidty4letri5cliri3colri5corri4craarb3lirid4aler3apyer3apier3aphera4doar4bularch5otwi5liri5gamaren5dri5l4aar5ettar3ev5ar5iff5tur5oequin4rima4gar4illrim3ate4putarimen4e3pur5ept3or5turitr4inetturf5iturb3aep5rimt4uranrins5itu5racep3rehtun5it5rioneepol3iepol3ari5p2ari5piear5iniep3licarm3erris4ise4peteris4paris4pear5mit4ristiri3tonr5it5rep5ertriv4alar3nalar3nisriv3enriv3il5ri5zoar5oidep5arceor4derk5atir5kellrk5enia5rotieol5ata5roucr3kiertud5ier5kin_r5kinsrks4meen4tusent5uptu5denr3l4icr3liner5linsen4tritu4binen5tiarma5cetuari4ent3arr4mancr4manor4marir4maryen4susars5alart5atarth4een4sumens5alrm4icar5m2iden3otyenit5ut4tupermin4erm3ingarth3rar5tizen5iere2n3euen4ettrmu3lie3nessen5esiener5var5un4as5conrn3ateas5cotrn5edlt3tlerr3nessrn5esttti3tuas3ectt5test3encept4tereen3as_rn4inee2n3arrn3isten4annash5ayem4preash5ilem5pesas5ilyempa5rask5erem3orras5ochrob3letstay4e3moniem3oloemod4uemo3birody4n4emnitem4maee4mitaem3ismem5ingem3inar4oledas4silassit5as4tatro5melro3mitas4tiaas3tisemet4eron4ac4ronalas4titron5chron4dorong5ir5onmeem5ero4asto2as3traas4trit5roto4atabiem3anaro3peltro3spem3agor5opteel5tieelp5inel5opsrosi4aro5solel5op_5troopros4tiatar3aro3tatata3t4ro4terelo4dieloc3uelo5caat3eautri3me4roussell5izel4labrow3erelit4ttri3lie4li4seli3onr3pentrp5er_el3ingat3echr3pholrp3ingat5eerrpol3ar2p5ouele3vi3tricuelev3at5ricla5tel_e5lesstres4sele5phel3enor4reo4el5eni4e4ledelea5grricu4tre5prate5lerri4oseld3ertre4moat3entat3eraelast3el5ancel5age4traddeiv3ereit5ertra4co4atesse4ins_to3warehyd5re5g4oneg5nabefut5arsell5rs3er_rs3ersa3thene4fiteath3odr4shier5si2ato3temto5stra5thonrs3ingeem5eree2l1ieed3ere4d5urrstor4to3s4ped3ulo4a3tiator5oitor5ered3imeed5igrrt3ageto5radr4tareed5icsto4posr4tedlr3tel4r5tendrt3enito5piaa2t3in4atinaat5ingede3teton5earth3rir1t4icr4ticlr5tietr5tilar5tilltom5osrt5ilyedes3tr3tinart3ingr3titirti5tue4delee5dansrt5lete5culito4mogec4titrt5ridecti4cec4teratit3urtwis4e4cremtoma4nec3ratec5oroec3oratom3acat4iviec3lipruis5iecip5i4toledec5ath5at5odrun4clruncu42t3oidrun2d4e4caporu5netecal5ea4topsec3adea4toryebus5iebot3oe4belstode5cat3ronat5rouat4tagru3tale4bel_eav5our4vanceavi4ervel4ie3atrirven4erv5er_t4nerer3vestat3uraeatit4e3atifeat5ieeat3ertmo4t5east5iat3urge1as1s3ryngoau5ceraud5ereas5erryth4iaudic4ear4tee5ar2rear4liear3ereap5eream3ersac4teeam4blea3logeal3eread3liead3ersain4teac4tedy4ad_sa5lacdwell3sa3lies4al4t5tletrdvert3sa5minault5id5un4cdum4be5tledrs4an4etlant4san5ifdu5ettau5reodu5elldu5eliau5rordrunk3tiv3isaus5erdri4g3aut3ars5ativti3tradrast4d5railsau5ciaut3erdossi4sa3voudo5simdon4atdom5itt3itisdomin5doman4tit5ildo4lonscar4cdol5ittith4edol3endo4c3u4s4ces5dlestt4istrdi4val1di1v2ditor3av3ageava5latish5idithe4av5alr3tisand4iterd4itas3disiadisen34d5irodi4oladi5nossec5andin5gisecon4dimet4di5mersed4itdi3gamdig3al3di3evdi4ersd5icurse3lecselen55dicul2s4emedic4tesemi5dav5antdic5oldic5amt3iristi5quaav3end5sentmti3pliav3ernti5omosep4side4voisep3tiser4antiol3aser4to4servode3vitde3visdev3ils5estade3tesdes3tid3est_sev3enaviol4aw5er_de3sidde3sectin3uetin4tedes4casfor5esfran5der5os3dero45dernesh4abiaw5ersder4miaw5nieay5sta3dererde5reg4deredde3raiderac4si4allsiast5tin3ets3icatdepen42s5icldeont5si5cul4tinedba5birdens5aside5lsid3enbalm5ideni4eba5lonsi4ersde1n2ade4mosde3morba5nan5tilindemo4nti4letsin5etbardi44demiedel5lisi5nolsi3nusba5romdeli4esi5o5sde3lat5de3isde4fy_bar3onde4cilsist3asist3otigi5odeb5itsit5omdeac3td3dlerd4derebas4tedaugh3dativ4dast5a3d4as2d1an4ts3kierba4th4sk5ily3baticba5tiod4a4gid5ache3ti2encys5toc3utivbat5on4cur4oti3diecur4er1c2ultb4batab4bonecul5abcu5itycub3atctro5tbcord4ti3colct5olo3smithbdeac5tic5asct5ivec4tityc4tituc3t2isbed5elc3tinict5ing4s3oid4te3loct4in_so5lansol4erso3lic3solvebe5dra5ti5bube3lit3some_bend5ac4ticsbe5nigson5atbicen5son5orc4tentbi4ers5soriosor4its5orizc2t5eec3tato5bilesct5antc5ta5gctac5u5c4ruscrost4spast45thoug3b2ill3sperms5pero4thoptcre4to5creti3spher4t5hoocre4p3sp5id_s5pierspil4lcre3atsp3ingspi5nith3oli4creancra4tecras3tbimet55crani5bin4d3spons3spoonspru5dbind3ecous5t3co3trth4is_srep5ucost3aco5rolco3rels5sam24coreds5sengs3sent5th4ioss3er_s5seriss3ers3thinkt5hillbin5etcon4iecon4eyth3eryss4in_s4siness4is_s3s2itss4ivicon4chth3ernco3mo4co5masssol3ut5herds4soreth5erc5colouco3logco3inc4c3oidco3difco3dicsta3bic4lotrs4talebin5i4s3tas_theo3lc3lingbi3re4ste5arste5atbi5rusbisul54s1teds4tedls4tedn4stereth5eas3bituas3terost5est5blastcine5a4cinabs3ti3a3sticks3ticuthal3ms4tilyst3ing5s4tir5cimenth5al_st3lercigar5ci3estch5ousstone3bla5tu5blespblim3as4tose4chotis4tray4chosostrep33strucstru5dbment4tew3arch5oid5chlorstur4echizz4ch3innch4in_ch3ily3chicoche5va3chetech4erltetr5och4eriche3olcha3pa4boledbon4iesu5ingces5trcest5oce3remcer4bites5tusu3pinsupra3sur4ascept3a5testesur3pltest3aboni4ft3ess_bon4spcent4ab3oratbor5eebor5etbor5icter5nobor5iocen5cice4metce5lomter3itt4erinsy4chrcel3aice3darcci3d4ter5ifsy5photer5idcav3ilter3iabot3an3tablica3t2rta3bolta4bout4a3cete3reota3chyta4cidc4atom3casu35t2adjta5dor5terel3cas3scashi4tage5ota5gogca3roucar5oocar5oncar3olcar3nicar3ifter5ecca3reeter3ebta5lept4aliat4alin2tere45tallut2alo43ter3bt4eragtera4c3brachtan5atbran4db4reas5taneltan5iet5aniz4b2rescap3tica5piltent4atark5ican4trte5nog5brief5tennaca3noec2an4eta3stabring5t4ateu3tatist4ato_tat4ouca5nartat3uttau3tobri4osca5lefcal5ar4tenarcab5inb5ut5obut4ivten4ag3butiob5utinbu5tarte5cha5technbus5sibusi4ete5d2abur4rite5monb4ulosb5rist5tegicb5tletbro4mab4stacbso3lubsol3e4teledtel5izbscon4ct4ina",7:"mor4atobstupe5buf5ferb5u5nattch5ettm3orat4call5inmor5talcan5tarcan5tedcan4tictar5ia_brev5ettant5anca3ra5ctand5er_ad4din5ta3mettam5arit4eratocar5ameboun5tital4l3atal5entmonolo4cas5tigta5chom3teres4ta5blemcaulk4iccent5rcces4sacel5ib5mpel5licel5lincen5ded5ternit4sweredswell5icend5encend5ersvest5isvers5acen5tedt5esses_ama5tem5perercen5testest5ertest5intest5orcep5ticmpet5itchan5gi5cherin4choredchor5olmphal5os5toratblem5atston4iecil5lin4mologu4mologss4tern_ster4iaci5nesscla5rifclemat45static4molog_5therapmogast4ssolu4b4theredcon4aticond5erconta5dcor5dedcord5ermpol5itcost5ercraft5ispon5gicra5niuspital5spic5ulspers5a4thorescret5orspens5ac5tariabi4fid_4sor3iecter4iab5ertinberga5mc5ticiabend5erso5metesoma5toctifi4esolv5erc5tin5o_an4on_ct4ivittici5ar3ti3cint4icityc5torisc5toriz4ticulecull5ercull5inbattle5cur5ialmmel5lislang5idal5lersk5iness5kiest4tific_daun5tede5cantdefor5edel5ler_an3ti34dem4issim4plyb4aniti_ant4icde4mons_an4t5osid5eri5timet4dens5er5ti5nadden5titdeposi4zin4c3i_aph5orshil5lider5minsfact5otin5tedtint5erde5scalmis4tindes5ponse5renedevol5u4tionemdiat5omti5plexseo5logsent5eemi5racu_ar4isedic5tat4scuras4scura__ar4isi5scopic3s4cope5t4istedi5vineti5t4ando5linesca5lendom5inodot4tins5atorydress5oaus4tedtiv5allsassem4dropho4duci5ansant5risan5garaun4dresan4ded_ar5sendust5erault5erdvoc5ataul5tedearth5iea4soni4ryngoleassem4eat5enieat4iturv5ers_rus4t5urus5ticrust5eeatric5urust5at_as5sibrup5licminth5oecad5enruncul5ru4moreecent5oa5tivizecon4sc_ateli4_au3g4uec5rean_aur4e5ect5atiec4t5usrtil5le4at4is__av5erar4theneedeter5edi4alsr5terered5icala4t1i4lediges4at5icizediv5idtori4asrswear4ati5citat5icisedu5cerrstrat4eer4ineefact5oming5li_ba5sicef5ereemin4ersath5eteath5eromin4er__be5r4ae5ignitr5salizmind5err5salisejudic44traistmil5iestrarch4tra5ven_blaz5o4m5iliee4lates_bos5omat5enatelch5errrin5getrend5irri4fy_rran5gie4lesteel3et3o_boun4d_bra5chtri5fli_burn5ieli4ers_ca4ginrou5sel_can5tamigh5tiros5tita5talisro5stattro4pharop4ineemarc5aem5atizemat5ole4m3eraron4tonro5nateem4icisnaffil4romant4emig5rarol5iteass5iblassa5giemon5ola4sonedem5orise4moticempara54empli_en3am3o_cen5sot5tereren4cileen4d5alen4dedlttitud45n4a3grend5ritrn5atine5nellee5nereor4mite_r4ming_en3ig3rmet5icirma5tocr4m3atinannot4en4tersen4tifyarp5ersent5rinr5kiesteol5ar_eologi4aro4mas_clem5eriv5eliri5vallris5ternan5teda5rishi3mesti4epolit5tup5lettup5lic_cop5roepres5erink5erme5si4aring5ie_co5terrim5an4equi5noment5or4tut4ivna5turiera4cierig5ant5rifugaar4donear5dinarif5tiear5chetrift5er4erati_4eratimrick4enrich5omrica5tuaran5teer5esteer5estieres5trre5termar4aged_dea5coaract4irest5erre5stalapu5lareri4ciduant5isuant5itres5ist5er5ickapo5strer4imet_de5lecuar4t5iua5terneri5staren4ter5ernaclmend5errem5atoreman4d_del5egerre5laer5sinere5galiert5er_ert5ersrec4t3rr4e1c2rreci5simelt5er_deli5ran4tone_de5nitan4tinges5idenesi5diur4d1an4rcriti4es3ol3urci5nogant5abludi4cinrch4ieru5dinisrch5ateu5ditiorch5ardes3per3mel5lerrcen5eres5piraanis5teesplen5uen4teres4s3anest5ifi_de5resues5trin4cept_rav5elianel5li4r4atom5ra5tolan4donirat4in_r4as5teand5istrass5in5meg2a1et3al5oand5eerrar5ia_an3d4atrant5inuicent55rantelran5teduild5erran4gennch5oloetell5irad4inencid5enra5culorac5ulaet3er3aet5eria3ra3binet5itivui5val5amphi5gam5peri_de5sirqua5tio4e4trala4mium_et5ressetrib5aaminos4am5inizamini4fp5u5tis5ulchrepush4ieev5eratev5eren4ulenciever4erpu5lar_puff5erevictu4evis5in_de5sisfall5inncip5ie_di4al_fend5erpros5trpropyl5proph5eul4l5ibp3roc3apris5inpring5imbival5nco5pat5pressiyllab5iulp5ingpre5matylin5dem4b3ingnct4ivife5veriffec4te_du4al_pprob5am5bererum4bar__echin5fi5anceal5tatipparat5pout5ern4curviumi5liaumin4aru4minedu4m3ingpoult5epor5tieal4orim4poratopon4i4eflo5rical4lish_ed4it_foment4_ed4itialli5anplum4befor4m3a_el3ev3fratch4pla5t4oma5turem4atizafrost5ipis5tilmat4itifuel5ligal5lerpill5ingang5ergariz4aunho5lial5ipotgass5inph5oriz4phonedgest5atg5gererphant5ipha5gedgiv5en_5glass_unk5eripet5allal5endepes5tilpert5isper5tinper4os_al5ance5p4er3nperem5indeleg4gna5turndepre4aint5eruodent4pend5er4gogram_en4dedpearl5indes5crgth5enimas4tinpat4richad4inepas4tinnd5is4ihak4inehal5anthan4crohar5dieha5rismhar4tedaet4or_aerody5pag4atihaught5_er5em5hearch44urantiheav5enurb5ingoxic5olowhith4ur5den_ur5deniowel5lih5erettovid5ennd5ism_her5ialh5erineout5ishoun5ginound5elhet4tedact5oryu5ri5cuheumat5ur5ifieact5ileought5ihi3c4anuri4os_h4i4ersh4manicurl5ingact5atemast4ichnocen5_men5taaci4erso5thermmar4shimantel5ot5estaurpen5tach5isma5chinihol4is_ot4atioot4anico5talito5stome5acanthost5icaosten5tost5ageh4op4te3house3hras5eoy4chosen5ectom4abolicht5eneror5tes_man4icay5chedei5a4g5oori5cidialect4or5este_escal5iatur4aorator5_wine5s_vo5lutich5ingo5quial_etern5us5ticiic4tedloplast4ophy5laid4ines4operag2i4d1itoost5eriff5leronvo5lui4ficaconti5fiman5dar_vic5to_fal4lemament4mal4is__ver4ieila5telonical4i5later_feoff5ili4arl_va5ledil4ificond5ent_ur5eth5ond5arut4toneil5ine_on5ativonast5i_under5ompt5eromot5ivi4matedi4matin_fi5liaimpar5a_fil5tro5lunte4inalit_tular5olon5el5neringinator5_tro4ph_fis4c5inc4tua_trin4aol4lopeoli4f3eol5ies_mal5ari_tran4c_tit4isnerv5inval4iseol5icizinfilt5olat5erin4itud_gam5etxter4m3ink4inein4sch5_tell5evas5el5insect5insec5uinsolv5int5essvat4inaoher4erint5res_tamar5xtens5o_tact4iinvol5ui4omani_gen4et_gen5iave5linei5pheriip5torivel5lerir4alinvel5opiir4alliirassi4nfortu5irl5ingirwo4meo4ducts4lut5arv5en5ue_stat4o_si5gnoverde5v4v4ere4o4duct_odu5cerodis5iaocus5siis5onerist5encxotrop4_ser4ie5vialitist5entochro4n_gnost4_sec5tovi5cariocess4iis4t3iclum4brio5calli4is4tom4itioneit5ress3vili4av5ilisev5ilizevil5linoast5eritu4als_han4de_hast5ii4vers__sa5linlsi4fiai5vilit5ivist_5ivistsnvoc5at_ho5rol_rol4lakinema4ni4cul4nultim5_re5strloth4ie5la5collos5sienight5ilor4ife_re5spolor5iatntup5li5lo5pen_re5sen_res5ci_re5linnt5ressn4trant_re5garloom5erxhort4a_ran5gilong5invol4ubi_ra5cem_put4ten5tition4tiparlo4cus__pos5si_lash4e_len5tint5ing_nit5res_le5vanxecut5o_plica4n4tify__plast45latini_phon4illow5er_li4onslligat4_peri5nntic4u4_pen5dewall5ern5ticizwan5gliwank5erwar5dedward5ern5ticisnth5ine_lo4giawar5thinmater4_pec3t4_pa4tiowav4ine_lous5i_para5t_par5af_lov5ernmor5ti_orner4nt5ativ_or5che_ma5lin_mar5ti_or4at4le5ation5tasiswel4izint4ariun4t3antntan5eon4t3ancleav5erl3eb5rannel5li_nucle5_no5ticlem5enclen5darwill5in_ni5tronsec4tewing5er4lentio5l4eriannerv5a_nas5tinres5tr5le5tu5lev5itano5blemnovel5el3ic3onwol5ver_mor5tilift5erlight5ilimet4e_mo5lec5lin3ealin4er_lin4erslin4gern5ocula_min5uenobser4_met4er_me5rin_me5ridmas4ted",8:"_musi5cobserv5anwith5erilect5icaweight5ica5laman_mal5ad5l5di5nestast5i4cntend5enntern5alnter5nat_perse5c_pe5titi_phe5nomxe5cutio5latiliz_librar5nt5ilati_les5son_po5lite_ac5tiva5latilisnis5tersnis5ter_tamorph5_pro5batvo5litiolan5tine_ref5eremophil5ila5melli_re5statca3r4i3c5lamandrcen5ter_5visecti5numentanvers5aniver5saliv5eling_salt5ercen5ters_ha5bilio4c5ativlunch5eois5terer_sev5era_glor5io_stra5tocham5perstor5ianstil5ler_ge5neti_sulph5a_tac5ticnform5eroin4t5erneuma5to_te5ra5tma5chinecine5mat_tri5bal_fran5ch_tri5sti_fi5n4it_troph5o_fin5essimparad5stant5iv_vent5il4o5nomicssor5ialight5ersight5er__evol5utm5ament_ont5ane_icotyle5orest5atiab5oliziab5olismod5ifiehrill5inothalam5oth5erinnduct5ivrth5ing_otherm5a5ot5inizov5elinghav5ersipass5ivessent5ermater5n4ain5dersuo5tatiopens5atipercent5slav5eriplant5er5sing5erfortu5naplumb5erpo5lemicpound5erffranch5ppress5oa5lumnia_domest5pref5ereprel5atea5marinepre5scina5m4aticpring5ertil4l5agmmand5er5sid5u4a_de5spoievol5utee5tometeetend5erting5ingmed5icatran5dishm5ed5ieset5allis_de5servsh5inessmlo5cutiuest5ratncent5rincarn5atdes5ignareact5ivr5ebratereced5ennbarric5sen5sorier5nalisuar5tersre4t4er3_custom5naugh5tirill5er_sen5sati5scripti_cotyle5e4p5rob5a5ri5netaun5chierin4t5errip5lica_art5icl5at5ressepend5entu4al5lir5ma5tolttitu5di_cent5ria5torianena5ture5na5geri_cas5ualromolec5elom5ateatitud5i_ca5pituround5ernac5tiva_at5omizrpass5intomat5oltrifu5gae4l3ica4rpret5erel5ativetrav5esttra5versat5ernisat5ernizefor5estath5erinef5initeto5talizto5talis_barri5c_authen5mass5ing",9:"_bap5tismna5cious_econstit5na5ciousl_at5omisena5culari_cen5tena_clima5toepe5titionar5tisti_cri5ticirill5ingserpent5inrcen5tenaest5igati_de5scrib_de5signe_determ5ifals5ifiefan5tasizplas5ticiundeter5msmu5tatiopa5triciaosclero5s_fec5unda_ulti5matindeterm5ipart5ite_string5i5lutionizltramont5_re5storeter5iorit_invest5imonolog5introl5ler_lam5enta_po5sitio_para5dis_ora5tori_me5lodio"},patternChars:"_abcdefghijklmnopqrstuvwxyz",patternArrayLength:181888,valueStoreLength:35544};Hyphenator.languages['en-us']=Hyphenator.languages['en']={leftmin:2,rightmin:3,specialChars:"",patterns:{3:"x1qei2e1je1f1to2tlou2w3c1tue1q4tvtw41tyo1q4tz4tcd2yd1wd1v1du1ta4eu1pas4y1droo2d1psw24sv1dod1m1fad1j1su4fdo2n4fh1fi4fm4fn1fopd42ft3fu1fy1ga2sss1ru5jd5cd1bg3bgd44uk2ok1cyo5jgl2g1m4pf4pg1gog3p1gr1soc1qgs2oi2g3w1gysk21coc5nh1bck1h1fh1h4hk1zo1ci4zms2hh1w2ch5zl2idc3c2us2igi3hi3j4ik1cab1vsa22btr1w4bp2io2ipu3u4irbk4b1j1va2ze2bf4oar1p4nz4zbi1u2iv4iy5ja1jeza1y1wk1bk3fkh4k1ikk4k1lk1mk5tk1w2ldr1mn1t2lfr1lr3j4ljl1l2lm2lp4ltn1rrh4v4yn1q1ly1maw1brg2r1fwi24ao2mhw4kr1cw5p4mkm1m1mo4wtwy4x1ar1ba2nn5mx1ex1h4mtx3i1muqu2p3wx3o4mwa1jx3p1naai2x1ua2fxx4y1ba2dn1jy1cn3fpr2y1dy1i",4:"4dryn2itni4on1inn1im_up3nik4ni4dy5giye4tyes4_ye44ab_nhe4nha4abe2n2gyn1guy1ery5eep2pe4abry3lay3lone4wne4v1nesy3chn1erne2q3neo1nenp2seps4hy2cey5lu2nedne2cyme44nk2y5at2adine2b2ne_y5ac2p1tp2ten1den1cun1cryn5dp2th4adup4twpub3ae4rxu3ayn5gaff4pue4n2au4p1ppuf4n2atag1ipu4mag1na2gon4asx3tix1t2pu2na4gya3haa3heah4la3ho_ti2a5ian2an5puspu2tnak4_th2n1kl_te4_ta4mu4u4mupmun23mum2alex4ob_sy25ynxal1i_st4y1o4xi5cxi5a4alm_si2_sh2m5sixhu4m4sh4m3r4amam2py2rabm2pixhi2yo5dr2ai4m1pmo2vmos2x2edmo2r4n1la2mor2asx3c2xas5yom4x4apxam3nme44mokrbi2nne44andy4osp4ot3noemn4omn4a4m1n4nog4m1l2angws4l1posw3shwri4wra4yp3iwom11wo2m2izrb4ow4nopo4pr2cem2isrd2iano4mig4y3pomi3awiz55mi_no4n4m1fme4v2re_wir42mes1menme2mme2gre1o2med4me_4nop4m5c4m1bwil21noureu2whi4w3ev4maprev2w1era2plpo4crfu4r4fyy5pu2maha3pu2mab2a2rn1p4npi44lyb4lya2p3nwam42l1w1lut4luplu3or1glluf4lu5a2wacltu2y3rol1tr4vv4r3guyr4rl1te4rh_nru4ar1il2sel4sc4l1rl5prl4plys4c4lovri3ar4ib4lof3lo_ar2par3q_os3ll4oll2i4as_ri1o3vokl2levoi44p1mlka35vo_ns4cas4ll1izr4iqr2is3vivl1it3lika2tan2sen2slrle42l3hlgo3l5gal5frns3mvi4p3ley_od2r2meles24athr4myle2al3drv1inldi4l2de2vilnt2il3civik4lce42l1b4lavv3ifrno4r3nua1trr2ocnt4sy4sok4syks4la2tuk4sck3ouko5ryss4a2tyau4b4klyys1tnu1akis4au3rki4pro4ek4ima2va5ki_nu4dn4umn3uokes4k1erav1irok2ke4g1keek2ed_me2aw3ikal4aws4k5agk3ab3ka_aye4ays4veg3jo4p5ba_4vedjew3n1v24ve_ja4pzar23vatizi4n1w41batba4z2b1bb2beix4o4i5w4b1d4be_rox5nym4nyp4n3za4ittr3por1r4i1ti1bel2ith2itei2su4rs2r1sars4cr2seis1p3betvag4i2sor1shbe3wr1sioad34b3hbi2bbi4d3bie3isf4ise2is_1bilr1sp5va_r5sw_le2uz4eir1ibi2tuxu3r1tiu1v2i1raze4nze4pb2l2uu4mo1biip3iz1eripe4b4louts44b1m4b3no3br3bodi4osbo4eru3aio4mi1ol4io_3booo1ce4inyin1u2insru2n2inn4inl4inkrv4e2inioch42iner3vo4indpi2np4idbt4lb4tob3trry4cry3t2in_o4elbu4ni2muim1i5saiil3v4ilnil1iil5fs1apo3er4b5w5by_bys4_in1sau4i1lazet4u2suo3ev2z1ii2go4igius1p5saw4s5bo2fi4ifti3fl4if_i3etsch2usc22ie4i2dui4dri2diid5dpi3au3ruz4ils1cuz4is4s5d4se_se4a2ce_2ici4ich3ceii1bri5bo1ceni1blse2g5seiibe43cepi2aniam4ur2li2al2i1acet4hy2scew41phy4ch_5phuhu4thu4gche2h4tyh4shur1durc44hr44h5p5sev5sexu1ra4s3fup3p2s3gph3t2sh_ho4g2h1n_he23ciau3pl4h1mci5ch2lozo4m4ciihi2vhi4p2cim2cin4phsu1peu1ouo1geu5osheu4sho4he4th1es4shwun5zun5ysi1bunu45cizo4glck3ihep5he2nh4ed1sioph2l5hazsi2rcly4zte4_ge21siscoe22cog5siu1siv5siz_ga24skes1l2s2leha4m2s1ms3ma1ogyo1h2u1ni3gus3gun2guegu4acov1gth3_eu3g4ros1n4_es3u2nez4zyum2pu1mi3som_ev2oig4cri2gov15goos4opgon2ul5v5goeu3lugob53go_2c1t4ph_g1nog1nic2te4sov4ulsgn4ag4myc4twcud5c4ufc4uipe2t3glo1gleul2igla4_eg23giz3cun5givgi4u3gir5gio1cusul4e2spagil4g1ic5gi__eb4cze41d2a5da_u1laggo44daf2dagg2gege4v1geo1gen2ged3dato1la2ge_ol2dol2i5daypek4p4eed1d42de_4gazol2tuiv3ol2vo2lys1sa2gamgaf4o2meui4n2ui2pe2cd4em4fugi4jku3fl3ufaf2tyf4to1denu4du4pe_2f3sfri2de1ps1si4f5pfos5d3eqs4sls4snfo2rss2tdes25fon4p1b_ci23payss5w2st_de1tf4l2de1v2fin4dey4d1fd4gast2idg1id2gyd1h25di_ud5dfi3au4cy_ch4pav43didu3cud1iff2fyu3crd1inst4r4f1ffev4fer11dio2fedfe4bdir2s2ty4fe_dis1on1au3ca4f5bon1c2ondd5k25far4fagpa1peys45eyc1exps4ul2dlyp4ale3whon3s3do_e1wa5doee5vud4oge1visu2msu2nub4euav4su2rp4ai6rk_d4or3dosu1atdo4v3doxp4adoo4k4swoo2padre4eus4e3upe5un2ophet5z4syc3syl4y3hoy1ads4pd4swd4syd2tho4wo3ta_du2c4etn2tabta2luac4es4wdu4g2ess4uabdu4n4duptav4st5bow1io1pr5dyn2tawe1sp2t1bop1uead1tz4et4chopy5ea4l4t1d4te_2tyle1si4esh1tee4tyat1cr4twoteg4es2c4eru1teoer1s2eroea2tte4po1rat1wh3tusea2v3teu3texer1i2e1ber1h4tey2t1f4t1ge3br2th_th2e4thle1ce3tumec2i2ths2erb1tia4tueer1aou5vtud2tif22tige1potu1aou4lttu41timt5toos4le1cre2pat4swe5owe1cue4ottsh4eos4e1ort4sce3ol4edieo2ge5of1tio4eno4enn5tiq4edoti4u1tive3my1tiz4othee2ct5laee2ft5lo4t1mee2mtme4e1meem5bcoi4to3be5exo1ry2tof1effel2iel2ftos24t1pe1la1traos2ceig2ei5de5ico2soe1h45egyeg5n",5:"_ach4e4go_e4goseg1ule5gurtre5feg4iceher4eg5ibeger44egaltre4mei5gle3imbe3infe1ingtra3beir4deit3eei3the5ity5triae4jud3efiteki4nek4la2trime4la_e4lactri4v4toute4law5toure3leaefil45elece4ledto2rae5len4tonye1lestro3ve4fic4tonoto3mytom4bto2mato5ice5limto2gre3lioe2listru5i4todo4ellaee4tyello4e5locel5ogeest4el2shel4tae5ludel5uge4mace4mage5man2t1n2ee2s4ee4p1e2mele4metee4naemi4eee4lyeel3i3tled3tle_e4mistlan4eed3iem3iztrus4emo4gti3zaem3pie4mule4dulemu3ne4dritiv4aedon2e4dolti3tle5neae5neeen3emtis4pti5sotis4m3tisee3newti3sae5niee5nile3nioedi5zen3ite5niu5enized1ited3imeno4ge4nosen3oven4swti5oc4t1s2en3uaen5ufe3ny_4en3zed3ibe3diae4oi4ede4s3tini4ed3deo3ret2ina2e2dae4culeo4toe5outec4te4t3t2t4tes2t1ine5pel4timpe2corephe4e4plie2col5tigutu3arti5fytu4bie3pro3tienep4sh5tidie4putt4icoeci4t4tick2ti2bec3imera4bti4aber3ar4tuf45tu3ier4bler3che4cib2ere_4thooecca54thil3thet4thea3turethan4e4cade4bitere4qe4ben5turieret4tur5oeav5oeav5itu5ry4tess4tes_ter5ve1rio4eriter4iueri4v1terier3m4ter3cte5pe4t1waer3noeast3er5obe5rocero4rer1oue3assea5sp1tent4ertler3twtwis4eru4t3tende1s4a3tenc5telsear2te2scateli4e3scres5cue1s2ee2sec3tel_te5giear5kear4cte5diear3ae3sha2t1ede5ande2sice2sid5tecttece44teattype3ty5phesi4uea4gees4mie2sole3acte2sone1a4bdys5pdy4sedu4petaun4d3uleta5sytas4e4tare4tarctar4ata5pl2estrta5mo4talke2surtal3idu5eleta4bta5lae3teoua5naet1ic4taf4etin4ta5doe5tir4taciuan4id1ucad1u1ae3trae3tre2d1s2syn5ouar2d4drowet3uaet5ymdro4pdril4dri4b5dreneu3rouar3ieute44draieu5truar3te2vasdop4pe5veadoo3ddoni4u4belsum3iev1erdoli4do4laev3idevi4le4vinevi4ve5voc2d5ofdo5dee4wage5wee4d1n4ewil54d5lue3wit2d3lou3ber5eye_u1b4i3dledfa3blfab3rfa4ce3dle_fain4suit3su5issu2g34d5lasu4b3fa3tasu1al4fato1di1vd2iti5disiuci4bfeas4di1redi4pl4feca5fectdio5gfe3life4mofen2d4st3wuc4it5ferr5diniucle3f4fesf4fie4stry1dinaf4flydi3ge3dictd4icedia5bs4tops1tle5stirs3tifs4ties1ticfic4is5tias4ti_4ficsfi3cuud3ers3thefil5iste2w4filyudev45finas4tedfi2nes2talfin4ns2tagde2tode4suflin4u1dicf2ly5ud5isu5ditde1scd2es_der5sfon4tu4don5dermss4lid4erhfor4is4siede2pudepi4fra4tf5reade3pade3nufril4frol5ud4side3nou4eneuens4ug5infu5el5dem_s5setfu5nefu3rifusi4fus4s4futade5lode5if4dee_5gal_3galiga3lo2d1eds3selg5amos2s5cssas3u1ing4ganouir4mgass4gath3uita4deaf5dav5e5dav44dato4darygeez44spotspor4s4pon4gelydark5s4ply4spio4geno4genydard5ge3omg4ery5gesigeth54getoge4tydan3g4g1g2da2m2g3gergglu5dach4gh3inspil4gh4to4cutr1gi4agia5rula5bspho5g4icogien5s2pheulch42sperspa4n5spai3c4utu1lenul4gigir4lg3islcu5pycu3picu4mic3umecu2maso5vi5glasu5liagli4bg3lig5culiglo3r4ul3mctu4ru1l4og4na_c3terul1tig2ning4nio4ultug4noncta4b4c3s2cru4dul5ulsor5dgo3isum5absor5ccris4go3nic4rinson4gsona45gos_cri5fcre4vum4bi5credg4raigran25solvsoft3so4ceunat44graygre4nco5zi4gritcoz5egruf4cow5ag5stecove4cos4es5menun4ersmel44corbco4pl4gu4tco3pacon5tsman3gy5racon3ghach4hae4mhae4th5aguha3lac4onecon4aun4ims3latu2ninhan4gs3ket5colocol5ihan4kuni3vhap3lhap5ttras4co4grhar2dco5agsir5aclim45sionhas5shaun44clichaz3acle4m1head3hearun3s4s3ingun4sws2ina2s1in4silysil4eh5elohem4p4clarhena45sidiheo5r1c4l4h4eras5icc2c1itu4orsh3ernshor4h3eryci3phshon34cipecion45cinoc1ingc4inahi5anhi4cohigh5h4il2shiv5h4ina3ship3cilihir4lhi3rohir4phir4rsh3iohis4ssh1inci4lau5pia4h1l4hlan44cier5shevcia5rhmet4ch4tish1erh5ods3cho2hoge4chi2z3chitho4mahome3hon4aho5ny3hoodhoon45chiouptu44ura_ho5ruhos4esew4ihos1p1housu4ragses5tu4rasur4behree5se5shs1e4s4h1s24chedh4tarht1enht5esur4fru3rifser4os4erlhun4tsen5gur1inu3riosen4dhy3pehy3phu1ritces5tur3iz4cesa4sencur4no4iancian3i4semeia5peiass45selv5selfi4atu3centse1le4ceniib5iaib3inseg3ruros43cencib3li3cell5cel_s5edli5bun4icam5icap4icar4s4ed3secticas5i4cayiccu44iceour4pe4ced_i5cidsea5wi2cipseas4i4clyur4pi4i1cr5icrai4cryic4teictu2ccon4urti4ic4umic5uoi3curcci4ai4daiccha5ca4thscof4ide4s4casys4cliscle5i5dieid3ios4choid1itid5iui3dlei4domid3owu5sadu5sanid5uous4apied4ecany4ield3s4cesien4ei5enn4sceii1er_i3esci1estus3ciuse5as4cedscav5if4frsca4pi3fieu5siau3siccan4eiga5bcan5d4calous5sli3gibig3ilig3inig3iti4g4lus1trig3orig5oti5greigu5iig1ur2c5ah4i5i44cag4cach4ca1blusur4sat3usa5tab5utoi3legil1erilev4uta4b4butail3iail2ibil3io3sanc2ilitil2izsal4t5bustil3oqil4tyil5uru3tati4magsa5losal4m4ute_4imetbu3res3act5sack2s1ab4imitim4nii3mon4utelbumi4bu3libu4ga4inav4utenbsor42b5s2u4tis4briti3neervi4vr3vic4inga4inger3vey4ingir3ven4ingo4inguu4t1li5ni_i4niain3ioin1isbo4tor5uscrunk5both5b5ota5bos42i1no5boriino4si4not5borein3seru3in2int_ru4glbor5di5nusut5of5bor_uto5gioge4io2grbon4au5tonru3enu4touion3iio5phior3ibod3iio5thi5otiio4toi4ourbne5gb3lisrt4shblen4ip4icr3triip3uli3quar4tivr3tigrti4db4le_b5itzira4bi4racird5ert5ibi4refbi3tri4resir5gibi5ourte5oir4isr3tebr4tagbin4diro4gvac3uir5ul2b3ifis5agis3arisas52is1cis3chbi4eris3erbi5enrson3be5yor5shais3ibisi4di5sisbe3tw4is4krs3es4ismsbe5trr3secva4geis2piis4py4is1sbe3sp4bes4be5nuval5ois1teis1tirrys4rros44be5mis5us4ita_rron4i4tagrri4vi3tani3tatbe3lorri4or4reoit4esbe1libe5gu4itiarre4frre4cbe3giit3igbe3dii2tim2itio4itisrp4h4r3pet4itonr4peait5rybe3debe3dai5tudit3ul4itz_4be2dbeat3beak4ro4varo4tyros4sro5roiv5ioiv1itror3i5root1roomval1ub3berva5mo4izarva5piron4eban3ijac4qban4ebal1ajer5srom4prom4iba4geazz5i5judgay5alax4idax4ickais4aw4ly3awaya1vorav5ocav3igke5liv3el_ve4lov4elyro1feke4tyv4erdv4e2sa5vanav3ag5k2ick4illkilo5au1thk4in_4ves_ro3crkin4gve4teaun5dk5ishau4l2au3gu4kleyaugh3ve4tyk5nes1k2noat3ulkosh4at5uekro5n4k1s2at5uaat4that5te5vianat4sk5vidil4abolaci4l4adela3dylag4nlam3o3landrob3la4tosr4noular4glar3ilas4ea4topr3nivr3nita2tomr5nica4toglbin44l1c2vi5gnat3ifat1ica5tiar3neyr5net4ati_ld5isat4hol4driv2incle4bileft55leg_5leggr4nerr3nel4len_3lencr4nar1lentle3phle4prvin5dler4e3lergr3mitl4eroat5evr4mio5lesq3lessr3menl3eva4vingrma5cvio3lvi1ou4leyevi5rovi3so4l1g4vi3sulgar3l4gesate5cat5apli4agli2amr3lo4li4asr4lisli5bir4ligr2led4lics4vitil4icul3icyl3idaat5ac3lidirk4lel4iffli4flr3ket3lighvit3r4vityriv3iri2tulim3ili4moris4pl4inar3ishris4clin3ir4is_li5og4l4iqlis4pas1trl2it_as4shas5phri2pla4socask3ia3sicl3kallka4ta3sibl4lawashi4l5leal3lecl3legl3lel5riphas4abar2shrin4grin4ear4sarin4dr2inal5lowarre4l5met3rimol4modlmon42l1n2a3roorim5ilo4civo4la5rigil5ogo3loguri5et5longlon4iri1erlood5r4icolop3il3opmlora44ricir4icerib3a5los_v5oleri4agria4blos4tlo4taar2mi2loutar2izar3iolpa5bl3phal5phi4rhall3pit5voltar4im3volv2l1s2vom5ivori4l4siear4fllt5agar4fivo4rylten4vo4talth3ia3reeltis4ar4drw5ablrgo4naraw4lu3brluch4lu3cilu3enwag5olu5idlu4ma5lumia5raur5gitwait5luo3rw5al_luss4r5gisar4atl5venrgi4nara3pwar4tar3alwas4tly5mely3no2lys4l5ysewa1teaque5ma2car3gicma4clr3get5magnwed4nmaid54maldrg3erweet3wee5vwel4lapoc5re4whwest3ap3in4aphires2tr4es_mar3vre5rumas4emas1t5matemath3rero4r4eriap5atr1er4m5bilre1pumbi4vapar4a5nuran3ul4med_an3uare5lure1lian4twre5itmel4tan2trre4fy4antomen4are3fire2fe4menemen4imens4re1de3ment2r2edme5onre4awwin4g5reavme4tare3anme1tere1alm4etr3wiserdin4rdi4aan4stwith3an2span4snan2samid4amid4gan5otwl4esr4dalm4illmin4a3mindrcum3rc4itr3charcen4min4tm4inumiot4wl3ina3niumis5lan3ita3nip4mithan3ioan1gla3neuws4per2bina3nena5neem4ninw5s4tan1dl4mocrrbi4fmo2d1mo4gomois2xac5ex4agor4bagmo3mer4baba3narrau4ta5monrare4rar5cra5nor4aniam1inr2amiam5ifra4lomo3spmoth3m5ouf3mousam3icxer4ixe5roraf4tr5aclm3petra3bixhil5mpi4aam3ag3quetm5pirmp5is3quer2que_qua5vmpov5mp4tram5ab3alyz4m1s25alyt4alysa4ly_ali4exi5di5multx4ime4aldia4laral3adal5abak1enain5opu3trn4abu4nac_na4can5act5putexpe3dna4lia4i4n4naltai5lya3ic_pur4rag5ulnank4nar3c4narenar3inar4ln5arm3agognas4c4ag4l4ageupul3cage4oaga4na4gab3nautnav4e4n1b4ncar5ad5umn3chaa3ducptu4rpti3mnc1innc4itad4suad3owad4len4dain5dana5diua3ditndi4ba3dion1ditn3dizn5ducndu4rnd2we3yar4n3eara3dianeb3uac4um5neckac3ulp4siba3cio5negene4laac1inne5mine4moa3cie4nene4a2cine4poyc5erac1er2p1s2pro1tn2erepro3lner4rych4e2nes_4nesp2nest4neswpri4sycom4n5evea4carab3uln4gabn3gelpre3vpre3rycot4ng5han3gibng1inn5gitn4glangov4ng5shabi5an4gumy4erf4n1h4a5bannhab3a5bal3n4iani3anni4apni3bani4bl_us5ani5dini4erni2fip3petn5igr_ure3_un3up3per_un5op3pennin4g_un5k5nis_p5pel_un1en4ithp4ped_un1ani3tr_to4pympa3_til4n3ketnk3inyn5ic_se2ny4o5gy4onsnmet44n1n2_ru4d5pounnni4vnob4lpo4tan5ocly4ped_ro4qyper5noge4pos1s_ri4gpo4ry1p4or_res2no4mono3my_ree2po4ninon5ipoin2y4poc5po4gpo5em5pod_4noscnos4enos5tno5tayp2ta3noun_ra4cnowl3_pi2tyra5m_pi4eyr5ia_out3_oth32n1s2ns5ab_or3t_or1d_or3cplu4mnsid1nsig4y3s2eys3ion4socns4pen5spiploi4_odd5nta4bpli4n_ni4cn5tib4plignti2fpli3a3plannti4p1p2l23ysis2p3k2ys3ta_mis1nu5enpi2tun3uinp3ithysur4nu1men5umi3nu4nyt3icnu3trz5a2b_li4t_li3o_li2n_li4g_lev1_lep5_len4pion4oard3oas4e3pi1ooat5ip4inoo5barobe4l_la4mo2binpind4_ju3rob3ul_is4i_ir5rp4in_ocif3o4cil_in3so4codpi3lopi3enocre33piec5pidipi3dep5ida_in2kod3icodi3oo2do4odor3pi4cypian4_ine2o5engze3rooe4ta_im3m_id4l_hov5_hi3b_het3_hes3_go4r_gi4bpho4ro5geoo4gero3gie3phobog3it_gi5azo5ol3phizo4groogu5i4z1z22ogyn_fes3ohab5_eye55phieph1icoiff4_en3sph4ero3ing_en3go5ism_to2qans3v_el5d_eer4bbi4to3kenok5iebio5mo4lanper1v4chs_old1eol3erpe5ruo3letol4fi_du4co3liaper3op4ernp4erio5lilpe5ono5liop4encpe4la_do4tpee4do5livcin2q3pediolo4rol5pld3tabol3ub3pedeol3uno5lusedg1le1loaom5ahoma5l2p2edom2beom4bl_de3o3fich3pe4ao4met_co4ro3mia_co3ek3shao5midom1inll1fll3teapa2teo4monom3pi3pare_ca4tlue1pon4aco3nanm2an_pa4pum2en_on5doo3nenng1hoon4guon1ico3nioon1iso5niupa3nypan4ao3nou_bri2pain4ra1oronsu4rk1hopac4tpa4ceon5umonva5_ber4ood5eood5i6rks_oop3io3ordoost5rz1scope5dop1erpa4ca_ba4g_awn4_av4i_au1down5io3pito5pon1sync_as1s_as1p_as3ctch1c_ar5so5ra_ow3elo3visov4enore5auea1mor3eioun2d_ant4orew4or4guou5etou3blo5rilor1ino1rio_ang4o3riuor2miorn2eo5rofoto5sor5pe3orrhor4seo3tisorst4o3tif_an5cor4tyo5rum_al3tos3al_af1tos4ceo4teso4tano5scros2taos4poos4paz2z3wosi4ue3pai",6:"os3ityos3itoz3ian_os4i4ey1stroos5tilos5titxquis3_am5atot3er_ot5erso3scopor3thyweek1noth3i4ot3ic_ot5icao3ticeor3thiors5enor3ougor3ityor3icaouch5i4o5ria_ani5mv1ativore5sho5realus2er__an3teover3sov4erttot3icoviti4o5v4olow3dero4r3agow5esto4posiop3ingo5phero5phanthy3sc3operaontif5on3t4ionten45paganp3agattele2gonspi4on3omyon4odipan3elpan4tyon3keyon5est3oncil_ar4tyswimm6par5diompro5par5elp4a4ripar4isomo4gepa5terst5scrpa5thy_atom5sta1tio5miniom3icaom3ic_ss3hatsky1scpear4lom3ena_ba5naol3umer1veilpedia4ped4icolli4er1treuo5liteol3ishpeli4epe4nano5lis_pen4thol3ingp4era_r1thoup4erago3li4f_bas4er1krauperme5ol5id_o3liceper3tio3lescolass4oi3terpe5tenpe5tiz_be5raoi5son_be3smphar5iphe3nooi5letph4es_oi3deroic3esph5ingr3ial_3ognizo5g2ly1o1gis3phone5phonio5geneo4gatora3mour2amenofit4tof5itera3chupi4ciepoly1eod5dedo5cureoc3ula1pole_5ocritpee2v1param4oc3raco4clamo3chetob5ingob3a3boast5eoke1st3nu3itpi5thanuf4fentu3meoerst2o3chasplas5tn3tinepli5ernti4ernter3sntre1pn4s3esplum4bnsati4npre4cns4moonon1eqnor5abpo3et5n5lessn5oniz5pointpoly5tnon4agnk3rup3nomicng1sprno5l4inois5i4n3o2dno3blenni3aln5keroppa5ran3itor3nitionis4ta5nine_ni3miznd3thrmu2dron3geripray4e5precipre5copre3emm3ma1bpre4lan5gerep3rese3press_can5cmedi2c5pri4e_ce4la3neticpris3op3rocal3chain4er5ipros3en4erarnera5bnel5iz_cit5rne4gatn5d2ifpt5a4bjanu3aign4itn3chisn5chiln5cheon4ces_nau3seid4iosna3talnas5tinan4itnanci4na5mitna5liahnau3zput3er2n1a2bhex2a3hatch1multi3hair1sm4pousg1utanmpo3rim4p1inmp5iesmphas4rach4empar5iraf5figriev1mpara5mo5seyram3et4mora_rane5oran4gemo3ny_monol4rap3er3raphymo3nizgno5morar5ef4raril1g2nacg1leadmoni3ara5vairav3elra5ziemon5gemon5etght1wemoi5sege3o1dmma5ryr5bine3fluoren1dixmis4ti_de3ra_de3rie3chasrch4err4ci4bm4inglm5ineedu2al_3miliame3tryrdi4er_des4crd3ingdi2rerme5thimet3alre5arr3mestim5ersadi2rende2ticdes3icre4cremen4temensu5re3disred5itre4facmen4dede2mosmen5acmem1o3reg3ismel5onm5e5dyme3died2d5ibren4te5mediare5pindd5a5bdata1bmba4t5cle4arma3tisma5scemar4lyre4spichs3huma5riz_dumb5re3strre4terbrus4qre3tribio1rhre5utiman3izre4valrev3elbi1orbbe2vie_eas3ire5vilba1thyman5is5maniamal4tymal4lima5linma3ligmag5inav3ioul5vet4rg3inglus3teanti1dl5umn_ltur3a_el3emltera4ltane5lp5ingloun5dans5gra2cabllos5etlor5ouric5aslo5rie_enam35ricidri4cie5lope_rid5erri3encri3ent_semi5lom3errig5an3logicril3iz5rimanlob5allm3ingrim4pell5out5rina__er4ril5linal2lin4l3le4tl3le4nriph5eliv3er_ge5og_han5k_hi3er_hon3olin3ea1l4inel4im4p_idol3_in3ci_la4cy_lath5rit3iclim4blrit5urriv5elriv3et4l4i4lli4gra_leg5elif3errk4linlid5er4lict_li4cor5licioli4atorl5ish_lig5a_mal5o_man5a_mer3c5less_rm5ersrm3ingy3thinle5sco3l4erilera5b5lene__mon3ele4matld4erild4erela4v4ar1nis44lativ_mo3rola5tanlan4telan5etlan4dllab3ic_mu5takin4dek3est_ro5filk3en4dro5ker5role__of5te4jestyys3icaron4al5izont_os4tlron4tai4v3ot_pe5tero3pelrop3ici5voreiv5il__pio5n_pre3mro4the_ran4tiv3en_rov5eliv3ellit3uati4tramr5pentrp5er__rit5ui4tismrp3ingit5ill_ros5tit3ica4i2tici5terirre4stit3era4ita5mita4bi_row5dist4lyis4ta_is4sesrsa5tiissen4is4sal_sci3erse4crrs5er_islan4rse5v2yo5netish5opis3honr4si4bis5han5iron_ir4minrtach4_self5iri3turten4diri5dei4rel4ire4de_sell5r4tieriq3uidrtil3irtil4lr4tilyr4tistiq5uefip4re4_sing4_ting4yn3chrru3e4lion3at2in4th_tin5krum3pli4no4cin3ityrun4ty_ton4aruti5nymbol5rvel4i_top5irv5er_r5vestin5geni5ness_tou5s_un3cein3cerincel45ryngei4n3auim3ulai5miniimi5lesac3riim5ida_ve5rasalar4ima5ryim3ageill5abil4istsan4deila5rai2l5am_wil5ii4ladeil3a4bsa5voright3iig3eraab5erd4ific_iff5enif5eroi3entiien5a45ie5gaidi5ou3s4cieab5latidi4arid5ianide3al4scopyab5rogid5ancic3ulaac5ardi2c5ocic3ipaic5inase2c3oi4carai4car_se4d4ei2b5riib5iteib5it_ib5ertib3eraac5aroi4ativ4ian4tse4molsen5ata5ceouh4warts5enedhus3t4s5enin4sentd4sentlsep3a34s1er_hun5kehu4min4servohro3poa5chethov5el5se5umhouse3sev3enho5senhort3eho5rishor5at3hol4ehol5arh5odizhlo3riac5robhis3elhion4ehimer4het4edsh5oldhe2s5ph5eroushort5here5aher4bahera3p3side_5sideshen5atsi5diz4signahel4lyact5ifhe3l4ihe5do55sine_h5ecathe4canad4dinsion5aad5er_har4lehard3e3sitioha5rasha3ranhan4tead3icahang5oadi4ersk5inesk5ing5hand_han4cyhan4cislith5hala3mh3ab4lsmall32g5y3n5gui5t3guard5smithad5ranaeri4eag5ellag3onia5guerso4labsol3d2so3licain5in4grada3s4on_gor5ougo5rizgondo5xpan4dait5ens5ophyal3end3g4o4ggnet4tglad5i5g4insgin5ge3g4in_spen4d2s5peog3imen5gies_3spher5giciagh5outsp5ingge5nizge4natge5lizge5lisgel4inxi5miz4gativgar5n4a5le5oga3nizgan5isga5mets5sengs4ses_fu4minfres5cfort5assi4erss5ilyfore5tfor5ayfo5ratal4ia_fon4dessur5aflo3ref5lessfis4tif1in3gstam4i5stands4ta4p5stat_fin2d5al5levs5tero4allicstew5afight5fi5del5ficie5ficiafi3cer5stickf3icena5log_st3ingf3icanama5ra5stockstom3a5stone2f3ic_3storef2f5iss4tradam5ascs4trays4tridf5fin_fend5efeath3fault5fa3thefar5thfam5is4fa4mafall5eew3inge5verbeven4ie5vengevel3oev3ellev5asteva2p5euti5let5roset3roget5rifsy5rinet3ricet5onaam5eraam5ilyami4noamor5ieti4noe5tidetai5loethod3eten4dtal5enes5urramp5enan3ageta5loge5strotan4detanta3ta5pere3ston4es2toes5times3tigta3rizestan43analy4taticta4tures4prean3arces3pertax4ises5onaes3olue5skintch5etanar4ies4i4ntead4ie2s5ima3natiande4sesh5enan3disan4dowang5iete5geres5ences5ecres5cana4n1icte2ma2tem3at3tenanwrita45erwau4tenesert3era3nieser3set5erniz4erniter4nis5ter3de4rivaan3i3fter3isan4imewo5vener3ineeri4ere3rient3ess_teth5e5ericke1ria4er3ester5esser3ent4erenea5nimier5enaer3emoth3easthe5atthe3iser5el_th5ic_th5icaere3in5thinkere5coth5odea5ninee3realan3ishan4klier4che5anniz4erandti4atoanoth5equi3lep5utat4ic1uan4scoe4probep3rehe4predans3poe4precan4surantal4e3penttim5ulep5anceo5rol3tine_eop3aran4tiewin4deap5eroen3ishen5icsen3etren5esten5esien5eroa3pheren3dicap3itae4nanten5amoem5ulaa3pituti3zen5emnize5missem5ishap5olaem5ine3tles_t5let_em1in2apor5iem3icaem5anael3op_el4labapos3te3liv3el5ishaps5esweath3e3lierel3icaar3actwa5verto3nate3libee4l1erel3egato3rietor5iza5radeelaxa4aran4gto3warelan4dej5udie5insttra5chtraci4ar5av4wa5gere5git5arbal4ar5easeg5ing4voteetrem5iar3enta5ressar5ial4tricsvor5abe3finetro5mitron5i4tronyar3iantro3sp5eficia3rieted5uloed3icae4d1erec3ulaec4tane4cremeco5roec3orae4concar5o5de4comme4cluse4clame5citeec5ifya5ronias3anta5sia_tu4nis2t3up_ecan5ce4belstur3ise4bel_eav3ene4a3tue5atifeath3ieat5eneart3eear4ilear4icear5eseam3ereal3oueal5erea5geread5iedum4be4ducts4duct_duc5eras3tenasur5adrea5rat3abl4d5outdo3natdom5izdo5lor4dlessu4bero3dles_at3alou3ble_d4is3tdirt5idi5niz3dine_at5ech5di3endi4cam1d4i3ad3ge4tud5estdev3ilde3strud3iedud3iesdes3tide2s5oat3egovis3itde4nardemor5at3en_uen4teuer4ilde5milat3eraugh3en3demicater5nuil5izdeli4ede5comde4cildecan4de4bonv3io4rdeb5it4dativ2d3a4bat3estu5laticu4tie5ulcheul3dercuss4icu5riaath5em3cultua5thenul3ingul5ishul4lar4vi4naul4liscu5ityctim3ic4ticuuls5esc5tantultra3ct5angcros4ecrop5ocro4pl5critiath5omum4blycre3at5vilitumor5oat5i5b5crat_cras5tcoro3ncop3iccom5ercol3orun5ishco3inc5clareat3ituunt3abat5ropun4tescit3iz4cisti4cista4cipicc5ing_cin3em3cinatuper5s5videsup3ingci2a5b5chini5videdupt5ib5vide_at4tag4ch1inch3ersch3er_ch5ene3chemiche5loure5atur4fercheap3vi5aliat3uravet3er4ch3abc5e4taau5sib3cessives4tece5ram2cen4e4cedenccou3turs5erur5tesur3theaut5enur4tiecav5al4cativave4nover3thcar5omca5percan4tycan3izcan5iscan4icus4lin3versecal4laver3ieca3latca5dencab3in3butiobuss4ebus5iebunt4iv4eresuten4i4u1t2iv3erenu3tineut3ingv4erelbroth35u5tizbound34b1orabon5at5vere_bom4bibol3icblun4t5blespblath5av3erav5enuebi3ogrbi5netven3om2v1a4bvac5ilbi3lizbet5izbe5strva5liebe5nigbbi4nabas4siva5nizbari4aav5ernbarbi5av5eryvel3liazi4eravi4er",7:"_dri5v4ban5dagvar5iedbina5r43bi3tio3bit5ua_ad4derution5auti5lizver5encbuf4ferus5terevermi4ncall5incast5ercas5tigccompa5z3o1phros5itiv5chanicuri4fico5stati5chine_y5che3dupport54v3iden5cific_un4ter_at5omiz4oscopiotele4g5craticu4m3ingv3i3liz4c3retaul4li4bcul4tiscur5a4b4c5utiva5ternauiv4er_del5i5qdem5ic_de4monsdenti5fdern5izdi4latou4b5ingdrag5on5drupliuar5ant5a5si4tec5essawo4k1enec5ifiee4compear5inate4f3eretro5phewide5sp5triciatri5cesefor5ese4fuse_oth5esiar5dinear4chantra5ventrac4tetrac4itar5ativa5ratioel5ativor5est_ar5adisel5ebraton4alie4l5ic_wea5rieel5igibe4l3ingto5cratem5igraem3i3niemoni5oench4erwave1g4a4pillavoice1ption5eewill5inent5age4enthesvaude3vtill5inep5recaep5ti5bva6guer4erati_tho5rizthor5it5thodicer5ence5ternitteri5zater5iesten4tage4sage_e4sagese4sert_an5est_e4sertse4servaes5idenes5ignaesis4tees5piraes4si4btal4lisestruc5e5titioounc5erxe4cutota5bleset5itiva4m5atoa4matis5stratu4f3ical5a5lyst4ficatefill5instern5isspend4gani5zasqual4la4lenti4g3o3nas5ophiz5sophicxpecto55graph_or5angeuri4al_4graphy4gress_smol5d4hang5erh5a5nizharp5enhar5terhel4lishith5erhro5niziam5eteia4tricic4t3uascour5au2r1al_5scin4dover4nescan4t55sa3tiou5do3ny_ven4de_under5ty2p5al_anti5sylla5bliner4arturn3ari5nite_5initioinsur5aion4eryiphras4_tim5o5_ten5an_sta5blrtroph4_se5rieiq3ui3t5i5r2izis5itiviso5mer4istral5i5ticki2t5o5mtsch3ie_re5mittro3fiti4v3er_i4vers_ros5per_pe5titiv3o3ro_ped5alro5n4is_or5ato4jestierom5ete_muta5bk5iness4latelitr4ial__mist5i_me5terr4ming_lev4er__mar5tilev4eralev4ers_mag5a5liar5iz5ligaterit5ers_lat5errit5er_r5ited__im5pinri3ta3blink5er_hon5ey5litica_hero5ior5aliz_hand5irip5lic_gen3t4tolo2gylloqui5_con5grt1li2erbad5ger4operag_eu4lertho3donter2ic__ar4tie_ge4ome_ge5ot1_he3mo1_he3p6a_he3roe_in5u2tpara5bl5tar2rht1a1mintalk1a5ta3gon_par5age_aster5_ne6o3f_noe1thstyl1is_poly1s5pathic_pre1ampa4tricl3o3niz_sem4ic_semid6_semip4_semir45ommend_semiv4lea4s1a_spin1oom5etryspher1o_to6poglo4ratospe3cio3s2paceso2lute_we2b1l_re1e4ca5bolicom5erseaf6fishside5swanal6ysano5a2cside5stl5ties_5lumniasid2ed_anti1reshoe1stscy4th1s4chitzsales5wsales3cat6tes_augh4tlau5li5fom5atizol5ogizo5litiorev5olure5vertre5versbi5d2ifbil2lab_earth5pera5blro1tronro3meshblan2d1blin2d1blon2d2bor1no5ro1bot1re4ti4zr5le5quperi5stper4malbut2ed_but4tedcad5e1moist5enre5stalress5ibchie5vocig3a3roint5er4matizariv1o1lcous2ticri3tie5phisti_be5stoog5ativo2g5a5rr3a3digm4b3ingre4posir4en4tade4als_od5uctsquasis6quasir6re5fer_p5trol3rec5olldic1aiddif5fra3pseu2dr5ebrat5metric2d1lead2d1li2epro2g1epre1neuod5uct_octor5apoin3came5triem5i5liepli5narpara3memin5glim5inglypi4grappal6matmis4er_m5istryeo3graporth1riop1ism__but4tio3normaonom1icfeb1ruafermi1o_de4moio5a5lesodit1icodel3lirb5ing_gen2cy_n4t3ingmo5lestration4get2ic_4g1lishobli2g1mon4ismnsta5blmon4istg2n1or_nov3el3ns5ceivno1vembmpa5rabno4rarymula5r4nom1a6lput4tinput4tedn5o5miz_cam4penag5er_nge5nesh2t1eoun1dieck2ne1skiifac1etncour5ane3backmono1s6mono3chmol1e5cpref5ac3militapre5tenith5i2lnge4n4end5est__capa5bje1re1mma1la1ply5styr1kovian_car5olprin4t3lo2ges_l2l3ishprof5it1s2tamp",8:"lead6er_url5ing_ces5si5bch5a5nis1le1noidlith1o5g_chill5ilar5ce1nym5e5trych5inessation5arload4ed_load6er_la4c3i5elth5i2lyneg5ativ1lunk3erwrit6er_wrap3arotrav5es51ke6linga5rameteman3u1scmar1gin1ap5illar5tisticamedio6c1me3gran3i1tesima3mi3da5bves1titemil2l1agv1er1eigmi6n3is_1verely_e4q3ui3s5tabolizg5rapher5graphicmo5e2lasinfra1s2mon4ey1lim3ped3amo4no1enab5o5liz_cor5nermoth4et2m1ou3sinm5shack2ppo5sitemul2ti5uab5it5abimenta5rignit1ernato5mizhypo1thani5ficatuad1ratu4n5i4an_ho6r1ic_ua3drati5nologishite3sidin5dling_trib5utin5glingnom5e1non1o1mistmpos5itenon1i4so_re5stattro1p2istrof4ic_g2norespgnet1ism5glo5binlem5aticflow2er_fla1g6elntrol5lifit5ted_treach1etra1versl5i5ticso3mecha6_for5mer_de5rivati2n3o1me3spac6i2t3i4an_thy4l1antho1k2er_eq5ui5to4s3phertha4l1amt3ess2es3ter1geiou3ba3dotele1r6ooxi6d1iceli2t1isonspir5apar4a1leed1ulingea4n3iesoc5ratiztch3i1er_kil2n3ipi2c1a3dpli2c1abt6ap6athdrom3e5d_le6icesdrif2t1a_me4ga1l1prema3cdren1a5lpres2plipro2cess_met4ala3do5word1syth3i2_non1e2m_post1ampto3mat4rec5ompepu5bes5cstrib5utqu6a3si31stor1ab_sem6is4star3tliqui3v4arr1abolic_sph6in1de5clar12d3aloneradi1o6gs3qui3tosports3wsports3cra5n2hascro5e2cor3bin1gespokes5wspi2c1il_te3legrcroc1o1d_un3at5t_dictio5cat1a1s2buss4ingbus6i2esbus6i2erbo2t1u1lro5e2las1s2pacinb1i3tivema5rine_r3pau5li_un5err5r5ev5er__vi2c3arback2er_ma5chinesi5resid5losophyan3ti1n2sca6p1ersca2t1olar2rangesep3temb1sci2uttse3mes1tar3che5tsem1a1ph",9:"re4t1ribuuto5maticl3chil6d1a4pe5able1lec3ta6bas5ymptotyes5ter1yl5mo3nell5losophizlo1bot1o1c5laratioba6r1onierse1rad1iro5epide1co6ph1o3nscrap4er_rec5t6angre2c3i1prlai6n3ess1lum5bia_3lyg1a1miec5ificatef5i5nites2s3i4an_1ki5neticjapan1e2smed3i3cinirre6v3ocde2c5linao3les3termil5li5listrat1a1gquain2t1eep5etitiostu1pi4d1v1oir5du1su2per1e6_mi1s4ers3di1methy_mim5i2c1i5nitely_5maph1ro15moc1ra1tmoro6n5isdu1op1o1l_ko6r1te1n3ar4chs_phi2l3ant_ga4s1om1teach4er_parag6ra4o6v3i4an_oth3e1o1sn3ch2es1to5tes3toro5test1eror5tively5nop5o5liha2p3ar5rttrib1ut1_eth1y6l1e2r3i4an_5nop1oly_graph5er_5eu2clid1o1lo3n4omtrai3tor1_ratio5na5mocratiz_rav5en1o",10:"se1mi6t5ic3tro1le1um5sa3par5iloli3gop1o1am1en3ta5bath3er1o1s3slova1kia3s2og1a1myo3no2t1o3nc2tro3me6c1cu2r1ance5noc3er1osth1o5gen1ih3i5pel1a4nfi6n3ites_ever5si5bs2s1a3chu1d1ri3pleg5_ta5pes1trproc3i3ty_s5sign5a3b3rab1o1loiitin5er5arwaste3w6a2mi1n2ut1erde3fin3itiquin5tes5svi1vip3a3r",11:"pseu3d6o3f2s2t1ant5shimi1n2ut1estpseu3d6o3d25tab1o1lismpo3lyph1onophi5lat1e3ltravers3a3bschro1ding12g1o4n3i1zat1ro1pol3it3trop1o5lis3trop1o5lesle3g6en2dreeth1y6l1eneor4tho3ni4t",12:"3ra4m5e1triz1e6p3i3neph1"},patternChars:"_abcdefghijklmnopqrstuvwxyz",patternArrayLength:113949,valueStoreLength:20195};Hyphenator.languages['fr']={leftmin:3,rightmin:3,specialChars:"àâçèéêëîïôûüœæ’'",patterns:{2:"1ç1j1q",3:"1gè’â41zu1zo1zi1zè1zé1ze1za’y4_y41wu1wo1wi1we1wa1vy1vû1vu1vô1vo1vî1vi1vê1vè1vé1ve1vâ1va’û4_û4’u4_u41ba1bâ1ty1be1bé1bè1bê1tû1tu1tô1bi1bî1to1tî1ti1tê1tè1té1te1tà1tâ1ta1bo1bô1sy1sû1su1sœ1bu1bû1by2’21ca1câ1sô1ce1cé1cè1cê1so1sî1si1sê1sè1sé1se1sâ1sa1ry1rû1ru1rô1ro1rî1ri1rê1rè1ré1re1râ1ra’a41py1pû1pu1pô1po1pî1pi1pê1pè1pé1pe1pâ1pa_ô41ci1cî’ô4’o4_o41nyn1x1nû1nu1nœ1nô1no1nî1ni1nê1nè1né1ne1nâ1co1cô1na1my1mû1mu1mœ1mô1mo1mî1mi1cœ1mê1mè1mé1me1mâ1ma1ly1lû1lu1lô1lo1lî1li1lê1lè1cu1cû1cy1lé1d’1da1dâ1le1là1de1dé1dè1dê1lâ1la1ky1kû1ku1kô1ko1kî1ki1kê1kè1ké1ke1kâ1ka2jk_a4’î4_î4’i4_i41hy1hû1hu1hô1ho1hî1hi1hê1hè1hé1he1hâ1ha1gy1gû1gu1gô1go1gî1gi1gê_â41gé1ge1gâ1ga1fy1di1dî1fû1fu1fô1fo’e41fî1fi1fê1fè1do1dô1fé1fe1fâ1fa’è41du1dû1dy_è4’é4_é4’ê4_ê4_e41zy",4:"1f2lab2h2ckg2ckp2cksd1s22ckb4ck_1c2k2chw4ze_4ne_2ckt1c2lad2hm1s22cht2chsch2r2chp4pe_1t2r1p2h_ph44ph_ph2l2phnph2r2phs1d2r2pht2chn4fe_2chm1p2l1p2r4me_1w2rch2l2chg1c2r2chb4ch_1f2r4le_4re_4de_f1s21k2r4we_1r2h_kh44kh_1k2h4ke_1c2h_ch44ge_4je_4se_1v2r_sh41s2h4ve_4sh_2shm2shr2shs4ce_il2l1b2r4be_1b2l4he_4te__th41t2h4th_g1s21g2r2thl1g2l2thm2thnth2r1g2n2ths2ckf",5:"2ck3h4rhe_4kes_4wes_4res_4cke_éd2hi4vre_4jes_4tre_4zes_4ges_4des_i1oxy4gle_d1d2h_cul44gne_4fre_o1d2l_sch44nes_4les_4gre_1s2ch_réu24sch_4the_1g2hy4gue_2schs4cle_1g2ho1g2hi1g2he4ses_4tes_1g2ha4ves_4she_4che_4cre_4ces_t1t2l4hes_l1s2t4bes_4ble__con4xil3lco1ap4que_vil3l4fle_co1arco1exco1enco1auco1axco1ef4pes_co1é2per3h4mes__pe4r4bre_4pre_4phe_1p2né4ple__dé2smil3llil3lhil3l4dre_cil3lgil3l4fes_",6:"’in1o2rcil4l4phre_4dres_l3lioni1algi2fent_émil4l4phle_rmil4l4ples_4phes_1p2neuextra14pres_y1asthpé2nul2xent__mé2sa2pent_y1algi4chre_1m2nès4bres_1p2tèr1p2tér4chle_’en1o24fles_oxy1a2avil4l_en1o24ques_uvil4lco1a2d4bles__in1a2’in1a21s2por_cons4_bi1u2’as2ta_in1e2’in1e2_in1é2’in1é21s2lov1s2lavco1acq2cent__as2ta_co1o24ches_hémi1é_in2er’in2er2s3homo1ioni_in1i2’in1i22went_4shes__ré1a2_ré1é2_ré1e2_ré2el_in1o2ucil4lco1accu2s3tr_ré2er_ré2èr4cles_2vent__ré1i22sent_2tent_2gent__ré1o24gues__re1s24sche_4thes_’en1a2e2s3ch4gres_1s2cop2lent__en1a22nent__in1u2’in1u24gnes_4cres_wa2g3n4fres_4tres_4gles_1octet_dé1o2_dé1io4thre__bi1au2jent__dé1a22zent_4vres_2dent_4ckes_4rhes__dy2s3sub1s22kent_2rent_2bent_3d2hal",7:"a2g3nos3d2houdé3rent__dé3s2t_dé3s2pé3dent_2r3heur2r3hydri1s2tat2frent_io1a2ctla2w3re’in2u3l_in2u3l2crent_’in2uit_in2uit1s2caph1s2clér_ré2ussi2s3ché_re2s3t_re2s3s4sches_é3cent__seu2le’in2ond_in2ond’in2i3t_in2i3t’in2i3q_ré2aux_in2i3q2shent__di1alduni1a2x’in2ept2flent__in2eptuni1o2v2brent_co2nurb2chent_2quent_1s2perm1s2phèr_ma2c3kuevil4l1s2phér1s2piel1s2tein1s2tigm4chles_1s2tock1s2tyle1p2sych_pro1é2_ma2r1x_stil3lpusil3libril3lcyril3l_pré1s2thril3l_mé3san_pré1u2_mé2s1i_pré1o2_pré1i2piril3lpupil3lâ2ment__pré1e2_pré1é2_pré2au_pré1a22prent_2vrent_supero2_di1e2npoly1u2è2ment_poly1s2poly1o2poly1i2poly1è2poly1é2poly1e2poly1a2supe4r1capil3l2plent_armil5lsemil4lmil4letvacil4l_di2s3h3ph2tis2dlent_a2s3tro4phres_l2ment_i1è2drei1arthr2drent_4phles_supers2ô2ment_extra2i2phent_su3r2ah_su2r3hextra2chypo1u21alcool_per1u2_per1o2_per1i2_per1é2hypo1s2_per1a2hypo1o2hypo1i2hypo1é2_pen2tahypo1e2hypo1a2y1s2tome2s3cophyperu2hype4r1hypers2hypero21m2némohyperi21m2nési4chres_a1è2drehyperé2hypere2hypera2’oua1ou_oua1ouo1s2tomo1s2timo1s2tato1s2tasomni1s2tung2s3_dé3s2c2blent__bio1a2télé1e2télé1i22clent_télé1s22guent_1é2nerg2grent_2trent__dé2s1œ2t3heuro1è2dre2gnent_2glent_4thres__bi1a2t1é2drie_bi1a2c_i2g3nin3s2at_’i2g3ni2ckent__i2g3né’ab3réa’i2g3né_ab3réa_per1e2",8:"_ma2l1ap_dy2s1u2_dy2s1o2_dy2s1i2n3s2ats__dy2s1a2distil3l1é2lectrinstil3l1s2trophe2n1i2vro2b3long1s2tomos_ae3s4ch’ae3s4ch_eu2r1a2ombud2s3’eu2r1a2_mono1s2_mono1u2o1s2téro_mono1o2eu1s2tato1s2tradfritil3la2l1algi_mono1i2_mono1é2_ovi1s2c’ovi1s2c_mono1e2_mono1a2co1assocpaléo1é2boutil3l1s2piros_ré2i3fi_pa2n1ischevil4l1s2patiaca3ou3t2_di1a2cé_para1s2_pa2r3héco1assur_su2b1é2tu2ment_su2ment__su2b1in_su2b3lupapil3lire3pent_’inte4r3_su2b1urab3sent__su2b1a2di2s3cophu2ment_fu2ment__intera2au2ment_as2ment_or2ment_’intera2_intere2pé1r2é2q_péri1os_péri1s2ja3cent__anti1a2_péri1u2’anti1a2er2ment__anti1e2ac3cent_ar2ment_to2ment_’intere2ré3gent_papil3leom2ment_’anti1e2photo1s2_anti1é2_interé2’anti1é2_anti1s2’anti1s23ph2talé’interé2ri2ment__interi2’interi2mi2ment_apo2s3tri2s3chio_pluri1ai2s3chia_intero2’intero2_inte4r3po1astre_interu2’interu2_inters2ai2ment_’inters2papil3la_tri1o2n_su2r1a2_pon2tet_pos2t3h_dés2a3mes3cent__pos2t3r_post1s2_tri1a2tta2ment__tri1a2nra2ment_is3cent__su2r1e2_tri1a2cfa2ment_da2ment__su3r2et_su2r1é2_mé2s1es_mé2g1oh_su2r1of_su2r1ox_re3s4ty_re3s4tu_ma2l1oc’a2g3nat_dé2s1é2_ma2l1entachy1a2_pud1d2ltchin3t2_re3s4trtran2s3p_bi2s1a2tran2s3hhémo1p2té3quent__a2g3nat_dé2s1i2télé1o2bo2g3nosiradio1a2télé1o2ppu2g3nacru3lent__sta2g3nre3lent__ré2a3le_di1a2mi",9:"_ré2a3lit_dé3s2o3lthermo1s2_dé3s2ist_dé3s2i3rmit3tent_éni3tent__do3lent__ré2a3lisopu3lent__pa3tent__re2s3cap_la3tent__co2o3lie_re2s3cou_re2s3cri_ma2g3num_re2s3pir_dé3s2i3dco2g3nititran2s1a2tran2s1o2_dé3s2exu_re3s4tab_re3s4tag_dé3s2ert_re3s4tat_re3s4tén_re3s4tér_re3s4tim_re3s4tip_re3s4toc_re3s4toptran2s1u2_no2n1obs_ma2l1a2v_ma2l1int_prou3d2hpro2s3tativa3lent__ta3lent__rétro1a2_pro1s2cé_ma2l1o2dcci3dent__pa3rent__su2r1int_su2r1inf_su2r1i2mtor3rent_cur3rent__mé2s1u2stri3dent__dé3s2orm_su3r2ell_ar3dent__su3r2eaupru3dent__pré2a3lacla2ment__su3r2a3t_pos2t1o2_pos2t1inqua2ment_ter3gent_ser3gent_rai3ment_abî2ment_éci2ment_’ar3gent__ar3gent_rin3gent_tan3gent_éli2ment_ani2ment_’apo2s3ta_apo2s3tavélo1s2kivol2t1amp_dé3s2orp_dé2s1u2n_péri2s3ssesqui1a2’ana3s4trfir2ment_écu2ment_ser3pent_pré3sent_’ar3pent__ar3pent_’in1s2tab_in1s2tab’in2o3cul_in2o3culplu2ment_bou2ment_’in2exora_in2exora_su2b3linbru2ment__su3b2é3r_milli1am’in2effab_in2effab’in2augur_di1a2cid_in2augur_pa2n1opt’in2a3nit_in2a3nit1informat_ana3s4trvanil3lis_di1a2tom_su3b2altvanil3linstéréo1s2_pa2n1a2fo1s2tratuépi2s3cop_ci2s1alp1s2tructu1é2lément1é2driquepapil3lomllu2ment_",10:"1s2tandardimmi3nent__émi3nent_imma3nent_réma3nent_épi3s4cope_in2i3miti’in2i3miti_res3sent_moye2n1â2gréti3cent__dé3s2a3crmon2t3réalinno3cent__mono1ï2dé_pa2n1a2méimpu3dent__pa2n1a2ra_amino1a2c’amino1a2c_pa2n1o2phinci3dent__ser3ment_appa3rent_déca3dent__dacryo1a2_dé3s2astr_re4s5trin_dé3s2é3gr_péri2s3ta_sar3ment__dé3s2oufr_re3s4tandchro2ment__com3ment__re2s3quil_re2s3pons_gem2ment__re2s3pect_re2s3ciso_dé3s2i3gn_dé3s2i3ligram2ment__dé3s2invo_re2s3cisitran3s2act’anti2enneindo3lent__sou3vent_indi3gent_dili3gent_flam2ment_impo3tent_inso3lent_esti2ment_’on3guent__on3guent_inti2ment__dé3s2o3défécu3lent_veni2ment_reli2ment_vidi2ment_chlo2r3é2tpu2g3nablechlo2r3a2cryth2ment_o2g3nomonicarê2ment__méta1s2ta_ma2l1aisé_macro1s2célo3quent_tran3s2ats_anti2enne",11:"_contre1s2cperti3nent_conti3nent__ma2l1a2dro_in2é3lucta_psycho1a2n_dé3s2o3pil’in2é3luctaperma3nent__in2é3narratesta3ment__su2b3liminrésur3gent_’in2é3narraimmis4cent__pro2g3nathchien3dent_sporu4lent_dissi3dent_corpu3lent_archi1é2pissubli2ment_indul3gent_confi3dent__syn2g3nathtrucu3lent_détri3ment_nutri3ment_succu3lent_turbu3lent__pa2r1a2che_pa2r1a2chèfichu3ment_entre3gent_conni3vent_mécon3tent_compé3tent__re4s5trict_dé3s2i3nen_re2s3plend1a2nesthésislalo2ment__dé3s2ensib_re4s5trein_phalan3s2tabsti3nent_",12:"polyva3lent_équiva4lent_monova3lent_amalga2ment_omnipo3tent__ma2l1a2dreséquipo3tent__dé3s2a3tellproémi3nent_contin3gent_munifi3cent__ma2g3nicideo1s2trictionsurémi3nent_préémi3nent__bai2se3main",13:"acquies4cent_intelli3gent_tempéra3ment_transpa3rent__ma2g3nificatantifer3ment_",14:"privatdo3cent_diaphrag2ment_privatdo3zent_ventripo3tent__contre3maître",15:"grandilo3quent_",16:"_chè2vre3feuille"},patternChars:"_abcdefghijklmnopqrstuvwxyzàâçèéêîïôûœ’",patternArrayLength:79410,valueStoreLength:5134};Hyphenator.config({dohyphenation:false,remoteloading:false,enablecache:false,unhide:'progressive',useCSS3hyphenation:true});Hyphenator.run();

// TODO reduce hyphenator size

// Fr ' to ’ converter
var France = France ||
{
  franchize: function(text) {
    // Apostrophes
    text = France.apostrophize(text);

    // Espaces insécables
    text = text.replace(/\s:/g, '\u00A0:');
    text = text.replace(/\s\?/g, '\u00A0\?');
    text = text.replace(/\s!/g, '\u00A0!');
    text = text.replace(/\s;/g, '\u00A0;');

    // Espaces insécables et guillemets
    text = text.replace(/\s"/g, ' «\u00A0');
    text = text.replace(/"\s/g, '\u00A0» ');
    text = text.replace(/",\s/g, '\u00A0», ');
    text = text.replace(/"\./g, '\u00A0».');
    return text;
  },

  apostrophize: function(text) {
    text = text.replace(/'/g, '’');
    return text;
  },

  guillemetize: function(text) {
    // Espaces insécables et guillemets
    text = text.replace(/\s"/g, ' “');
    text = text.replace(/"\s/g, '” ');
    text = text.replace(/",\s/g, '”, ');
    text = text.replace(/"\./g, '”.');
    return text;
  }
};

var HyphenatorProxy = HyphenatorProxy || {
  hyphenate: function(string, language)
  {
    if (string.length < 1) return;

    // Filter: don't hyphenate hrefs.
    var splits = string.split('href="');
    var outputString = '';
    outputString += Hyphenator.hyphenate(splits[0], language);
    for (var i = 1; i < splits.length; ++i) {
      outputString += 'href="';
      var splits2 = splits[i].split('">');
      if (splits2.length <= 1 || splits2.length > 2) {
        console.log('BRAIN ERROR ');
        console.log(splits2);
      }
      outputString += splits2[0] + '">';
      outputString += Hyphenator.hyphenate(splits2[1], language);
    }

    return outputString;
  }
};

// Only expose the one function we need from Hyphenator.
module.exports = { Hyphenator: { hyphenate: HyphenatorProxy.hyphenate }, France: France };


/***/ }),

/***/ "./src/app/app.module.ts":
/*!*******************************!*\
  !*** ./src/app/app.module.ts ***!
  \*******************************/
/*! exports provided: hljsLanguages, AppModule */
/***/ (function(module, __webpack_exports__, __webpack_require__) {

"use strict";
__webpack_require__.r(__webpack_exports__);
/* harmony export (binding) */ __webpack_require__.d(__webpack_exports__, "hljsLanguages", function() { return hljsLanguages; });
/* harmony export (binding) */ __webpack_require__.d(__webpack_exports__, "AppModule", function() { return AppModule; });
/* harmony import */ var tslib__WEBPACK_IMPORTED_MODULE_0__ = __webpack_require__(/*! tslib */ "./node_modules/tslib/tslib.es6.js");
/* harmony import */ var _angular_platform_browser__WEBPACK_IMPORTED_MODULE_1__ = __webpack_require__(/*! @angular/platform-browser */ "./node_modules/@angular/platform-browser/fesm5/platform-browser.js");
/* harmony import */ var _angular_core__WEBPACK_IMPORTED_MODULE_2__ = __webpack_require__(/*! @angular/core */ "./node_modules/@angular/core/fesm5/core.js");
/* harmony import */ var _angular_forms__WEBPACK_IMPORTED_MODULE_3__ = __webpack_require__(/*! @angular/forms */ "./node_modules/@angular/forms/fesm5/forms.js");
/* harmony import */ var _angular_common_http__WEBPACK_IMPORTED_MODULE_4__ = __webpack_require__(/*! @angular/common/http */ "./node_modules/@angular/common/fesm5/http.js");
/* harmony import */ var _app_component__WEBPACK_IMPORTED_MODULE_5__ = __webpack_require__(/*! ./app.component */ "./src/app/app.component.ts");
/* harmony import */ var _articles_articles_component__WEBPACK_IMPORTED_MODULE_6__ = __webpack_require__(/*! ./articles/articles.component */ "./src/app/articles/articles.component.ts");
/* harmony import */ var _article_detail_article_detail_component__WEBPACK_IMPORTED_MODULE_7__ = __webpack_require__(/*! ./article-detail/article-detail.component */ "./src/app/article-detail/article-detail.component.ts");
/* harmony import */ var _messages_messages_component__WEBPACK_IMPORTED_MODULE_8__ = __webpack_require__(/*! ./messages/messages.component */ "./src/app/messages/messages.component.ts");
/* harmony import */ var _app_routing_module__WEBPACK_IMPORTED_MODULE_9__ = __webpack_require__(/*! ./app-routing.module */ "./src/app/app-routing.module.ts");
/* harmony import */ var _dashboard_dashboard_component__WEBPACK_IMPORTED_MODULE_10__ = __webpack_require__(/*! ./dashboard/dashboard.component */ "./src/app/dashboard/dashboard.component.ts");
/* harmony import */ var _article_search_article_search_component__WEBPACK_IMPORTED_MODULE_11__ = __webpack_require__(/*! ./article-search/article-search.component */ "./src/app/article-search/article-search.component.ts");
/* harmony import */ var _ng_bootstrap_ng_bootstrap__WEBPACK_IMPORTED_MODULE_12__ = __webpack_require__(/*! @ng-bootstrap/ng-bootstrap */ "./node_modules/@ng-bootstrap/ng-bootstrap/fesm5/ng-bootstrap.js");
/* harmony import */ var ngx_highlightjs__WEBPACK_IMPORTED_MODULE_13__ = __webpack_require__(/*! ngx-highlightjs */ "./node_modules/ngx-highlightjs/fesm5/ngx-highlightjs.js");
/* harmony import */ var highlight_js_lib_languages_javascript__WEBPACK_IMPORTED_MODULE_14__ = __webpack_require__(/*! highlight.js/lib/languages/javascript */ "./node_modules/highlight.js/lib/languages/javascript.js");
/* harmony import */ var highlight_js_lib_languages_javascript__WEBPACK_IMPORTED_MODULE_14___default = /*#__PURE__*/__webpack_require__.n(highlight_js_lib_languages_javascript__WEBPACK_IMPORTED_MODULE_14__);
/* harmony import */ var highlight_js_lib_languages_typescript__WEBPACK_IMPORTED_MODULE_15__ = __webpack_require__(/*! highlight.js/lib/languages/typescript */ "./node_modules/highlight.js/lib/languages/typescript.js");
/* harmony import */ var highlight_js_lib_languages_typescript__WEBPACK_IMPORTED_MODULE_15___default = /*#__PURE__*/__webpack_require__.n(highlight_js_lib_languages_typescript__WEBPACK_IMPORTED_MODULE_15__);
/* harmony import */ var highlight_js_lib_languages_cpp__WEBPACK_IMPORTED_MODULE_16__ = __webpack_require__(/*! highlight.js/lib/languages/cpp */ "./node_modules/highlight.js/lib/languages/cpp.js");
/* harmony import */ var highlight_js_lib_languages_cpp__WEBPACK_IMPORTED_MODULE_16___default = /*#__PURE__*/__webpack_require__.n(highlight_js_lib_languages_cpp__WEBPACK_IMPORTED_MODULE_16__);
/* harmony import */ var highlight_js_lib_languages_bash__WEBPACK_IMPORTED_MODULE_17__ = __webpack_require__(/*! highlight.js/lib/languages/bash */ "./node_modules/highlight.js/lib/languages/bash.js");
/* harmony import */ var highlight_js_lib_languages_bash__WEBPACK_IMPORTED_MODULE_17___default = /*#__PURE__*/__webpack_require__.n(highlight_js_lib_languages_bash__WEBPACK_IMPORTED_MODULE_17__);
/* harmony import */ var highlight_js_lib_languages_cmake__WEBPACK_IMPORTED_MODULE_18__ = __webpack_require__(/*! highlight.js/lib/languages/cmake */ "./node_modules/highlight.js/lib/languages/cmake.js");
/* harmony import */ var highlight_js_lib_languages_cmake__WEBPACK_IMPORTED_MODULE_18___default = /*#__PURE__*/__webpack_require__.n(highlight_js_lib_languages_cmake__WEBPACK_IMPORTED_MODULE_18__);
/* harmony import */ var highlight_js_lib_languages_python__WEBPACK_IMPORTED_MODULE_19__ = __webpack_require__(/*! highlight.js/lib/languages/python */ "./node_modules/highlight.js/lib/languages/python.js");
/* harmony import */ var highlight_js_lib_languages_python__WEBPACK_IMPORTED_MODULE_19___default = /*#__PURE__*/__webpack_require__.n(highlight_js_lib_languages_python__WEBPACK_IMPORTED_MODULE_19__);
/* harmony import */ var highlight_js_lib_languages_glsl__WEBPACK_IMPORTED_MODULE_20__ = __webpack_require__(/*! highlight.js/lib/languages/glsl */ "./node_modules/highlight.js/lib/languages/glsl.js");
/* harmony import */ var highlight_js_lib_languages_glsl__WEBPACK_IMPORTED_MODULE_20___default = /*#__PURE__*/__webpack_require__.n(highlight_js_lib_languages_glsl__WEBPACK_IMPORTED_MODULE_20__);
/* harmony import */ var ng_katex__WEBPACK_IMPORTED_MODULE_21__ = __webpack_require__(/*! ng-katex */ "./node_modules/ng-katex/ng-katex.esm.js");
/* harmony import */ var _navbar_navbar_component__WEBPACK_IMPORTED_MODULE_22__ = __webpack_require__(/*! ./navbar/navbar.component */ "./src/app/navbar/navbar.component.ts");
/* harmony import */ var _fortawesome_angular_fontawesome__WEBPACK_IMPORTED_MODULE_23__ = __webpack_require__(/*! @fortawesome/angular-fontawesome */ "./node_modules/@fortawesome/angular-fontawesome/fesm5/angular-fontawesome.js");
/* harmony import */ var _footer_footer_component__WEBPACK_IMPORTED_MODULE_24__ = __webpack_require__(/*! ./footer/footer.component */ "./src/app/footer/footer.component.ts");
/* harmony import */ var _angular_flex_layout__WEBPACK_IMPORTED_MODULE_25__ = __webpack_require__(/*! @angular/flex-layout */ "./node_modules/@angular/flex-layout/esm5/flex-layout.es5.js");













// Code rendering








// Style rendering





/**
 * Import every language you wish to highlight here
 */
function hljsLanguages() {
    return [
        { name: 'javascript', func: highlight_js_lib_languages_javascript__WEBPACK_IMPORTED_MODULE_14___default.a },
        { name: 'typescript', func: highlight_js_lib_languages_typescript__WEBPACK_IMPORTED_MODULE_15___default.a },
        { name: 'cpp', func: highlight_js_lib_languages_cpp__WEBPACK_IMPORTED_MODULE_16___default.a },
        { name: 'bash', func: highlight_js_lib_languages_bash__WEBPACK_IMPORTED_MODULE_17___default.a },
        { name: 'glsl', func: highlight_js_lib_languages_glsl__WEBPACK_IMPORTED_MODULE_20___default.a },
        { name: 'cmake', func: highlight_js_lib_languages_cmake__WEBPACK_IMPORTED_MODULE_18___default.a },
        { name: 'python', func: highlight_js_lib_languages_python__WEBPACK_IMPORTED_MODULE_19___default.a }
    ];
}
var AppModule = /** @class */ (function () {
    function AppModule() {
    }
    AppModule = tslib__WEBPACK_IMPORTED_MODULE_0__["__decorate"]([
        Object(_angular_core__WEBPACK_IMPORTED_MODULE_2__["NgModule"])({
            declarations: [
                _app_component__WEBPACK_IMPORTED_MODULE_5__["AppComponent"],
                _articles_articles_component__WEBPACK_IMPORTED_MODULE_6__["ArticlesComponent"],
                _article_detail_article_detail_component__WEBPACK_IMPORTED_MODULE_7__["ArticleDetailComponent"],
                _messages_messages_component__WEBPACK_IMPORTED_MODULE_8__["MessagesComponent"],
                _dashboard_dashboard_component__WEBPACK_IMPORTED_MODULE_10__["DashboardComponent"],
                _article_search_article_search_component__WEBPACK_IMPORTED_MODULE_11__["ArticleSearchComponent"],
                _navbar_navbar_component__WEBPACK_IMPORTED_MODULE_22__["NavbarComponent"],
                _footer_footer_component__WEBPACK_IMPORTED_MODULE_24__["FooterComponent"]
            ],
            imports: [
                _angular_platform_browser__WEBPACK_IMPORTED_MODULE_1__["BrowserModule"],
                _angular_forms__WEBPACK_IMPORTED_MODULE_3__["FormsModule"],
                _app_routing_module__WEBPACK_IMPORTED_MODULE_9__["AppRoutingModule"],
                _angular_common_http__WEBPACK_IMPORTED_MODULE_4__["HttpClientModule"],
                _ng_bootstrap_ng_bootstrap__WEBPACK_IMPORTED_MODULE_12__["NgbModule"],
                ng_katex__WEBPACK_IMPORTED_MODULE_21__["KatexModule"],
                _fortawesome_angular_fontawesome__WEBPACK_IMPORTED_MODULE_23__["FontAwesomeModule"],
                _angular_flex_layout__WEBPACK_IMPORTED_MODULE_25__["FlexLayoutModule"],
                ngx_highlightjs__WEBPACK_IMPORTED_MODULE_13__["HighlightModule"].forRoot({
                    languages: hljsLanguages
                }),
            ],
            providers: [],
            bootstrap: [_app_component__WEBPACK_IMPORTED_MODULE_5__["AppComponent"]]
        })
    ], AppModule);
    return AppModule;
}());



/***/ }),

/***/ "./src/app/article-detail/article-detail.component.css":
/*!*************************************************************!*\
  !*** ./src/app/article-detail/article-detail.component.css ***!
  \*************************************************************/
/*! no static exports found */
/***/ (function(module, exports) {

module.exports = ".before-section {\r\n  height: 1px;\r\n  padding: 0;\r\n  margin: 0;\r\n}\r\n\r\n.article-section-title {\r\n  padding-top: 50px;\r\n  text-align: justify;\r\n  padding-bottom: 20px;\r\n}\r\n\r\n.article-subtitle-comment {\r\n  font-size: small;\r\n  text-align: right;\r\n  padding-right: 5%;\r\n  padding-bottom: 5px;\r\n  text-indent: 15%;\r\n}\r\n\r\n.author-footer {\r\n  font-size: small;\r\n  text-align: right;\r\n  padding-bottom: 20px;\r\n}\r\n\r\n.article-paragraph {\r\n  text-align: justify;\r\n  /*text-indent: 20px;*/\r\n  margin: 10px 0 10px;\r\n}\r\n\r\n.list-item, .list-item-nofrench {\r\n  text-align: justify;\r\n}\r\n\r\n.list-item-nofrench>a {\r\n  color: lightskyblue;\r\n}\r\n\r\n.list-item-nofrench>a:hover {\r\n  text-decoration: underline;\r\n}\r\n\r\n.list-item>small>span, .list-item-nofrench>small>span,\r\n.article-paragraph>small>span {\r\n  font-family: Menlo, Monaco, Consolas, \"Courier New\", monospace;\r\n  color: #8f600f;\r\n  /*color:rgb(51, 51, 51);*/\r\n  /*background-color: rgba(255, 255, 255, 0.7);*/\r\n}\r\n\r\n.article-code > pre  {\r\n  padding: 5px;\r\n  margin: 0;\r\n}\r\n\r\np.article-code {\r\n  padding: 0;\r\n  margin: 0;\r\n  background-color: transparent;\r\n}\r\n\r\n.panel-collapse > .panel-body {\r\n  padding: 5px;\r\n}\r\n\r\n/*.article-code > pre > code.hljs.crystal {*/\r\n\r\n/*background-color: rgba(255, 255, 255, 0.7);*/\r\n\r\n/*}*/\r\n\r\ndiv.panel, div.panel-body, .panel-default > div.panel-heading, .panel-group {\r\n  background: transparent;\r\n}\r\n\r\n.lead {\r\n  padding-top: 20px;\r\n}\r\n\r\n.panel-title {\r\n  color: #8f600f;;\r\n}\r\n\r\n/*pre,code {*/\r\n\r\n/*background: #3f3f3f;*/\r\n\r\n/*!*background:#474949;*!*/\r\n\r\n/*color:#dcdcdc;*/\r\n\r\n/*}*/\r\n\r\n.article-image, .article-video {\r\n  width: 100%;\r\n}\r\n\r\na:visited {\r\n  text-decoration: none;\r\n}\r\n\r\n.lead {\r\n  background-color: #F4F1EA;\r\n  color: #454545;\r\n  box-shadow: 1px 1px 1px rgba(118, 105, 94, 1);\r\n}\r\n\r\n.container {\r\n  margin-top: 9px;\r\n  margin-bottom: 10px;\r\n}\r\n\r\n.article-heading {\r\n  color: #454545;\r\n  background-color: #f6edda;\r\n  box-shadow: 1px 1px 1px rgba(118, 105, 94, 1);\r\n}\r\n\r\n.article-view-title {\r\n  text-align: center;\r\n  padding-top: 15px;\r\n  padding-bottom: 15px;\r\n}\r\n\r\nblockquote {\r\n  font-size: 15pt;\r\n  padding-top: 20px;\r\n}\r\n\r\nblockquote>p::before {\r\n  font-family: FontAwesome;\r\n  content: '\\f10d';\r\n  color: lightgrey;\r\n  padding-right: 10px;\r\n}\r\n\r\np.article-paragraph, a.article-paragraph,\r\np.list-item, p.list-item-nofrench { font-size: 14pt; }\r\n\r\nh1 { font-weight: 400; }\r\n\r\nh1.article-section-title { font-size: 20pt; font-weight: 500; }\r\n\r\nh2.article-section-subtitle { font-size: 18pt; font-weight: bold; }\r\n\r\n@media (max-width: 768px) {\r\n  .katex {\r\n    font-size: small;\r\n  }\r\n}\r\n\r\n@media (min-width: 768px) {\r\n  .katex {\r\n    font-size: medium;\r\n  }\r\n}\r\n\r\n.article-code {\r\n  font-size: medium;\r\n}\r\n\r\n.collapsible-title {\r\n  font-size: medium;\r\n  color: #454545;\r\n}\r\n\r\n.image-left {\r\n  padding-right: 0;\r\n}\r\n\r\n.image-right {\r\n  padding-left: 0;\r\n}\r\n\r\n.katex-eqn {\r\n  text-align:center;\r\n  display: block;\r\n}\r\n\r\n.nopadding {\r\n  padding-left: 0;\r\n  padding-right: 0;\r\n}\r\n\r\n.nomargin {\r\n  margin-left: 0;\r\n  margin-right: 0;\r\n}\r\n\r\n.padding-lr-20 {\r\n  padding-left: 20px;\r\n  padding-right: 20px;\r\n}\r\n\r\n/*# sourceMappingURL=data:application/json;base64,eyJ2ZXJzaW9uIjozLCJzb3VyY2VzIjpbInNyYy9hcHAvYXJ0aWNsZS1kZXRhaWwvYXJ0aWNsZS1kZXRhaWwuY29tcG9uZW50LmNzcyJdLCJuYW1lcyI6W10sIm1hcHBpbmdzIjoiQUFBQTtFQUNFLFdBQVc7RUFDWCxVQUFVO0VBQ1YsU0FBUztBQUNYOztBQUVBO0VBQ0UsaUJBQWlCO0VBQ2pCLG1CQUFtQjtFQUNuQixvQkFBb0I7QUFDdEI7O0FBRUE7RUFDRSxnQkFBZ0I7RUFDaEIsaUJBQWlCO0VBQ2pCLGlCQUFpQjtFQUNqQixtQkFBbUI7RUFDbkIsZ0JBQWdCO0FBQ2xCOztBQUVBO0VBQ0UsZ0JBQWdCO0VBQ2hCLGlCQUFpQjtFQUNqQixvQkFBb0I7QUFDdEI7O0FBRUE7RUFDRSxtQkFBbUI7RUFDbkIscUJBQXFCO0VBQ3JCLG1CQUFtQjtBQUNyQjs7QUFFQTtFQUNFLG1CQUFtQjtBQUNyQjs7QUFFQTtFQUNFLG1CQUFtQjtBQUNyQjs7QUFFQTtFQUNFLDBCQUEwQjtBQUM1Qjs7QUFFQTs7RUFFRSw4REFBOEQ7RUFDOUQsY0FBYztFQUNkLHlCQUF5QjtFQUN6Qiw4Q0FBOEM7QUFDaEQ7O0FBRUE7RUFDRSxZQUFZO0VBQ1osU0FBUztBQUNYOztBQUVBO0VBQ0UsVUFBVTtFQUNWLFNBQVM7RUFDVCw2QkFBNkI7QUFDL0I7O0FBRUE7RUFDRSxZQUFZO0FBQ2Q7O0FBRUEsNENBQTRDOztBQUMxQyw4Q0FBOEM7O0FBQ2hELElBQUk7O0FBRUo7RUFDRSx1QkFBdUI7QUFDekI7O0FBRUE7RUFDRSxpQkFBaUI7QUFDbkI7O0FBRUE7RUFDRSxjQUFjO0FBQ2hCOztBQUVBLGFBQWE7O0FBQ1gsdUJBQXVCOztBQUN2QiwwQkFBMEI7O0FBQzFCLGlCQUFpQjs7QUFDbkIsSUFBSTs7QUFFSjtFQUNFLFdBQVc7QUFDYjs7QUFFQTtFQUNFLHFCQUFxQjtBQUN2Qjs7QUFFQTtFQUNFLHlCQUF5QjtFQUN6QixjQUFjO0VBQ2QsNkNBQTZDO0FBQy9DOztBQUVBO0VBQ0UsZUFBZTtFQUNmLG1CQUFtQjtBQUNyQjs7QUFFQTtFQUNFLGNBQWM7RUFDZCx5QkFBeUI7RUFDekIsNkNBQTZDO0FBQy9DOztBQUVBO0VBQ0Usa0JBQWtCO0VBQ2xCLGlCQUFpQjtFQUNqQixvQkFBb0I7QUFDdEI7O0FBRUE7RUFDRSxlQUFlO0VBQ2YsaUJBQWlCO0FBQ25COztBQUVBO0VBQ0Usd0JBQXdCO0VBQ3hCLGdCQUFnQjtFQUNoQixnQkFBZ0I7RUFDaEIsbUJBQW1CO0FBQ3JCOztBQUVBO29DQUNvQyxlQUFlLEVBQUU7O0FBQ3JELEtBQUssZ0JBQWdCLEVBQUU7O0FBQ3ZCLDJCQUEyQixlQUFlLEVBQUUsZ0JBQWdCLEVBQUU7O0FBQzlELDhCQUE4QixlQUFlLEVBQUUsaUJBQWlCLEVBQUU7O0FBRWxFO0VBQ0U7SUFDRSxnQkFBZ0I7RUFDbEI7QUFDRjs7QUFDQTtFQUNFO0lBQ0UsaUJBQWlCO0VBQ25CO0FBQ0Y7O0FBRUE7RUFDRSxpQkFBaUI7QUFDbkI7O0FBRUE7RUFDRSxpQkFBaUI7RUFDakIsY0FBYztBQUNoQjs7QUFFQTtFQUNFLGdCQUFnQjtBQUNsQjs7QUFFQTtFQUNFLGVBQWU7QUFDakI7O0FBRUE7RUFDRSxpQkFBaUI7RUFDakIsY0FBYztBQUNoQjs7QUFFQTtFQUNFLGVBQWU7RUFDZixnQkFBZ0I7QUFDbEI7O0FBRUE7RUFDRSxjQUFjO0VBQ2QsZUFBZTtBQUNqQjs7QUFFQTtFQUNFLGtCQUFrQjtFQUNsQixtQkFBbUI7QUFDckIiLCJmaWxlIjoic3JjL2FwcC9hcnRpY2xlLWRldGFpbC9hcnRpY2xlLWRldGFpbC5jb21wb25lbnQuY3NzIiwic291cmNlc0NvbnRlbnQiOlsiLmJlZm9yZS1zZWN0aW9uIHtcclxuICBoZWlnaHQ6IDFweDtcclxuICBwYWRkaW5nOiAwO1xyXG4gIG1hcmdpbjogMDtcclxufVxyXG5cclxuLmFydGljbGUtc2VjdGlvbi10aXRsZSB7XHJcbiAgcGFkZGluZy10b3A6IDUwcHg7XHJcbiAgdGV4dC1hbGlnbjoganVzdGlmeTtcclxuICBwYWRkaW5nLWJvdHRvbTogMjBweDtcclxufVxyXG5cclxuLmFydGljbGUtc3VidGl0bGUtY29tbWVudCB7XHJcbiAgZm9udC1zaXplOiBzbWFsbDtcclxuICB0ZXh0LWFsaWduOiByaWdodDtcclxuICBwYWRkaW5nLXJpZ2h0OiA1JTtcclxuICBwYWRkaW5nLWJvdHRvbTogNXB4O1xyXG4gIHRleHQtaW5kZW50OiAxNSU7XHJcbn1cclxuXHJcbi5hdXRob3ItZm9vdGVyIHtcclxuICBmb250LXNpemU6IHNtYWxsO1xyXG4gIHRleHQtYWxpZ246IHJpZ2h0O1xyXG4gIHBhZGRpbmctYm90dG9tOiAyMHB4O1xyXG59XHJcblxyXG4uYXJ0aWNsZS1wYXJhZ3JhcGgge1xyXG4gIHRleHQtYWxpZ246IGp1c3RpZnk7XHJcbiAgLyp0ZXh0LWluZGVudDogMjBweDsqL1xyXG4gIG1hcmdpbjogMTBweCAwIDEwcHg7XHJcbn1cclxuXHJcbi5saXN0LWl0ZW0sIC5saXN0LWl0ZW0tbm9mcmVuY2gge1xyXG4gIHRleHQtYWxpZ246IGp1c3RpZnk7XHJcbn1cclxuXHJcbi5saXN0LWl0ZW0tbm9mcmVuY2g+YSB7XHJcbiAgY29sb3I6IGxpZ2h0c2t5Ymx1ZTtcclxufVxyXG5cclxuLmxpc3QtaXRlbS1ub2ZyZW5jaD5hOmhvdmVyIHtcclxuICB0ZXh0LWRlY29yYXRpb246IHVuZGVybGluZTtcclxufVxyXG5cclxuLmxpc3QtaXRlbT5zbWFsbD5zcGFuLCAubGlzdC1pdGVtLW5vZnJlbmNoPnNtYWxsPnNwYW4sXHJcbi5hcnRpY2xlLXBhcmFncmFwaD5zbWFsbD5zcGFuIHtcclxuICBmb250LWZhbWlseTogTWVubG8sIE1vbmFjbywgQ29uc29sYXMsIFwiQ291cmllciBOZXdcIiwgbW9ub3NwYWNlO1xyXG4gIGNvbG9yOiAjOGY2MDBmO1xyXG4gIC8qY29sb3I6cmdiKDUxLCA1MSwgNTEpOyovXHJcbiAgLypiYWNrZ3JvdW5kLWNvbG9yOiByZ2JhKDI1NSwgMjU1LCAyNTUsIDAuNyk7Ki9cclxufVxyXG5cclxuLmFydGljbGUtY29kZSA+IHByZSAge1xyXG4gIHBhZGRpbmc6IDVweDtcclxuICBtYXJnaW46IDA7XHJcbn1cclxuXHJcbnAuYXJ0aWNsZS1jb2RlIHtcclxuICBwYWRkaW5nOiAwO1xyXG4gIG1hcmdpbjogMDtcclxuICBiYWNrZ3JvdW5kLWNvbG9yOiB0cmFuc3BhcmVudDtcclxufVxyXG5cclxuLnBhbmVsLWNvbGxhcHNlID4gLnBhbmVsLWJvZHkge1xyXG4gIHBhZGRpbmc6IDVweDtcclxufVxyXG5cclxuLyouYXJ0aWNsZS1jb2RlID4gcHJlID4gY29kZS5obGpzLmNyeXN0YWwgeyovXHJcbiAgLypiYWNrZ3JvdW5kLWNvbG9yOiByZ2JhKDI1NSwgMjU1LCAyNTUsIDAuNyk7Ki9cclxuLyp9Ki9cclxuXHJcbmRpdi5wYW5lbCwgZGl2LnBhbmVsLWJvZHksIC5wYW5lbC1kZWZhdWx0ID4gZGl2LnBhbmVsLWhlYWRpbmcsIC5wYW5lbC1ncm91cCB7XHJcbiAgYmFja2dyb3VuZDogdHJhbnNwYXJlbnQ7XHJcbn1cclxuXHJcbi5sZWFkIHtcclxuICBwYWRkaW5nLXRvcDogMjBweDtcclxufVxyXG5cclxuLnBhbmVsLXRpdGxlIHtcclxuICBjb2xvcjogIzhmNjAwZjs7XHJcbn1cclxuXHJcbi8qcHJlLGNvZGUgeyovXHJcbiAgLypiYWNrZ3JvdW5kOiAjM2YzZjNmOyovXHJcbiAgLyohKmJhY2tncm91bmQ6IzQ3NDk0OTsqISovXHJcbiAgLypjb2xvcjojZGNkY2RjOyovXHJcbi8qfSovXHJcblxyXG4uYXJ0aWNsZS1pbWFnZSwgLmFydGljbGUtdmlkZW8ge1xyXG4gIHdpZHRoOiAxMDAlO1xyXG59XHJcblxyXG5hOnZpc2l0ZWQge1xyXG4gIHRleHQtZGVjb3JhdGlvbjogbm9uZTtcclxufVxyXG5cclxuLmxlYWQge1xyXG4gIGJhY2tncm91bmQtY29sb3I6ICNGNEYxRUE7XHJcbiAgY29sb3I6ICM0NTQ1NDU7XHJcbiAgYm94LXNoYWRvdzogMXB4IDFweCAxcHggcmdiYSgxMTgsIDEwNSwgOTQsIDEpO1xyXG59XHJcblxyXG4uY29udGFpbmVyIHtcclxuICBtYXJnaW4tdG9wOiA5cHg7XHJcbiAgbWFyZ2luLWJvdHRvbTogMTBweDtcclxufVxyXG5cclxuLmFydGljbGUtaGVhZGluZyB7XHJcbiAgY29sb3I6ICM0NTQ1NDU7XHJcbiAgYmFja2dyb3VuZC1jb2xvcjogI2Y2ZWRkYTtcclxuICBib3gtc2hhZG93OiAxcHggMXB4IDFweCByZ2JhKDExOCwgMTA1LCA5NCwgMSk7XHJcbn1cclxuXHJcbi5hcnRpY2xlLXZpZXctdGl0bGUge1xyXG4gIHRleHQtYWxpZ246IGNlbnRlcjtcclxuICBwYWRkaW5nLXRvcDogMTVweDtcclxuICBwYWRkaW5nLWJvdHRvbTogMTVweDtcclxufVxyXG5cclxuYmxvY2txdW90ZSB7XHJcbiAgZm9udC1zaXplOiAxNXB0O1xyXG4gIHBhZGRpbmctdG9wOiAyMHB4O1xyXG59XHJcblxyXG5ibG9ja3F1b3RlPnA6OmJlZm9yZSB7XHJcbiAgZm9udC1mYW1pbHk6IEZvbnRBd2Vzb21lO1xyXG4gIGNvbnRlbnQ6ICdcXGYxMGQnO1xyXG4gIGNvbG9yOiBsaWdodGdyZXk7XHJcbiAgcGFkZGluZy1yaWdodDogMTBweDtcclxufVxyXG5cclxucC5hcnRpY2xlLXBhcmFncmFwaCwgYS5hcnRpY2xlLXBhcmFncmFwaCxcclxucC5saXN0LWl0ZW0sIHAubGlzdC1pdGVtLW5vZnJlbmNoIHsgZm9udC1zaXplOiAxNHB0OyB9XHJcbmgxIHsgZm9udC13ZWlnaHQ6IDQwMDsgfVxyXG5oMS5hcnRpY2xlLXNlY3Rpb24tdGl0bGUgeyBmb250LXNpemU6IDIwcHQ7IGZvbnQtd2VpZ2h0OiA1MDA7IH1cclxuaDIuYXJ0aWNsZS1zZWN0aW9uLXN1YnRpdGxlIHsgZm9udC1zaXplOiAxOHB0OyBmb250LXdlaWdodDogYm9sZDsgfVxyXG5cclxuQG1lZGlhIChtYXgtd2lkdGg6IDc2OHB4KSB7XHJcbiAgLmthdGV4IHtcclxuICAgIGZvbnQtc2l6ZTogc21hbGw7XHJcbiAgfVxyXG59XHJcbkBtZWRpYSAobWluLXdpZHRoOiA3NjhweCkge1xyXG4gIC5rYXRleCB7XHJcbiAgICBmb250LXNpemU6IG1lZGl1bTtcclxuICB9XHJcbn1cclxuXHJcbi5hcnRpY2xlLWNvZGUge1xyXG4gIGZvbnQtc2l6ZTogbWVkaXVtO1xyXG59XHJcblxyXG4uY29sbGFwc2libGUtdGl0bGUge1xyXG4gIGZvbnQtc2l6ZTogbWVkaXVtO1xyXG4gIGNvbG9yOiAjNDU0NTQ1O1xyXG59XHJcblxyXG4uaW1hZ2UtbGVmdCB7XHJcbiAgcGFkZGluZy1yaWdodDogMDtcclxufVxyXG5cclxuLmltYWdlLXJpZ2h0IHtcclxuICBwYWRkaW5nLWxlZnQ6IDA7XHJcbn1cclxuXHJcbi5rYXRleC1lcW4ge1xyXG4gIHRleHQtYWxpZ246Y2VudGVyO1xyXG4gIGRpc3BsYXk6IGJsb2NrO1xyXG59XHJcblxyXG4ubm9wYWRkaW5nIHtcclxuICBwYWRkaW5nLWxlZnQ6IDA7XHJcbiAgcGFkZGluZy1yaWdodDogMDtcclxufVxyXG5cclxuLm5vbWFyZ2luIHtcclxuICBtYXJnaW4tbGVmdDogMDtcclxuICBtYXJnaW4tcmlnaHQ6IDA7XHJcbn1cclxuXHJcbi5wYWRkaW5nLWxyLTIwIHtcclxuICBwYWRkaW5nLWxlZnQ6IDIwcHg7XHJcbiAgcGFkZGluZy1yaWdodDogMjBweDtcclxufVxyXG4iXX0= */"

/***/ }),

/***/ "./src/app/article-detail/article-detail.component.html":
/*!**************************************************************!*\
  !*** ./src/app/article-detail/article-detail.component.html ***!
  \**************************************************************/
/*! no static exports found */
/***/ (function(module, exports) {

module.exports = "<app-navbar\r\n  [currentTitle]=\"currentTitle\"\r\n  [currentId]=\"currentId\">\r\n</app-navbar>\r\n\r\n<div class=\"before-section\"></div>\r\n<section *ngIf=\"article\" class=\"container col-sm-12 col-12 nopadding nomargin\">\r\n\r\n  <!-- Head -->\r\n  <div class=\"article-heading\">\r\n    <h1 class=\"article-view-title\">\r\n      <!-- Title -->\r\n      <p class=\"padding-lr-20\">{{article.title}}</p>\r\n    </h1>\r\n\r\n    <!-- Subtitle -->\r\n    <!--<p class=\"article-subtitle-comment\">-->\r\n      <!--<em class=\"text-muted\">-->\r\n        <!--authored by-->\r\n        <!--<span *ngIf=\"article.author\">{{article.author}}</span>-->\r\n        <!--<span *ngIf=\"!article.author\">unknown</span>.-->\r\n      <!--</em>-->\r\n    <!--</p>-->\r\n  </div>\r\n\r\n  <!-- Body -->\r\n  <!--<p></p>-->\r\n  <div *ngIf=\"article.body\" class=\"lead\">\r\n    <div *ngFor=\"let o of article.body\" class=\"row\">\r\n\r\n      <div class=\"col-1\"></div>\r\n\r\n      <div class=\"col-10\">\r\n\r\n        <!-- Headings -->\r\n        <h1 *ngIf=\"o.type==paragraphTypes.Title\"\r\n            class=\"article-section-title\" [id]=\"o.id\">\r\n          <br/>{{o.content}}\r\n        </h1>\r\n        <h2 *ngIf=\"o.type==paragraphTypes.Subtitle\"\r\n            class=\"article-section-subtitle\">\r\n          {{o.content}}\r\n        </h2>\r\n\r\n        <!-- Body -->\r\n        <p *ngIf=\"o.type==paragraphTypes.Paragraph\"\r\n           class=\"article-paragraph\" [innerHTML]=\"o.content\">\r\n        </p>\r\n        <p *ngIf=\"o.type==paragraphTypes.ListItemFR\"\r\n           class=\"list-item\" [innerHTML]=\"o.content\">\r\n        </p>\r\n        <p *ngIf=\"o.type==paragraphTypes.ListItem\"\r\n           class=\"list-item-nofrench\"\r\n           [innerHTML]=\"o.content\">\r\n        </p>\r\n        <a *ngIf=\"o.type==paragraphTypes.Link\"\r\n           [href]=\"o.content\">\r\n          {{o.heading}}\r\n        </a>\r\n\r\n        <!-- Quotations -->\r\n        <div *ngIf=\"o.type==paragraphTypes.Quotation\">\r\n          <blockquote class=\"blockquote text-center\" >\r\n\r\n            <p class=\"mb-0\">{{o.content}}</p>\r\n            <footer class=\"blockquote-footer\">Attributed to {{o.origin}}\r\n              <span *ngIf=\"o.source\">in <cite>{{o.source}}</cite></span>\r\n            </footer>\r\n          </blockquote>\r\n        </div>\r\n\r\n        <!-- Formulas -->\r\n        <div *ngIf=\"o.type==paragraphTypes.Equation\"\r\n             class=\"katex-eqn\">\r\n          <ng-katex class=\"katex\" [equation]=\"o.content\"></ng-katex>\r\n        </div>\r\n\r\n        <!-- Media -->\r\n        <img *ngIf=\"o.type==paragraphTypes.Image\"\r\n             [src]=\"o.content\" class=\"article-image card-header\"/>\r\n        <div class=\"row\" *ngIf=\"o.type==paragraphTypes.TwoImages\" >\r\n          <div class=\"col-6 image-left\">\r\n            <img [src]=\"o.image1\" class=\"article-image card-header\"/>\r\n          </div>\r\n          <div class=\"col-6 image-right\">\r\n            <img [src]=\"o.image2\" class=\"article-image card-header\"/>\r\n          </div>\r\n        </div>\r\n\r\n        <div *ngIf=\"o.type==paragraphTypes.Video && !isNotIE()\"\r\n             class=\"article-video\">\r\n          <p style=\"color:red; font-family: 'Source Sans Pro'\">\r\n            Error: video tag available on Chrome/Firefox.</p>\r\n        </div>\r\n        <div *ngIf=\"o.type==paragraphTypes.Video && isNotIE()\"\r\n             class=\"article-video\">\r\n          <video width=\"100%\" height=\"100%\" controls>\r\n            <source src=\"{{o.content}}\"\r\n                    type=\"video/webm; codecs=vp8,vorbis;\">\r\n            Your browser does not support the video tag.\r\n          </video>\r\n          <div><br/></div>\r\n        </div>\r\n\r\n        <!-- Code -->\r\n        <div *ngIf=\"o.type==paragraphTypes.Code\">\r\n          <pre class=\"article-code {{o.language}}\">\r\n            <code [highlight]=\"o.content\"\r\n                  (highlighted)=\"onHighlight($event)\">\r\n            </code>\r\n          </pre>\r\n        </div>\r\n\r\n        <div *ngIf=\"o.type==paragraphTypes.CollapsibleCode\">\r\n          <ngb-accordion>\r\n            <ngb-panel>\r\n              <ng-template ngbPanelTitle>\r\n                <span class=\"collapsible-title\">{{o.heading}}</span>\r\n              </ng-template>\r\n              <ng-template ngbPanelContent>\r\n                <pre class=\"article-code {{o.language}}\">\r\n                  <code [highlight]=\"o.content\"\r\n                        (highlighted)=\"onHighlight($event)\">\r\n                  </code>\r\n                </pre>\r\n              </ng-template>\r\n            </ngb-panel>\r\n          </ngb-accordion>\r\n        </div>\r\n      </div>\r\n\r\n      <div class=\"col-lg-2\"></div>\r\n    </div>\r\n\r\n    <!-- Author's mark -->\r\n    <br/>\r\n    <div class=\"row\">\r\n      <div class=\"col-9\"></div>\r\n      <div class=\"col-2 author-footer\">\r\n        <p><em class=\"text-muted\">\r\n        <span *ngIf=\"article.author\">{{article.author}}</span>\r\n        <span *ngIf=\"!article.author\">unknown</span>.\r\n        </em></p>\r\n\r\n      </div>\r\n      <div class=\"col-1\"></div>\r\n    </div>\r\n  </div>\r\n  <!-- /Body -->\r\n\r\n</section>\r\n"

/***/ }),

/***/ "./src/app/article-detail/article-detail.component.ts":
/*!************************************************************!*\
  !*** ./src/app/article-detail/article-detail.component.ts ***!
  \************************************************************/
/*! exports provided: ArticleDetailComponent */
/***/ (function(module, __webpack_exports__, __webpack_require__) {

"use strict";
__webpack_require__.r(__webpack_exports__);
/* harmony export (binding) */ __webpack_require__.d(__webpack_exports__, "ArticleDetailComponent", function() { return ArticleDetailComponent; });
/* harmony import */ var tslib__WEBPACK_IMPORTED_MODULE_0__ = __webpack_require__(/*! tslib */ "./node_modules/tslib/tslib.es6.js");
/* harmony import */ var _angular_core__WEBPACK_IMPORTED_MODULE_1__ = __webpack_require__(/*! @angular/core */ "./node_modules/@angular/core/fesm5/core.js");
/* harmony import */ var _angular_router__WEBPACK_IMPORTED_MODULE_2__ = __webpack_require__(/*! @angular/router */ "./node_modules/@angular/router/fesm5/router.js");
/* harmony import */ var _angular_common__WEBPACK_IMPORTED_MODULE_3__ = __webpack_require__(/*! @angular/common */ "./node_modules/@angular/common/fesm5/common.js");
/* harmony import */ var _services_article_service__WEBPACK_IMPORTED_MODULE_4__ = __webpack_require__(/*! ../services/article.service */ "./src/app/services/article.service.ts");
/* harmony import */ var _model_article__WEBPACK_IMPORTED_MODULE_5__ = __webpack_require__(/*! ../model/article */ "./src/app/model/article.ts");
/* harmony import */ var ngx_highlightjs__WEBPACK_IMPORTED_MODULE_6__ = __webpack_require__(/*! ngx-highlightjs */ "./node_modules/ngx-highlightjs/fesm5/ngx-highlightjs.js");
/* harmony import */ var _app_hyphenation__WEBPACK_IMPORTED_MODULE_7__ = __webpack_require__(/*! ../app.hyphenation */ "./src/app/app.hyphenation.js");
/* harmony import */ var _app_hyphenation__WEBPACK_IMPORTED_MODULE_7___default = /*#__PURE__*/__webpack_require__.n(_app_hyphenation__WEBPACK_IMPORTED_MODULE_7__);
/* harmony import */ var rxjs_operators__WEBPACK_IMPORTED_MODULE_8__ = __webpack_require__(/*! rxjs/operators */ "./node_modules/rxjs/_esm5/operators/index.js");










var ArticleDetailComponent = /** @class */ (function () {
    function ArticleDetailComponent(route, router, articleService, hljs, location) {
        var _this = this;
        this.route = route;
        this.router = router;
        this.articleService = articleService;
        this.hljs = hljs;
        this.location = location;
        this.currentTitle = '';
        this.currentId = '';
        this.paragraphTypes = _model_article__WEBPACK_IMPORTED_MODULE_5__["ParagraphType"];
        this.subbed = router.events
            .pipe(Object(rxjs_operators__WEBPACK_IMPORTED_MODULE_8__["filter"])(function (e) { return e instanceof _angular_router__WEBPACK_IMPORTED_MODULE_2__["NavigationEnd"]; }))
            .subscribe(function () { return _this.getArticle(); });
    }
    ArticleDetailComponent.prototype.ngOnInit = function () { };
    ArticleDetailComponent.prototype.ngOnDestroy = function () {
        this.subbed.unsubscribe();
    };
    ArticleDetailComponent.prototype.getArticle = function () {
        var _this = this;
        var id = +this.route.snapshot.paramMap.get('id');
        this.articleService.getArticle(id)
            .subscribe(function (article) {
            // this.article = article;
            var newArticle = new _model_article__WEBPACK_IMPORTED_MODULE_5__["Article"](article);
            var newTitle = _app_hyphenation__WEBPACK_IMPORTED_MODULE_7__["France"].apostrophize(article.title);
            newArticle.title = newTitle;
            if (!article.body) {
                _this.article = newArticle;
                return;
            }
            var newBody = [];
            article.body.forEach(function (item) {
                var newItem = {};
                newItem['type'] = item.type;
                newItem['id'] = item.id;
                newItem['language'] = item.language;
                newItem['heading'] = item.heading;
                var newContent = '';
                switch (item.type) {
                    case _model_article__WEBPACK_IMPORTED_MODULE_5__["ParagraphType"].Title:
                    case _model_article__WEBPACK_IMPORTED_MODULE_5__["ParagraphType"].Subtitle:
                    case _model_article__WEBPACK_IMPORTED_MODULE_5__["ParagraphType"].Paragraph:
                        // TODO Remove hyphenator load overload.
                        // item.content = France.franchize(item.content);
                        newContent = _app_hyphenation__WEBPACK_IMPORTED_MODULE_7__["France"].apostrophize(item.content);
                        newContent = _app_hyphenation__WEBPACK_IMPORTED_MODULE_7__["Hyphenator"].hyphenate(newContent, 'en');
                        newContent = _app_hyphenation__WEBPACK_IMPORTED_MODULE_7__["France"].guillemetize(newContent);
                        break;
                    case _model_article__WEBPACK_IMPORTED_MODULE_5__["ParagraphType"].Quotation:
                        newContent = _app_hyphenation__WEBPACK_IMPORTED_MODULE_7__["France"].apostrophize(item.content);
                        newContent = _app_hyphenation__WEBPACK_IMPORTED_MODULE_7__["France"].guillemetize(newContent);
                        newItem['origin'] = item['origin'];
                        newItem['source'] = item['source'];
                        break;
                    case _model_article__WEBPACK_IMPORTED_MODULE_5__["ParagraphType"].ListItemFR:
                        // item.content = Hyphenator.hyphenate(item.content, 'fr');
                        newContent = _app_hyphenation__WEBPACK_IMPORTED_MODULE_7__["France"].franchize(item.content);
                        // item.content = $sce.trustAsHtml(item.content);
                        break;
                    case _model_article__WEBPACK_IMPORTED_MODULE_5__["ParagraphType"].ListItem:
                        newContent = _app_hyphenation__WEBPACK_IMPORTED_MODULE_7__["France"].apostrophize(item.content);
                        // item.content = $sce.trustAsHtml(item.content);
                        break;
                    case _model_article__WEBPACK_IMPORTED_MODULE_5__["ParagraphType"].Code:
                    case _model_article__WEBPACK_IMPORTED_MODULE_5__["ParagraphType"].CollapsibleCode:
                        // item.content = item.content.replace(/</g, '&lt;');
                        // item.content = item.content.replace(/>/g, '&gt;');
                        // item.content = '<pre><code>' + item.content + '</code></pre>';
                        // item.content = $sce.trustAsHtml(item.content);
                        newContent = item.content;
                        break;
                    case _model_article__WEBPACK_IMPORTED_MODULE_5__["ParagraphType"].Equation:
                        newContent = item.content;
                        break;
                    case _model_article__WEBPACK_IMPORTED_MODULE_5__["ParagraphType"].Link:
                        newContent = item.content;
                        break;
                    case _model_article__WEBPACK_IMPORTED_MODULE_5__["ParagraphType"].TwoImages:
                        newItem['image1'] = item['image1'];
                        newItem['image2'] = item['image2'];
                        newContent = item.content;
                        break;
                    default:
                        newContent = item.content;
                }
                newItem['content'] = newContent;
                newBody.push(newItem);
            });
            newArticle.body = newBody;
            _this.article = newArticle;
        });
    };
    ArticleDetailComponent.prototype.onScroll = function () {
        var titles = document.getElementsByClassName('article-section-title');
        if (window.pageYOffset < 100) {
            this.currentTitle = '';
            return;
        }
        for (var t in titles) {
            if (titles.hasOwnProperty(t)) {
                var title = titles[t];
                var top_1 = title.getBoundingClientRect().top;
                if (top_1 <= 100) {
                    this.currentTitle = title.innerText;
                    this.currentId = title.id;
                }
            }
        }
    };
    ArticleDetailComponent.prototype.onHighlight = function (e) {
        console.log(e);
    };
    ArticleDetailComponent.prototype.isNotIE = function () {
        var ua = window.navigator.userAgent;
        var msie = ua.indexOf('MSIE ');
        return !(msie > 0 || !!navigator.userAgent.match(/Trident.*rv\:11\./));
    };
    tslib__WEBPACK_IMPORTED_MODULE_0__["__decorate"]([
        Object(_angular_core__WEBPACK_IMPORTED_MODULE_1__["HostListener"])('window:scroll', ['$event']),
        tslib__WEBPACK_IMPORTED_MODULE_0__["__metadata"]("design:type", Function),
        tslib__WEBPACK_IMPORTED_MODULE_0__["__metadata"]("design:paramtypes", []),
        tslib__WEBPACK_IMPORTED_MODULE_0__["__metadata"]("design:returntype", void 0)
    ], ArticleDetailComponent.prototype, "onScroll", null);
    ArticleDetailComponent = tslib__WEBPACK_IMPORTED_MODULE_0__["__decorate"]([
        Object(_angular_core__WEBPACK_IMPORTED_MODULE_1__["Component"])({
            selector: 'app-article-detail',
            template: __webpack_require__(/*! ./article-detail.component.html */ "./src/app/article-detail/article-detail.component.html"),
            styles: [__webpack_require__(/*! ./article-detail.component.css */ "./src/app/article-detail/article-detail.component.css")]
        }),
        tslib__WEBPACK_IMPORTED_MODULE_0__["__metadata"]("design:paramtypes", [_angular_router__WEBPACK_IMPORTED_MODULE_2__["ActivatedRoute"],
            _angular_router__WEBPACK_IMPORTED_MODULE_2__["Router"],
            _services_article_service__WEBPACK_IMPORTED_MODULE_4__["ArticleService"],
            ngx_highlightjs__WEBPACK_IMPORTED_MODULE_6__["HighlightJS"],
            _angular_common__WEBPACK_IMPORTED_MODULE_3__["Location"]])
    ], ArticleDetailComponent);
    return ArticleDetailComponent;
}());



/***/ }),

/***/ "./src/app/article-search/article-search.component.css":
/*!*************************************************************!*\
  !*** ./src/app/article-search/article-search.component.css ***!
  \*************************************************************/
/*! no static exports found */
/***/ (function(module, exports) {

module.exports = "/* ArticleSearch private styles */\r\n@media (max-width: 800px) {\r\n  #search-component {\r\n    display: none;\r\n  }\r\n  .search-result {\r\n    display: none;\r\n  }\r\n}\r\n#search-box {\r\n  opacity: 0.7;\r\n  margin: 0 1px;\r\n  width: 98px;\r\n  height: 30px;\r\n  font-family: FontAwesome, sans-serif;\r\n  font-style: normal;\r\n  font-weight: normal;\r\n  font-size: small;\r\n  text-decoration: inherit;\r\n  text-align: right;\r\n  padding-right: 10px;\r\n}\r\na {\r\n  padding: 5px;\r\n}\r\n.search-result li {\r\n  border-bottom: 1px solid gray;\r\n  border-left: 1px solid gray;\r\n  border-right: 1px solid gray;\r\n  /*width: 195px;*/\r\n  /*height: 16px;*/\r\n  /*padding: 5px;*/\r\n  background-color: white;\r\n  cursor: pointer;\r\n  list-style-type: none;\r\n}\r\n/*.search-result li:hover {*/\r\n/*background-color: #454545;*/\r\n/*}*/\r\n.search-result li  {\r\n  color: #454545;\r\n  display: block;\r\n  text-decoration: none;\r\n}\r\n.search-result li:hover {\r\n  background-color: #454545;\r\n  color: white;\r\n}\r\n.search-result li a:active {\r\n  color: white;\r\n}\r\nul.search-result {\r\n  /*border-top: 1px solid gray;*/\r\n  position: absolute;\r\n  width: 250px;\r\n  /*right: 50px;*/\r\n  left: 50px;\r\n  /* = 35 (col-12 + 20) + 15 (col-12) */\r\n  margin-top: 0;\r\n  padding-left: 0;\r\n}\r\n\r\n/*# sourceMappingURL=data:application/json;base64,eyJ2ZXJzaW9uIjozLCJzb3VyY2VzIjpbInNyYy9hcHAvYXJ0aWNsZS1zZWFyY2gvYXJ0aWNsZS1zZWFyY2guY29tcG9uZW50LmNzcyJdLCJuYW1lcyI6W10sIm1hcHBpbmdzIjoiQUFBQSxpQ0FBaUM7QUFDakM7RUFDRTtJQUNFLGFBQWE7RUFDZjtFQUNBO0lBQ0UsYUFBYTtFQUNmO0FBQ0Y7QUFFQTtFQUNFLFlBQVk7RUFDWixhQUFhO0VBQ2IsV0FBVztFQUNYLFlBQVk7RUFDWixvQ0FBb0M7RUFDcEMsa0JBQWtCO0VBQ2xCLG1CQUFtQjtFQUNuQixnQkFBZ0I7RUFDaEIsd0JBQXdCO0VBQ3hCLGlCQUFpQjtFQUNqQixtQkFBbUI7QUFDckI7QUFFQTtFQUNFLFlBQVk7QUFDZDtBQUVBO0VBQ0UsNkJBQTZCO0VBQzdCLDJCQUEyQjtFQUMzQiw0QkFBNEI7RUFDNUIsZ0JBQWdCO0VBQ2hCLGdCQUFnQjtFQUNoQixnQkFBZ0I7RUFDaEIsdUJBQXVCO0VBQ3ZCLGVBQWU7RUFDZixxQkFBcUI7QUFDdkI7QUFFQSw0QkFBNEI7QUFDMUIsNkJBQTZCO0FBQy9CLElBQUk7QUFFSjtFQUNFLGNBQWM7RUFDZCxjQUFjO0VBQ2QscUJBQXFCO0FBQ3ZCO0FBRUE7RUFDRSx5QkFBeUI7RUFDekIsWUFBWTtBQUNkO0FBQ0E7RUFDRSxZQUFZO0FBQ2Q7QUFFQTtFQUNFLDhCQUE4QjtFQUM5QixrQkFBa0I7RUFDbEIsWUFBWTtFQUNaLGVBQWU7RUFDZixVQUFVO0VBQ1YscUNBQXFDO0VBQ3JDLGFBQWE7RUFDYixlQUFlO0FBQ2pCIiwiZmlsZSI6InNyYy9hcHAvYXJ0aWNsZS1zZWFyY2gvYXJ0aWNsZS1zZWFyY2guY29tcG9uZW50LmNzcyIsInNvdXJjZXNDb250ZW50IjpbIi8qIEFydGljbGVTZWFyY2ggcHJpdmF0ZSBzdHlsZXMgKi9cclxuQG1lZGlhIChtYXgtd2lkdGg6IDgwMHB4KSB7XHJcbiAgI3NlYXJjaC1jb21wb25lbnQge1xyXG4gICAgZGlzcGxheTogbm9uZTtcclxuICB9XHJcbiAgLnNlYXJjaC1yZXN1bHQge1xyXG4gICAgZGlzcGxheTogbm9uZTtcclxuICB9XHJcbn1cclxuXHJcbiNzZWFyY2gtYm94IHtcclxuICBvcGFjaXR5OiAwLjc7XHJcbiAgbWFyZ2luOiAwIDFweDtcclxuICB3aWR0aDogOThweDtcclxuICBoZWlnaHQ6IDMwcHg7XHJcbiAgZm9udC1mYW1pbHk6IEZvbnRBd2Vzb21lLCBzYW5zLXNlcmlmO1xyXG4gIGZvbnQtc3R5bGU6IG5vcm1hbDtcclxuICBmb250LXdlaWdodDogbm9ybWFsO1xyXG4gIGZvbnQtc2l6ZTogc21hbGw7XHJcbiAgdGV4dC1kZWNvcmF0aW9uOiBpbmhlcml0O1xyXG4gIHRleHQtYWxpZ246IHJpZ2h0O1xyXG4gIHBhZGRpbmctcmlnaHQ6IDEwcHg7XHJcbn1cclxuXHJcbmEge1xyXG4gIHBhZGRpbmc6IDVweDtcclxufVxyXG5cclxuLnNlYXJjaC1yZXN1bHQgbGkge1xyXG4gIGJvcmRlci1ib3R0b206IDFweCBzb2xpZCBncmF5O1xyXG4gIGJvcmRlci1sZWZ0OiAxcHggc29saWQgZ3JheTtcclxuICBib3JkZXItcmlnaHQ6IDFweCBzb2xpZCBncmF5O1xyXG4gIC8qd2lkdGg6IDE5NXB4OyovXHJcbiAgLypoZWlnaHQ6IDE2cHg7Ki9cclxuICAvKnBhZGRpbmc6IDVweDsqL1xyXG4gIGJhY2tncm91bmQtY29sb3I6IHdoaXRlO1xyXG4gIGN1cnNvcjogcG9pbnRlcjtcclxuICBsaXN0LXN0eWxlLXR5cGU6IG5vbmU7XHJcbn1cclxuXHJcbi8qLnNlYXJjaC1yZXN1bHQgbGk6aG92ZXIgeyovXHJcbiAgLypiYWNrZ3JvdW5kLWNvbG9yOiAjNDU0NTQ1OyovXHJcbi8qfSovXHJcblxyXG4uc2VhcmNoLXJlc3VsdCBsaSAge1xyXG4gIGNvbG9yOiAjNDU0NTQ1O1xyXG4gIGRpc3BsYXk6IGJsb2NrO1xyXG4gIHRleHQtZGVjb3JhdGlvbjogbm9uZTtcclxufVxyXG5cclxuLnNlYXJjaC1yZXN1bHQgbGk6aG92ZXIge1xyXG4gIGJhY2tncm91bmQtY29sb3I6ICM0NTQ1NDU7XHJcbiAgY29sb3I6IHdoaXRlO1xyXG59XHJcbi5zZWFyY2gtcmVzdWx0IGxpIGE6YWN0aXZlIHtcclxuICBjb2xvcjogd2hpdGU7XHJcbn1cclxuXHJcbnVsLnNlYXJjaC1yZXN1bHQge1xyXG4gIC8qYm9yZGVyLXRvcDogMXB4IHNvbGlkIGdyYXk7Ki9cclxuICBwb3NpdGlvbjogYWJzb2x1dGU7XHJcbiAgd2lkdGg6IDI1MHB4O1xyXG4gIC8qcmlnaHQ6IDUwcHg7Ki9cclxuICBsZWZ0OiA1MHB4O1xyXG4gIC8qID0gMzUgKGNvbC0xMiArIDIwKSArIDE1IChjb2wtMTIpICovXHJcbiAgbWFyZ2luLXRvcDogMDtcclxuICBwYWRkaW5nLWxlZnQ6IDA7XHJcbn1cclxuIl19 */"

/***/ }),

/***/ "./src/app/article-search/article-search.component.html":
/*!**************************************************************!*\
  !*** ./src/app/article-search/article-search.component.html ***!
  \**************************************************************/
/*! no static exports found */
/***/ (function(module, exports) {

module.exports = "<span id=\"search-component\">\r\n  <!--<h4>Article Search</h4>-->\r\n  <input #searchBox id=\"search-box\"\r\n    (input)=\"search(searchBox.value)\"\r\n    [(ngModel)]=\"searchField\"\r\n    placeholder=\"&#xF002;\"/>\r\n</span>\r\n<ul class=\"search-result\">\r\n  <li *ngFor=\"let article of articlesObservable | async\">\r\n    <a (click)=\"nav(article.id)\">\r\n    <!--<a routerLink=\"/detail/{{article.id}}\">-->\r\n      {{ article.title.length > 25 ? article.title.substr(0,25) + '...' : article.title }}\r\n    </a>\r\n  </li>\r\n</ul>\r\n"

/***/ }),

/***/ "./src/app/article-search/article-search.component.ts":
/*!************************************************************!*\
  !*** ./src/app/article-search/article-search.component.ts ***!
  \************************************************************/
/*! exports provided: ArticleSearchComponent */
/***/ (function(module, __webpack_exports__, __webpack_require__) {

"use strict";
__webpack_require__.r(__webpack_exports__);
/* harmony export (binding) */ __webpack_require__.d(__webpack_exports__, "ArticleSearchComponent", function() { return ArticleSearchComponent; });
/* harmony import */ var tslib__WEBPACK_IMPORTED_MODULE_0__ = __webpack_require__(/*! tslib */ "./node_modules/tslib/tslib.es6.js");
/* harmony import */ var _angular_core__WEBPACK_IMPORTED_MODULE_1__ = __webpack_require__(/*! @angular/core */ "./node_modules/@angular/core/fesm5/core.js");
/* harmony import */ var rxjs__WEBPACK_IMPORTED_MODULE_2__ = __webpack_require__(/*! rxjs */ "./node_modules/rxjs/_esm5/index.js");
/* harmony import */ var rxjs_operators__WEBPACK_IMPORTED_MODULE_3__ = __webpack_require__(/*! rxjs/operators */ "./node_modules/rxjs/_esm5/operators/index.js");
/* harmony import */ var _services_article_service__WEBPACK_IMPORTED_MODULE_4__ = __webpack_require__(/*! ../services/article.service */ "./src/app/services/article.service.ts");
/* harmony import */ var _angular_router__WEBPACK_IMPORTED_MODULE_5__ = __webpack_require__(/*! @angular/router */ "./node_modules/@angular/router/fesm5/router.js");






var ArticleSearchComponent = /** @class */ (function () {
    function ArticleSearchComponent(articleService, router) {
        var _this = this;
        this.articleService = articleService;
        this.router = router;
        this.searchTerms = new rxjs__WEBPACK_IMPORTED_MODULE_2__["Subject"]();
        this.init();
        this.subbed = router.events
            .pipe(Object(rxjs_operators__WEBPACK_IMPORTED_MODULE_3__["filter"])(function (e) { return e instanceof _angular_router__WEBPACK_IMPORTED_MODULE_5__["NavigationEnd"]; }))
            .subscribe(function () { return _this.init(); });
    }
    ArticleSearchComponent.prototype.ngOnInit = function () { };
    ArticleSearchComponent.prototype.ngOnDestroy = function () {
        this.subbed.unsubscribe();
    };
    // Put a search term on the observable stream
    ArticleSearchComponent.prototype.search = function (term) {
        this.searchTerms.next(term);
    };
    ArticleSearchComponent.prototype.nav = function (id) {
        this.searchField = '';
        this.router.navigate(['/detail/' + id]);
        this.search('');
    };
    ArticleSearchComponent.prototype.init = function () {
        var _this = this;
        this.articlesObservable = this.searchTerms.pipe(
        // Wait 300ms after each keystroke
        Object(rxjs_operators__WEBPACK_IMPORTED_MODULE_3__["debounceTime"])(300), 
        // Ignore new term if same as previous
        Object(rxjs_operators__WEBPACK_IMPORTED_MODULE_3__["distinctUntilChanged"])(), 
        // Switch to new search observable each time the term changes
        Object(rxjs_operators__WEBPACK_IMPORTED_MODULE_3__["switchMap"])(function (term) {
            return _this.articleService.searchArticleTitles(term);
        }));
    };
    ArticleSearchComponent = tslib__WEBPACK_IMPORTED_MODULE_0__["__decorate"]([
        Object(_angular_core__WEBPACK_IMPORTED_MODULE_1__["Component"])({
            selector: 'app-article-search',
            template: __webpack_require__(/*! ./article-search.component.html */ "./src/app/article-search/article-search.component.html"),
            styles: [__webpack_require__(/*! ./article-search.component.css */ "./src/app/article-search/article-search.component.css")]
        }),
        tslib__WEBPACK_IMPORTED_MODULE_0__["__metadata"]("design:paramtypes", [_services_article_service__WEBPACK_IMPORTED_MODULE_4__["ArticleService"],
            _angular_router__WEBPACK_IMPORTED_MODULE_5__["Router"]])
    ], ArticleSearchComponent);
    return ArticleSearchComponent;
}());



/***/ }),

/***/ "./src/app/articles/articles.component.css":
/*!*************************************************!*\
  !*** ./src/app/articles/articles.component.css ***!
  \*************************************************/
/*! no static exports found */
/***/ (function(module, exports) {

module.exports = "/* ArticleComponent's private CSS styles */\r\n.selected {\r\n  background-color: #CFD8DC !important;\r\n  color: white;\r\n}\r\n.articles {\r\n  font-family: 'Dosis', sans-serif;\r\n  margin: 0 0 0 0;\r\n  list-style-type: none;\r\n  padding: 0;\r\n  width: 100%;\r\n}\r\n.articles li {\r\n  cursor: pointer;\r\n  position: relative;\r\n  left: 0;\r\n  background-color: #EEE;\r\n  margin: .1em;\r\n  padding: .3em 0;\r\n  height: 2em;\r\n  border-radius: 4px;\r\n}\r\n.articles li.selected:hover {\r\n  background-color: #BBD8DC !important;\r\n  color: white;\r\n}\r\n.articles li:hover {\r\n  color: #607D8B;\r\n  background-color: #DDD;\r\n  left: .1em;\r\n}\r\n.articles .text {\r\n  position: relative;\r\n  top: -3px;\r\n}\r\n.articles a {\r\n  color: #888;\r\n  text-decoration: none;\r\n  position: relative;\r\n  display: block;\r\n  /*width: 250px;*/\r\n}\r\nh4 {\r\n  font-family: 'Dosis', sans-serif;\r\n  padding-top: 10px;\r\n  padding-bottom: 5px;\r\n  text-align: center;\r\n  color: #454545;\r\n}\r\n.articles a:hover {\r\n  color:#607D8B;\r\n}\r\n.articles .badge {\r\n  display: inline-block;\r\n  /*font-size: small;*/\r\n  color: darkgray;\r\n  /*padding: 0.8em 0.7em 0 0.7em;*/\r\n  /*background-color: #607D8B;*/\r\n  /*line-height: 1em;*/\r\n  position: relative;\r\n  /*left: -1px;*/\r\n  /*top: -11px;*/\r\n  /*height: 40px;*/\r\n  margin-left: .4em;\r\n  margin-right: .8em;\r\n  border-radius: 4px 0 0 4px;\r\n}\r\nbutton {\r\n  background-color: #eee;\r\n  border: none;\r\n  padding: 5px 10px;\r\n  border-radius: 4px;\r\n  cursor: pointer;\r\n  cursor: hand;\r\n  font-family: Arial;\r\n}\r\nbutton:hover {\r\n  background-color: #cfd8dc;\r\n}\r\nbutton.delete {\r\n  position: relative;\r\n  left: 194px;\r\n  top: -32px;\r\n  background-color: gray !important;\r\n  color: white;\r\n}\r\n\r\n/*# sourceMappingURL=data:application/json;base64,eyJ2ZXJzaW9uIjozLCJzb3VyY2VzIjpbInNyYy9hcHAvYXJ0aWNsZXMvYXJ0aWNsZXMuY29tcG9uZW50LmNzcyJdLCJuYW1lcyI6W10sIm1hcHBpbmdzIjoiQUFBQSwwQ0FBMEM7QUFDMUM7RUFDRSxvQ0FBb0M7RUFDcEMsWUFBWTtBQUNkO0FBQ0E7RUFDRSxnQ0FBZ0M7RUFDaEMsZUFBZTtFQUNmLHFCQUFxQjtFQUNyQixVQUFVO0VBQ1YsV0FBVztBQUNiO0FBRUE7RUFDRSxlQUFlO0VBQ2Ysa0JBQWtCO0VBQ2xCLE9BQU87RUFDUCxzQkFBc0I7RUFDdEIsWUFBWTtFQUNaLGVBQWU7RUFDZixXQUFXO0VBQ1gsa0JBQWtCO0FBQ3BCO0FBQ0E7RUFDRSxvQ0FBb0M7RUFDcEMsWUFBWTtBQUNkO0FBQ0E7RUFDRSxjQUFjO0VBQ2Qsc0JBQXNCO0VBQ3RCLFVBQVU7QUFDWjtBQUNBO0VBQ0Usa0JBQWtCO0VBQ2xCLFNBQVM7QUFDWDtBQUNBO0VBQ0UsV0FBVztFQUNYLHFCQUFxQjtFQUNyQixrQkFBa0I7RUFDbEIsY0FBYztFQUNkLGdCQUFnQjtBQUNsQjtBQUVBO0VBQ0UsZ0NBQWdDO0VBQ2hDLGlCQUFpQjtFQUNqQixtQkFBbUI7RUFDbkIsa0JBQWtCO0VBQ2xCLGNBQWM7QUFDaEI7QUFFQTtFQUNFLGFBQWE7QUFDZjtBQUVBO0VBQ0UscUJBQXFCO0VBQ3JCLG9CQUFvQjtFQUNwQixlQUFlO0VBQ2YsZ0NBQWdDO0VBQ2hDLDZCQUE2QjtFQUM3QixvQkFBb0I7RUFDcEIsa0JBQWtCO0VBQ2xCLGNBQWM7RUFDZCxjQUFjO0VBQ2QsZ0JBQWdCO0VBQ2hCLGlCQUFpQjtFQUNqQixrQkFBa0I7RUFDbEIsMEJBQTBCO0FBQzVCO0FBRUE7RUFDRSxzQkFBc0I7RUFDdEIsWUFBWTtFQUNaLGlCQUFpQjtFQUNqQixrQkFBa0I7RUFDbEIsZUFBZTtFQUNmLFlBQVk7RUFDWixrQkFBa0I7QUFDcEI7QUFFQTtFQUNFLHlCQUF5QjtBQUMzQjtBQUVBO0VBQ0Usa0JBQWtCO0VBQ2xCLFdBQVc7RUFDWCxVQUFVO0VBQ1YsaUNBQWlDO0VBQ2pDLFlBQVk7QUFDZCIsImZpbGUiOiJzcmMvYXBwL2FydGljbGVzL2FydGljbGVzLmNvbXBvbmVudC5jc3MiLCJzb3VyY2VzQ29udGVudCI6WyIvKiBBcnRpY2xlQ29tcG9uZW50J3MgcHJpdmF0ZSBDU1Mgc3R5bGVzICovXHJcbi5zZWxlY3RlZCB7XHJcbiAgYmFja2dyb3VuZC1jb2xvcjogI0NGRDhEQyAhaW1wb3J0YW50O1xyXG4gIGNvbG9yOiB3aGl0ZTtcclxufVxyXG4uYXJ0aWNsZXMge1xyXG4gIGZvbnQtZmFtaWx5OiAnRG9zaXMnLCBzYW5zLXNlcmlmO1xyXG4gIG1hcmdpbjogMCAwIDAgMDtcclxuICBsaXN0LXN0eWxlLXR5cGU6IG5vbmU7XHJcbiAgcGFkZGluZzogMDtcclxuICB3aWR0aDogMTAwJTtcclxufVxyXG5cclxuLmFydGljbGVzIGxpIHtcclxuICBjdXJzb3I6IHBvaW50ZXI7XHJcbiAgcG9zaXRpb246IHJlbGF0aXZlO1xyXG4gIGxlZnQ6IDA7XHJcbiAgYmFja2dyb3VuZC1jb2xvcjogI0VFRTtcclxuICBtYXJnaW46IC4xZW07XHJcbiAgcGFkZGluZzogLjNlbSAwO1xyXG4gIGhlaWdodDogMmVtO1xyXG4gIGJvcmRlci1yYWRpdXM6IDRweDtcclxufVxyXG4uYXJ0aWNsZXMgbGkuc2VsZWN0ZWQ6aG92ZXIge1xyXG4gIGJhY2tncm91bmQtY29sb3I6ICNCQkQ4REMgIWltcG9ydGFudDtcclxuICBjb2xvcjogd2hpdGU7XHJcbn1cclxuLmFydGljbGVzIGxpOmhvdmVyIHtcclxuICBjb2xvcjogIzYwN0Q4QjtcclxuICBiYWNrZ3JvdW5kLWNvbG9yOiAjREREO1xyXG4gIGxlZnQ6IC4xZW07XHJcbn1cclxuLmFydGljbGVzIC50ZXh0IHtcclxuICBwb3NpdGlvbjogcmVsYXRpdmU7XHJcbiAgdG9wOiAtM3B4O1xyXG59XHJcbi5hcnRpY2xlcyBhIHtcclxuICBjb2xvcjogIzg4ODtcclxuICB0ZXh0LWRlY29yYXRpb246IG5vbmU7XHJcbiAgcG9zaXRpb246IHJlbGF0aXZlO1xyXG4gIGRpc3BsYXk6IGJsb2NrO1xyXG4gIC8qd2lkdGg6IDI1MHB4OyovXHJcbn1cclxuXHJcbmg0IHtcclxuICBmb250LWZhbWlseTogJ0Rvc2lzJywgc2Fucy1zZXJpZjtcclxuICBwYWRkaW5nLXRvcDogMTBweDtcclxuICBwYWRkaW5nLWJvdHRvbTogNXB4O1xyXG4gIHRleHQtYWxpZ246IGNlbnRlcjtcclxuICBjb2xvcjogIzQ1NDU0NTtcclxufVxyXG5cclxuLmFydGljbGVzIGE6aG92ZXIge1xyXG4gIGNvbG9yOiM2MDdEOEI7XHJcbn1cclxuXHJcbi5hcnRpY2xlcyAuYmFkZ2Uge1xyXG4gIGRpc3BsYXk6IGlubGluZS1ibG9jaztcclxuICAvKmZvbnQtc2l6ZTogc21hbGw7Ki9cclxuICBjb2xvcjogZGFya2dyYXk7XHJcbiAgLypwYWRkaW5nOiAwLjhlbSAwLjdlbSAwIDAuN2VtOyovXHJcbiAgLypiYWNrZ3JvdW5kLWNvbG9yOiAjNjA3RDhCOyovXHJcbiAgLypsaW5lLWhlaWdodDogMWVtOyovXHJcbiAgcG9zaXRpb246IHJlbGF0aXZlO1xyXG4gIC8qbGVmdDogLTFweDsqL1xyXG4gIC8qdG9wOiAtMTFweDsqL1xyXG4gIC8qaGVpZ2h0OiA0MHB4OyovXHJcbiAgbWFyZ2luLWxlZnQ6IC40ZW07XHJcbiAgbWFyZ2luLXJpZ2h0OiAuOGVtO1xyXG4gIGJvcmRlci1yYWRpdXM6IDRweCAwIDAgNHB4O1xyXG59XHJcblxyXG5idXR0b24ge1xyXG4gIGJhY2tncm91bmQtY29sb3I6ICNlZWU7XHJcbiAgYm9yZGVyOiBub25lO1xyXG4gIHBhZGRpbmc6IDVweCAxMHB4O1xyXG4gIGJvcmRlci1yYWRpdXM6IDRweDtcclxuICBjdXJzb3I6IHBvaW50ZXI7XHJcbiAgY3Vyc29yOiBoYW5kO1xyXG4gIGZvbnQtZmFtaWx5OiBBcmlhbDtcclxufVxyXG5cclxuYnV0dG9uOmhvdmVyIHtcclxuICBiYWNrZ3JvdW5kLWNvbG9yOiAjY2ZkOGRjO1xyXG59XHJcblxyXG5idXR0b24uZGVsZXRlIHtcclxuICBwb3NpdGlvbjogcmVsYXRpdmU7XHJcbiAgbGVmdDogMTk0cHg7XHJcbiAgdG9wOiAtMzJweDtcclxuICBiYWNrZ3JvdW5kLWNvbG9yOiBncmF5ICFpbXBvcnRhbnQ7XHJcbiAgY29sb3I6IHdoaXRlO1xyXG59XHJcbiJdfQ== */"

/***/ }),

/***/ "./src/app/articles/articles.component.html":
/*!**************************************************!*\
  !*** ./src/app/articles/articles.component.html ***!
  \**************************************************/
/*! no static exports found */
/***/ (function(module, exports) {

module.exports = "<app-navbar></app-navbar>\r\n\r\n<h4 class=\"col-11\">{{catDesc}}</h4>\r\n\r\n<ul class=\"articles col-12\">\r\n  <li *ngFor=\"let article of articles\">\r\n    <span class=\"nobreak\">\r\n      <a routerLink=\"/detail/{{article.id}}\">\r\n        <span class=\"badge\">{{article.date}}</span> {{article.title}}\r\n      </a>\r\n    </span>\r\n  </li>\r\n</ul>\r\n"

/***/ }),

/***/ "./src/app/articles/articles.component.ts":
/*!************************************************!*\
  !*** ./src/app/articles/articles.component.ts ***!
  \************************************************/
/*! exports provided: ArticlesComponent */
/***/ (function(module, __webpack_exports__, __webpack_require__) {

"use strict";
__webpack_require__.r(__webpack_exports__);
/* harmony export (binding) */ __webpack_require__.d(__webpack_exports__, "ArticlesComponent", function() { return ArticlesComponent; });
/* harmony import */ var tslib__WEBPACK_IMPORTED_MODULE_0__ = __webpack_require__(/*! tslib */ "./node_modules/tslib/tslib.es6.js");
/* harmony import */ var _angular_core__WEBPACK_IMPORTED_MODULE_1__ = __webpack_require__(/*! @angular/core */ "./node_modules/@angular/core/fesm5/core.js");
/* harmony import */ var _services_article_service__WEBPACK_IMPORTED_MODULE_2__ = __webpack_require__(/*! ../services/article.service */ "./src/app/services/article.service.ts");
/* harmony import */ var _angular_router__WEBPACK_IMPORTED_MODULE_3__ = __webpack_require__(/*! @angular/router */ "./node_modules/@angular/router/fesm5/router.js");
/* harmony import */ var rxjs_operators__WEBPACK_IMPORTED_MODULE_4__ = __webpack_require__(/*! rxjs/operators */ "./node_modules/rxjs/_esm5/operators/index.js");





var ArticlesComponent = /** @class */ (function () {
    function ArticlesComponent(route, router, articleService) {
        var _this = this;
        this.route = route;
        this.router = router;
        this.articleService = articleService;
        this.subbed = router.events
            .pipe(Object(rxjs_operators__WEBPACK_IMPORTED_MODULE_4__["filter"])(function (e) { return e instanceof _angular_router__WEBPACK_IMPORTED_MODULE_3__["NavigationEnd"]; }))
            .subscribe(function () {
            return _this.articleService.getCategories()
                .subscribe(function (categories) {
                _this.categories = categories;
                _this.getArticles();
            });
        });
    }
    ArticlesComponent.prototype.ngOnInit = function () { };
    ArticlesComponent.prototype.ngOnDestroy = function () {
        this.subbed.unsubscribe();
    };
    ArticlesComponent.prototype.getArticles = function () {
        var _this = this;
        var cat = this.route.snapshot.paramMap.get('cat');
        this.cat = cat;
        var cats = this.categories;
        switch (cat) {
            case cats[0][0].toString():
                this.catDesc = cats[0][1];
                break;
            case cats[1][0]:
                this.catDesc = cats[1][1];
                break;
            case cats[2][0]:
                this.catDesc = cats[2][1];
                break;
            case cats[3][0]:
                this.catDesc = cats[3][1];
                break;
        }
        this.articleService.getArticles(cat)
            .subscribe(function (articles) {
            _this.articles = articles;
        });
    };
    ArticlesComponent = tslib__WEBPACK_IMPORTED_MODULE_0__["__decorate"]([
        Object(_angular_core__WEBPACK_IMPORTED_MODULE_1__["Component"])({
            selector: 'app-articles',
            template: __webpack_require__(/*! ./articles.component.html */ "./src/app/articles/articles.component.html"),
            styles: [__webpack_require__(/*! ./articles.component.css */ "./src/app/articles/articles.component.css")]
        }),
        tslib__WEBPACK_IMPORTED_MODULE_0__["__metadata"]("design:paramtypes", [_angular_router__WEBPACK_IMPORTED_MODULE_3__["ActivatedRoute"],
            _angular_router__WEBPACK_IMPORTED_MODULE_3__["Router"],
            _services_article_service__WEBPACK_IMPORTED_MODULE_2__["ArticleService"]])
    ], ArticlesComponent);
    return ArticlesComponent;
}());



/***/ }),

/***/ "./src/app/dashboard/dashboard.component.css":
/*!***************************************************!*\
  !*** ./src/app/dashboard/dashboard.component.css ***!
  \***************************************************/
/*! no static exports found */
/***/ (function(module, exports) {

module.exports = "/* DashboardComponent's private CSS styles */\r\n\r\n[class*='col-'] {\r\n  float: left;\r\n  padding-left: 2px;\r\n  padding-right: 2px;\r\n  padding-bottom: 10px;\r\n}\r\n\r\n[class*='col-']:last-of-type {\r\n  padding-right: 0;\r\n}\r\n\r\n#dash-links {\r\n  position: absolute;\r\n  left: 0;\r\n  top: 0;\r\n}\r\n\r\n#windy {\r\n  left:0;\r\n  top:0;\r\n  opacity: 1.0;\r\n  width: 100%;\r\n}\r\n\r\n/* The following is for pixelated rendering */\r\n\r\n/*canvas {*/\r\n\r\n/*image-rendering: optimizeSpeed;             !* Older versions of FF          *!*/\r\n\r\n/*image-rendering: -moz-crisp-edges;          !* FF 6.0+                       *!*/\r\n\r\n/*image-rendering: -webkit-optimize-contrast; !* Safari                        *!*/\r\n\r\n/*image-rendering: -o-crisp-edges;            !* OS X & Windows Opera (12.02+) *!*/\r\n\r\n/*image-rendering: pixelated;                 !* Awesome future-browsers       *!*/\r\n\r\n/*-ms-interpolation-mode: nearest-neighbor;   !* IE                            *!*/\r\n\r\n/*}*/\r\n\r\n*, *:after, *:before {\r\n  box-sizing: border-box;\r\n}\r\n\r\nh3 {\r\n  color: #454545;\r\n  text-align: center;\r\n  margin-bottom: 0;\r\n}\r\n\r\n.grid {\r\n  margin: 0;\r\n}\r\n\r\n.col-1-4 {\r\n  width: 25%;\r\n}\r\n\r\n.module {\r\n  font-family: 'Dosis', sans-serif;\r\n  padding: 20px;\r\n  text-align: center;\r\n  color: #454545;\r\n  max-height: 120px;\r\n  min-width: 120px;\r\n  border-radius: 2px;\r\n}\r\n\r\n.cmodule {\r\n  cursor: pointer;\r\n  font-family: 'Dosis', sans-serif;\r\n  padding: 2px;\r\n  text-align: center;\r\n  color: #454545;\r\n  max-height: 120px;\r\n  min-width: 120px;\r\n  border-radius: 2px;\r\n}\r\n\r\np.ctooltip-item {\r\n  margin: 0;\r\n  position: relative;\r\n}\r\n\r\n.module>p.ctooltip-item:after {\r\n  color: #454545;\r\n  content: '';\r\n  position: absolute;\r\n  left: 25%;\r\n  display: inline-block;\r\n  height: 1em;\r\n  width: 50%;\r\n  border-bottom: 1px solid;\r\n  margin-top: 10px;\r\n  opacity: 0.3;\r\n  transition: opacity 0.35s, transform 0.35s, -webkit-transform 0.35s;\r\n  -webkit-transform: scale(0,1);\r\n  transform: scale(0,1);\r\n}\r\n\r\n.module>p.ctooltip-item:hover:after {\r\n  opacity: 1;\r\n  -webkit-transform: scale(1);\r\n  transform: scale(1);\r\n}\r\n\r\n.grid-pad {\r\n  padding: 10px 0;\r\n}\r\n\r\n.grid-pad > [class*='col-']:last-of-type {\r\n  padding-right: 20px;\r\n}\r\n\r\n.ctooltip {\r\n  display: inline;\r\n  position: relative;\r\n  z-index: 96;\r\n}\r\n\r\n.ctooltip-item {\r\n  width: 100%;\r\n  color: #b3b8b6;\r\n  cursor: pointer;\r\n  z-index: 999;\r\n  position: relative;\r\n  display: inline-block;\r\n  transition: background-color 0.3s, color 0.3s;\r\n}\r\n\r\n.ctooltip:hover .ctooltip-item {\r\n  color: #eee;\r\n}\r\n\r\n.ctooltip:hover {\r\n  z-index: 998;\r\n}\r\n\r\n.ctooltip-item {\r\n  z-index: 997;\r\n  background-color: #454545;\r\n  padding: 8px 0;\r\n}\r\n\r\n.ctooltip-content {\r\n  z-index: 99;\r\n  width: 100%;\r\n  left: 0;\r\n  top: -5px;\r\n  text-align: left;\r\n  background: #ddd;\r\n  opacity: 0.7;\r\n  font-size: 0.8em;\r\n  line-height: 1.5;\r\n  padding: 5px;\r\n  color: #454545;\r\n  pointer-events: none;\r\n  border-radius: 2px;\r\n  transition: opacity 0.3s;\r\n}\r\n\r\n.ctooltip-content>p {\r\n  padding: 5px 10px;\r\n  margin-bottom: 0;\r\n}\r\n\r\n.ctooltip-content p.ctooltip-item {\r\n  color: #32434f;\r\n}\r\n\r\n.ctooltip-text {\r\n  opacity: 0;\r\n  transition: opacity 0.3s, color 0.3s;\r\n}\r\n\r\n.ctooltip:hover .ctooltip-content,\r\n.ctooltip:hover .ctooltip-text\r\n{\r\n  z-index: 100;\r\n  pointer-events: auto;\r\n  opacity: 1;\r\n  color: black;\r\n  background: #edaf8c;\r\n  transition: background 0.3s, color 0.3s;\r\n}\r\n\r\n.dashboard-image-tooltip {\r\n  max-width: 20%;\r\n  max-height: 30%;\r\n}\r\n\r\n#pusher {\r\n  height: 200px;\r\n}\r\n\r\n#pusher-wrapper {\r\n  opacity: 0.5;\r\n  z-index: -10;\r\n}\r\n\r\n#dashboard-cards-container {\r\n  margin-top: 10px;\r\n}\r\n\r\n@media (max-width: 768px) {\r\n  .blockquote-container {\r\n    display: none;\r\n    visibility: hidden;\r\n  }\r\n  .ctooltip-content {\r\n    opacity: 1;\r\n  }\r\n}\r\n\r\n@media (max-width: 576px) {\r\n  #pusher-wrapper {\r\n    margin-top: 10px;\r\n  }\r\n}\r\n\r\n#texture-windy {\r\n  visibility: hidden;\r\n  display: none;\r\n}\r\n\r\nblockquote {\r\n  font-size: 0.8em;\r\n  padding-right: 5px;\r\n  padding-left: 5px;\r\n  color: black;\r\n}\r\n\r\nblockquote>footer {\r\n  padding-bottom: 5px;\r\n  color: #454545;\r\n}\r\n\r\nblockquote>p::before {\r\n  font-family: FontAwesome;\r\n  content: '\\f10d';\r\n  color: #454545;\r\n  padding-right: 10px;\r\n}\r\n\r\n.ctooltip:hover>.blockquote-container {\r\n  opacity: 1;\r\n  -webkit-transform: translate3d(0, 80px, 0);\r\n  transform: translate3d(0, 80px, 0);\r\n  transition: opacity 0.3s, -webkit-transform 0.3s;\r\n  transition: opacity 0.3s, transform 0.3s;\r\n  transition: opacity 0.3s, transform 0.3s, -webkit-transform 0.3s;\r\n}\r\n\r\n.blockquote-container {\r\n  margin-top: -100px;\r\n  opacity: 0;\r\n  padding: 0;\r\n  background-color: #edaf8c;\r\n  z-index: 10020;\r\n}\r\n\r\n.blockquote-container>hr {\r\n  margin-bottom: 5px;\r\n}\r\n\r\n/*# sourceMappingURL=data:application/json;base64,eyJ2ZXJzaW9uIjozLCJzb3VyY2VzIjpbInNyYy9hcHAvZGFzaGJvYXJkL2Rhc2hib2FyZC5jb21wb25lbnQuY3NzIl0sIm5hbWVzIjpbXSwibWFwcGluZ3MiOiJBQUFBLDRDQUE0Qzs7QUFFNUM7RUFDRSxXQUFXO0VBQ1gsaUJBQWlCO0VBQ2pCLGtCQUFrQjtFQUNsQixvQkFBb0I7QUFDdEI7O0FBRUE7RUFDRSxnQkFBZ0I7QUFDbEI7O0FBRUE7RUFDRSxrQkFBa0I7RUFDbEIsT0FBTztFQUNQLE1BQU07QUFDUjs7QUFFQTtFQUNFLE1BQU07RUFDTixLQUFLO0VBQ0wsWUFBWTtFQUNaLFdBQVc7QUFDYjs7QUFFQSw2Q0FBNkM7O0FBQzdDLFdBQVc7O0FBQ1Qsa0ZBQWtGOztBQUNsRixrRkFBa0Y7O0FBQ2xGLGtGQUFrRjs7QUFDbEYsa0ZBQWtGOztBQUNsRixrRkFBa0Y7O0FBQ2xGLGtGQUFrRjs7QUFDcEYsSUFBSTs7QUFFSjtFQUdFLHNCQUFzQjtBQUN4Qjs7QUFFQTtFQUNFLGNBQWM7RUFDZCxrQkFBa0I7RUFDbEIsZ0JBQWdCO0FBQ2xCOztBQUVBO0VBQ0UsU0FBUztBQUNYOztBQUVBO0VBQ0UsVUFBVTtBQUNaOztBQUVBO0VBQ0UsZ0NBQWdDO0VBQ2hDLGFBQWE7RUFDYixrQkFBa0I7RUFDbEIsY0FBYztFQUNkLGlCQUFpQjtFQUNqQixnQkFBZ0I7RUFDaEIsa0JBQWtCO0FBQ3BCOztBQUVBO0VBQ0UsZUFBZTtFQUNmLGdDQUFnQztFQUNoQyxZQUFZO0VBQ1osa0JBQWtCO0VBQ2xCLGNBQWM7RUFDZCxpQkFBaUI7RUFDakIsZ0JBQWdCO0VBQ2hCLGtCQUFrQjtBQUNwQjs7QUFFQTtFQUNFLFNBQVM7RUFDVCxrQkFBa0I7QUFDcEI7O0FBRUE7RUFDRSxjQUFjO0VBQ2QsV0FBVztFQUNYLGtCQUFrQjtFQUNsQixTQUFTO0VBQ1QscUJBQXFCO0VBQ3JCLFdBQVc7RUFDWCxVQUFVO0VBQ1Ysd0JBQXdCO0VBQ3hCLGdCQUFnQjtFQUNoQixZQUFZO0VBQ1osbUVBQW1FO0VBQ25FLDZCQUE2QjtFQUM3QixxQkFBcUI7QUFDdkI7O0FBRUE7RUFDRSxVQUFVO0VBQ1YsMkJBQTJCO0VBQzNCLG1CQUFtQjtBQUNyQjs7QUFFQTtFQUNFLGVBQWU7QUFDakI7O0FBRUE7RUFDRSxtQkFBbUI7QUFDckI7O0FBRUE7RUFDRSxlQUFlO0VBQ2Ysa0JBQWtCO0VBQ2xCLFdBQVc7QUFDYjs7QUFFQTtFQUNFLFdBQVc7RUFDWCxjQUFjO0VBQ2QsZUFBZTtFQUNmLFlBQVk7RUFDWixrQkFBa0I7RUFDbEIscUJBQXFCO0VBRXJCLDZDQUE2QztBQUMvQzs7QUFFQTtFQUNFLFdBQVc7QUFDYjs7QUFFQTtFQUNFLFlBQVk7QUFDZDs7QUFFQTtFQUNFLFlBQVk7RUFDWix5QkFBeUI7RUFDekIsY0FBYztBQUNoQjs7QUFFQTtFQUNFLFdBQVc7RUFDWCxXQUFXO0VBQ1gsT0FBTztFQUNQLFNBQVM7RUFDVCxnQkFBZ0I7RUFDaEIsZ0JBQWdCO0VBQ2hCLFlBQVk7RUFDWixnQkFBZ0I7RUFDaEIsZ0JBQWdCO0VBQ2hCLFlBQVk7RUFDWixjQUFjO0VBQ2Qsb0JBQW9CO0VBQ3BCLGtCQUFrQjtFQUVsQix3QkFBd0I7QUFDMUI7O0FBRUE7RUFDRSxpQkFBaUI7RUFDakIsZ0JBQWdCO0FBQ2xCOztBQUVBO0VBQ0UsY0FBYztBQUNoQjs7QUFFQTtFQUNFLFVBQVU7RUFFVixvQ0FBb0M7QUFDdEM7O0FBRUE7OztFQUdFLFlBQVk7RUFDWixvQkFBb0I7RUFDcEIsVUFBVTtFQUNWLFlBQVk7RUFDWixtQkFBbUI7RUFFbkIsdUNBQXVDO0FBQ3pDOztBQUVBO0VBQ0UsY0FBYztFQUNkLGVBQWU7QUFDakI7O0FBRUE7RUFDRSxhQUFhO0FBQ2Y7O0FBQ0E7RUFDRSxZQUFZO0VBQ1osWUFBWTtBQUNkOztBQUNBO0VBQ0UsZ0JBQWdCO0FBQ2xCOztBQUVBO0VBQ0U7SUFDRSxhQUFhO0lBQ2Isa0JBQWtCO0VBQ3BCO0VBQ0E7SUFDRSxVQUFVO0VBQ1o7QUFDRjs7QUFFQTtFQUNFO0lBQ0UsZ0JBQWdCO0VBQ2xCO0FBQ0Y7O0FBRUE7RUFDRSxrQkFBa0I7RUFDbEIsYUFBYTtBQUNmOztBQUVBO0VBQ0UsZ0JBQWdCO0VBQ2hCLGtCQUFrQjtFQUNsQixpQkFBaUI7RUFDakIsWUFBWTtBQUNkOztBQUNBO0VBQ0UsbUJBQW1CO0VBQ25CLGNBQWM7QUFDaEI7O0FBRUE7RUFDRSx3QkFBd0I7RUFDeEIsZ0JBQWdCO0VBQ2hCLGNBQWM7RUFDZCxtQkFBbUI7QUFDckI7O0FBRUE7RUFDRSxVQUFVO0VBQ1YsMENBQTBDO0VBQzFDLGtDQUFrQztFQUVsQyxnREFBd0M7RUFBeEMsd0NBQXdDO0VBQXhDLGdFQUF3QztBQUMxQzs7QUFFQTtFQUNFLGtCQUFrQjtFQUNsQixVQUFVO0VBQ1YsVUFBVTtFQUNWLHlCQUF5QjtFQUN6QixjQUFjO0FBQ2hCOztBQUVBO0VBQ0Usa0JBQWtCO0FBQ3BCIiwiZmlsZSI6InNyYy9hcHAvZGFzaGJvYXJkL2Rhc2hib2FyZC5jb21wb25lbnQuY3NzIiwic291cmNlc0NvbnRlbnQiOlsiLyogRGFzaGJvYXJkQ29tcG9uZW50J3MgcHJpdmF0ZSBDU1Mgc3R5bGVzICovXHJcblxyXG5bY2xhc3MqPSdjb2wtJ10ge1xyXG4gIGZsb2F0OiBsZWZ0O1xyXG4gIHBhZGRpbmctbGVmdDogMnB4O1xyXG4gIHBhZGRpbmctcmlnaHQ6IDJweDtcclxuICBwYWRkaW5nLWJvdHRvbTogMTBweDtcclxufVxyXG5cclxuW2NsYXNzKj0nY29sLSddOmxhc3Qtb2YtdHlwZSB7XHJcbiAgcGFkZGluZy1yaWdodDogMDtcclxufVxyXG5cclxuI2Rhc2gtbGlua3Mge1xyXG4gIHBvc2l0aW9uOiBhYnNvbHV0ZTtcclxuICBsZWZ0OiAwO1xyXG4gIHRvcDogMDtcclxufVxyXG5cclxuI3dpbmR5IHtcclxuICBsZWZ0OjA7XHJcbiAgdG9wOjA7XHJcbiAgb3BhY2l0eTogMS4wO1xyXG4gIHdpZHRoOiAxMDAlO1xyXG59XHJcblxyXG4vKiBUaGUgZm9sbG93aW5nIGlzIGZvciBwaXhlbGF0ZWQgcmVuZGVyaW5nICovXHJcbi8qY2FudmFzIHsqL1xyXG4gIC8qaW1hZ2UtcmVuZGVyaW5nOiBvcHRpbWl6ZVNwZWVkOyAgICAgICAgICAgICAhKiBPbGRlciB2ZXJzaW9ucyBvZiBGRiAgICAgICAgICAqISovXHJcbiAgLyppbWFnZS1yZW5kZXJpbmc6IC1tb3otY3Jpc3AtZWRnZXM7ICAgICAgICAgICEqIEZGIDYuMCsgICAgICAgICAgICAgICAgICAgICAgICohKi9cclxuICAvKmltYWdlLXJlbmRlcmluZzogLXdlYmtpdC1vcHRpbWl6ZS1jb250cmFzdDsgISogU2FmYXJpICAgICAgICAgICAgICAgICAgICAgICAgKiEqL1xyXG4gIC8qaW1hZ2UtcmVuZGVyaW5nOiAtby1jcmlzcC1lZGdlczsgICAgICAgICAgICAhKiBPUyBYICYgV2luZG93cyBPcGVyYSAoMTIuMDIrKSAqISovXHJcbiAgLyppbWFnZS1yZW5kZXJpbmc6IHBpeGVsYXRlZDsgICAgICAgICAgICAgICAgICEqIEF3ZXNvbWUgZnV0dXJlLWJyb3dzZXJzICAgICAgICohKi9cclxuICAvKi1tcy1pbnRlcnBvbGF0aW9uLW1vZGU6IG5lYXJlc3QtbmVpZ2hib3I7ICAgISogSUUgICAgICAgICAgICAgICAgICAgICAgICAgICAgKiEqL1xyXG4vKn0qL1xyXG5cclxuKiwgKjphZnRlciwgKjpiZWZvcmUge1xyXG4gIC13ZWJraXQtYm94LXNpemluZzogYm9yZGVyLWJveDtcclxuICAtbW96LWJveC1zaXppbmc6IGJvcmRlci1ib3g7XHJcbiAgYm94LXNpemluZzogYm9yZGVyLWJveDtcclxufVxyXG5cclxuaDMge1xyXG4gIGNvbG9yOiAjNDU0NTQ1O1xyXG4gIHRleHQtYWxpZ246IGNlbnRlcjtcclxuICBtYXJnaW4tYm90dG9tOiAwO1xyXG59XHJcblxyXG4uZ3JpZCB7XHJcbiAgbWFyZ2luOiAwO1xyXG59XHJcblxyXG4uY29sLTEtNCB7XHJcbiAgd2lkdGg6IDI1JTtcclxufVxyXG5cclxuLm1vZHVsZSB7XHJcbiAgZm9udC1mYW1pbHk6ICdEb3NpcycsIHNhbnMtc2VyaWY7XHJcbiAgcGFkZGluZzogMjBweDtcclxuICB0ZXh0LWFsaWduOiBjZW50ZXI7XHJcbiAgY29sb3I6ICM0NTQ1NDU7XHJcbiAgbWF4LWhlaWdodDogMTIwcHg7XHJcbiAgbWluLXdpZHRoOiAxMjBweDtcclxuICBib3JkZXItcmFkaXVzOiAycHg7XHJcbn1cclxuXHJcbi5jbW9kdWxlIHtcclxuICBjdXJzb3I6IHBvaW50ZXI7XHJcbiAgZm9udC1mYW1pbHk6ICdEb3NpcycsIHNhbnMtc2VyaWY7XHJcbiAgcGFkZGluZzogMnB4O1xyXG4gIHRleHQtYWxpZ246IGNlbnRlcjtcclxuICBjb2xvcjogIzQ1NDU0NTtcclxuICBtYXgtaGVpZ2h0OiAxMjBweDtcclxuICBtaW4td2lkdGg6IDEyMHB4O1xyXG4gIGJvcmRlci1yYWRpdXM6IDJweDtcclxufVxyXG5cclxucC5jdG9vbHRpcC1pdGVtIHtcclxuICBtYXJnaW46IDA7XHJcbiAgcG9zaXRpb246IHJlbGF0aXZlO1xyXG59XHJcblxyXG4ubW9kdWxlPnAuY3Rvb2x0aXAtaXRlbTphZnRlciB7XHJcbiAgY29sb3I6ICM0NTQ1NDU7XHJcbiAgY29udGVudDogJyc7XHJcbiAgcG9zaXRpb246IGFic29sdXRlO1xyXG4gIGxlZnQ6IDI1JTtcclxuICBkaXNwbGF5OiBpbmxpbmUtYmxvY2s7XHJcbiAgaGVpZ2h0OiAxZW07XHJcbiAgd2lkdGg6IDUwJTtcclxuICBib3JkZXItYm90dG9tOiAxcHggc29saWQ7XHJcbiAgbWFyZ2luLXRvcDogMTBweDtcclxuICBvcGFjaXR5OiAwLjM7XHJcbiAgdHJhbnNpdGlvbjogb3BhY2l0eSAwLjM1cywgdHJhbnNmb3JtIDAuMzVzLCAtd2Via2l0LXRyYW5zZm9ybSAwLjM1cztcclxuICAtd2Via2l0LXRyYW5zZm9ybTogc2NhbGUoMCwxKTtcclxuICB0cmFuc2Zvcm06IHNjYWxlKDAsMSk7XHJcbn1cclxuXHJcbi5tb2R1bGU+cC5jdG9vbHRpcC1pdGVtOmhvdmVyOmFmdGVyIHtcclxuICBvcGFjaXR5OiAxO1xyXG4gIC13ZWJraXQtdHJhbnNmb3JtOiBzY2FsZSgxKTtcclxuICB0cmFuc2Zvcm06IHNjYWxlKDEpO1xyXG59XHJcblxyXG4uZ3JpZC1wYWQge1xyXG4gIHBhZGRpbmc6IDEwcHggMDtcclxufVxyXG5cclxuLmdyaWQtcGFkID4gW2NsYXNzKj0nY29sLSddOmxhc3Qtb2YtdHlwZSB7XHJcbiAgcGFkZGluZy1yaWdodDogMjBweDtcclxufVxyXG5cclxuLmN0b29sdGlwIHtcclxuICBkaXNwbGF5OiBpbmxpbmU7XHJcbiAgcG9zaXRpb246IHJlbGF0aXZlO1xyXG4gIHotaW5kZXg6IDk2O1xyXG59XHJcblxyXG4uY3Rvb2x0aXAtaXRlbSB7XHJcbiAgd2lkdGg6IDEwMCU7XHJcbiAgY29sb3I6ICNiM2I4YjY7XHJcbiAgY3Vyc29yOiBwb2ludGVyO1xyXG4gIHotaW5kZXg6IDk5OTtcclxuICBwb3NpdGlvbjogcmVsYXRpdmU7XHJcbiAgZGlzcGxheTogaW5saW5lLWJsb2NrO1xyXG4gIC13ZWJraXQtdHJhbnNpdGlvbjogYmFja2dyb3VuZC1jb2xvciAwLjNzLCBjb2xvciAwLjNzO1xyXG4gIHRyYW5zaXRpb246IGJhY2tncm91bmQtY29sb3IgMC4zcywgY29sb3IgMC4zcztcclxufVxyXG5cclxuLmN0b29sdGlwOmhvdmVyIC5jdG9vbHRpcC1pdGVtIHtcclxuICBjb2xvcjogI2VlZTtcclxufVxyXG5cclxuLmN0b29sdGlwOmhvdmVyIHtcclxuICB6LWluZGV4OiA5OTg7XHJcbn1cclxuXHJcbi5jdG9vbHRpcC1pdGVtIHtcclxuICB6LWluZGV4OiA5OTc7XHJcbiAgYmFja2dyb3VuZC1jb2xvcjogIzQ1NDU0NTtcclxuICBwYWRkaW5nOiA4cHggMDtcclxufVxyXG5cclxuLmN0b29sdGlwLWNvbnRlbnQge1xyXG4gIHotaW5kZXg6IDk5O1xyXG4gIHdpZHRoOiAxMDAlO1xyXG4gIGxlZnQ6IDA7XHJcbiAgdG9wOiAtNXB4O1xyXG4gIHRleHQtYWxpZ246IGxlZnQ7XHJcbiAgYmFja2dyb3VuZDogI2RkZDtcclxuICBvcGFjaXR5OiAwLjc7XHJcbiAgZm9udC1zaXplOiAwLjhlbTtcclxuICBsaW5lLWhlaWdodDogMS41O1xyXG4gIHBhZGRpbmc6IDVweDtcclxuICBjb2xvcjogIzQ1NDU0NTtcclxuICBwb2ludGVyLWV2ZW50czogbm9uZTtcclxuICBib3JkZXItcmFkaXVzOiAycHg7XHJcbiAgLXdlYmtpdC10cmFuc2l0aW9uOiBvcGFjaXR5IDAuM3M7XHJcbiAgdHJhbnNpdGlvbjogb3BhY2l0eSAwLjNzO1xyXG59XHJcblxyXG4uY3Rvb2x0aXAtY29udGVudD5wIHtcclxuICBwYWRkaW5nOiA1cHggMTBweDtcclxuICBtYXJnaW4tYm90dG9tOiAwO1xyXG59XHJcblxyXG4uY3Rvb2x0aXAtY29udGVudCBwLmN0b29sdGlwLWl0ZW0ge1xyXG4gIGNvbG9yOiAjMzI0MzRmO1xyXG59XHJcblxyXG4uY3Rvb2x0aXAtdGV4dCB7XHJcbiAgb3BhY2l0eTogMDtcclxuICAtd2Via2l0LXRyYW5zaXRpb246IG9wYWNpdHkgMC4zcywgY29sb3IgMC4zcztcclxuICB0cmFuc2l0aW9uOiBvcGFjaXR5IDAuM3MsIGNvbG9yIDAuM3M7XHJcbn1cclxuXHJcbi5jdG9vbHRpcDpob3ZlciAuY3Rvb2x0aXAtY29udGVudCxcclxuLmN0b29sdGlwOmhvdmVyIC5jdG9vbHRpcC10ZXh0XHJcbntcclxuICB6LWluZGV4OiAxMDA7XHJcbiAgcG9pbnRlci1ldmVudHM6IGF1dG87XHJcbiAgb3BhY2l0eTogMTtcclxuICBjb2xvcjogYmxhY2s7XHJcbiAgYmFja2dyb3VuZDogI2VkYWY4YztcclxuICAtd2Via2l0LXRyYW5zaXRpb246IGJhY2tncm91bmQgMC4zcywgY29sb3IgMC4zcztcclxuICB0cmFuc2l0aW9uOiBiYWNrZ3JvdW5kIDAuM3MsIGNvbG9yIDAuM3M7XHJcbn1cclxuXHJcbi5kYXNoYm9hcmQtaW1hZ2UtdG9vbHRpcCB7XHJcbiAgbWF4LXdpZHRoOiAyMCU7XHJcbiAgbWF4LWhlaWdodDogMzAlO1xyXG59XHJcblxyXG4jcHVzaGVyIHtcclxuICBoZWlnaHQ6IDIwMHB4O1xyXG59XHJcbiNwdXNoZXItd3JhcHBlciB7XHJcbiAgb3BhY2l0eTogMC41O1xyXG4gIHotaW5kZXg6IC0xMDtcclxufVxyXG4jZGFzaGJvYXJkLWNhcmRzLWNvbnRhaW5lciB7XHJcbiAgbWFyZ2luLXRvcDogMTBweDtcclxufVxyXG5cclxuQG1lZGlhIChtYXgtd2lkdGg6IDc2OHB4KSB7XHJcbiAgLmJsb2NrcXVvdGUtY29udGFpbmVyIHtcclxuICAgIGRpc3BsYXk6IG5vbmU7XHJcbiAgICB2aXNpYmlsaXR5OiBoaWRkZW47XHJcbiAgfVxyXG4gIC5jdG9vbHRpcC1jb250ZW50IHtcclxuICAgIG9wYWNpdHk6IDE7XHJcbiAgfVxyXG59XHJcblxyXG5AbWVkaWEgKG1heC13aWR0aDogNTc2cHgpIHtcclxuICAjcHVzaGVyLXdyYXBwZXIge1xyXG4gICAgbWFyZ2luLXRvcDogMTBweDtcclxuICB9XHJcbn1cclxuXHJcbiN0ZXh0dXJlLXdpbmR5IHtcclxuICB2aXNpYmlsaXR5OiBoaWRkZW47XHJcbiAgZGlzcGxheTogbm9uZTtcclxufVxyXG5cclxuYmxvY2txdW90ZSB7XHJcbiAgZm9udC1zaXplOiAwLjhlbTtcclxuICBwYWRkaW5nLXJpZ2h0OiA1cHg7XHJcbiAgcGFkZGluZy1sZWZ0OiA1cHg7XHJcbiAgY29sb3I6IGJsYWNrO1xyXG59XHJcbmJsb2NrcXVvdGU+Zm9vdGVyIHtcclxuICBwYWRkaW5nLWJvdHRvbTogNXB4O1xyXG4gIGNvbG9yOiAjNDU0NTQ1O1xyXG59XHJcblxyXG5ibG9ja3F1b3RlPnA6OmJlZm9yZSB7XHJcbiAgZm9udC1mYW1pbHk6IEZvbnRBd2Vzb21lO1xyXG4gIGNvbnRlbnQ6ICdcXGYxMGQnO1xyXG4gIGNvbG9yOiAjNDU0NTQ1O1xyXG4gIHBhZGRpbmctcmlnaHQ6IDEwcHg7XHJcbn1cclxuXHJcbi5jdG9vbHRpcDpob3Zlcj4uYmxvY2txdW90ZS1jb250YWluZXIge1xyXG4gIG9wYWNpdHk6IDE7XHJcbiAgLXdlYmtpdC10cmFuc2Zvcm06IHRyYW5zbGF0ZTNkKDAsIDgwcHgsIDApO1xyXG4gIHRyYW5zZm9ybTogdHJhbnNsYXRlM2QoMCwgODBweCwgMCk7XHJcbiAgLXdlYmtpdC10cmFuc2l0aW9uOiBvcGFjaXR5IDAuM3MsIC13ZWJraXQtdHJhbnNmb3JtIDAuM3M7XHJcbiAgdHJhbnNpdGlvbjogb3BhY2l0eSAwLjNzLCB0cmFuc2Zvcm0gMC4zcztcclxufVxyXG5cclxuLmJsb2NrcXVvdGUtY29udGFpbmVyIHtcclxuICBtYXJnaW4tdG9wOiAtMTAwcHg7XHJcbiAgb3BhY2l0eTogMDtcclxuICBwYWRkaW5nOiAwO1xyXG4gIGJhY2tncm91bmQtY29sb3I6ICNlZGFmOGM7XHJcbiAgei1pbmRleDogMTAwMjA7XHJcbn1cclxuXHJcbi5ibG9ja3F1b3RlLWNvbnRhaW5lcj5ociB7XHJcbiAgbWFyZ2luLWJvdHRvbTogNXB4O1xyXG59XHJcbiJdfQ== */"

/***/ }),

/***/ "./src/app/dashboard/dashboard.component.html":
/*!****************************************************!*\
  !*** ./src/app/dashboard/dashboard.component.html ***!
  \****************************************************/
/*! no static exports found */
/***/ (function(module, exports) {

module.exports = "<app-navbar></app-navbar>\r\n\r\n<!--<div class=\"row flex-grow-1\">-->\r\n<!--<canvas width=\"1000\" height=\"500\" id=\"windy\" class=\"col-12 flex-fill\"></canvas>-->\r\n<!--</div>-->\r\n\r\n<div id=\"dash-links\" class=\"container-fluid\">\r\n\r\n  <div id=\"dashboard-links\">\r\n    <div class=\"row\" id=\"pusher-wrapper\">\r\n      <div class=\"col-12\" id=\"pusher\">\r\n        <canvas width=\"1000\" height=\"200\" id=\"windy\" class=\"col-12\"></canvas>\r\n        <img src=\"assets/img/frontpage/inkt.jpg\" id=\"texture-windy\"/>\r\n      </div>\r\n    </div>\r\n\r\n    <div class=\"row\" id=\"dashboard-cards-container\">\r\n      <div *ngFor=\"let category of categories\"\r\n           class=\"col-md-3 col-sm-12 no-select\">\r\n        <div class=\"cmodule\" routerLink=\"/articles/{{category[3]}}\">\r\n          <div class=\"ctooltip ctooltip-effect-1\">\r\n            <p class=\"ctooltip-item\" >{{category[0]}}</p>\r\n            <div class=\"ctooltip-content clearfix justify-content-around\">\r\n\r\n              <p>\r\n                {{ category[2] }}\r\n              </p>\r\n\r\n            </div>\r\n\r\n            <div *ngIf=\"category[5]\" class=\"blockquote-container\">\r\n              <hr/>\r\n              <blockquote class=\"blockquote text-center\" >\r\n\r\n                <p class=\"mb-0\">{{ category[5] }}</p>\r\n                <footer class=\"blockquote-footer\">{{ category[6] }}\r\n                  <span *ngIf=\"category[7]\">, <cite>{{ category[7] }}</cite></span>\r\n                </footer>\r\n              </blockquote>\r\n            </div>\r\n\r\n          </div>\r\n        </div>\r\n\r\n      </div>\r\n    </div>\r\n  </div>\r\n\r\n</div>\r\n\r\n<!--<app-footer></app-footer>-->\r\n"

/***/ }),

/***/ "./src/app/dashboard/dashboard.component.ts":
/*!**************************************************!*\
  !*** ./src/app/dashboard/dashboard.component.ts ***!
  \**************************************************/
/*! exports provided: DashboardComponent */
/***/ (function(module, __webpack_exports__, __webpack_require__) {

"use strict";
__webpack_require__.r(__webpack_exports__);
/* harmony export (binding) */ __webpack_require__.d(__webpack_exports__, "DashboardComponent", function() { return DashboardComponent; });
/* harmony import */ var tslib__WEBPACK_IMPORTED_MODULE_0__ = __webpack_require__(/*! tslib */ "./node_modules/tslib/tslib.es6.js");
/* harmony import */ var _angular_core__WEBPACK_IMPORTED_MODULE_1__ = __webpack_require__(/*! @angular/core */ "./node_modules/@angular/core/fesm5/core.js");
/* harmony import */ var _services_article_service__WEBPACK_IMPORTED_MODULE_2__ = __webpack_require__(/*! ../services/article.service */ "./src/app/services/article.service.ts");
/* harmony import */ var _animation_wind_windy__WEBPACK_IMPORTED_MODULE_3__ = __webpack_require__(/*! ../animation/wind/windy */ "./src/app/animation/wind/windy.js");
/* harmony import */ var _animation_wind_windy__WEBPACK_IMPORTED_MODULE_3___default = /*#__PURE__*/__webpack_require__.n(_animation_wind_windy__WEBPACK_IMPORTED_MODULE_3__);
/* harmony import */ var _angular_common_http__WEBPACK_IMPORTED_MODULE_4__ = __webpack_require__(/*! @angular/common/http */ "./node_modules/@angular/common/fesm5/http.js");





var DashboardComponent = /** @class */ (function () {
    function DashboardComponent(articleService, http) {
        this.articleService = articleService;
        this.http = http;
        this.timeoutLimit = 200;
        this.articles = [];
        this.glContext = null;
    }
    DashboardComponent.prototype.ngOnInit = function () {
        this.getCategories();
    };
    DashboardComponent.prototype.ngOnDestroy = function () {
        _animation_wind_windy__WEBPACK_IMPORTED_MODULE_3__["Windy"].end();
    };
    DashboardComponent.prototype.getCategories = function () {
        var _this = this;
        this.articleService.getCategories()
            .subscribe(function (categories) {
            _this.categories = categories;
            setTimeout(_this.animatePart1.bind(_this), 16);
        });
    };
    DashboardComponent.prototype.onResize = function () {
        this.tryToAnimate();
    };
    DashboardComponent.prototype.tryToAnimate = function () {
        clearTimeout(this.timeout);
        this.timeout = setTimeout(this.animatePart1.bind(this), this.timeoutLimit);
    };
    DashboardComponent.prototype.animatePart1 = function () {
        // Warning! This is quick and dirty, be indulgent.
        var element = document.getElementById('windy');
        var hr = document.getElementById('dashboard-hr');
        var hrStyle = window.getComputedStyle(hr);
        var navHeight = hr.offsetHeight +
            parseInt(hrStyle.marginTop, 10) +
            parseInt(hrStyle.marginBottom, 10);
        var winHeight = window.innerHeight;
        var dashHeight = document
            .getElementById('dashboard-links').clientHeight;
        var paddingBottom = 20;
        var width = element.offsetWidth;
        var height = width / 1.87; // winHeight - navHeight - dashHeight - paddingBottom;
        element.style.height = height.toString();
        var nbSamples = 10000;
        // Math.floor(Math.min(25 * width * height / 2000, 100000));
        console.log('[Dashboard] Particle tracing starting using ' +
            nbSamples + ' samples.');
        element.setAttribute('width', width.toString());
        element.setAttribute('height', height.toString());
        var wrapper = document.getElementById('pusher-wrapper');
        var mix = Math.min(300, Math.max(150, height));
        wrapper.setAttribute('style', 'height: ' + mix.toString() + 'px;');
        var canvas = document.createElement('canvas');
        var context = canvas.getContext('2d');
        var myData = null;
        if (!myData) {
            var img = document.getElementById('texture-windy');
            this.imageWidth = img.width;
            this.imageHeight = img.height;
            canvas.width = img.width;
            canvas.height = img.height;
            context.drawImage(img, 0, 0);
            myData = context.getImageData(0, 0, img.width, img.height);
        }
        if (myData && myData.data[0] !== 0) {
            this.animatePart2(element, width, height, nbSamples, myData);
        }
        else {
            console.log('[Dashboard] The browser did some fancy stuff that prevented me from getting that ' +
                'blasted image. Trying again later.');
            this.tryToAnimate();
        }
    };
    DashboardComponent.prototype.animatePart2 = function (element, width, height, nbSamples, myData) {
        var gl = this.glContext;
        // Firefox, Chrome
        if (!gl) {
            element.getContext('webgl', { antialiasing: true });
        }
        // IE
        if (!gl) {
            try {
                gl = element.getContext('experimental-webgl');
            }
            catch (error) {
                var msg = '[Dashboard] Error while creating WebGL context: ' + error.toString();
                throw Error(msg);
            }
        }
        // Unsupported
        if (!gl) {
            if (!document.getElementById('webgl-unsupported')) {
                var image = document.createElement('div');
                image.setAttribute('id', 'webgl-unsupported');
                image.setAttribute('style', 'position: absolute;left:0;');
                image.innerText = 'Your browser does not support WebGL :/';
                document.getElementById('pusher').appendChild(image);
            }
            return;
        }
        this.glContext = gl;
        _animation_wind_windy__WEBPACK_IMPORTED_MODULE_3__["Windy"].start(gl, element, width, height, nbSamples, null, myData, this.imageWidth, this.imageHeight);
    };
    tslib__WEBPACK_IMPORTED_MODULE_0__["__decorate"]([
        Object(_angular_core__WEBPACK_IMPORTED_MODULE_1__["HostListener"])('window:resize'),
        tslib__WEBPACK_IMPORTED_MODULE_0__["__metadata"]("design:type", Function),
        tslib__WEBPACK_IMPORTED_MODULE_0__["__metadata"]("design:paramtypes", []),
        tslib__WEBPACK_IMPORTED_MODULE_0__["__metadata"]("design:returntype", void 0)
    ], DashboardComponent.prototype, "onResize", null);
    DashboardComponent = tslib__WEBPACK_IMPORTED_MODULE_0__["__decorate"]([
        Object(_angular_core__WEBPACK_IMPORTED_MODULE_1__["Component"])({
            selector: 'app-dashboard',
            template: __webpack_require__(/*! ./dashboard.component.html */ "./src/app/dashboard/dashboard.component.html"),
            styles: [__webpack_require__(/*! ./dashboard.component.css */ "./src/app/dashboard/dashboard.component.css")]
        }),
        tslib__WEBPACK_IMPORTED_MODULE_0__["__metadata"]("design:paramtypes", [_services_article_service__WEBPACK_IMPORTED_MODULE_2__["ArticleService"],
            _angular_common_http__WEBPACK_IMPORTED_MODULE_4__["HttpClient"]])
    ], DashboardComponent);
    return DashboardComponent;
}());



/***/ }),

/***/ "./src/app/footer/footer.component.css":
/*!*********************************************!*\
  !*** ./src/app/footer/footer.component.css ***!
  \*********************************************/
/*! no static exports found */
/***/ (function(module, exports) {

module.exports = "\r\n.module {\r\n  font-family: 'Dosis', sans-serif;\r\n  padding: 15px;\r\n  text-align: center;\r\n  color: #454545;\r\n  max-height: 120px;\r\n  min-width: 120px;\r\n  border-radius: 2px;\r\n}\r\n\r\na {\r\n  color: #454545;\r\n  margin: 0;\r\n  position: relative;\r\n}\r\n\r\n.module>a:after {\r\n  color: #454545;\r\n  content: '';\r\n  position: absolute;\r\n  left: 25%;\r\n  display: inline-block;\r\n  height: 1em;\r\n  width: 50%;\r\n  border-bottom: 1px solid;\r\n  margin-top: 10px;\r\n  opacity: 0;\r\n  transition: opacity 0.35s, -webkit-transform 0.35s;\r\n  transition: opacity 0.35s, transform 0.35s;\r\n  transition: opacity 0.35s, transform 0.35s, -webkit-transform 0.35s;\r\n  -webkit-transform: scale(0,1);\r\n  transform: scale(0,1);\r\n}\r\n\r\n.module>a:hover:after {\r\n  opacity: 1;\r\n  -webkit-transform: scale(1);\r\n  transform: scale(1);\r\n}\r\n\r\n.cfoot {\r\n  /*background-color: rgba(214, 214, 214, 0.16);*/\r\n  /*border-top: 1px solid #d6d6d6;*/\r\n  z-index: 1;\r\n}\r\n\r\n#footer-right-align {\r\n  margin-right: 50px;\r\n  color: #bbbbbb;\r\n}\r\n\r\n#footer-right-align:hover {\r\n  color: #454545;\r\n  transition: color 0.3s;\r\n}\r\n\r\n/*# sourceMappingURL=data:application/json;base64,eyJ2ZXJzaW9uIjozLCJzb3VyY2VzIjpbInNyYy9hcHAvZm9vdGVyL2Zvb3Rlci5jb21wb25lbnQuY3NzIl0sIm5hbWVzIjpbXSwibWFwcGluZ3MiOiI7QUFDQTtFQUNFLGdDQUFnQztFQUNoQyxhQUFhO0VBQ2Isa0JBQWtCO0VBQ2xCLGNBQWM7RUFDZCxpQkFBaUI7RUFDakIsZ0JBQWdCO0VBQ2hCLGtCQUFrQjtBQUNwQjs7QUFFQTtFQUNFLGNBQWM7RUFDZCxTQUFTO0VBQ1Qsa0JBQWtCO0FBQ3BCOztBQUVBO0VBQ0UsY0FBYztFQUNkLFdBQVc7RUFDWCxrQkFBa0I7RUFDbEIsU0FBUztFQUNULHFCQUFxQjtFQUNyQixXQUFXO0VBQ1gsVUFBVTtFQUNWLHdCQUF3QjtFQUN4QixnQkFBZ0I7RUFDaEIsVUFBVTtFQUNWLGtEQUFrRDtFQUNsRCwwQ0FBMEM7RUFDMUMsbUVBQW1FO0VBQ25FLDZCQUE2QjtFQUM3QixxQkFBcUI7QUFDdkI7O0FBRUE7RUFDRSxVQUFVO0VBQ1YsMkJBQTJCO0VBQzNCLG1CQUFtQjtBQUNyQjs7QUFFQTtFQUNFLCtDQUErQztFQUMvQyxpQ0FBaUM7RUFDakMsVUFBVTtBQUNaOztBQUVBO0VBQ0Usa0JBQWtCO0VBQ2xCLGNBQWM7QUFDaEI7O0FBRUE7RUFDRSxjQUFjO0VBQ2Qsc0JBQXNCO0FBQ3hCIiwiZmlsZSI6InNyYy9hcHAvZm9vdGVyL2Zvb3Rlci5jb21wb25lbnQuY3NzIiwic291cmNlc0NvbnRlbnQiOlsiXHJcbi5tb2R1bGUge1xyXG4gIGZvbnQtZmFtaWx5OiAnRG9zaXMnLCBzYW5zLXNlcmlmO1xyXG4gIHBhZGRpbmc6IDE1cHg7XHJcbiAgdGV4dC1hbGlnbjogY2VudGVyO1xyXG4gIGNvbG9yOiAjNDU0NTQ1O1xyXG4gIG1heC1oZWlnaHQ6IDEyMHB4O1xyXG4gIG1pbi13aWR0aDogMTIwcHg7XHJcbiAgYm9yZGVyLXJhZGl1czogMnB4O1xyXG59XHJcblxyXG5hIHtcclxuICBjb2xvcjogIzQ1NDU0NTtcclxuICBtYXJnaW46IDA7XHJcbiAgcG9zaXRpb246IHJlbGF0aXZlO1xyXG59XHJcblxyXG4ubW9kdWxlPmE6YWZ0ZXIge1xyXG4gIGNvbG9yOiAjNDU0NTQ1O1xyXG4gIGNvbnRlbnQ6ICcnO1xyXG4gIHBvc2l0aW9uOiBhYnNvbHV0ZTtcclxuICBsZWZ0OiAyNSU7XHJcbiAgZGlzcGxheTogaW5saW5lLWJsb2NrO1xyXG4gIGhlaWdodDogMWVtO1xyXG4gIHdpZHRoOiA1MCU7XHJcbiAgYm9yZGVyLWJvdHRvbTogMXB4IHNvbGlkO1xyXG4gIG1hcmdpbi10b3A6IDEwcHg7XHJcbiAgb3BhY2l0eTogMDtcclxuICB0cmFuc2l0aW9uOiBvcGFjaXR5IDAuMzVzLCAtd2Via2l0LXRyYW5zZm9ybSAwLjM1cztcclxuICB0cmFuc2l0aW9uOiBvcGFjaXR5IDAuMzVzLCB0cmFuc2Zvcm0gMC4zNXM7XHJcbiAgdHJhbnNpdGlvbjogb3BhY2l0eSAwLjM1cywgdHJhbnNmb3JtIDAuMzVzLCAtd2Via2l0LXRyYW5zZm9ybSAwLjM1cztcclxuICAtd2Via2l0LXRyYW5zZm9ybTogc2NhbGUoMCwxKTtcclxuICB0cmFuc2Zvcm06IHNjYWxlKDAsMSk7XHJcbn1cclxuXHJcbi5tb2R1bGU+YTpob3ZlcjphZnRlciB7XHJcbiAgb3BhY2l0eTogMTtcclxuICAtd2Via2l0LXRyYW5zZm9ybTogc2NhbGUoMSk7XHJcbiAgdHJhbnNmb3JtOiBzY2FsZSgxKTtcclxufVxyXG5cclxuLmNmb290IHtcclxuICAvKmJhY2tncm91bmQtY29sb3I6IHJnYmEoMjE0LCAyMTQsIDIxNCwgMC4xNik7Ki9cclxuICAvKmJvcmRlci10b3A6IDFweCBzb2xpZCAjZDZkNmQ2OyovXHJcbiAgei1pbmRleDogMTtcclxufVxyXG5cclxuI2Zvb3Rlci1yaWdodC1hbGlnbiB7XHJcbiAgbWFyZ2luLXJpZ2h0OiA1MHB4O1xyXG4gIGNvbG9yOiAjYmJiYmJiO1xyXG59XHJcblxyXG4jZm9vdGVyLXJpZ2h0LWFsaWduOmhvdmVyIHtcclxuICBjb2xvcjogIzQ1NDU0NTtcclxuICB0cmFuc2l0aW9uOiBjb2xvciAwLjNzO1xyXG59XHJcbiJdfQ== */"

/***/ }),

/***/ "./src/app/footer/footer.component.html":
/*!**********************************************!*\
  !*** ./src/app/footer/footer.component.html ***!
  \**********************************************/
/*! no static exports found */
/***/ (function(module, exports) {

module.exports = "\r\n<div class=\"cfoot fixed-bottom\">\r\n  <div class=\"row\">\r\n    <div class=\"col-12 no-select module article\">\r\n      <a routerLink=\"/detail/-3\" id=\"footer-right-align\" class=\"pull-right\">about</a>\r\n    </div>\r\n  </div>\r\n</div>\r\n"

/***/ }),

/***/ "./src/app/footer/footer.component.ts":
/*!********************************************!*\
  !*** ./src/app/footer/footer.component.ts ***!
  \********************************************/
/*! exports provided: FooterComponent */
/***/ (function(module, __webpack_exports__, __webpack_require__) {

"use strict";
__webpack_require__.r(__webpack_exports__);
/* harmony export (binding) */ __webpack_require__.d(__webpack_exports__, "FooterComponent", function() { return FooterComponent; });
/* harmony import */ var tslib__WEBPACK_IMPORTED_MODULE_0__ = __webpack_require__(/*! tslib */ "./node_modules/tslib/tslib.es6.js");
/* harmony import */ var _angular_core__WEBPACK_IMPORTED_MODULE_1__ = __webpack_require__(/*! @angular/core */ "./node_modules/@angular/core/fesm5/core.js");


var FooterComponent = /** @class */ (function () {
    function FooterComponent() {
    }
    FooterComponent.prototype.ngOnInit = function () {
    };
    FooterComponent = tslib__WEBPACK_IMPORTED_MODULE_0__["__decorate"]([
        Object(_angular_core__WEBPACK_IMPORTED_MODULE_1__["Component"])({
            selector: 'app-footer',
            template: __webpack_require__(/*! ./footer.component.html */ "./src/app/footer/footer.component.html"),
            styles: [__webpack_require__(/*! ./footer.component.css */ "./src/app/footer/footer.component.css")]
        }),
        tslib__WEBPACK_IMPORTED_MODULE_0__["__metadata"]("design:paramtypes", [])
    ], FooterComponent);
    return FooterComponent;
}());



/***/ }),

/***/ "./src/app/messages/messages.component.css":
/*!*************************************************!*\
  !*** ./src/app/messages/messages.component.css ***!
  \*************************************************/
/*! no static exports found */
/***/ (function(module, exports) {

module.exports = "\n/*# sourceMappingURL=data:application/json;base64,eyJ2ZXJzaW9uIjozLCJzb3VyY2VzIjpbXSwibmFtZXMiOltdLCJtYXBwaW5ncyI6IiIsImZpbGUiOiJzcmMvYXBwL21lc3NhZ2VzL21lc3NhZ2VzLmNvbXBvbmVudC5jc3MifQ== */"

/***/ }),

/***/ "./src/app/messages/messages.component.html":
/*!**************************************************!*\
  !*** ./src/app/messages/messages.component.html ***!
  \**************************************************/
/*! no static exports found */
/***/ (function(module, exports) {

module.exports = "<div *ngIf=\"messageService.messages.length\">\r\n\r\n  <h2>Messages</h2>\r\n  <button class=\"clear\"\r\n          (click)=\"messageService.clear()\">clear</button>\r\n  <div *ngFor=\"let message of messageService.messages\">{{message}}</div>\r\n\r\n</div>\r\n"

/***/ }),

/***/ "./src/app/messages/messages.component.ts":
/*!************************************************!*\
  !*** ./src/app/messages/messages.component.ts ***!
  \************************************************/
/*! exports provided: MessagesComponent */
/***/ (function(module, __webpack_exports__, __webpack_require__) {

"use strict";
__webpack_require__.r(__webpack_exports__);
/* harmony export (binding) */ __webpack_require__.d(__webpack_exports__, "MessagesComponent", function() { return MessagesComponent; });
/* harmony import */ var tslib__WEBPACK_IMPORTED_MODULE_0__ = __webpack_require__(/*! tslib */ "./node_modules/tslib/tslib.es6.js");
/* harmony import */ var _angular_core__WEBPACK_IMPORTED_MODULE_1__ = __webpack_require__(/*! @angular/core */ "./node_modules/@angular/core/fesm5/core.js");
/* harmony import */ var _services_message_service__WEBPACK_IMPORTED_MODULE_2__ = __webpack_require__(/*! ../services/message.service */ "./src/app/services/message.service.ts");



var MessagesComponent = /** @class */ (function () {
    function MessagesComponent(messageService) {
        this.messageService = messageService;
    }
    MessagesComponent.prototype.ngOnInit = function () {
    };
    MessagesComponent = tslib__WEBPACK_IMPORTED_MODULE_0__["__decorate"]([
        Object(_angular_core__WEBPACK_IMPORTED_MODULE_1__["Component"])({
            selector: 'app-messages',
            template: __webpack_require__(/*! ./messages.component.html */ "./src/app/messages/messages.component.html"),
            styles: [__webpack_require__(/*! ./messages.component.css */ "./src/app/messages/messages.component.css")]
        }),
        tslib__WEBPACK_IMPORTED_MODULE_0__["__metadata"]("design:paramtypes", [_services_message_service__WEBPACK_IMPORTED_MODULE_2__["MessageService"]])
    ], MessagesComponent);
    return MessagesComponent;
}());



/***/ }),

/***/ "./src/app/model/article.ts":
/*!**********************************!*\
  !*** ./src/app/model/article.ts ***!
  \**********************************/
/*! exports provided: ParagraphType, Article */
/***/ (function(module, __webpack_exports__, __webpack_require__) {

"use strict";
__webpack_require__.r(__webpack_exports__);
/* harmony export (binding) */ __webpack_require__.d(__webpack_exports__, "ParagraphType", function() { return ParagraphType; });
/* harmony export (binding) */ __webpack_require__.d(__webpack_exports__, "Article", function() { return Article; });
var ParagraphType;
(function (ParagraphType) {
    ParagraphType[ParagraphType["Title"] = 0] = "Title";
    ParagraphType[ParagraphType["Subtitle"] = 1] = "Subtitle";
    ParagraphType[ParagraphType["Paragraph"] = 2] = "Paragraph";
    ParagraphType[ParagraphType["Link"] = 3] = "Link";
    ParagraphType[ParagraphType["Image"] = 4] = "Image";
    ParagraphType[ParagraphType["TwoImages"] = 5] = "TwoImages";
    ParagraphType[ParagraphType["Quotation"] = 6] = "Quotation";
    ParagraphType[ParagraphType["Equation"] = 7] = "Equation";
    ParagraphType[ParagraphType["Lettrine"] = 8] = "Lettrine";
    ParagraphType[ParagraphType["Video"] = 9] = "Video";
    ParagraphType[ParagraphType["Code"] = 10] = "Code";
    ParagraphType[ParagraphType["CollapsibleCode"] = 11] = "CollapsibleCode";
    ParagraphType[ParagraphType["ListItemFR"] = 12] = "ListItemFR";
    ParagraphType[ParagraphType["ListItem"] = 13] = "ListItem";
    ParagraphType[ParagraphType["ClickableDemo"] = 14] = "ClickableDemo";
})(ParagraphType || (ParagraphType = {}));
var Article = /** @class */ (function () {
    function Article(copy) {
        this.id = copy.id;
        this.title = copy.title;
        this.author = copy.author;
        this.style = copy.style;
        this.date = copy.date;
        this.body = copy.body;
    }
    return Article;
}());



/***/ }),

/***/ "./src/app/navbar/navbar.component.css":
/*!*********************************************!*\
  !*** ./src/app/navbar/navbar.component.css ***!
  \*********************************************/
/*! no static exports found */
/***/ (function(module, exports) {

module.exports = "h1 {\r\n  font-size: 1.2em;\r\n  color: #999;\r\n  margin-bottom: 0;\r\n}\r\n\r\nh2 {\r\n  font-size: 2em;\r\n  margin-top: 0;\r\n  padding-top: 0;\r\n}\r\n\r\nnav a {\r\n  color: #454545;\r\n  padding: 5px 10px;\r\n  margin: 10px 2px;\r\n  text-decoration: none;\r\n  display: inline-block;\r\n  background-color: #eee;\r\n  border-radius: 2px;\r\n}\r\n\r\nnav a:hover {\r\n  color: #000;\r\n  background-color: #d6d6d6;\r\n}\r\n\r\nnav a.active {\r\n}\r\n\r\n#navbar {\r\n  border-bottom: solid 1px #d6d6d6;\r\n}\r\n\r\n#dropdown-search-wrapper,\r\n#dropdown-scroll-wrapper,\r\n#dropdown-about-wrapper\r\n{\r\n  margin-top: -200px;\r\n  height: 215px;\r\n  width: 45px;\r\n  transition: transform 0.3s, -webkit-transform 0.3s;\r\n}\r\n\r\n#about-col,\r\n#dropdown-scroll-wrapper,\r\n#dropdown-about-wrapper {\r\n  z-index: 2;\r\n}\r\n\r\n#search-col,\r\n#dropdown-search-wrapper {\r\n  z-index: 1;\r\n}\r\n\r\n#dropdown-scroll-wrapper,\r\n#dropdown-scroll-wrapper-mobile {\r\n  cursor: pointer;\r\n  background-color: rgba(210, 101, 27, 0.64);\r\n}\r\n\r\n#dropdown-scroll-wrapper-mobile,\r\n#dropdown-about-wrapper-mobile\r\n{\r\n  padding: 0;\r\n}\r\n\r\n#dropdown-scroll-wrapper-mobile {\r\n  margin-right: 5px;\r\n}\r\n\r\n#dropdown-search-wrapper,\r\n#dropdown-search-wrapper-mobile {\r\n  background-color: rgba(35, 146, 41, 0.38);\r\n}\r\n\r\n#dropdown-about-wrapper,\r\n#dropdown-about-wrapper-mobile {\r\n  cursor:pointer;\r\n  background-color: rgba(55, 0, 146, 0.38);\r\n}\r\n\r\n#dropdown-search-wrapper>#dropdown-search-engine {\r\n  visibility: hidden;\r\n}\r\n\r\n#dropdown-search-wrapper:hover>#dropdown-search-engine {\r\n  visibility: visible;\r\n}\r\n\r\n#dropdown-scroll-wrapper:hover,\r\n#dropdown-search-wrapper:hover,\r\n#dropdown-about-wrapper:hover\r\n{\r\n  height: 215px;\r\n  width: 45px;\r\n  transition: transform 0s, -webkit-transform 0s;\r\n  transform: translateY(200px);\r\n  -webkit-transform: translateY(200px);\r\n}\r\n\r\n#dropdown-scroll-image,\r\n#dropdown-about-image\r\n{\r\n  left: 0;\r\n  margin-left: -1px;\r\n  cursor: pointer;\r\n  height: 200px;\r\n  width: 47px;\r\n  border-left: solid 1px transparent;\r\n  border-right: solid 1px transparent;\r\n}\r\n\r\n#dropdown-about-image-mobile,\r\n#dropdown-scroll-image-mobile {\r\n  left: 0;\r\n  margin-left: -1px;\r\n  cursor: pointer;\r\n  height: auto;\r\n  width: 102%;\r\n  /*object-fit: fill;*/\r\n  border-left: solid 1px transparent;\r\n  border-right: solid 1px transparent;\r\n}\r\n\r\n#dropdown-scroll-image-mobile {\r\n  height: 50px;\r\n}\r\n\r\n#dropdown-about-image-mobile {\r\n  height: 50px;\r\n}\r\n\r\n#dropdown-scroll-wrapper-mobile {\r\n  margin-top: 0px;\r\n  height: 65px;\r\n  width: 45px;\r\n}\r\n\r\n#dropdown-about-wrapper-mobile {\r\n  margin-top: 0px;\r\n  height: 65px;\r\n  width: 45px;\r\n}\r\n\r\n#dropdown-search-image {\r\n  left: 0;\r\n  margin-left: -1px;\r\n  height: 200px;\r\n  width: 102px;\r\n  border-left: solid 1px white;\r\n}\r\n\r\n#dropdown-search-engine {\r\n  position: relative;\r\n  top: -200px;\r\n}\r\n\r\n.dropdown-title {\r\n  margin-top: -5px;\r\n  text-align: center;\r\n  font-size: 10pt;\r\n  font-family: 'Dosis', sans-serif;\r\n}\r\n\r\n.dropdown-title-mobile {\r\n  margin-top: -5px;\r\n  text-align: center;\r\n  font-size: 10pt;\r\n  font-family: 'Dosis', sans-serif;\r\n}\r\n\r\n.dropdown-title-find {\r\n  margin-top: -37px;\r\n  text-align: center;\r\n  font-size: 10pt;\r\n  font-family: 'Dosis', sans-serif;\r\n}\r\n\r\n@media (max-width: 934px) {\r\n  #dropdown-search-wrapper {\r\n    display: none;\r\n  }\r\n}\r\n\r\n@media (min-width: 934px) {\r\n  .display-if-lesser-than-934p {\r\n    display: none;\r\n  }\r\n}\r\n\r\n.display-if-lesser-than-934p {\r\n  z-index: -999;\r\n}\r\n\r\n/*# sourceMappingURL=data:application/json;base64,eyJ2ZXJzaW9uIjozLCJzb3VyY2VzIjpbInNyYy9hcHAvbmF2YmFyL25hdmJhci5jb21wb25lbnQuY3NzIl0sIm5hbWVzIjpbXSwibWFwcGluZ3MiOiJBQUFBO0VBQ0UsZ0JBQWdCO0VBQ2hCLFdBQVc7RUFDWCxnQkFBZ0I7QUFDbEI7O0FBRUE7RUFDRSxjQUFjO0VBQ2QsYUFBYTtFQUNiLGNBQWM7QUFDaEI7O0FBRUE7RUFDRSxjQUFjO0VBQ2QsaUJBQWlCO0VBQ2pCLGdCQUFnQjtFQUNoQixxQkFBcUI7RUFDckIscUJBQXFCO0VBQ3JCLHNCQUFzQjtFQUN0QixrQkFBa0I7QUFDcEI7O0FBRUE7RUFDRSxXQUFXO0VBQ1gseUJBQXlCO0FBQzNCOztBQUVBO0FBQ0E7O0FBRUE7RUFDRSxnQ0FBZ0M7QUFDbEM7O0FBRUE7Ozs7RUFJRSxrQkFBa0I7RUFDbEIsYUFBYTtFQUNiLFdBQVc7RUFDWCxrREFBa0Q7QUFDcEQ7O0FBRUE7OztFQUdFLFVBQVU7QUFDWjs7QUFDQTs7RUFFRSxVQUFVO0FBQ1o7O0FBRUE7O0VBRUUsZUFBZTtFQUNmLDBDQUEwQztBQUM1Qzs7QUFDQTs7O0VBR0UsVUFBVTtBQUNaOztBQUNBO0VBQ0UsaUJBQWlCO0FBQ25COztBQUVBOztFQUVFLHlDQUF5QztBQUMzQzs7QUFFQTs7RUFFRSxjQUFjO0VBQ2Qsd0NBQXdDO0FBQzFDOztBQUVBO0VBQ0Usa0JBQWtCO0FBQ3BCOztBQUNBO0VBQ0UsbUJBQW1CO0FBQ3JCOztBQUVBOzs7O0VBSUUsYUFBYTtFQUNiLFdBQVc7RUFDWCw4Q0FBOEM7RUFDOUMsNEJBQTRCO0VBQzVCLG9DQUFvQztBQUN0Qzs7QUFFQTs7O0VBR0UsT0FBTztFQUNQLGlCQUFpQjtFQUNqQixlQUFlO0VBQ2YsYUFBYTtFQUNiLFdBQVc7RUFDWCxrQ0FBa0M7RUFDbEMsbUNBQW1DO0FBQ3JDOztBQUVBOztFQUVFLE9BQU87RUFDUCxpQkFBaUI7RUFDakIsZUFBZTtFQUNmLFlBQVk7RUFDWixXQUFXO0VBQ1gsb0JBQW9CO0VBQ3BCLGtDQUFrQztFQUNsQyxtQ0FBbUM7QUFDckM7O0FBRUE7RUFDRSxZQUFZO0FBQ2Q7O0FBRUE7RUFDRSxZQUFZO0FBQ2Q7O0FBRUE7RUFDRSxlQUFlO0VBQ2YsWUFBWTtFQUNaLFdBQVc7QUFDYjs7QUFFQTtFQUNFLGVBQWU7RUFDZixZQUFZO0VBQ1osV0FBVztBQUNiOztBQUVBO0VBQ0UsT0FBTztFQUNQLGlCQUFpQjtFQUNqQixhQUFhO0VBQ2IsWUFBWTtFQUNaLDRCQUE0QjtBQUM5Qjs7QUFFQTtFQUNFLGtCQUFrQjtFQUNsQixXQUFXO0FBQ2I7O0FBRUE7RUFDRSxnQkFBZ0I7RUFDaEIsa0JBQWtCO0VBQ2xCLGVBQWU7RUFDZixnQ0FBZ0M7QUFDbEM7O0FBRUE7RUFDRSxnQkFBZ0I7RUFDaEIsa0JBQWtCO0VBQ2xCLGVBQWU7RUFDZixnQ0FBZ0M7QUFDbEM7O0FBRUE7RUFDRSxpQkFBaUI7RUFDakIsa0JBQWtCO0VBQ2xCLGVBQWU7RUFDZixnQ0FBZ0M7QUFDbEM7O0FBRUE7RUFDRTtJQUNFLGFBQWE7RUFDZjtBQUNGOztBQUVBO0VBQ0U7SUFDRSxhQUFhO0VBQ2Y7QUFDRjs7QUFFQTtFQUNFLGFBQWE7QUFDZiIsImZpbGUiOiJzcmMvYXBwL25hdmJhci9uYXZiYXIuY29tcG9uZW50LmNzcyIsInNvdXJjZXNDb250ZW50IjpbImgxIHtcclxuICBmb250LXNpemU6IDEuMmVtO1xyXG4gIGNvbG9yOiAjOTk5O1xyXG4gIG1hcmdpbi1ib3R0b206IDA7XHJcbn1cclxuXHJcbmgyIHtcclxuICBmb250LXNpemU6IDJlbTtcclxuICBtYXJnaW4tdG9wOiAwO1xyXG4gIHBhZGRpbmctdG9wOiAwO1xyXG59XHJcblxyXG5uYXYgYSB7XHJcbiAgY29sb3I6ICM0NTQ1NDU7XHJcbiAgcGFkZGluZzogNXB4IDEwcHg7XHJcbiAgbWFyZ2luOiAxMHB4IDJweDtcclxuICB0ZXh0LWRlY29yYXRpb246IG5vbmU7XHJcbiAgZGlzcGxheTogaW5saW5lLWJsb2NrO1xyXG4gIGJhY2tncm91bmQtY29sb3I6ICNlZWU7XHJcbiAgYm9yZGVyLXJhZGl1czogMnB4O1xyXG59XHJcblxyXG5uYXYgYTpob3ZlciB7XHJcbiAgY29sb3I6ICMwMDA7XHJcbiAgYmFja2dyb3VuZC1jb2xvcjogI2Q2ZDZkNjtcclxufVxyXG5cclxubmF2IGEuYWN0aXZlIHtcclxufVxyXG5cclxuI25hdmJhciB7XHJcbiAgYm9yZGVyLWJvdHRvbTogc29saWQgMXB4ICNkNmQ2ZDY7XHJcbn1cclxuXHJcbiNkcm9wZG93bi1zZWFyY2gtd3JhcHBlcixcclxuI2Ryb3Bkb3duLXNjcm9sbC13cmFwcGVyLFxyXG4jZHJvcGRvd24tYWJvdXQtd3JhcHBlclxyXG57XHJcbiAgbWFyZ2luLXRvcDogLTIwMHB4O1xyXG4gIGhlaWdodDogMjE1cHg7XHJcbiAgd2lkdGg6IDQ1cHg7XHJcbiAgdHJhbnNpdGlvbjogdHJhbnNmb3JtIDAuM3MsIC13ZWJraXQtdHJhbnNmb3JtIDAuM3M7XHJcbn1cclxuXHJcbiNhYm91dC1jb2wsXHJcbiNkcm9wZG93bi1zY3JvbGwtd3JhcHBlcixcclxuI2Ryb3Bkb3duLWFib3V0LXdyYXBwZXIge1xyXG4gIHotaW5kZXg6IDI7XHJcbn1cclxuI3NlYXJjaC1jb2wsXHJcbiNkcm9wZG93bi1zZWFyY2gtd3JhcHBlciB7XHJcbiAgei1pbmRleDogMTtcclxufVxyXG5cclxuI2Ryb3Bkb3duLXNjcm9sbC13cmFwcGVyLFxyXG4jZHJvcGRvd24tc2Nyb2xsLXdyYXBwZXItbW9iaWxlIHtcclxuICBjdXJzb3I6IHBvaW50ZXI7XHJcbiAgYmFja2dyb3VuZC1jb2xvcjogcmdiYSgyMTAsIDEwMSwgMjcsIDAuNjQpO1xyXG59XHJcbiNkcm9wZG93bi1zY3JvbGwtd3JhcHBlci1tb2JpbGUsXHJcbiNkcm9wZG93bi1hYm91dC13cmFwcGVyLW1vYmlsZVxyXG57XHJcbiAgcGFkZGluZzogMDtcclxufVxyXG4jZHJvcGRvd24tc2Nyb2xsLXdyYXBwZXItbW9iaWxlIHtcclxuICBtYXJnaW4tcmlnaHQ6IDVweDtcclxufVxyXG5cclxuI2Ryb3Bkb3duLXNlYXJjaC13cmFwcGVyLFxyXG4jZHJvcGRvd24tc2VhcmNoLXdyYXBwZXItbW9iaWxlIHtcclxuICBiYWNrZ3JvdW5kLWNvbG9yOiByZ2JhKDM1LCAxNDYsIDQxLCAwLjM4KTtcclxufVxyXG5cclxuI2Ryb3Bkb3duLWFib3V0LXdyYXBwZXIsXHJcbiNkcm9wZG93bi1hYm91dC13cmFwcGVyLW1vYmlsZSB7XHJcbiAgY3Vyc29yOnBvaW50ZXI7XHJcbiAgYmFja2dyb3VuZC1jb2xvcjogcmdiYSg1NSwgMCwgMTQ2LCAwLjM4KTtcclxufVxyXG5cclxuI2Ryb3Bkb3duLXNlYXJjaC13cmFwcGVyPiNkcm9wZG93bi1zZWFyY2gtZW5naW5lIHtcclxuICB2aXNpYmlsaXR5OiBoaWRkZW47XHJcbn1cclxuI2Ryb3Bkb3duLXNlYXJjaC13cmFwcGVyOmhvdmVyPiNkcm9wZG93bi1zZWFyY2gtZW5naW5lIHtcclxuICB2aXNpYmlsaXR5OiB2aXNpYmxlO1xyXG59XHJcblxyXG4jZHJvcGRvd24tc2Nyb2xsLXdyYXBwZXI6aG92ZXIsXHJcbiNkcm9wZG93bi1zZWFyY2gtd3JhcHBlcjpob3ZlcixcclxuI2Ryb3Bkb3duLWFib3V0LXdyYXBwZXI6aG92ZXJcclxue1xyXG4gIGhlaWdodDogMjE1cHg7XHJcbiAgd2lkdGg6IDQ1cHg7XHJcbiAgdHJhbnNpdGlvbjogdHJhbnNmb3JtIDBzLCAtd2Via2l0LXRyYW5zZm9ybSAwcztcclxuICB0cmFuc2Zvcm06IHRyYW5zbGF0ZVkoMjAwcHgpO1xyXG4gIC13ZWJraXQtdHJhbnNmb3JtOiB0cmFuc2xhdGVZKDIwMHB4KTtcclxufVxyXG5cclxuI2Ryb3Bkb3duLXNjcm9sbC1pbWFnZSxcclxuI2Ryb3Bkb3duLWFib3V0LWltYWdlXHJcbntcclxuICBsZWZ0OiAwO1xyXG4gIG1hcmdpbi1sZWZ0OiAtMXB4O1xyXG4gIGN1cnNvcjogcG9pbnRlcjtcclxuICBoZWlnaHQ6IDIwMHB4O1xyXG4gIHdpZHRoOiA0N3B4O1xyXG4gIGJvcmRlci1sZWZ0OiBzb2xpZCAxcHggdHJhbnNwYXJlbnQ7XHJcbiAgYm9yZGVyLXJpZ2h0OiBzb2xpZCAxcHggdHJhbnNwYXJlbnQ7XHJcbn1cclxuXHJcbiNkcm9wZG93bi1hYm91dC1pbWFnZS1tb2JpbGUsXHJcbiNkcm9wZG93bi1zY3JvbGwtaW1hZ2UtbW9iaWxlIHtcclxuICBsZWZ0OiAwO1xyXG4gIG1hcmdpbi1sZWZ0OiAtMXB4O1xyXG4gIGN1cnNvcjogcG9pbnRlcjtcclxuICBoZWlnaHQ6IGF1dG87XHJcbiAgd2lkdGg6IDEwMiU7XHJcbiAgLypvYmplY3QtZml0OiBmaWxsOyovXHJcbiAgYm9yZGVyLWxlZnQ6IHNvbGlkIDFweCB0cmFuc3BhcmVudDtcclxuICBib3JkZXItcmlnaHQ6IHNvbGlkIDFweCB0cmFuc3BhcmVudDtcclxufVxyXG5cclxuI2Ryb3Bkb3duLXNjcm9sbC1pbWFnZS1tb2JpbGUge1xyXG4gIGhlaWdodDogNTBweDtcclxufVxyXG5cclxuI2Ryb3Bkb3duLWFib3V0LWltYWdlLW1vYmlsZSB7XHJcbiAgaGVpZ2h0OiA1MHB4O1xyXG59XHJcblxyXG4jZHJvcGRvd24tc2Nyb2xsLXdyYXBwZXItbW9iaWxlIHtcclxuICBtYXJnaW4tdG9wOiAwcHg7XHJcbiAgaGVpZ2h0OiA2NXB4O1xyXG4gIHdpZHRoOiA0NXB4O1xyXG59XHJcblxyXG4jZHJvcGRvd24tYWJvdXQtd3JhcHBlci1tb2JpbGUge1xyXG4gIG1hcmdpbi10b3A6IDBweDtcclxuICBoZWlnaHQ6IDY1cHg7XHJcbiAgd2lkdGg6IDQ1cHg7XHJcbn1cclxuXHJcbiNkcm9wZG93bi1zZWFyY2gtaW1hZ2Uge1xyXG4gIGxlZnQ6IDA7XHJcbiAgbWFyZ2luLWxlZnQ6IC0xcHg7XHJcbiAgaGVpZ2h0OiAyMDBweDtcclxuICB3aWR0aDogMTAycHg7XHJcbiAgYm9yZGVyLWxlZnQ6IHNvbGlkIDFweCB3aGl0ZTtcclxufVxyXG5cclxuI2Ryb3Bkb3duLXNlYXJjaC1lbmdpbmUge1xyXG4gIHBvc2l0aW9uOiByZWxhdGl2ZTtcclxuICB0b3A6IC0yMDBweDtcclxufVxyXG5cclxuLmRyb3Bkb3duLXRpdGxlIHtcclxuICBtYXJnaW4tdG9wOiAtNXB4O1xyXG4gIHRleHQtYWxpZ246IGNlbnRlcjtcclxuICBmb250LXNpemU6IDEwcHQ7XHJcbiAgZm9udC1mYW1pbHk6ICdEb3NpcycsIHNhbnMtc2VyaWY7XHJcbn1cclxuXHJcbi5kcm9wZG93bi10aXRsZS1tb2JpbGUge1xyXG4gIG1hcmdpbi10b3A6IC01cHg7XHJcbiAgdGV4dC1hbGlnbjogY2VudGVyO1xyXG4gIGZvbnQtc2l6ZTogMTBwdDtcclxuICBmb250LWZhbWlseTogJ0Rvc2lzJywgc2Fucy1zZXJpZjtcclxufVxyXG5cclxuLmRyb3Bkb3duLXRpdGxlLWZpbmQge1xyXG4gIG1hcmdpbi10b3A6IC0zN3B4O1xyXG4gIHRleHQtYWxpZ246IGNlbnRlcjtcclxuICBmb250LXNpemU6IDEwcHQ7XHJcbiAgZm9udC1mYW1pbHk6ICdEb3NpcycsIHNhbnMtc2VyaWY7XHJcbn1cclxuXHJcbkBtZWRpYSAobWF4LXdpZHRoOiA5MzRweCkge1xyXG4gICNkcm9wZG93bi1zZWFyY2gtd3JhcHBlciB7XHJcbiAgICBkaXNwbGF5OiBub25lO1xyXG4gIH1cclxufVxyXG5cclxuQG1lZGlhIChtaW4td2lkdGg6IDkzNHB4KSB7XHJcbiAgLmRpc3BsYXktaWYtbGVzc2VyLXRoYW4tOTM0cCB7XHJcbiAgICBkaXNwbGF5OiBub25lO1xyXG4gIH1cclxufVxyXG5cclxuLmRpc3BsYXktaWYtbGVzc2VyLXRoYW4tOTM0cCB7XHJcbiAgei1pbmRleDogLTk5OTtcclxufVxyXG4iXX0= */"

/***/ }),

/***/ "./src/app/navbar/navbar.component.html":
/*!**********************************************!*\
  !*** ./src/app/navbar/navbar.component.html ***!
  \**********************************************/
/*! no static exports found */
/***/ (function(module, exports) {

module.exports = "<div *ngIf=\"currentTitle\" class=\"col-12 fixed-top\" id=\"navbar\">\r\n  <div class=\"row col-12\">\r\n    <div class=\"col-2 hide-if-sm\"></div>\r\n    <div class=\"col-8\">\r\n      <nav>\r\n        <span class=\"nobreak\">\r\n          <a (click)=\"scrollToTop()\"><fa-icon [icon]=\"faAngleDoubleUp\"></fa-icon></a>\r\n          <a>{{currentTitle}}</a>\r\n        </span>\r\n      </nav>\r\n    </div>\r\n  </div>\r\n</div>\r\n\r\n<div *ngIf=\"!currentTitle\" class=\"col-12 fixed-top\">\r\n\r\n    <div class=\"row col-12\">\r\n\r\n      <!-- Desktop header -->\r\n      <div class=\"col-1 hide-if-sm\"></div>\r\n\r\n      <div class=\"col-11 row hide-if-sm\">\r\n        <!--<div class=\"col-1\"></div>-->\r\n        <div class=\"col-1\">\r\n          <div id=\"dropdown-scroll-wrapper\">\r\n            <!--<div id=\"color-wrapper\">-->\r\n            <img routerLink=\"/\" id=\"dropdown-scroll-image\" src=\"assets/img/frontpage/bn.jpg\"/>\r\n            <!--</div>-->\r\n            <p routerLink=\"/\" class=\"dropdown-title\">m. s.</p>\r\n          </div>\r\n        </div>\r\n\r\n        <div class=\"col-1 display-if-lesser-than-934p\">\r\n        </div>\r\n\r\n        <div class=\"col-1\" id=\"about-col\">\r\n          <div id=\"dropdown-about-wrapper\">\r\n            <img routerLink=\"/detail/0\" id=\"dropdown-about-image\" src=\"assets/img/frontpage/ab.jpg\"/>\r\n            <p routerLink=\"/detail/0\" class=\"dropdown-title\">about</p>\r\n          </div>\r\n        </div>\r\n\r\n        <div class=\"col-1\" id=\"search-col\">\r\n          <div id=\"dropdown-search-wrapper\">\r\n            <img id=\"dropdown-search-image\" src=\"assets/img/frontpage/sc.jpg\"/>\r\n            <app-article-search id=\"dropdown-search-engine\"></app-article-search>\r\n            <p class=\"dropdown-title-find\">search</p>\r\n          </div>\r\n        </div>\r\n\r\n        <!--<nav>-->\r\n        <!--<a routerLink=\"/dashboard\">test</a>-->\r\n        <!--</nav>-->\r\n      </div>\r\n      <!-- / Desktop header -->\r\n\r\n      <!-- Mobile header -->\r\n      <div class=\"reset-mainstyle hide-if-larger-than-sm col-12\">\r\n\r\n        <div class=\"reset-mainstyle row col-12\">\r\n          <!--<div class=\"col-1\"></div>-->\r\n          <div class=\"col-2\" id=\"dropdown-scroll-wrapper-mobile\">\r\n            <img routerLink=\"/\" id=\"dropdown-scroll-image-mobile\" src=\"assets/img/frontpage/bn-mobile.jpg\"/>\r\n            <!--</div>-->\r\n            <p routerLink=\"/\" class=\"dropdown-title-mobile\">m. s.</p>\r\n          </div>\r\n\r\n          <div class=\"col-2\" id=\"dropdown-about-wrapper-mobile\">\r\n            <div>\r\n              <img routerLink=\"/detail/-3\" id=\"dropdown-about-image-mobile\"  src=\"assets/img/frontpage/ab-mobile.jpg\"/>\r\n              <p routerLink=\"/detail/-3\" class=\"dropdown-title-mobile\">about</p>\r\n            </div>\r\n          </div>\r\n        </div>\r\n\r\n      </div>\r\n      <!-- / Mobile header -->\r\n\r\n    </div>\r\n\r\n</div>\r\n"

/***/ }),

/***/ "./src/app/navbar/navbar.component.ts":
/*!********************************************!*\
  !*** ./src/app/navbar/navbar.component.ts ***!
  \********************************************/
/*! exports provided: NavbarComponent */
/***/ (function(module, __webpack_exports__, __webpack_require__) {

"use strict";
__webpack_require__.r(__webpack_exports__);
/* harmony export (binding) */ __webpack_require__.d(__webpack_exports__, "NavbarComponent", function() { return NavbarComponent; });
/* harmony import */ var tslib__WEBPACK_IMPORTED_MODULE_0__ = __webpack_require__(/*! tslib */ "./node_modules/tslib/tslib.es6.js");
/* harmony import */ var _angular_core__WEBPACK_IMPORTED_MODULE_1__ = __webpack_require__(/*! @angular/core */ "./node_modules/@angular/core/fesm5/core.js");
/* harmony import */ var _fortawesome_free_solid_svg_icons__WEBPACK_IMPORTED_MODULE_2__ = __webpack_require__(/*! @fortawesome/free-solid-svg-icons */ "./node_modules/@fortawesome/free-solid-svg-icons/index.es.js");



var NavbarComponent = /** @class */ (function () {
    function NavbarComponent() {
        this.faAngleDoubleUp = _fortawesome_free_solid_svg_icons__WEBPACK_IMPORTED_MODULE_2__["faAngleDoubleUp"];
    }
    NavbarComponent.prototype.ngOnInit = function () { };
    NavbarComponent.prototype.ngOnChanges = function () { };
    NavbarComponent.prototype.scrollToTop = function () {
        window.scrollTo(0, 0);
        console.log('Scrolling to top.');
    };
    tslib__WEBPACK_IMPORTED_MODULE_0__["__decorate"]([
        Object(_angular_core__WEBPACK_IMPORTED_MODULE_1__["Input"])('currentTitle'),
        tslib__WEBPACK_IMPORTED_MODULE_0__["__metadata"]("design:type", String)
    ], NavbarComponent.prototype, "currentTitle", void 0);
    tslib__WEBPACK_IMPORTED_MODULE_0__["__decorate"]([
        Object(_angular_core__WEBPACK_IMPORTED_MODULE_1__["Input"])('currentId'),
        tslib__WEBPACK_IMPORTED_MODULE_0__["__metadata"]("design:type", String)
    ], NavbarComponent.prototype, "currentId", void 0);
    NavbarComponent = tslib__WEBPACK_IMPORTED_MODULE_0__["__decorate"]([
        Object(_angular_core__WEBPACK_IMPORTED_MODULE_1__["Component"])({
            selector: 'app-navbar',
            template: __webpack_require__(/*! ./navbar.component.html */ "./src/app/navbar/navbar.component.html"),
            styles: [__webpack_require__(/*! ./navbar.component.css */ "./src/app/navbar/navbar.component.css")]
        }),
        tslib__WEBPACK_IMPORTED_MODULE_0__["__metadata"]("design:paramtypes", [])
    ], NavbarComponent);
    return NavbarComponent;
}());



/***/ }),

/***/ "./src/app/services/article.service.ts":
/*!*********************************************!*\
  !*** ./src/app/services/article.service.ts ***!
  \*********************************************/
/*! exports provided: ArticleService */
/***/ (function(module, __webpack_exports__, __webpack_require__) {

"use strict";
__webpack_require__.r(__webpack_exports__);
/* harmony export (binding) */ __webpack_require__.d(__webpack_exports__, "ArticleService", function() { return ArticleService; });
/* harmony import */ var tslib__WEBPACK_IMPORTED_MODULE_0__ = __webpack_require__(/*! tslib */ "./node_modules/tslib/tslib.es6.js");
/* harmony import */ var _angular_core__WEBPACK_IMPORTED_MODULE_1__ = __webpack_require__(/*! @angular/core */ "./node_modules/@angular/core/fesm5/core.js");
/* harmony import */ var rxjs__WEBPACK_IMPORTED_MODULE_2__ = __webpack_require__(/*! rxjs */ "./node_modules/rxjs/_esm5/index.js");
/* harmony import */ var _message_service__WEBPACK_IMPORTED_MODULE_3__ = __webpack_require__(/*! ./message.service */ "./src/app/services/message.service.ts");
/* harmony import */ var _angular_common_http__WEBPACK_IMPORTED_MODULE_4__ = __webpack_require__(/*! @angular/common/http */ "./node_modules/@angular/common/fesm5/http.js");
/* harmony import */ var rxjs_operators__WEBPACK_IMPORTED_MODULE_5__ = __webpack_require__(/*! rxjs/operators */ "./node_modules/rxjs/_esm5/operators/index.js");






var httpOptions = {
    headers: new _angular_common_http__WEBPACK_IMPORTED_MODULE_4__["HttpHeaders"]({ 'Content-Type': 'application/json' })
};
var ArticleService = /** @class */ (function () {
    function ArticleService(http, messageService) {
        this.http = http;
        this.messageService = messageService;
        this.dbUrl = 'https://raw.githubusercontent.com/madblade/folio-db/master/db/';
        this.articlesUrl = 'all-titles.json';
        this.categoriesUrl = 'categories.json';
    }
    ArticleService.prototype.getCategories = function () {
        var _this = this;
        var url = this.dbUrl + this.categoriesUrl;
        return this.http.get(url)
            .pipe(Object(rxjs_operators__WEBPACK_IMPORTED_MODULE_5__["tap"])(function () { return _this.log('fetched categories'); }), Object(rxjs_operators__WEBPACK_IMPORTED_MODULE_5__["catchError"])(this.handleError('getCategories', [])));
    };
    ArticleService.prototype.getArticles = function (cat) {
        var _this = this;
        var url = this.dbUrl + cat + '.json';
        return this.http.get(url)
            .pipe(Object(rxjs_operators__WEBPACK_IMPORTED_MODULE_5__["tap"])(function () { return _this.log('fetched articles'); }), Object(rxjs_operators__WEBPACK_IMPORTED_MODULE_5__["catchError"])(this.handleError('getArticles', [])));
    };
    ArticleService.prototype.getArticle = function (id) {
        var _this = this;
        var url = this.dbUrl + 'articles/' + id + '.json';
        return this.http.get(url)
            .pipe(Object(rxjs_operators__WEBPACK_IMPORTED_MODULE_5__["tap"])(function () { return _this.log("fetched article " + id); }), Object(rxjs_operators__WEBPACK_IMPORTED_MODULE_5__["catchError"])(this.handleError('getArticle id=${id}')));
    };
    // GET articles whose name contains search term
    ArticleService.prototype.searchArticleTitles = function (term) {
        var _this = this;
        if (!term.trim()) {
            return Object(rxjs__WEBPACK_IMPORTED_MODULE_2__["of"])([]); // Empty article array
        }
        var url = this.dbUrl + this.articlesUrl;
        return this.http.get(url)
            .pipe(Object(rxjs_operators__WEBPACK_IMPORTED_MODULE_5__["map"])(function (articles) {
            var filtered = [];
            for (var i = 0; i < articles.length; ++i) {
                if (articles[i].title.toLowerCase().includes(term.toLowerCase())) {
                    filtered.push(articles[i]);
                }
            }
            return filtered;
        }), Object(rxjs_operators__WEBPACK_IMPORTED_MODULE_5__["catchError"])(this.handleError('searchArticles', [])))
            .pipe(Object(rxjs_operators__WEBPACK_IMPORTED_MODULE_5__["tap"])(function () { return _this.log("found articles matching \"" + term + "\""); }), Object(rxjs_operators__WEBPACK_IMPORTED_MODULE_5__["catchError"])(this.handleError('searchArticles', [])));
    };
    ArticleService.prototype.log = function (message) {
        this.messageService.add("ArticleService: " + message);
    };
    /**
     * Handle Http operation that failed.
     * Let the app continue.
     * @param operation - name of the operation that failed
     * @param result - optional value to return as the observable result
     */
    ArticleService.prototype.handleError = function (operation, result) {
        var _this = this;
        if (operation === void 0) { operation = 'operation'; }
        return function (error) {
            // [Enhancement] Should send the error to remote logging infrastructure
            console.error(error); // log to console instead
            // Should transform error for user consumption
            _this.log(operation + " failed: " + error.message);
            // [Enhancement] Let the app keep running by returning an empty result
            return Object(rxjs__WEBPACK_IMPORTED_MODULE_2__["of"])(result);
        };
    };
    ArticleService = tslib__WEBPACK_IMPORTED_MODULE_0__["__decorate"]([
        Object(_angular_core__WEBPACK_IMPORTED_MODULE_1__["Injectable"])({
            providedIn: 'root'
        }),
        tslib__WEBPACK_IMPORTED_MODULE_0__["__metadata"]("design:paramtypes", [_angular_common_http__WEBPACK_IMPORTED_MODULE_4__["HttpClient"],
            _message_service__WEBPACK_IMPORTED_MODULE_3__["MessageService"]])
    ], ArticleService);
    return ArticleService;
}());



/***/ }),

/***/ "./src/app/services/message.service.ts":
/*!*********************************************!*\
  !*** ./src/app/services/message.service.ts ***!
  \*********************************************/
/*! exports provided: MessageService */
/***/ (function(module, __webpack_exports__, __webpack_require__) {

"use strict";
__webpack_require__.r(__webpack_exports__);
/* harmony export (binding) */ __webpack_require__.d(__webpack_exports__, "MessageService", function() { return MessageService; });
/* harmony import */ var tslib__WEBPACK_IMPORTED_MODULE_0__ = __webpack_require__(/*! tslib */ "./node_modules/tslib/tslib.es6.js");
/* harmony import */ var _angular_core__WEBPACK_IMPORTED_MODULE_1__ = __webpack_require__(/*! @angular/core */ "./node_modules/@angular/core/fesm5/core.js");


var MessageService = /** @class */ (function () {
    function MessageService() {
        this.messages = [];
    }
    MessageService.prototype.add = function (message) {
        this.messages.push(message);
    };
    MessageService.prototype.clear = function () {
        this.messages = [];
    };
    MessageService = tslib__WEBPACK_IMPORTED_MODULE_0__["__decorate"]([
        Object(_angular_core__WEBPACK_IMPORTED_MODULE_1__["Injectable"])({
            providedIn: 'root'
        }),
        tslib__WEBPACK_IMPORTED_MODULE_0__["__metadata"]("design:paramtypes", [])
    ], MessageService);
    return MessageService;
}());



/***/ }),

/***/ "./src/environments/environment.ts":
/*!*****************************************!*\
  !*** ./src/environments/environment.ts ***!
  \*****************************************/
/*! exports provided: environment */
/***/ (function(module, __webpack_exports__, __webpack_require__) {

"use strict";
__webpack_require__.r(__webpack_exports__);
/* harmony export (binding) */ __webpack_require__.d(__webpack_exports__, "environment", function() { return environment; });
// This file can be replaced during build by using the `fileReplacements` array.
// `ng build --prod` replaces `environment.ts` with `environment.prod.ts`.
// The list of file replacements can be found in `angular.json`.
var environment = {
    production: false
};
/*
 * For easier debugging in development mode, you can import the following file
 * to ignore zone related error stack frames such as `zone.run`, `zoneDelegate.invokeTask`.
 *
 * This import should be commented out in production mode because it will have a negative impact
 * on performance if an error is thrown.
 */
// import 'zone.js/dist/zone-error';  // Included with Angular CLI.


/***/ }),

/***/ "./src/main.ts":
/*!*********************!*\
  !*** ./src/main.ts ***!
  \*********************/
/*! no exports provided */
/***/ (function(module, __webpack_exports__, __webpack_require__) {

"use strict";
__webpack_require__.r(__webpack_exports__);
/* harmony import */ var _angular_core__WEBPACK_IMPORTED_MODULE_0__ = __webpack_require__(/*! @angular/core */ "./node_modules/@angular/core/fesm5/core.js");
/* harmony import */ var _angular_platform_browser_dynamic__WEBPACK_IMPORTED_MODULE_1__ = __webpack_require__(/*! @angular/platform-browser-dynamic */ "./node_modules/@angular/platform-browser-dynamic/fesm5/platform-browser-dynamic.js");
/* harmony import */ var _app_app_module__WEBPACK_IMPORTED_MODULE_2__ = __webpack_require__(/*! ./app/app.module */ "./src/app/app.module.ts");
/* harmony import */ var _environments_environment__WEBPACK_IMPORTED_MODULE_3__ = __webpack_require__(/*! ./environments/environment */ "./src/environments/environment.ts");




if (_environments_environment__WEBPACK_IMPORTED_MODULE_3__["environment"].production) {
    Object(_angular_core__WEBPACK_IMPORTED_MODULE_0__["enableProdMode"])();
}
Object(_angular_platform_browser_dynamic__WEBPACK_IMPORTED_MODULE_1__["platformBrowserDynamic"])().bootstrapModule(_app_app_module__WEBPACK_IMPORTED_MODULE_2__["AppModule"])
    .catch(function (err) { return console.error(err); });


/***/ }),

/***/ 0:
/*!***************************!*\
  !*** multi ./src/main.ts ***!
  \***************************/
/*! no static exports found */
/***/ (function(module, exports, __webpack_require__) {

module.exports = __webpack_require__(/*! D:\Repositories\blog\homepage\src\main.ts */"./src/main.ts");


/***/ })

},[[0,"runtime","vendor"]]]);
//# sourceMappingURL=main.js.map