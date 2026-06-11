/*
 * Minimal self-contained replacement for the GLM types used by the
 * rasterizer (vec3, vec4, mat3, mat4). Matrices are column-major and
 * constructors, indexing (m[col][row]) and multiplication follow GLM
 * semantics, so this is a drop-in replacement for third_party/glm.
 */

#ifndef CUDA_RASTERIZER_VEC_MATH_H_INCLUDED
#define CUDA_RASTERIZER_VEC_MATH_H_INCLUDED

#include <cmath>

#if defined(__CUDACC__)
#define VM_FUNC __host__ __device__ __forceinline__
#else
#define VM_FUNC inline
#endif

struct vec3
{
	float x, y, z;

	VM_FUNC vec3() : x(0.f), y(0.f), z(0.f) {}
	VM_FUNC explicit vec3(float v) : x(v), y(v), z(v) {}
	VM_FUNC vec3(float x, float y, float z) : x(x), y(y), z(z) {}

	VM_FUNC float& operator[](int i) { return (&x)[i]; }
	VM_FUNC const float& operator[](int i) const { return (&x)[i]; }

	VM_FUNC vec3& operator+=(const vec3& v) { x += v.x; y += v.y; z += v.z; return *this; }
	VM_FUNC vec3& operator-=(const vec3& v) { x -= v.x; y -= v.y; z -= v.z; return *this; }
	VM_FUNC vec3& operator+=(float s) { x += s; y += s; z += s; return *this; }
	VM_FUNC vec3& operator*=(float s) { x *= s; y *= s; z *= s; return *this; }
};

VM_FUNC vec3 operator+(const vec3& a, const vec3& b) { return vec3(a.x + b.x, a.y + b.y, a.z + b.z); }
VM_FUNC vec3 operator-(const vec3& a, const vec3& b) { return vec3(a.x - b.x, a.y - b.y, a.z - b.z); }
VM_FUNC vec3 operator-(const vec3& a) { return vec3(-a.x, -a.y, -a.z); }
VM_FUNC vec3 operator*(const vec3& a, float s) { return vec3(a.x * s, a.y * s, a.z * s); }
VM_FUNC vec3 operator*(float s, const vec3& a) { return vec3(s * a.x, s * a.y, s * a.z); }
VM_FUNC vec3 operator/(const vec3& a, float s) { return vec3(a.x / s, a.y / s, a.z / s); }

VM_FUNC float dot(const vec3& a, const vec3& b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
VM_FUNC float length(const vec3& a) { return sqrtf(dot(a, a)); }
VM_FUNC vec3 max(const vec3& a, float s)
{
	return vec3(a.x < s ? s : a.x, a.y < s ? s : a.y, a.z < s ? s : a.z);
}

struct vec4
{
	float x, y, z, w;

	VM_FUNC vec4() : x(0.f), y(0.f), z(0.f), w(0.f) {}
	VM_FUNC explicit vec4(float v) : x(v), y(v), z(v), w(v) {}
	VM_FUNC vec4(float x, float y, float z, float w) : x(x), y(y), z(z), w(w) {}

	VM_FUNC float& operator[](int i) { return (&x)[i]; }
	VM_FUNC const float& operator[](int i) const { return (&x)[i]; }

	VM_FUNC vec4& operator+=(const vec4& v) { x += v.x; y += v.y; z += v.z; w += v.w; return *this; }
	VM_FUNC vec4& operator*=(float s) { x *= s; y *= s; z *= s; w *= s; return *this; }
};

VM_FUNC vec4 operator+(const vec4& a, const vec4& b) { return vec4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w); }
VM_FUNC vec4 operator-(const vec4& a, const vec4& b) { return vec4(a.x - b.x, a.y - b.y, a.z - b.z, a.w - b.w); }
VM_FUNC vec4 operator*(const vec4& a, float s) { return vec4(a.x * s, a.y * s, a.z * s, a.w * s); }
VM_FUNC vec4 operator*(float s, const vec4& a) { return vec4(s * a.x, s * a.y, s * a.z, s * a.w); }
VM_FUNC vec4 operator/(const vec4& a, float s) { return vec4(a.x / s, a.y / s, a.z / s, a.w / s); }

VM_FUNC float dot(const vec4& a, const vec4& b) { return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w; }
VM_FUNC float length(const vec4& a) { return sqrtf(dot(a, a)); }

// Column-major 3x3 matrix: m[i] is column i, m[i][j] addresses column i, row j.
struct mat3
{
	vec3 cols[3];

	VM_FUNC mat3() : mat3(1.0f) {}
	VM_FUNC explicit mat3(float d)
	{
		cols[0] = vec3(d, 0.f, 0.f);
		cols[1] = vec3(0.f, d, 0.f);
		cols[2] = vec3(0.f, 0.f, d);
	}
	VM_FUNC mat3(
		float x0, float y0, float z0,
		float x1, float y1, float z1,
		float x2, float y2, float z2)
	{
		cols[0] = vec3(x0, y0, z0);
		cols[1] = vec3(x1, y1, z1);
		cols[2] = vec3(x2, y2, z2);
	}

	VM_FUNC vec3& operator[](int i) { return cols[i]; }
	VM_FUNC const vec3& operator[](int i) const { return cols[i]; }
};

VM_FUNC vec3 operator*(const mat3& m, const vec3& v)
{
	return m[0] * v.x + m[1] * v.y + m[2] * v.z;
}

VM_FUNC mat3 operator*(const mat3& a, const mat3& b)
{
	mat3 r(0.0f);
	r[0] = a * b[0];
	r[1] = a * b[1];
	r[2] = a * b[2];
	return r;
}

VM_FUNC mat3 operator*(const mat3& m, float s)
{
	mat3 r(0.0f);
	r[0] = m[0] * s;
	r[1] = m[1] * s;
	r[2] = m[2] * s;
	return r;
}

VM_FUNC mat3 operator*(float s, const mat3& m) { return m * s; }

VM_FUNC mat3 transpose(const mat3& m)
{
	return mat3(
		m[0].x, m[1].x, m[2].x,
		m[0].y, m[1].y, m[2].y,
		m[0].z, m[1].z, m[2].z);
}

// Column-major 4x4 matrix, same conventions as mat3.
struct mat4
{
	vec4 cols[4];

	VM_FUNC mat4() : mat4(1.0f) {}
	VM_FUNC explicit mat4(float d)
	{
		cols[0] = vec4(d, 0.f, 0.f, 0.f);
		cols[1] = vec4(0.f, d, 0.f, 0.f);
		cols[2] = vec4(0.f, 0.f, d, 0.f);
		cols[3] = vec4(0.f, 0.f, 0.f, d);
	}
	VM_FUNC mat4(
		float x0, float y0, float z0, float w0,
		float x1, float y1, float z1, float w1,
		float x2, float y2, float z2, float w2,
		float x3, float y3, float z3, float w3)
	{
		cols[0] = vec4(x0, y0, z0, w0);
		cols[1] = vec4(x1, y1, z1, w1);
		cols[2] = vec4(x2, y2, z2, w2);
		cols[3] = vec4(x3, y3, z3, w3);
	}

	VM_FUNC vec4& operator[](int i) { return cols[i]; }
	VM_FUNC const vec4& operator[](int i) const { return cols[i]; }
};

VM_FUNC vec4 operator*(const mat4& m, const vec4& v)
{
	return m[0] * v.x + m[1] * v.y + m[2] * v.z + m[3] * v.w;
}

VM_FUNC mat4 operator*(const mat4& a, const mat4& b)
{
	mat4 r(0.0f);
	r[0] = a * b[0];
	r[1] = a * b[1];
	r[2] = a * b[2];
	r[3] = a * b[3];
	return r;
}

VM_FUNC mat4 operator*(const mat4& m, float s)
{
	mat4 r(0.0f);
	r[0] = m[0] * s;
	r[1] = m[1] * s;
	r[2] = m[2] * s;
	r[3] = m[3] * s;
	return r;
}

VM_FUNC mat4 operator*(float s, const mat4& m) { return m * s; }

VM_FUNC mat4 transpose(const mat4& m)
{
	return mat4(
		m[0].x, m[1].x, m[2].x, m[3].x,
		m[0].y, m[1].y, m[2].y, m[3].y,
		m[0].z, m[1].z, m[2].z, m[3].z,
		m[0].w, m[1].w, m[2].w, m[3].w);
}

#undef VM_FUNC

#endif
