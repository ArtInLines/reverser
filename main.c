#define AIL_TYPES_IMPL
#define AIL_BENCH_IMPL
#define AIL_BENCH_PROFILE
#include "ail.h"       // For typedefs and some useful macros
#include "ail_bench.h" // For benchmarking
#include <stdio.h>     // For printf
#include <xmmintrin.h> // For SIMD instructions

#if defined(_WIN32) || defined(__WIN32__)
#include <Windows.h> // For VirtualAlloc
#else
#include <sys/mman.h> // For mmap
#endif

#define BENCH
#define BUFFER_SIZE (AIL_GB(1))
#define ITER_COUNT 10

#ifdef ALL
#define TEST
#define BENCH
#endif

#ifndef BENCH
#undef  AIL_BENCH_PROFILE_START
#undef  AIL_BENCH_PROFILE_END
#define AIL_BENCH_PROFILE_START(name) do {} while(false)
#define AIL_BENCH_PROFILE_END(name) do {} while(false)
#endif

typedef struct {
	u64 size;
	u8 *data;
} Buffer;

static Buffer get_buffer(u64 size) {
	Buffer buf;
	buf.size = size;
#if defined(_WIN32) || defined(__WIN32__)
    buf.data = VirtualAlloc(0, size, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE);
#else
    buf.data = mmap(0, size, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_ANON, -1, 0);
#endif
	return buf;
}

// @Note: Fills the buffer with a repeating pattern of increasing bytes between 0 and 255. This makes it very easy to see if the buffer was reversed correctly
static void fill_buffer(Buffer buf) {
	u8 x = 0;
	for (u64 i = 0; i < buf.size; i++) {
		buf.data[i] = x++;
	}
}

// Checks whether the buffer contains the reversed pattern of fill_buffer
static b32 test_buffer(Buffer buf) {
	u64 i = buf.size;
	u8  x = 0;
	while (i > 0) {
		if (buf.data[--i] != x++) {
			printf("\033[31mError at index %lld - Expected: %d, but received: %d\033[0m\n", i, x-1, buf.data[i]);
			return 0;
		}
	}
	return 1;
}

static void scalar(Buffer src, Buffer dst) {
	AIL_BENCH_PROFILE_START(scalar);
	for (u64 i = 0; i < src.size; i++) {
		dst.data[dst.size - i - 1] = src.data[i];
	}
	AIL_BENCH_PROFILE_END(scalar);
}

static void scalar_in_place(Buffer buf) {
	AIL_BENCH_PROFILE_START(scalar_in_place);
	u8 tmp;
	for (u64 i = 0; i < buf.size/2; i++) {
		tmp = buf.data[i];
		buf.data[i] = buf.data[buf.size - i - 1];
		buf.data[buf.size - i - 1] = tmp;
	}
	AIL_BENCH_PROFILE_END(scalar_in_place);
}

static void simd_shuffle(Buffer src, Buffer dst) {
	AIL_BENCH_PROFILE_START(simd_shuffle);
	u64 n   = src.size / sizeof(__m128);
	u64 rem = src.size % sizeof(__m128);
	__m128i mask = _mm_set_epi8(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15); // Requires SSE2
	__m128i tmp;
	__m128i *s = (__m128i*)src.data;
	__m128i *d = (__m128i*)(dst.data + rem);
	for (u64 i = 0; i < n; i++) {
		tmp = _mm_loadu_si128(&s[i]);         // Requires SSE2
		tmp = _mm_shuffle_epi8(tmp, mask);    // Requires SSSE3
		_mm_storeu_si128(&d[n - i - 1], tmp); // Requires SSE2
	}
	for (u64 i = 0; i < rem; i++) {
		dst.data[i] = src.data[src.size - i - 1];
	}
	AIL_BENCH_PROFILE_END(simd_shuffle);
}

static void simd_shuffle_in_place(Buffer buf) {
	AIL_BENCH_PROFILE_START(simd_shuffle_in_place);
	u64 n   = buf.size / (sizeof(__m128) * 2);
	u64 rem = buf.size % (sizeof(__m128) * 2);
	__m128i mask = _mm_set_epi8(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15); // Requires SSE2
	__m128i a, b;
	__m128i *start = (__m128i*)buf.data;
	__m128i *end   = (__m128i*)&buf.data[buf.size];
	for (u64 i = 0; i < n; i++) {
		a = _mm_loadu_si128(start + i);   // Requires SSE2
		b = _mm_loadu_si128(end - i - 1); // Requires SSE2
		a = _mm_shuffle_epi8(a, mask);    // Requires SSSE3
		b = _mm_shuffle_epi8(b, mask);    // Requires SSSE3
		_mm_storeu_si128(start + i, b);   // Requires SSE2
		_mm_storeu_si128(end - i - 1, a); // Requires SSE2
	}
	for (u64 i = 0; i < rem/2; i++) {
		u8 tmp = buf.data[n*sizeof(__m128) + i];
		buf.data[n*sizeof(__m128) + i] = buf.data[n*sizeof(__m128) + rem - i - 1];
		buf.data[n*sizeof(__m128) + rem - i - 1] = tmp;
	}
	AIL_BENCH_PROFILE_END(simd_shuffle_in_place);
}

int main(void)
{
	Buffer buf, cpy;

#ifdef TEST
	buf = get_buffer(BUFFER_SIZE);
	cpy = get_buffer(BUFFER_SIZE);
	fill_buffer(buf);
	fill_buffer(cpy);
	scalar(buf, cpy);
	if (test_buffer(cpy)) printf("\033[32mscalar reverses buffers correctly :)\033[0m\n");
	else printf("\033[31mscalar failed to reverse the buffer :(\033[0m\n");

	fill_buffer(buf);
	scalar_in_place(buf);
	if (test_buffer(buf)) printf("\033[32mscalar_in_place reverses buffers correctly :)\033[0m\n");
	else printf("\033[31mscalar_in_place failed to reverse the buffer :(\033[0m\n");

	fill_buffer(buf);
	fill_buffer(cpy);
	simd_shuffle(buf, cpy);
	if (test_buffer(cpy)) printf("\033[32msimd_shuffle reverses buffers correctly :)\033[0m\n");
	else printf("\033[31msimd_shuffle failed to reverse the buffer :(\033[0m\n");

	fill_buffer(buf);
	simd_shuffle_in_place(buf);
	if (test_buffer(buf)) printf("\033[32msimd_shuffle_in_place reverses buffers correctly :)\033[0m\n");
	else printf("\033[31msimd_shuffle_in_place failed to reverse the buffer :(\033[0m\n");
#endif

#ifdef BENCH
	buf = get_buffer(BUFFER_SIZE);
	cpy = get_buffer(BUFFER_SIZE);
	fill_buffer(buf);
	ail_bench_begin_profile();
	for (u32 i = 0; i < ITER_COUNT; i++) {
		scalar(buf, cpy);
		scalar_in_place(buf);
		simd_shuffle(buf, cpy);
		simd_shuffle_in_place(buf);
	}
	printf("-----------\n");
	if (BUFFER_SIZE > AIL_GB(1))      printf("Benchmark Results for Reversing %lldGB of memory\n", BUFFER_SIZE/AIL_GB(1));
	else if (BUFFER_SIZE > AIL_MB(1)) printf("Benchmark Results for Reversing %lldMB of memory\n", BUFFER_SIZE/AIL_MB(1));
	else                              printf("Benchmark Results for Reversing %lldKB of memory\n", BUFFER_SIZE/AIL_KB(1));
	ail_bench_end_and_print_profile(0);
	AIL_BENCH_END_OF_COMPILATION_UNIT();
#endif
}