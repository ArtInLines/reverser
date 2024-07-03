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

#define ALL
#define ITER_COUNT 4

#ifdef ALL
#define TEST
#define BENCH
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

static void free_buffer(Buffer buf) {
#if defined(_WIN32) || defined(__WIN32__)
    VirtualFree(buf.data, buf.size, MEM_DECOMMIT);
#else
    munmap(buf.data, buf.size);
#endif
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

static void scalar_wide(Buffer src, Buffer dst) {
	AIL_BENCH_PROFILE_START(scalar_wide);
	for (u64 i = 0; i < src.size; i += 4) {
		dst.data[dst.size - i - 1] = src.data[i + 0];
		dst.data[dst.size - i - 2] = src.data[i + 1];
		dst.data[dst.size - i - 3] = src.data[i + 2];
		dst.data[dst.size - i - 4] = src.data[i + 3];
	}
	for (u32 i = 0; i < src.size % 4; i++) {
		dst.data[i] = src.data[src.size - i - 1];
	}
	AIL_BENCH_PROFILE_END(scalar_wide);
}

static void scalar_wide_in_place(Buffer buf) {
	AIL_BENCH_PROFILE_START(scalar_wide_in_place);
	u8 tmp[4];
	u64 n = buf.size/2;
	for (u64 i = 0; i < n; i += 4) {
		tmp[0] = buf.data[i + 0];
		tmp[1] = buf.data[i + 1];
		tmp[2] = buf.data[i + 2];
		tmp[3] = buf.data[i + 3];
		buf.data[i + 0] = buf.data[buf.size - i - 1];
		buf.data[i + 1] = buf.data[buf.size - i - 2];
		buf.data[i + 2] = buf.data[buf.size - i - 3];
		buf.data[i + 3] = buf.data[buf.size - i - 4];
		buf.data[buf.size - i - 1] = tmp[0];
		buf.data[buf.size - i - 2] = tmp[1];
		buf.data[buf.size - i - 3] = tmp[2];
		buf.data[buf.size - i - 4] = tmp[3];
	}
	for (u32 i = 0; i < buf.size % 4; i++) {
		u8 tmp = buf.data[n + i];
		buf.data[n + i] = buf.data[n + 7 - i];
		buf.data[n + 7 - i] = tmp;
	}
	AIL_BENCH_PROFILE_END(scalar_wide_in_place);
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

#define FUNCTIONS \
	X(scalar, scalar_in_place) \
	X(scalar_wide, scalar_wide_in_place) \
	X(simd_shuffle, simd_shuffle_in_place)

void bench(u64 buffer_size)
{
	Buffer buf = get_buffer(buffer_size);
	Buffer cpy = get_buffer(buffer_size);
	fill_buffer(buf);
	ail_bench_begin_profile();
	for (u64 i = 0; i < ITER_COUNT; i++) {
		#define X(func, func_in_place) { func(buf, cpy); func_in_place(buf); }
			FUNCTIONS
		#undef X
	}
	printf("-----------\n");
	if      (buffer_size >= AIL_GB(1)) printf("Benchmark Results for Reversing %lldGB of memory\n", buffer_size/AIL_GB(1));
	else if (buffer_size >= AIL_MB(1)) printf("Benchmark Results for Reversing %lldMB of memory\n", buffer_size/AIL_MB(1));
	else if (buffer_size >= AIL_KB(1)) printf("Benchmark Results for Reversing %lldKB of memory\n", buffer_size/AIL_KB(1));
	else                               printf("Benchmark Results for Reversing %lld bytes of memory\n", buffer_size);
	ail_bench_end_and_print_profile(true);
	free_buffer(buf);
	free_buffer(cpy);
}

static u64 test_buffer_sizes[] = { 1, 15, 16, 17, 25, 31, 32, 33, 511, 512, 513, AIL_KB(1) + 15, AIL_KB(1) + 17 };
typedef Buffer BufferList[AIL_ARRLEN(test_buffer_sizes)][2];
typedef void (FuncType)(Buffer src, Buffer dst);
typedef void (FuncInPlaceType)(Buffer buf);

static void test(BufferList buffers, FuncType func, FuncInPlaceType func_in_place, char *func_name, char *func_in_place_name) {
	for (u64 i = 0; i < AIL_ARRLEN(buffers); i++) {
		fill_buffer(buffers[i][0]);
		fill_buffer(buffers[i][1]);
		func(buffers[i][0], buffers[i][1]);
		if (!test_buffer(buffers[i][1])) {
			printf("\033[31m%s failed test for buffer-size %lld :(\033[0m\n", func_name, test_buffer_sizes[i]);
			return;
		}

		fill_buffer(buffers[i][0]);
		func_in_place(buffers[i][0]);
		if (!test_buffer(buffers[i][0])) {
			printf("\033[31m%s failed test for buffer-size %lld :(\033[0m\n", func_in_place_name, test_buffer_sizes[i]);
			return;
		}
	}
	printf("\033[32m%s succeeded all tests :)\033[0m\n", func_name);
	printf("\033[32m%s succeeded all tests :)\033[0m\n", func_in_place_name);
}

int main(void)
{
#ifdef TEST
	Buffer buffers[AIL_ARRLEN(test_buffer_sizes)][2];
	for (u64 i = 0; i < AIL_ARRLEN(test_buffer_sizes); i++) {
		buffers[i][0] = get_buffer(test_buffer_sizes[i]);
		buffers[i][1] = get_buffer(test_buffer_sizes[i]);
	}
	#define X(func, func_in_place) test(buffers, func, func_in_place, AIL_STRINGIZE(func), AIL_STRINGIZE(func_in_place));
		FUNCTIONS
	#undef X
	for (u64 i = 0; i < AIL_ARRLEN(buffers); i++) {
		free_buffer(buffers[i][0]);
		free_buffer(buffers[i][1]);
	}
#endif

#ifdef BENCH
	for (u64 buffer_size = 128; buffer_size < AIL_GB(1); buffer_size <<= 4) {
		bench(buffer_size);
	}
	AIL_BENCH_END_OF_COMPILATION_UNIT();
#endif
}