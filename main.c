#define AIL_TYPES_IMPL
#define AIL_BENCH_IMPL
#define AIL_BENCH_PROFILE
#include "ail.h"
#include "ail_bench.h"
#include <stdint.h>
#include <stdio.h>
#include <xmmintrin.h>
#include <Windows.h>

#define KB(x) (((u64)(x)) << 10)
#define MB(x) (((u64)(x)) << 20)
#define GB(x) (((u64)(x)) << 30)
#define TB(x) (((u64)(x)) << 40)

// #define TEST
#define BUFFER_SIZE MB(1)
#define ITER_COUNT 100

#ifdef TEST
#undef AIL_BENCH_PROFILE_START
#undef AIL_BENCH_PROFILE_END
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
#if defined(_WIN32)
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
		if (buf.data[--i] != x++) return 0;
	}
	return 1;
}

static void naive_u8(Buffer src, Buffer dst) {
	AIL_BENCH_PROFILE_START(naive_u8);
	for (u64 i = 0; i < src.size; i++) {
		dst.data[dst.size - i - 1] = src.data[i];
	}
	AIL_BENCH_PROFILE_END(naive_u8);
}

static void naive_in_place_u8(Buffer buf) {
	AIL_BENCH_PROFILE_START(naive_in_place_u8);
	u8 tmp;
	for (u64 i = 0; i < buf.size/2; i++) {
		tmp = buf.data[i];
		buf.data[i] = buf.data[buf.size - i - 1];
		buf.data[buf.size - i - 1] = tmp;
	}
	AIL_BENCH_PROFILE_END(naive_in_place_u8);
}

int main(void)
{
#ifdef TEST
	Buffer buf = get_buffer(BUFFER_SIZE);
	Buffer cpy = get_buffer(BUFFER_SIZE);
	fill_buffer(buf);
	naive_u8(buf, cpy);
	if (test_buffer(cpy)) printf("\033[32mnaive_u8 reverses buffers correctly :)\033[0m\n");
	else printf("\033[31mnaive_u8 failed to reverse the buffer :(\033[0m\n");

	fill_buffer(buf);
	naive_in_place_u8(buf);
	if (test_buffer(buf)) printf("\033[32mnaive_in_place_u8 reverses buffers correctly :)\033[0m\n");
	else printf("\033[31mnaive_in_place_u8 failed to reverse the buffer :(\033[0m\n");
#else
	Buffer buf = get_buffer(BUFFER_SIZE);
	Buffer cpy = get_buffer(BUFFER_SIZE);
	fill_buffer(buf);
	ail_bench_begin_profile();
	for (u32 i = 0; i < ITER_COUNT; i++) {
		naive_u8(buf, cpy);
		naive_in_place_u8(buf);
	}
	ail_bench_end_and_print_profile(0);
	AIL_BENCH_END_OF_COMPILATION_UNIT();
#endif
}