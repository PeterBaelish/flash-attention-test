#include <cuda.h>
#include <chrono>
#include <errno.h>
#include <error.h>
#include <fcntl.h>
#include <stdint.h>
#include <stdio.h>
#include <unistd.h>

__device__ inline uint64_t GlobalTimer64(void) {
  // Due to a bug in CUDA's 64-bit globaltimer, the lower 32 bits can wrap
  // around after the upper bits have already been read. Work around this by
  // reading the high bits a second time. Use the second value to detect a
  // rollover, and set the lower bits of the 64-bit "timer reading" to 0, which
  // would be valid, it's passed over during the duration of the reading. If no
  // rollover occurred, just return the initial reading.
  volatile uint64_t first_reading;
  volatile uint32_t second_reading;
  uint32_t high_bits_first;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(first_reading));
  high_bits_first = first_reading >> 32;
  asm volatile("mov.u32 %0, %%globaltimer_hi;" : "=r"(second_reading));
  if (high_bits_first == second_reading) {
    return first_reading;
  }
  // Return the value with the updated high bits, but the low bits set to 0.
  return ((uint64_t) second_reading) << 32;
}

// Returns the ID of the SM this is executed on.
static __device__ __inline__ uint32_t GetSMID(void) {
  uint32_t to_return;
  asm volatile("mov.u32 %0, %%smid;" : "=r"(to_return));
  return to_return;
}


/**
 * Copyright 2023 Joshua Bakita
 * Library to control SM masks on CUDA launches. Co-opts preexisting debug
 * logic in the CUDA driver library, and thus requires a build with -lcuda.
 */

// In functions that do not return an error code, we favor terminating with an
// error rather than merely printing a warning and continuing.
#define abort(ret, errno, ...) error_at_line(ret, errno, __FILE__, __LINE__, \
                                             __VA_ARGS__)

// Layout of mask control fields to match CUDA's static global struct
struct global_sm_control {
	uint32_t enabled;
	uint64_t mask;
} __attribute__((packed));

/*** CUDA Globals Manipulation. CUDA 10.2 only ***/

// Ends up being 0x7fb7fa3408 in some binaries (CUDA 10.2, Jetson)
static struct global_sm_control* g_sm_control = NULL;

/* Find the location of CUDA's `globals` struct and the SM mask control fields
 * No symbols are exported from within `globals`, so this has to do a very
 * messy lookup, following the pattern of the assembly of `cuDeviceGetCount()`.
 * Don't call this before the CUDA library has been initialized.
 * (Note that this appears to work, even if built on CUDA > 10.2.)
 */


/*** QMD/TMD-based SM Mask Control via Debug Callback. CUDA 11+ ***/

// Tested working on CUDA x86_64 11.0-12.2.
// Tested not working on aarch64 or x86_64 10.2
static const CUuuid callback_funcs_id = {0x2c, (char)0x8e, 0x0a, (char)0xd8, 0x07, 0x10, (char)0xab, 0x4e, (char)0x90, (char)0xdd, 0x54, 0x71, (char)0x9f, (char)0xe5, (char)0xf7, 0x4b};
#define LAUNCH_DOMAIN 0x3
#define LAUNCH_PRE_UPLOAD 0x3
static uint64_t g_sm_mask = 0;
static __thread uint64_t g_next_sm_mask = 0;
static char sm_control_setup_called = 0;
static void launchCallback(void *ukwn, int domain, int cbid, const void *in_params) {
	if (*(uint32_t*)in_params < 0x50) {
		fprintf(stderr, "Unsupported CUDA version for callback-based SM masking. Aborting...\n");
		return;
	}
	if (!**((uintptr_t***)in_params+8)) {
		fprintf(stderr, "Called with NULL halLaunchDataAllocation\n");
		return;
	}
	//fprintf(stderr, "cta: %lx\n", *(uint64_t*)(**((char***)in_params + 8) + 74));
	// TODO: Check for supported QMD version (>XXX, <4.00)
	// TODO: Support QMD version 4 (Hopper), where offset starts at +304 (rather than +84) and is 72 bytes (rather than 8 bytes) wide
	uint32_t *lower_ptr = (uint32_t*)(**((char***)in_params + 8) + 84);
	uint32_t *upper_ptr = (uint32_t*)(**((char***)in_params + 8) + 88);
	if (g_next_sm_mask) {
		*lower_ptr = (uint32_t)g_next_sm_mask;
		*upper_ptr = (uint32_t)(g_next_sm_mask >> 32);
		g_next_sm_mask = 0;
	} else if (!*lower_ptr && !*upper_ptr){
		// Only apply the global mask if a per-stream mask hasn't been set
		*lower_ptr = (uint32_t)g_sm_mask;
		*upper_ptr = (uint32_t)(g_sm_mask >> 32);
	}
	//fprintf(stderr, "lower mask: %x\n", *lower_ptr);
	//fprintf(stderr, "upper mask: %x\n", *upper_ptr);
}

static void setup_sm_control_11() {
	int (*subscribe)(uint32_t* hndl, void(*callback)(void*, int, int, const void*), void* ukwn);
	int (*enable)(uint32_t enable, uint32_t hndl, int domain, int cbid);
	uintptr_t* tbl_base;
	uint32_t my_hndl;
	// Avoid race conditions (setup can only be called once)
	if (__atomic_test_and_set(&sm_control_setup_called, __ATOMIC_SEQ_CST))
		return;

	cuGetExportTable((const void**)&tbl_base, &callback_funcs_id);
	uintptr_t subscribe_func_addr = *(tbl_base + 3);
	uintptr_t enable_func_addr = *(tbl_base + 6);
	subscribe = (typeof(subscribe))subscribe_func_addr;
	enable = (typeof(enable))enable_func_addr;
	int res = 0;
	res = subscribe(&my_hndl, launchCallback, NULL);
	if (res) {
		fprintf(stderr, "libsmctrl: Error subscribing to launch callback. Error %d\n", res);
		return;
	}
	res = enable(1, my_hndl, LAUNCH_DOMAIN, LAUNCH_PRE_UPLOAD);
	if (res)
		fprintf(stderr, "libsmctrl: Error enabling launch callback. Error %d\n", res);
}

// Set default mask for all launches
void libsmctrl_set_global_mask(uint64_t mask) {
	int ver;
	cuDriverGetVersion(&ver);
	if (ver > 10020) {
		if (!sm_control_setup_called)
			setup_sm_control_11();
		g_sm_mask = mask;
	} else { // < CUDA 10.2
		abort(1, ENOSYS, "Global masking requires at least CUDA 10.2; "
		                 "this application is using CUDA %d.%d",
		                 ver / 1000, (ver % 100));
	}
}

// Set mask for next launch from this thread
void libsmctrl_set_next_mask(uint64_t mask) {
	if (!sm_control_setup_called)
		setup_sm_control_11();
	g_next_sm_mask = mask;
}


/*** Per-Stream SM Mask (unlikely to be forward-compatible) ***/

#define CU_8_0_MASK_OFF 0xec
#define CU_9_0_MASK_OFF 0x130
#define CU_9_0_MASK_OFF_TX2 0x128 // CUDA 9.0 is slightly different on the TX2
// CUDA 9.0 and 9.1 use the same offset
#define CU_9_2_MASK_OFF 0x140
#define CU_10_0_MASK_OFF 0x24c
// CUDA 10.0, 10.1 and 10.2 use the same offset
#define CU_11_0_MASK_OFF 0x274
#define CU_11_1_MASK_OFF 0x2c4
#define CU_11_2_MASK_OFF 0x37c
// CUDA 11.2, 11.3, 11.4, and 11.5 use the same offset
#define CU_11_6_MASK_OFF 0x38c
#define CU_11_7_MASK_OFF 0x3c4
#define CU_11_8_MASK_OFF 0x47c
#define CU_12_0_MASK_OFF 0x4cc
// CUDA 12.0 and 12.1 use the same offset

// Layout in CUDA's `stream` struct
struct stream_sm_mask {
	uint32_t upper;
	uint32_t lower;
} __attribute__((packed));

// Check if this system has a Parker SoC (TX2/PX2 chip)
// (CUDA 9.0 behaves slightly different on this platform.)
// @return 1 if detected, 0 if not, -cuda_err on error
#if __aarch64__
int detect_parker_soc() {
	int cap_major, cap_minor, err, dev_count;
	if (err = cuDeviceGetCount(&dev_count))
		return -err;
	// As CUDA devices are numbered by order of compute power, check every
	// device, in case a powerful discrete GPU is attached (such as on the
	// DRIVE PX2). We detect the Parker SoC via its unique CUDA compute
	// capability: 6.2.
	for (int i = 0; i < dev_count; i++) {
		if (err = cuDeviceGetAttribute(&cap_minor,
		                               CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR,
		                               i))
			return -err;
		if (err = cuDeviceGetAttribute(&cap_major,
		                               CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,
		                               i))
			return -err;
		if (cap_major == 6 && cap_minor == 2)
			return 1;
	}
	return 0;
}
#endif // __aarch64__

// Should work for CUDA 8.0 through 12.1
// A cudaStream_t is a CUstream*. We use void* to avoid a cuda.h dependency in
// our header
void libsmctrl_set_stream_mask(void* stream, uint64_t mask) {
	char* stream_struct_base = *(char**)stream;
	struct stream_sm_mask* hw_mask;
	int ver;
	cuDriverGetVersion(&ver);
	switch (ver) {
	case 8000:
		hw_mask = (struct stream_sm_mask*)(stream_struct_base + CU_8_0_MASK_OFF);
	case 9000:
	case 9010: {
		hw_mask = (struct stream_sm_mask*)(stream_struct_base + CU_9_0_MASK_OFF);
#if __aarch64__
		// Jetson TX2 offset is slightly different on CUDA 9.0.
		// Only compile the check into ARM64 builds.
		int is_parker;
		const char* err_str;
		if ((is_parker = detect_parker_soc()) < 0) {
			cuGetErrorName(-is_parker, &err_str);
			fprintf(stderr, "libsmctrl_set_stream_mask: CUDA call "
					"failed while doing compatibilty test."
			                "Error, '%s'. Not applying stream "
					"mask.\n", err_str);
		}

		if (is_parker)
			hw_mask = (struct stream_sm_mask*)(stream_struct_base + CU_9_0_MASK_OFF_TX2);
#endif
		break;
	}
	case 9020:
		hw_mask = (struct stream_sm_mask*)(stream_struct_base + CU_9_2_MASK_OFF);
		break;
	case 10000:
	case 10010:
	case 10020:
		hw_mask = (struct stream_sm_mask*)(stream_struct_base + CU_10_0_MASK_OFF);
		break;
	case 11000:
		hw_mask = (struct stream_sm_mask*)(stream_struct_base + CU_11_0_MASK_OFF);
		break;
	case 11010:
		hw_mask = (struct stream_sm_mask*)(stream_struct_base + CU_11_1_MASK_OFF);
		break;
	case 11020:
	case 11030:
	case 11040:
	case 11050:
		hw_mask = (struct stream_sm_mask*)(stream_struct_base + CU_11_2_MASK_OFF);
		break;
	case 11060:
		hw_mask = (struct stream_sm_mask*)(stream_struct_base + CU_11_6_MASK_OFF);
		break;
	case 11070:
		hw_mask = (struct stream_sm_mask*)(stream_struct_base + CU_11_7_MASK_OFF);
		break;
	case 11080:
		hw_mask = (struct stream_sm_mask*)(stream_struct_base + CU_11_8_MASK_OFF);
		break;
	case 12000:
	case 12010:
		hw_mask = (struct stream_sm_mask*)(stream_struct_base + CU_12_0_MASK_OFF);
		break;
	default: {
		// For experimenting to determine the right mask offset, set the MASK_OFF
		// environment variable (positive and negative numbers are supported)
		char* mask_off_str = getenv("MASK_OFF");
		fprintf(stderr, "libsmctrl: Stream masking unsupported on this CUDA version (%d)!\n", ver);
		if (mask_off_str) {
			int off = atoi(mask_off_str);
			fprintf(stderr, "libsmctrl: Attempting offset %d on CUDA 12.1 base %#x "
					"(total off: %#x)\n", off, CU_12_0_MASK_OFF, CU_12_0_MASK_OFF+off);
			hw_mask = (struct stream_sm_mask*)(stream_struct_base + CU_12_0_MASK_OFF + off);
		} else {
			return;
		}}
	}

	hw_mask->upper = mask >> 32;
	hw_mask->lower = mask;
}

/* INFORMATIONAL FUNCTIONS */

// Read an integer from a file in `/proc`
static int read_int_procfile(char* filename, uint64_t* out) {
	char f_data[18] = {0};
	int fd = open(filename, O_RDONLY);
	if (fd == -1)
		return errno;
	read(fd, f_data, 18);
	close(fd);
	*out = strtoll(f_data, NULL, 16);
	return 0;
}

// We support up to 12 GPCs per GPU, and up to 16 GPUs.
static uint64_t tpc_mask_per_gpc_per_dev[16][12];
// Output mask is vtpc-indexed (virtual TPC)
int libsmctrl_get_gpc_info(uint32_t* num_enabled_gpcs, uint64_t** tpcs_for_gpc, int dev) {
	uint32_t i, j, vtpc_idx = 0;
	uint64_t gpc_mask, num_tpc_per_gpc, max_gpcs, gpc_tpc_mask;
	int err;
	char filename[100];
	*num_enabled_gpcs = 0;
	// Maximum number of GPCs supported for this chip
	snprintf(filename, 100, "/proc/gpu%d/num_gpcs", dev);
	if (err = read_int_procfile(filename, &max_gpcs)) {
		fprintf(stderr, "libsmctrl: nvdebug module must be loaded into kernel before "
				"using libsmctrl_get_*_info() functions\n");
		return err;
	}
	// TODO: handle arbitrary-size GPUs
	if (dev > 16 || max_gpcs > 12) {
		fprintf(stderr, "libsmctrl: GPU possibly too large for preallocated map!\n");
		return ERANGE;
	}
	// Set bit = disabled GPC
	snprintf(filename, 100, "/proc/gpu%d/gpc_mask", dev);
	if (err = read_int_procfile(filename, &gpc_mask))
		return err;
	snprintf(filename, 100, "/proc/gpu%d/num_tpc_per_gpc", dev);
	if (err = read_int_procfile(filename, &num_tpc_per_gpc))
		return err;
	// For each enabled GPC
	for (i = 0; i < max_gpcs; i++) {
		// Skip this GPC if disabled
		if ((1 << i) & gpc_mask)
			continue;
		(*num_enabled_gpcs)++;
		// Get the bitstring of TPCs disabled for this GPC
		// Set bit = disabled TPC
		snprintf(filename, 100, "/proc/gpu%d/gpc%d_tpc_mask", dev, i);
		if (err = read_int_procfile(filename, &gpc_tpc_mask))
			return err;
		uint64_t* tpc_mask = &tpc_mask_per_gpc_per_dev[dev][*num_enabled_gpcs - 1];
		*tpc_mask = 0;
		for (j = 0; j < num_tpc_per_gpc; j++) {
				// Skip disabled TPCs
				if ((1 << j) & gpc_tpc_mask)
					continue;
				*tpc_mask |= (1ull << vtpc_idx);
				vtpc_idx++;
		}
	}
	*tpcs_for_gpc = tpc_mask_per_gpc_per_dev[dev];
	return 0;
}

int libsmctrl_get_tpc_info(uint32_t* num_tpcs, int dev) {
	uint32_t num_gpcs;
	uint64_t* tpcs_per_gpc;
	int res;
	if (res = libsmctrl_get_gpc_info(&num_gpcs, &tpcs_per_gpc, dev))
		return res;
	*num_tpcs = 0;
	for (int gpc = 0; gpc < num_gpcs; gpc++) {
		*num_tpcs += __builtin_popcountl(tpcs_per_gpc[gpc]);
	}
	return 0;
}

// @param dev Device index as understood by CUDA **can differ from nvdebug idx**
// This implementation is fragile, and could be incorrect for odd GPUs
int libsmctrl_get_tpc_info_cuda(uint32_t* num_tpcs, int cuda_dev) {
	int num_sms, major, minor, res = 0;
	const char* err_str;
	if (res = cuInit(0))
		goto abort_cuda;
	if (res = cuDeviceGetAttribute(&num_sms, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, cuda_dev))
		goto abort_cuda;
	if (res = cuDeviceGetAttribute(&minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, cuda_dev))
		goto abort_cuda;
	if (res = cuDeviceGetAttribute(&major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, cuda_dev))
		goto abort_cuda;
	// SM masking only works on sm_35+
	if (major < 3 || (major == 3 && minor < 5))
		return ENOTSUP;
	// Everything newer than Pascal (as of Hopper) has 2 SMs per TPC, as well
	// as the P100, which is uniquely sm_60
	int sms_per_tpc;
	if (major > 6 || (major == 6 && minor == 0))
		sms_per_tpc = 2;
	else
		sms_per_tpc = 1;
	// It looks like there may be some upcoming weirdness (TPCs with only one SM?)
	// with Hopper
	if (major >= 9)
		fprintf(stderr, "libsmctrl: WARNING, TPC masking is untested on Hopper,"
				" and will likely yield incorrect results! Proceed with caution.\n");
	*num_tpcs = num_sms/sms_per_tpc;
	return 0;
abort_cuda:
	//cuGetErrorName(res, &err_str);
	//fprintf(stderr, "libsmctrl: CUDA call failed due to %s. Failing with EIO...\n", err_str);
	return EIO;
}



inline __global__ void ker(int kernel_id, int stream_num) {
    uint64_t start_time = GlobalTimer64();
    uint32_t sm_id = GetSMID();
    int tidx = threadIdx.x;
    // The global block index.
    int block_id = blockIdx.x + blockIdx.y * gridDim.x + gridDim.x * gridDim.y * blockIdx.z;
    
    uint64_t start = clock64();
    uint64_t end = start + (uint64_t)(1000000000);
    while(clock64() < end) {
	    float dummy = 0.0;
	    for(int i=0;i<1000;i++){
		    dummy += tanf(dummy) + logf(dummy);
	    }
    }

    uint64_t end_time = GlobalTimer64();
    if(tidx == 0) {
        printf("kernel: %d, stream: %d, block: (%d, %d, %d), sm_id=%u, start_time=%llu, end_time=%llu, exec_time=%llu\n", 
           kernel_id, stream_num, blockIdx.x, blockIdx.y, blockIdx.z, sm_id, start_time, end_time, end_time-start_time);
    }
    
}


int main() {
	cudaStream_t streams[7];
        uint64_t mask = 0x3full;
        for(int i = 0; i < 7; i++) {
                        cudaStreamCreate(&streams[i]);
                        //CreateCUDAStreamWithPriorityAndMask(stream_priority, sm_mask, &streams[i]);
                        libsmctrl_set_stream_mask(streams[i], ~mask);
                        mask <<= 6;
        }
	auto kernel = &ker;
	dim3 grid(12,1,1);
       // #pragma unroll
        for (int i = 0; i < 1; i++){
                for (int j = 0; j < 5; j++){
		     auto now = std::chrono::high_resolution_clock::now();
                            auto duration = now.time_since_epoch();
                            auto nanoseconds = std::chrono::duration_cast<std::chrono::nanoseconds>(duration).count();
                            printf("start time of kernel(%d, %d) is %llu\n", i, j, nanoseconds);
                     kernel<<<grid, 256, 0, streams[(i*5+j)%7]>>>(i*5+j,(i*5+j)%7);
		     //ker<<<16,256>>>();
                }
        }
	#pragma unroll
        for(int i=0;i<7;i++){
                cudaStreamSynchronize(streams[i]);
                cudaStreamDestroy(streams[i]);
        }
        cudaDeviceSynchronize();
        return 0;
}
