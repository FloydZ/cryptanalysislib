#ifndef DECODING_CPUCYCLES_H
#define DECODING_CPUCYCLES_H

#include <stdio.h>
#include <stdlib.h>

#if defined(USE_ARM)
long long cpucycles(void) {
	unsigned long long result;
    __asm__ __volatile__ ("mrs %0, cntvct_el0" : "=r" (result));
	return result;
}
#elif defined(USE_AVX2)
long long cpucycles(void) {
	unsigned long long result;
	asm volatile(".byte 15;.byte 49;shlq $32,%%rdx;orq %%rdx,%%rax"
			: "=a"(result)::"%rdx");
	return result;
}
#else

// backup definition. If everything fails.
long long cpucycles(void) {
	return -1;
}
#endif

// taken from https://cpucycles.cr.yp.to/download.html
static double osfreq(void)
{
	FILE *f;
	char *x;
	double result;
	int s;

	f = fopen("/etc/cpucyclespersecond", "r");
	if (f) {
		s = fscanf(f,"%lf",&result);
		fclose(f);
		if (s > 0) return result;
	}

	f = fopen("/sys/devices/system/cpu/cpu0/cpufreq/scaling_setspeed", "r");
	if (f) {
		s = fscanf(f,"%lf",&result);
		fclose(f);
		if (s > 0) return 1000.0 * result;
	}

	f = fopen("/sys/devices/system/cpu/cpu0/cpufreq/scaling_max_freq", "r");
	if (f) {
		s = fscanf(f,"%lf",&result);
		fclose(f);
		if (s > 0) return 1000.0 * result;
	}

	f = fopen("/sys/devices/system/cpu/cpu0/clock_tick", "r");
	if (f) {
		s = fscanf(f,"%lf",&result);
		fclose(f);
		if (s > 0) return result;
	}

	f = fopen("/proc/cpuinfo","r");
	if (f) {
		for (;;) {
			s = fscanf(f,"cpu MHz : %lf",&result);
			if (s > 0) break;
			if (s == 0) s = fscanf(f,"%*[^\n]\n");
			if (s < 0) { result = 0; break; }
		}
		fclose(f);
		if (result) return 1000000.0 * result;
	}

	f = fopen("/proc/cpuinfo","r");
	if (f) {
		for (;;) {
			s = fscanf(f,"clock : %lf",&result);
			if (s > 0) break;
			if (s == 0) s = fscanf(f,"%*[^\n]\n");
			if (s < 0) { result = 0; break; }
		}
		fclose(f);
		if (result) return 1000000.0 * result;
	}

	f = fopen("/proc/cpuinfo","r");
	if (f) {
		for (;;) {
			s = fscanf(f,"cpu MHz static : %lf",&result);
			if (s > 0) break;
			if (s == 0) s = fscanf(f,"%*[^\n]\n");
			if (s < 0) { result = 0; break; }
		}
		fclose(f);
		if (result) return 1000000.0 * result;
	}

	f = popen("sysctl hw.cpufrequency 2>/dev/null","r");
	if (f) {
		s = fscanf(f,"hw.cpufrequency: %lf",&result);
		pclose(f);
		if (s > 0) if (result > 0) return result;
	}

	f = popen("/usr/sbin/lsattr -E -l proc0 -a frequency 2>/dev/null","r");
	if (f) {
		s = fscanf(f,"frequency %lf",&result);
		pclose(f);
		if (s > 0) return result;
	}

	f = popen("/usr/sbin/psrinfo -v 2>/dev/null","r");
	if (f) {
		for (;;) {
			s = fscanf(f," The %*s processor operates at %lf MHz",&result);
			if (s > 0) break;
			if (s == 0) s = fscanf(f,"%*[^\n]\n");
			if (s < 0) { result = 0; break; }
		}
		pclose(f);
		if (result) return 1000000.0 * result;
	}

	x = getenv("cpucyclespersecond");
	if (x) {
		s = sscanf(x,"%lf",&result);
		if (s > 0) return result;
	}

	return 2399987654.0;
}
#endif//DECODING_CPUCYCLES_H
