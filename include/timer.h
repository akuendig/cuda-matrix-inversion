#ifndef HEADER_TIMER_INCLUDED
#define HEADER_TIMER_INCLUDED

// http://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Online_algorithm

#define TIMER_BILLION 1000000000

#define TIMER_LOG(name, numMatrices, n) \
    printf(#name",%d,%d,%.4f,%lu\r\n", numMatrices, n, TIMER_ELAPSED(name), TIMER_ELAPSED_NS(name));

#ifdef __APPLE__
#include <sys/time.h>
#define TIMER_INIT(name) \
    struct timeval timer_start_##name = { 0 }; \
    struct timeval timer_time_##name = { 0 };

#define TIMER_ACC_INIT(name) \
    struct timeval timer_total_##name = { 0 }; \
    size_t timer_n_##name = 0; \
    double timer_delta_##name = 0; \
    double timer_mean_##name = 0; \
    double timer_M2_##name = 0;

#define TIMER_START(name) \
    gettimeofday(&timer_start_##name, NULL);

#define TIMER_STOP(name) \
    gettimeofday(&timer_time_##name, NULL); \
    timeval_sub(&timer_time_##name, &timer_start_##name);

#define TIMER_ACC(name) \
    timer_n_##name += 1; /* n = n+1 */ \
    timeval_add(&timer_total_##name, &timer_time_##name); /* total += time */ \
    timer_delta_##name = TIMER_ELAPSED(name) - timer_mean_##name; /* delta = (time-mean) */ \
    timer_mean_##name = timer_mean_##name + timer_delta_##name/timer_n_##name; /* mean = mean+delta/n */ \
    timer_M2_##name = timer_M2_##name + timer_delta_##name*(TIMER_ELAPSED(name) - timer_mean_##name); /* M2 = M2 + delta*(time-mean) */

#define TIMER_ELAPSED_NS(name) \
    ((timer_time_##name.tv_sec*TIMER_BILLION) + (timer_time_##name.tv_usec*1000))

#define TIMER_ELAPSED(name) \
    timer_to_ms(timer_time_##name)

#define TIMER_TOTAL(name) \
    timer_to_ms(timer_total_##name)

#define TIMER_MEAN(name) \
    (timer_mean_##name)

#define TIMER_VARIANCE(name) \
    (timer_M2_##name / (timer_n_##name - 1))

#define TIMER_ACC_RESET(name) \
    timer_n_##name = 0; \
    timer_total_##name = 0; \
    timer_mean_##name = 0; \
    timer_M2_##name = 0;

static void timeval_add(struct timeval *t1, const struct timeval *t2) {
    t1->tv_sec += t2->tv_sec;
    t1->tv_usec += t2->tv_usec;

    if (t1->tv_usec >= 1000*1000) {
        t1->tv_usec -= 1000*1000;
        t1->tv_sec++;
    }
}

static void timeval_sub(struct timeval *t1, const struct timeval *t2) {
    if (t1->tv_usec < t2->tv_usec) {
        ensure(t1->tv_sec >= 1, "No negative time possible");

        t1->tv_sec -= 1;
        t1->tv_usec += 1000*1000;
    }

    ensure(t1->tv_sec >= t2->tv_sec, "No negative time possible");
    t1->tv_sec -= t2->tv_sec;
    t1->tv_usec -= t2->tv_usec;
}

static double timer_to_ms(struct timeval tv) {
    return (tv.tv_sec*1000) + (tv.tv_usec/1000);
}

#else

#define TIMER_INIT(name) \
    struct timespec timer_start_##name = { 0 }; \
    struct timespec timer_time_##name = { 0 };

#define TIMER_ACC_INIT(name) \
    struct timespec timer_total_##name = { 0 }; \
    size_t timer_n_##name = 0; \
    double timer_delta_##name = 0; \
    double timer_mean_##name = 0; \
    double timer_M2_##name = 0;

#define TIMER_START(name) \
    clock_gettime(CLOCK_MONOTONIC, &timer_start_##name);

#define TIMER_STOP(name) \
    clock_gettime(CLOCK_MONOTONIC, &timer_time_##name); \
    timespec_sub(&timer_time_##name, &timer_start_##name);

#define TIMER_ACC(name) \
    timer_n_##name += 1; /* n = n+1 */ \
    timespec_add(&timer_total_##name, &timer_time_##name); /* total += time */ \
    timer_delta_##name = TIMER_ELAPSED(name) - timer_mean_##name; /* delta = time - mean */ \
    timer_mean_##name = timer_mean_##name + timer_delta_##name/timer_n_##name; /* mean = mean + delta/n */ \
    timer_M2_##name = timer_M2_##name + timer_delta_##name*(TIMER_ELAPSED(name) - timer_mean_##name); /* M2 = M2 + delta*(time-mean) */

#define TIMER_ELAPSED_NS(name) \
    (ts.tv_sec*TIMER_BILLION + ts.tv_nsec)

#define TIMER_ELAPSED(name) \
    timer_to_ms(timer_time_##name)

#define TIMER_TOTAL(name) \
    timer_to_ms(timer_total_##name)

#define TIMER_MEAN(name) \
    (timer_mean_##name)

#define TIMER_VARIANCE(name) \
    (timer_M2_##name / (timer_n_##name - 1))

#define TIMER_ACC_RESET(name) \
    timer_n_##name = 0; \
    memset(&timer_total_##name, 0, sizeof(struct timespec)); \
    timer_mean_##name = 0; \
    timer_M2_##name = 0;

static void timespec_add(struct timespec *t1, const struct timespec *t2) {
    t1->tv_sec += t2->tv_sec;
    t1->tv_nsec += t2->tv_nsec;

    if (t1->tv_nsec >= TIMER_BILLION) {
        t1->tv_nsec -= TIMER_BILLION;
        t1->tv_sec++;
    }
}

static void timespec_sub(struct timespec *t1, const struct timespec *t2) {
    if (t1->tv_nsec < t2->tv_nsec) {
        ensure(t1->tv_sec >= 1, "No negative time possible");

        t1->tv_sec -= 1;
        t1->tv_nsec += TIMER_BILLION;
    }

    ensure(t1->tv_sec >= t2->tv_sec, "No negative time possible");
    t1->tv_nsec -= t2->tv_nsec;
    t1->tv_sec -= t2->tv_sec;
}

static void timespec_mul(struct timespec *t1, double mul) {
    double sec = t1->tv_sec * mul;
    double nsec = (sec - floor(sec))*TIMER_BILLION + t1->tv_nsec * mul;

    t1->tv_sec = floor(sec);
    t1->tv_nsec = floor(nsec);
}

static void timespec_div(struct timespec *t1, double div) {
    double sec = t1->tv_sec / div;
    double nsec = (sec - floor(sec))*TIMER_BILLION + t1->tv_nsec / div;

    t1->tv_sec = floor(sec);
    t1->tv_nsec = floor(nsec);
}

static double timer_to_ms(struct timespec ts) {
    return ts.tv_sec*1000.0 + ts.tv_nsec/1000.0/1000.0;
}

static struct timespec ms_to_time(double ms) {
    struct timespec ts = { 0 };

    ts->tv_sec = floor(ms/1000.0);
    ts->tv_nsec = floor((ms - ts.tv_sec*1000)*1000.0*1000.0);

    return ts;
}
#endif // __APPLE__

#endif // HEADER_TIMER_INCLUDED