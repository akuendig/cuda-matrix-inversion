#ifndef HEADER_TIMER_INCLUDED
#define HEADER_TIMER_INCLUDED

// http://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Online_algorithm

#ifdef __APPLE__
#define TIMER_INIT(name) \
    clock_t timer_start_##name = 0; \
    clock_t timer_time_##name = 0;

#define TIMER_ACC_INIT(name) \
    clock_t timer_total_##name = 0; \
    size_t timer_n_##name = 0; \
    double timer_delta_##name = 0; \
    double timer_mean_##name = 0; \
    double timer_M2_##name = 0;

#define TIMER_START(name) \
    timer_start_##name = clock();

#define TIMER_STOP(name) \
    timer_time_##name = clock() - timer_start_##name;

#define TIMER_ACC(name) \
    timer_n_##name += 1; /* n = n+1 */ \
    timer_total_##name += timer_time_##name; /* total += time */ \
    timer_delta_##name = TIMER_ELAPSED(name) - timer_mean_##name; /* delta = (time-mean) */ \
    timer_mean_##name = timer_mean_##name + timer_delta_##name/timer_n_##name; /* mean = mean+delta/n */ \
    timer_M2_##name = timer_M2_##name + timer_delta_##name*(TIMER_ELAPSED(name) - timer_mean_##name); /* M2 = M2 + delta*(time-mean) */

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

#define TIMER_LOG(name) \
    printf("Timer " #name ": %lucycles\n", timer_time_##name);

static double timer_to_ms(clock_t cycles) {
    return (cycles*1000) / CLOCKS_PER_SEC;
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
    time_sub(&timer_time_##name, &timer_start_##name);

#define TIMER_ACC(name) \
    timer_n_##name += 1; /* n = n+1 */ \
    time_add(&timer_total_##name, &timer_time_##name); /* total += time */ \
    timer_delta_##name = TIMER_ELAPSED(name) - timer_mean_##name; /* delta = time - mean */ \
    timer_mean_##name = timer_mean_##name + timer_delta_##name/timer_n_##name; /* mean = mean + delta/n */ \
    timer_M2_##name = timer_M2_##name + timer_delta_##name*(TIMER_ELAPSED(name) - timer_mean_##name); /* M2 = M2 + delta*(time-mean) */

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

#define TIMER_LOG(name) \
    printf("Timer " #name ": %.4fms \n", TIMER_ELAPSED(name));

#define TIMER_BILLION 1000000000

static void time_add(struct timespec *t1, const struct timespec *t2) {
    t1->tv_sec += t2->tv_sec;
    t1->tv_nsec += t2->tv_nsec;

    if (t1->tv_nsec >= TIMER_BILLION) {
        t1->tv_nsec -= TIMER_BILLION;
        t1->tv_sec++;
    }
}

static void time_sub(struct timespec *t1, const struct timespec *t2) {
    if (t1->tv_nsec < t2->tv_nsec) {
        ensure(t1->tv_sec >= 1, "No negative time possible");

        t1->tv_sec -= 1;
        t1->tv_nsec += TIMER_BILLION;
    }

    ensure(t1->tv_sec >= t2->tv_sec, "No negative time possible");
    t1->tv_nsec -= t2->tv_nsec;
    t1->tv_sec -= t2->tv_sec;
}

static void time_mul(struct timespec *t1, double mul) {
    double sec = t1->tv_sec * mul;
    double nsec = (sec - floor(sec))*TIMER_BILLION + t1->tv_nsec * mul;

    t1->tv_sec = floor(sec);
    t1->tv_nsec = floor(nsec);
}

static void time_div(struct timespec *t1, double div) {
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
    ts->tv_nsec = floor(((ms/1000.0)-ts->tv_sec)*1000.0*1000.0);

    return ts;
}
#endif // __APPLE__

#endif // HEADER_TIMER_INCLUDED