#ifndef HEADER_TIMER_INCLUDED
#define HEADER_TIMER_INCLUDED

#ifdef __APPLE__
#define TIMER_INIT(name) \
    clock_t timer_start_##name = 0; \
    clock_t timer_time_##name = 0;

#define TIMER_ACC_INIT(name) \
    clock_t timer_total_##name = 0;

#define TIMER_START(name) \
    timer_start_##name = clock();

#define TIMER_STOP(name) \
    timer_time_##name = clock() - timer_start_##name;

#define TIMER_ACC(name) \
    timer_total_##name += timer_time_##name;

#define TIMER_ACC_RESET(name) \
    timer_total_##name = 0;

#define TIMER_LOG(name) \
    printf("Timer " #name ": %lucycles\n", timer_time_##name);

#else

#define TIMER_INIT(name) \
    struct timespec timer_start_##name = { 0 }; \
    struct timespec timer_time_##name = { 0 };

#define TIMER_ACC_INIT(name) \
    struct timespec timer_total_##name = { 0 };

#define TIMER_START(name) \
    clock_gettime(CLOCK_MONOTONIC, &timer_start_##name);

#define TIMER_STOP(name) \
    clock_gettime(CLOCK_MONOTONIC, &timer_time_##name); \
    time_sub(&timer_time_##name, &timer_start_##name);

#define TIMER_ACC(name) \
    time_add(&timer_total_##name, &timer_time_##name);

#define TIMER_ACC_RESET(name) \
    timer_total_##name = { 0 };

#define TIMER_LOG(name) \
    printf("Timer " #name ": %.4fms \n", time_to_ms(&timer_time_##name));

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

static void time_div(struct timespec *t1, double div) {
    double sec = t1->tv_sec / div;
    double nsec = (sec - floor(sec))*TIMER_BILLION + t1->tv_nsec / div;

    t1->tv_sec = floor(sec);
    t1->tv_nsec = floor(nsec);
}

static double time_to_ms(struct timespec *t1) {
    return t1->tv_sec*1000.0 + t1->tv_nsec/1000.0/1000.0;
}
#endif // __APPLE__

#endif // HEADER_TIMER_INCLUDED