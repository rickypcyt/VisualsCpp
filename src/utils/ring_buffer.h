#pragma once
#include <atomic>
#include <vector>
#include <cstddef>

// Lock-free SPSC ring buffer for POD types
// Capacity must be a power of two for fast modulo

template<typename T, size_t Capacity>
class RingBuffer {
    static_assert((Capacity & (Capacity - 1)) == 0, "Capacity must be a power of two");
public:
    RingBuffer() : head(0), tail(0) {}

    // Returns false if buffer is full
    bool push(const T& item) {
        size_t h = head.load(std::memory_order_relaxed);
        size_t n = (h + 1) & (Capacity - 1);
        if (n == tail.load(std::memory_order_acquire)) return false; // full
        buffer[h] = item;
        head.store(n, std::memory_order_release);
        return true;
    }

    // Returns false if buffer is empty
    bool pop(T& item) {
        size_t t = tail.load(std::memory_order_relaxed);
        if (t == head.load(std::memory_order_acquire)) return false; // empty
        item = buffer[t];
        tail.store((t + 1) & (Capacity - 1), std::memory_order_release);
        return true;
    }

    // Returns number of items available to pop
    size_t size() const {
        size_t h = head.load(std::memory_order_acquire);
        size_t t = tail.load(std::memory_order_acquire);
        return (h + Capacity - t) & (Capacity - 1);
    }

    // Returns true if empty
    bool empty() const { return head.load() == tail.load(); }
    // Returns true if full
    bool full() const { return ((head.load() + 1) & (Capacity - 1)) == tail.load(); }

    // Clear buffer
    void clear() { head.store(0); tail.store(0); }

private:
    T buffer[Capacity];
    std::atomic<size_t> head;
    std::atomic<size_t> tail;
}; 