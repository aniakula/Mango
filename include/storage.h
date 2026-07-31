#pragma once

#include <cstddef>

namespace mango {

// Untyped byte buffer shared by tensor views (ATen-style storage).
class Storage {
public:
  explicit Storage(size_t bytes);
  ~Storage();

  Storage(const Storage &) = delete;
  Storage &operator=(const Storage &) = delete;
  Storage(Storage &&other) noexcept;
  Storage &operator=(Storage &&other) noexcept;

  void *data();
  const void *data() const;
  size_t bytes() const;

private:
  void release();

  size_t bytes_;
  void *data_;
};

} // namespace mango
