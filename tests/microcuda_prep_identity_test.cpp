// Host-only regression for preparation submission failures. Compile with C++20,
// tools/microcuda/runtime.cpp's HIP/OpenSSL dependencies; no GPU is
// initialized.
#include "../tools/microcuda/runtime.cpp"
#include <cassert>
int main() {
  constexpr uint64_t address = 0x100000000ull;
  track_alloc(address, 4096);
  Function f{};
  f.prep = reinterpret_cast<hipFunction_t>(uintptr_t{1});
  f.prep_slots = 8;
  f.identity_at = 1;
  unsigned char args[8];
  std::memcpy(args, &address, 8);
  int result = 701, launches = 0;
  auto launch = [&](auto...) {
    ++launches;
    return result;
  };
  assert(identity_prep(&f, args, nullptr, launch) == 701);
  assert(f.prepared.empty());
  result = 0;
  assert(identity_prep(&f, args, nullptr, launch) == 0);
  assert(f.prepared.size() == 1 && launches == 2);
  assert(identity_prep(&f, args, nullptr, launch) == 0 && launches == 2);
  const uint64_t epoch = f.prepared[0].epoch;
  track_write(address + 8, 8);
  result = 701;
  assert(identity_prep(&f, args, nullptr, launch) == 701);
  assert(f.prepared[0].epoch == epoch);
  result = 0;
  assert(identity_prep(&f, args, nullptr, launch) == 0 && launches == 4);
  assert(f.prepared[0].epoch != epoch);
  assert(identity_prep(&f, args, nullptr, launch) == 0 && launches == 4);
  std::puts("failed initial/updated preparations retry; successful unchanged "
            "weights reuse");
}
