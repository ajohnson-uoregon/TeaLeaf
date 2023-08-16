#include <Kokkos_Core.hpp>
#include "ClangRewriteMacros.h"

std::vector<std::string> clang_rewrite_literal_names = {"intermed", "x", "y", "idx", "kk", "jj"};

class FakeClass {
  int* x;

  template<class T>
  auto decl_m();

  template<class T>
  auto decl_r();

  auto del_m();
};

// template <class T>
// [[clang::matcher("alloc")]]
// auto FakeClass::decl_m() {
// 	[[clang::rewrite_setup]]
// 		int align, array_size;
//
// 	[[clang::matcher_block]] {
// 		this->x = (T*)aligned_alloc(align, sizeof(T)*array_size);
// 	}
// }
//
// template <class T>
// [[clang::replace("alloc")]]
// auto FakeClass::decl_r() {
//   [[clang::rewrite_setup]]
//     int array_size;
//   [[clang::rewrite_setup]]
//     T* x;
//   [[clang::rewrite_setup]]
//     T* device_x = &clang_rewrite::code_literal("device_" + clang_rewrite::to_str(x));
//   [[clang::rewrite_setup]]
//     T* hostmirror_x = &clang_rewrite::code_literal("hostmirror_" + clang_rewrite::to_str(x));
//
//   [[clang::matcher_block]] {
//     device_x = new Kokkos::View<T*>("label change me", array_size);
//     hostmirror_x = new typename Kokkos::View<T*>::HostMirror();
//     *hostmirror_x = Kokkos::create_mirror_view(*device_x);
//   }
// }

// do not uncomment
// template <class T>
// [[clang::matcher("hpp")]]
// auto hpp_m() {
//
//   class Test {
//   [[clang::matcher_block]]
//     T* a;
//   };
// }
//
// template <class T>
// [[clang::replace("hpp")]]
// auto hpp_r() {
// 	[[clang::rewrite_setup]]
// 		T* a;
//   [[clang::rewrite_setup]]
//     std::string device_a = clang_rewrite::code_literal("device_" + clang_rewrite::to_str(a));
//   [[clang::rewrite_setup]]
//     std::string hostmirror_a = clang_rewrite::code_literal("hostmirror_" + clang_rewrite::to_str(a));
//
// 	[[clang::matcher_block]] {
// 		typename Kokkos::View<T*>* device_a;
//     typename Kokkos::View<T*>::HostMirror* hostmirror_a;
// 	}
// }

// [[clang::matcher("delete")]]
// auto FakeClass::del_m() {
//
// 	[[clang::matcher_block]] {
// 		free(x);
// 	}
// }
//
// [[clang::replace("delete")]]
// auto del_r() {
// 	[[clang::matcher_block]] {}
// }

[[clang::matcher("kokkos")]]
auto kokkos_m() {
	[[clang::rewrite_setup]]
		int k, N, l, M;

	[[clang::matcher_block]] {
		#pragma omp parallel for
		for (int i = k; i < N; ++i) {
      for (int j = l; j < M; ++j) {
        clang_rewrite::loop_body();
      }
    }
  }
}

[[clang::replace("kokkos")]]
auto kokkos_r() {
	[[clang::rewrite_setup]]
		int k, N, l, M, x, y;

	[[clang::matcher_block]] {
		Kokkos::parallel_for(x*y, KOKKOS_LAMBDA (const int idx) {
      const size_t kk = idx % x;
      const size_t jj = idx / x;

      if (k-1 < jj && jj < N && l-1 < kk && kk < M) {
        clang_rewrite::loop_body();
      }
		});
	}
}

[[clang::matcher("reduction")]]
auto red_m() {
	[[clang::rewrite_setup]]
		int k, N, l, M;
	[[clang::rewrite_setup]]
		double red;

	[[clang::matcher_block]] {
		#pragma omp parallel for reduction (+:red)
		for (int i = k; i < N; ++i) {
      for (int j = l; j < M; ++j) {
        clang_rewrite::loop_body();
      }
    }
  }
}

[[clang::replace("reduction")]]
auto red_r() {
	[[clang::rewrite_setup]]
		int k, N, l, M, x, y;
	[[clang::rewrite_setup]]
		double red;

	[[clang::matcher_block]] {
		Kokkos::parallel_reduce(x*y, KOKKOS_LAMBDA (const int idx, double& intermed) {
			const size_t kk = idx % x;
      const size_t jj = idx / x;

      if (k-1 < jj && jj < N && l-1 < kk && kk < M) {
        clang_rewrite::loop_body({{red, intermed}});
      }
		},
		red);
	}
}
