/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

// See docs in ../ops/string_ops.cc.
#include <algorithm>
#include <numeric>
#include "tensorflow/core/framework/kernel_def_builder.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/core/stringpiece.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/types.h"

#if defined(__AVX512F__)
#include <immintrin.h>
#endif

namespace tensorflow {
using thread::ThreadPool;
namespace {
// Split input string `str` based on a character delimiter.
// Returns a vector of StringPieces which are valid as long as input `str`
// is valid.
// Note: The single character delimiter is a common case and is implemented as
// a series of finds in the input string, making it much more effcient than
// SplitOnCharSet.

class WorkerInfo{
 public:
  explicit WorkerInfo(int num_threads, int64 batch_size)
      :output_size(0), counter_for_thread(0),
      thread_index(0), max_num_entries(0) {
      const int kReserveSize = 64;
      int numReserve = batch_size * kReserveSize / num_threads ?
                       batch_size * kReserveSize / num_threads : 1;
      tokens_buffer.reserve(numReserve);
      num_indices_buffer.assign(batch_size, 0);
  }

  std::vector<StringPiece> tokens_buffer;
  int64 output_size;
  int64 counter_for_thread;
  int64 thread_index;
  int64 max_num_entries;
  std::vector<int64> num_indices_buffer;
};

template <typename Predicate>
std::vector<StringPiece> SplitOnChar(const string& str,
                                     const char delim, Predicate p) {
  std::vector<StringPiece> result;
  StringPiece text(str);
  auto f = text.find(delim);
  while (f != StringPiece::npos) {
    StringPiece token = text.substr(0, f);
    if (p(token)) {
      result.emplace_back(token);
    }
    text.remove_prefix(f + 1);
    f = text.find(delim);
  }
  if (p(text)) {
    result.push_back(text);
  }
  return result;
}

// Split input string `str` based on a set of character delimiters.
// Returns a vector of StringPieces which are valid as long as input `str`
// is valid.
// Based on str_util::Split.
template <typename Predicate>
std::vector<StringPiece> SplitOnCharSet(const string& str,
                                        const string& delim_set, Predicate p) {
  std::vector<StringPiece> result;
  StringPiece text(str);
  StringPiece delims(delim_set);
  size_t token_start = 0;
  for (size_t i = 0; i < text.size() + 1; i++) {
    if ((i == text.size()) || (delims.find(text[i]) != StringPiece::npos)) {
      StringPiece token(text.data() + token_start, i - token_start);
      if (p(token)) {
        result.emplace_back(token);
      }
      token_start = i + 1;
    }
  }
  return result;
}

// Split input string `str` based on given delimiter.
// Returns a vector of StringPieces which are valid as long as input `str`
// is valid.
template <typename Predicate>
std::vector<StringPiece> Split(const string& str, const string& delimiter,
                               Predicate predicate) {
  if (str.empty()) {
    return std::vector<StringPiece>();
  }
  if (delimiter.empty()) {
    std::vector<StringPiece> result;
    result.resize(str.size());
    for (size_t i = 0; i < str.size(); ++i) {
      result[i] = StringPiece(str.data() + i, 1);
    }
    return result;
  }
  if (delimiter.size() == 1) {
    return SplitOnChar(str, delimiter[0], predicate);
  }
  return SplitOnCharSet(str, delimiter, predicate);
}

std::vector<StringPiece> SplitV2(const string& str, StringPiece sep,
                                 int maxsplit) {
  // This SplitV2 method matches the behavior of python's str.split:
  //   If sep is given, consecutive delimiters are not grouped together
  //   and are deemed to delimit empty strings (for example, '1,,2'.split(',')
  //   returns ['1', '', '2']). The sep argument may consist of multiple
  //   characters (for example, '1<>2<>3'.split('<>') returns ['1', '2', '3']).
  //   Splitting an empty string with a specified separator returns [''].
  //
  //   If sep is not specified or is None, a different splitting algorithm is
  //   applied: runs of consecutive whitespace are regarded as a single
  //   separator, and the result will contain no empty strings at the start or
  //   end if the string has leading or trailing whitespace. Consequently,
  //   splitting an empty string or a string consisting of just whitespace
  //   with a None separator returns [].

  std::vector<StringPiece> result;

  StringPiece text(str);
  if (maxsplit == 0) {
    result.emplace_back(text);
    return result;
  }

  if (sep.empty()) {
    StringPiece token;
    // Remove leading whitespaces.
    str_util::RemoveLeadingWhitespace(&text);
    int split = 0;
    while (str_util::ConsumeNonWhitespace(&text, &token)) {
      result.push_back(token);
      str_util::RemoveLeadingWhitespace(&text);
      ++split;
      if (maxsplit > 0 && split == maxsplit) {
        result.push_back(text);
        return result;
      }
    }
    return result;
  }
  auto p = std::search(text.begin(), text.end(), sep.begin(), sep.end());
  int split = 0;
  while (p != text.end()) {
    StringPiece token = text.substr(0, p - text.begin());
    result.push_back(token);
    text.remove_prefix(token.size());
    text.remove_prefix(sep.size());
    ++split;
    if (maxsplit > 0 && split == maxsplit) {
      result.push_back(StringPiece(text));
      return result;
    }
    p = std::search(text.begin(), text.end(), sep.begin(), sep.end());
  }
  result.push_back(text);
  return result;
}


int64 SplitV2Inplace(const string& str, StringPiece sep,
                     int maxsplit, std::vector<StringPiece>& result) {
  StringPiece text(str);
  if (maxsplit == 0) {
    result.emplace_back(text);
    return 1;
  }

  if (sep.empty()) {
    StringPiece token;
    // Remove leading whitespaces.
    str_util::RemoveLeadingWhitespace(&text);
    int split = 0;
    while (str_util::ConsumeNonWhitespace(&text, &token)) {
      result.push_back(token);
      str_util::RemoveLeadingWhitespace(&text);
      ++split;
      if (maxsplit > 0 && split == maxsplit) {
        result.push_back(text);
        return split + 1;
      }
    }
    return split;
  }
  auto p = std::search(text.begin(), text.end(), sep.begin(), sep.end());
  int split = 0;
  while (p != text.end()) {
    StringPiece token = text.substr(0, p - text.begin());
    result.push_back(token);
    text.remove_prefix(token.size());
    text.remove_prefix(sep.size());
    ++split;
    if (maxsplit > 0 && split == maxsplit) {
      result.push_back(StringPiece(text));
      return split + 1;
    }
    p = std::search(text.begin(), text.end(), sep.begin(), sep.end());
  }
  result.push_back(text);
  return split + 1;
}

#if defined(__AVX512F__)
int64 SplitCharV2InplaceAvx512(const string& str, char sep, std::vector<StringPiece>& result) {
  const int bytes_in_vec = 64;
  int64 n_vecs = str.size() / bytes_in_vec;
  int64 remaining = str.size() - bytes_in_vec * n_vecs;

  StringPiece text(str);
  __m512i sep64 = _mm512_set1_epi8(sep);

  int64 l = 0;
  int64 splits = 0;
  for (int i = 0; i < n_vecs; i++) {
    int r_base = i * bytes_in_vec;
    __m512i piece64 = _mm512_loadu_si512(&text[r_base]);
    __mmask64 eq = _mm512_cmpeq_epi8_mask(piece64, sep64);
    int64 tzcnt = _tzcnt_u64(eq);
    while (tzcnt != 64) {
      int64 r = r_base + tzcnt;
      ++splits;
      result.push_back(text.substr(l, r - l));
      l = r + 1;

      eq ^= 1ULL << tzcnt;
      tzcnt = _tzcnt_u64(eq);
    }
  }

  int64 r_base = n_vecs * bytes_in_vec;
  __mmask64 mask = (1ULL << remaining) - 1;
  __m512i piece64 = _mm512_maskz_loadu_epi8 (mask, &text[r_base]);
  __mmask64 eq = _mm512_cmpeq_epi8_mask(piece64, sep64);
  int64 tzcnt = _tzcnt_u64(eq);
  while (tzcnt != 64) {
    int64 r = r_base + tzcnt;
    ++splits;
    result.push_back(text.substr(l, r - l));
    l = r + 1;

    eq ^= 1ULL << tzcnt;
    tzcnt = _tzcnt_u64(eq);
  }

  result.push_back(text.substr(l, text.length() - l));
  return splits + 1;
}
#endif

int64 SplitCharV2InplaceAvx2(const string& str, char sep, std::vector<StringPiece>& result) {
  const int bytes_in_vec = 32;
  int64 n_vecs = str.size() / bytes_in_vec;
  int64 remaining = str.size() - bytes_in_vec * n_vecs;

  StringPiece text(str);
  __m256i sep32 = _mm256_set1_epi8(sep);

  int32 l = 0;
  int32 prev_size = result.size();
  int r_base = 0;
  for (int i = 0; i < n_vecs; i++) {
      __m256i piece32 = _mm256_loadu_si256((__m256i *) &text[r_base]);
      __mmask32 eq = _mm256_cmpeq_epi8_mask(piece32, sep32);
      int32 tzcnt = _tzcnt_u32(eq);
      while (tzcnt != 32) {
          result.emplace_back(&text[l], r_base + tzcnt - l);
          l = r_base + tzcnt + 1;

          eq ^= 1 << tzcnt;
          tzcnt = _tzcnt_u32(eq);
      }
      r_base += bytes_in_vec;
  }

  __mmask32 mask = (1ULL << remaining) - 1;
  __m256i piece32 = _mm256_maskz_loadu_epi8(mask, &text[r_base]);
  __mmask32 eq = _mm256_cmpeq_epi8_mask(piece32, sep32);
  int32 tzcnt = _tzcnt_u32(eq);
  while (tzcnt != 32) {
      result.emplace_back(&text[l], r_base + tzcnt - l);
      l = r_base + tzcnt + 1;

      eq ^= 1 << tzcnt;
      tzcnt = _tzcnt_u32(eq);
  }

  result.emplace_back(&text[l], text.size() - l);
  return result.size() - prev_size;
}

}  // namespace


class StringSplitOp : public OpKernel {
 public:
  explicit StringSplitOp(OpKernelConstruction* context)
      : OpKernel(context), skip_empty_(true),
      element_cost_(0), result_cost_(0) {
    bool skip_empty;
    // By default skip_empty_ is true. We only get the value from attr if it is
    // available, so that it is backward compatible.
    if (context->GetAttr("skip_empty", &skip_empty).ok()) {
      skip_empty_ = skip_empty;
    }
  }

  void ParallelSplit(OpKernelContext* ctx,
                      const Eigen::TensorMap<
                        Eigen::Tensor<const string, 1, 1, long>,
                        16, Eigen::MakePointer> input_vec,
                      const int64 batch_size, const string& delimiter) {
    ThreadPool* thread_pool =
    ctx->device()->tensorflow_cpu_worker_threads()->workers;
    const int64 num_threads = thread_pool->NumThreads() + 1;

    std::vector<int64> num_indices(batch_size);
    num_indices[0] = 0;

    std::vector<WorkerInfo> w_array;
    for (int i = 0; i < num_threads; i++) {
      WorkerInfo w(num_threads, batch_size);
      w_array.emplace_back(w);
    }
    std::vector<std::vector<int64>> id_to_worker(batch_size);

    thread_pool->ParallelForWithWorkerId(
      batch_size,
      element_cost_,
      [&w_array, &id_to_worker, &input_vec,
      &delimiter, ctx, this, &num_indices]
      (int64 start, int64 end, int64 worker_id){
        int64 position_in_worker = 0;
        for (int64 i = start; i < end; ++i) {
          std::vector<StringPiece> parts =
              skip_empty_ ?
               Split(input_vec(i), delimiter, str_util::SkipEmpty())
               : Split(input_vec(i), delimiter, str_util::AllowEmpty());
          int64 n_entries = parts.size();
          id_to_worker[i].emplace_back(worker_id);
          id_to_worker[i].emplace_back(w_array[worker_id].counter_for_thread);
          id_to_worker[i].emplace_back(w_array[worker_id].output_size);
          num_indices[i] = n_entries;
          position_in_worker += n_entries;
          w_array[worker_id].num_indices_buffer[
              w_array[worker_id].counter_for_thread] = n_entries;
          w_array[worker_id].output_size += n_entries;
          w_array[worker_id].max_num_entries =
              std::max(w_array[worker_id].max_num_entries, n_entries);
          w_array[worker_id].tokens_buffer.insert(
              w_array[worker_id].tokens_buffer.end(),
              std::make_move_iterator(parts.begin()),
              std::make_move_iterator(parts.end()));
          w_array[worker_id].counter_for_thread++;
       }
     });

    int64 output_size = 0;
    int64 max_num_entries = 0;
    for (int i = 0; i < num_threads; i++) {
      output_size += w_array[i].output_size;
      max_num_entries = std::max(w_array[i].max_num_entries, max_num_entries);
    }

    std::vector<int64> id_to_index(batch_size);
    id_to_index[0] = 0;
    for (size_t i = 1; i < batch_size; i++)
      id_to_index[i] = id_to_index[i-1] + num_indices[i-1];

    Tensor* sp_indices_t;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, TensorShape({output_size, 2}),
                                             &sp_indices_t));
    Tensor* sp_tokens_t;
    OP_REQUIRES_OK(
       ctx, ctx->allocate_output(1, TensorShape({output_size}), &sp_tokens_t));
    Tensor* sp_shape_t;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(2, TensorShape({2}), &sp_shape_t));

    auto sp_indices = sp_indices_t->matrix<int64>();
    auto sp_tokens = sp_tokens_t->vec<string>();
    auto sp_shape = sp_shape_t->vec<int64>();
    sp_shape(0) = batch_size;
    sp_shape(1) = max_num_entries;

    if (result_cost_ == 0) {
      uint64 start = 0;
      uint64 end = 0;
      size_t sample_id = rand() % batch_size;
      start = Env::Default()->NowNanos();
      int last_worker_id = id_to_worker[sample_id][0];
      int64 id_in_last_worker = id_to_worker[sample_id][1];
      int64 st = id_to_worker[sample_id][2];
      for (int64 j = 0;
          j < w_array[last_worker_id].num_indices_buffer[id_in_last_worker];
          j++) {
        size_t c = id_to_index[sample_id] + j;
        sp_indices(c, 0) = sample_id;
        sp_indices(c, 1) = j;
        sp_tokens(c).assign(
             w_array[last_worker_id].tokens_buffer[st+j].data(),
             w_array[last_worker_id].tokens_buffer[st+j].size());
      }
      end = Env::Default()->NowNanos();
      result_cost_ = end - start;
    }
    uint64 result_cost = result_cost_;

    thread_pool->ParallelForWithWorkerId(
      batch_size,
      result_cost,
      [&id_to_index, batch_size, &w_array,
       &id_to_worker, &sp_indices,
       &num_indices, &sp_tokens](int64 start, int64 end, int64 worker_id) {
        for (int64 i = start; i < end; i++) {
          int last_worker_id = id_to_worker[i][0];
          int64 id_in_last_worker = id_to_worker[i][1];
          int64 st = id_to_worker[i][2];
          for (int64 j = 0;
              j < w_array[last_worker_id].num_indices_buffer[id_in_last_worker];
              j++) {
              size_t c = id_to_index[i] + j;
              sp_indices(c, 0) = i;
              sp_indices(c, 1) = j;
              sp_tokens(c).assign(
                  w_array[last_worker_id].tokens_buffer[st+j].data(),
                  w_array[last_worker_id].tokens_buffer[st+j].size());
          }
        }
      });
  }

  void SequentialSplit(OpKernelContext* ctx,
                        const Eigen::TensorMap<
                        Eigen::Tensor<const string, 1, 1, long>,
                        16, Eigen::MakePointer> input_vec,
                        const int64 batch_size, const string& delimiter) {
    std::vector<StringPiece> tokens;
    static constexpr int kReserveSize = 4;
    tokens.reserve(batch_size * kReserveSize);
    int64 output_size = 0;
    int64 max_num_entries = 0;

    std::vector<int64> num_indices(batch_size);
    for (int64 i = 0; i < batch_size; ++i) {
      std::vector<StringPiece> parts =
          skip_empty_ ? Split(input_vec(i), delimiter, str_util::SkipEmpty())
                      : Split(input_vec(i), delimiter, str_util::AllowEmpty());
      int64 n_entries = parts.size();
      num_indices[i] = n_entries;
      output_size += n_entries;
      max_num_entries = std::max(max_num_entries, n_entries);
      tokens.insert(tokens.end(), std::make_move_iterator(parts.begin()),
                    std::make_move_iterator(parts.end()));
    }

    Tensor* sp_indices_t;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, TensorShape({output_size, 2}),
                                             &sp_indices_t));
    Tensor* sp_tokens_t;
    OP_REQUIRES_OK(
        ctx, ctx->allocate_output(1, TensorShape({output_size}), &sp_tokens_t));
    Tensor* sp_shape_t;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(2, TensorShape({2}), &sp_shape_t));

    auto sp_indices = sp_indices_t->matrix<int64>();
    auto sp_tokens = sp_tokens_t->vec<string>();
    auto sp_shape = sp_shape_t->vec<int64>();
    sp_shape(0) = batch_size;
    sp_shape(1) = max_num_entries;
    size_t c = 0;
    for (size_t i = 0; i < batch_size; ++i) {
      for (size_t j = 0; j < num_indices[i]; ++j) {
        sp_indices(c, 0) = i;
        sp_indices(c, 1) = j;
        sp_tokens(c).assign(tokens[c].data(), tokens[c].size());
        ++c;
      }
    }
  }

  void Compute(OpKernelContext* ctx) override {
    const Tensor* input_tensor;
    srand((unsigned)time(NULL));

    OP_REQUIRES_OK(ctx, ctx->input("input", &input_tensor));
    OP_REQUIRES(ctx, TensorShapeUtils::IsVector(input_tensor->shape()),
                errors::InvalidArgument("input must be a vector, got shape: ",
                                        input_tensor->shape().DebugString()));

    const auto input_vec = input_tensor->vec<string>();
    const int64 batch_size = input_vec.dimension(0);

    const Tensor* delimiter_tensor;
    OP_REQUIRES_OK(ctx, ctx->input("delimiter", &delimiter_tensor));
    OP_REQUIRES(
        ctx, TensorShapeUtils::IsScalar(delimiter_tensor->shape()),
        errors::InvalidArgument("delimiter must be a scalar, got shape: ",
                                delimiter_tensor->shape().DebugString()));
    const auto delimiter_vec = delimiter_tensor->flat<string>();
    const string& delimiter = delimiter_vec(0);

    uint64 start = 0;
    uint64 end = 0;

    if (element_cost_ == 0 && batch_size) {
      size_t sample_id = rand() % batch_size;
      std::vector<StringPiece> temp_for_warm_up =
            skip_empty_ ?
               Split(input_vec(sample_id), delimiter, str_util::SkipEmpty())
               : Split(input_vec(sample_id), delimiter, str_util::AllowEmpty());
      start = Env::Default()->NowNanos();
      temp_for_warm_up =
            skip_empty_ ?
             Split(input_vec(sample_id), delimiter, str_util::SkipEmpty())
             : Split(input_vec(sample_id), delimiter, str_util::AllowEmpty());
      end = Env::Default()->NowNanos();
      element_cost_ = end - start;
    }
    uint64 element_cost = element_cost_;

    if (element_cost * batch_size >= parallel_limit_) {
      ParallelSplit(ctx, input_vec, batch_size, delimiter);
    } else {
      SequentialSplit(ctx, input_vec, batch_size, delimiter);
    }
  }

 private:
  bool skip_empty_;
  uint64 element_cost_;
  uint64 result_cost_;
  const int64 parallel_limit_ = 240000;
};

class StringSplitV2Op : public OpKernel {
 public:
  explicit StringSplitV2Op(OpKernelConstruction* context)
      : OpKernel(context), maxsplit_(-1), element_cost_(0), result_cost_(0) {
    OP_REQUIRES_OK(context, context->GetAttr("maxsplit", &maxsplit_));
  }

  void ParallelSplitV2(OpKernelContext* ctx,
                      const Eigen::TensorMap<
                      Eigen::Tensor<const string, 1, 1, long>,
                      16, Eigen::MakePointer> input_vec,
                      const int64 batch_size, StringPiece sep) {
    ThreadPool* thread_pool =
        ctx->device()->tensorflow_cpu_worker_threads()->workers;
    const int64 num_threads = thread_pool->NumThreads() + 1;

    std::vector<int64> num_indices(batch_size);
    num_indices[0] = 0;

    std::vector<WorkerInfo> w_array;
    w_array.reserve(num_threads);
    for (int i = 0; i < num_threads; i++) {
      WorkerInfo w(num_threads, batch_size);
      w_array.emplace_back(w);
    }

    std::vector<std::vector<int64>> id_to_worker(batch_size);
    thread_pool->ParallelForWithWorkerId(
      batch_size,
      element_cost_,
      [&sep, &w_array, &id_to_worker,
       &input_vec, ctx, this,
       &num_indices](int64 start, int64 end, int64 worker_id) {
        int64 position_in_worker = 0;
        for (int64 i = start; i < end; ++i) {
          int64 n_entries = SplitV2Inplace(input_vec(i), sep, maxsplit_, 
                                           w_array[worker_id].tokens_buffer);
          id_to_worker[i].emplace_back(worker_id);
          id_to_worker[i].emplace_back(w_array[worker_id].counter_for_thread);
          id_to_worker[i].emplace_back(w_array[worker_id].output_size);
          num_indices[i] = n_entries;
          position_in_worker += n_entries;
          w_array[worker_id].num_indices_buffer[
              w_array[worker_id].counter_for_thread] = n_entries;
          w_array[worker_id].output_size += n_entries;
          w_array[worker_id].max_num_entries =
              std::max(w_array[worker_id].max_num_entries, n_entries);
          w_array[worker_id].counter_for_thread++;
       }
      });

    int64 output_size = 0;
    int64 max_num_entries = 0;
    for (int i = 0; i < num_threads; i++) {
      output_size += w_array[i].output_size;
      max_num_entries = std::max(w_array[i].max_num_entries, max_num_entries);
    }


    std::vector<int64> id_to_index(batch_size);
    id_to_index[0] = 0;
    for (size_t i = 1; i < batch_size; i++)
      id_to_index[i] = id_to_index[i-1] + num_indices[i-1];


    Tensor* sp_indices_t;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, TensorShape({output_size, 2}),
                                             &sp_indices_t));
    Tensor* sp_tokens_t;
    OP_REQUIRES_OK(
       ctx, ctx->allocate_output(1, TensorShape({output_size}), &sp_tokens_t));
    Tensor* sp_shape_t;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(2, TensorShape({2}), &sp_shape_t));

    auto sp_indices = sp_indices_t->matrix<int64>();
    auto sp_tokens = sp_tokens_t->vec<string>();
    auto sp_shape = sp_shape_t->vec<int64>();
    sp_shape(0) = batch_size;
    sp_shape(1) = max_num_entries;

    if (result_cost_ == 0) {
      uint64 start = 0;
      uint64 end = 0;
      size_t sample_id = rand() % batch_size;
      start = Env::Default()->NowNanos();
      int last_worker_id = id_to_worker[sample_id][0];
      int64 id_in_last_worker = id_to_worker[sample_id][1];
      int64 st = id_to_worker[sample_id][2];
      for (int64 j = 0;
       j < w_array[last_worker_id].num_indices_buffer[id_in_last_worker];
       j++) {
        size_t c = id_to_index[sample_id] + j;
        sp_indices(c, 0) = sample_id;
        sp_indices(c, 1) = j;
        sp_tokens(c).assign(
            w_array[last_worker_id].tokens_buffer[st+j].data(),
            w_array[last_worker_id].tokens_buffer[st+j].size());
      }
      end = Env::Default()->NowNanos();
      result_cost_ = end - start;
    }
    uint64 result_cost = result_cost_;

    thread_pool->ParallelForWithWorkerId(
      batch_size,
      result_cost,
      [&id_to_index, batch_size, &w_array,
       &id_to_worker, &sp_indices, &num_indices,
       &sp_tokens](int64 start, int64 end, int64 worker_id) {
        for (int64 i = start; i < end; i++) {
          int last_worker_id = id_to_worker[i][0];
          int64 id_in_last_worker = id_to_worker[i][1];
          int64 st = id_to_worker[i][2];
          for (int64 j = 0;
           j < w_array[last_worker_id].num_indices_buffer[id_in_last_worker];
           j++) {
              size_t c = id_to_index[i] + j;
              sp_indices(c, 0) = i;
              sp_indices(c, 1) = j;
              sp_tokens(c).assign(
                  w_array[last_worker_id].tokens_buffer[st+j].data(),
                  w_array[last_worker_id].tokens_buffer[st+j].size());
          }
        }
      });
  }

  void SequentialSplitV2(OpKernelContext* ctx,
                          const Eigen::TensorMap<
                          Eigen::Tensor<const string, 1, 1, long>,
                          16, Eigen::MakePointer> input_vec,
                          const int64 batch_size, StringPiece sep) {
    std::vector<StringPiece> tokens;
    static constexpr int kReserveSize = 64;
    const int64 reserved_size = batch_size * kReserveSize;
    tokens.reserve(reserved_size);
    int64 output_size = 0;
    int64 max_num_entries = 0;

    std::vector<int64> num_indices(batch_size);
    for (int64 i = 0; i < batch_size; ++i) {
      int64 n_entries = SplitV2Inplace(input_vec(i), sep, maxsplit_, tokens);
      num_indices[i] = n_entries;
      output_size += n_entries;
      max_num_entries = std::max(max_num_entries, n_entries);
    }

    Tensor* sp_indices_t;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, TensorShape({output_size, 2}),
                                             &sp_indices_t));
    Tensor* sp_tokens_t;
    OP_REQUIRES_OK(
        ctx, ctx->allocate_output(1, TensorShape({output_size}), &sp_tokens_t));
    Tensor* sp_shape_t;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(2, TensorShape({2}), &sp_shape_t));

    auto sp_indices = sp_indices_t->matrix<int64>();
    auto sp_tokens = sp_tokens_t->vec<string>();
    auto sp_shape = sp_shape_t->vec<int64>();
    sp_shape(0) = batch_size;
    sp_shape(1) = max_num_entries;
    size_t c = 0;
    for (size_t i = 0; i < batch_size; ++i) {
      for (size_t j = 0; j < num_indices[i]; ++j) {
        sp_indices(c, 0) = i;
        sp_indices(c, 1) = j;
        sp_tokens(c).assign(tokens[c].data(), tokens[c].size());
        ++c;
      }
    }
  }

  void SequentialSplitCharV2(OpKernelContext* ctx,
                          const Eigen::TensorMap<
                          Eigen::Tensor<const string, 1, 1, long>,
                          16, Eigen::MakePointer> input_vec,
                          const int64 batch_size, char sep) {
    std::vector<StringPiece> tokens;
    static constexpr int kReserveSize = 64;
    const int64 reserved_size = batch_size * kReserveSize;
    tokens.reserve(reserved_size);
    int64 output_size = 0;
    int64 max_num_entries = 0;

    std::vector<int64> num_indices(batch_size);
    // LOG(INFO) << "Batch size: " << batch_size;
    for (int64 i = 0; i < batch_size; i++) {
      int64 n_entries = SplitCharV2InplaceAvx2(input_vec(i), sep, tokens);
      num_indices[i] = n_entries;
      output_size += n_entries;
      max_num_entries = std::max(max_num_entries, n_entries);
    }

    Tensor* sp_indices_t;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, TensorShape({output_size, 2}),
                                             &sp_indices_t));
    Tensor* sp_tokens_t;
    OP_REQUIRES_OK(
        ctx, ctx->allocate_output(1, TensorShape({output_size}), &sp_tokens_t));
    Tensor* sp_shape_t;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(2, TensorShape({2}), &sp_shape_t));

    auto sp_indices = sp_indices_t->matrix<int64>();
    auto sp_tokens = sp_tokens_t->vec<string>();
    auto sp_shape = sp_shape_t->vec<int64>();
    sp_shape(0) = batch_size;
    sp_shape(1) = max_num_entries;
    size_t c = 0;
    for (size_t i = 0; i < batch_size; ++i) {
      for (size_t j = 0; j < num_indices[i]; ++j) {
        sp_indices(c, 0) = i;
        sp_indices(c, 1) = j;
        sp_tokens(c).assign(tokens[c].data(), tokens[c].size());
        ++c;
      }
    }
  }

  void Compute(OpKernelContext* ctx) override {
    const Tensor* input_tensor;
    OP_REQUIRES_OK(ctx, ctx->input("input", &input_tensor));
    OP_REQUIRES(ctx, TensorShapeUtils::IsVector(input_tensor->shape()),
                errors::InvalidArgument("input must be a vector, got shape: ",
                                        input_tensor->shape().DebugString()));

    const auto input_vec = input_tensor->vec<string>();
    const int64 batch_size = input_vec.dimension(0);

    const Tensor* sep_tensor;
    OP_REQUIRES_OK(ctx, ctx->input("sep", &sep_tensor));
    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(sep_tensor->shape()),
                errors::InvalidArgument("sep must be a scalar, got shape: ",
                                        sep_tensor->shape().DebugString()));
    const auto sep_vec = sep_tensor->flat<string>();
    StringPiece sep(sep_vec(0));

    if (sep.size() == 1 && maxsplit_ <= 0) {
      SequentialSplitCharV2(ctx, input_vec, batch_size, sep[0]);
    } else {
      SequentialSplitV2(ctx, input_vec, batch_size, sep);
    }

  //   uint64 start = 0;
  //   uint64 end = 0;

  //   if (element_cost_ == 0 && batch_size) {
  //     size_t sample_id = rand() % batch_size;
  //     std::vector<StringPiece> temp_for_warm_up =
  //         SplitV2(input_vec(sample_id), sep, maxsplit_);
  //     start = Env::Default()->NowNanos();
  //     temp_for_warm_up = SplitV2(input_vec(sample_id), sep, maxsplit_);
  //     end = Env::Default()->NowNanos();
  //     element_cost_ = end -start;
  //   }
  //   uint64 element_cost = element_cost_;
  //   if (element_cost * batch_size >= parallel_limit_) {
  //     ParallelSplitV2(ctx, input_vec, batch_size, sep);
  //   } else {
  //     SequentialSplitV2(ctx, input_vec, batch_size, sep);
  //   }
  }

 private:
  int maxsplit_;
  uint64 element_cost_;
  uint64 result_cost_;
  const int64 parallel_limit_ = 240000;
};

REGISTER_KERNEL_BUILDER(Name("StringSplit").Device(DEVICE_CPU), StringSplitOp);
REGISTER_KERNEL_BUILDER(Name("StringSplitV2").Device(DEVICE_CPU),
                        StringSplitV2Op);

}  // namespace tensorflow
