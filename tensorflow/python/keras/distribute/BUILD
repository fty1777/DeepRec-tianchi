# Description:
#   keras/distribute package is intended to serve as the centralized place for things
#   related to dist-strat used by Keras..

load("//tensorflow/core/platform:default/distribute.bzl", "distribute_py_test")
load("//tensorflow:tensorflow.bzl", "cuda_py_test")

package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],  # Apache 2.0
)

exports_files(["LICENSE"])

py_library(
    name = "distribute",
    srcs = [
        "__init__.py",
        "distributed_training_utils.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow/python:client_testlib",
        "//tensorflow/python/data",
        "//tensorflow/python/distribute:distribute_coordinator",
        "//tensorflow/python/distribute:distribute_lib",
        "//tensorflow/python/distribute:input_lib",
        "//tensorflow/python/distribute:one_device_strategy",
        "//tensorflow/python/distribute:reduce_util",
        "//tensorflow/python/distribute:values",
        "//tensorflow/python/eager:def_function",
        "//tensorflow/python/keras:activations",
        "//tensorflow/python/keras:backend",
        "//tensorflow/python/keras:callbacks",
        "//tensorflow/python/keras:callbacks_v1",
        "//tensorflow/python/keras:constraints",
        "//tensorflow/python/keras:engine_utils",
        "//tensorflow/python/keras:initializers",
        "//tensorflow/python/keras:losses",
        "//tensorflow/python/keras:mode_keys",
        "//tensorflow/python/keras:optimizers",
        "//tensorflow/python/keras:regularizers",
        "//tensorflow/python/keras:saving",
        "//tensorflow/python/keras/distribute:multi_worker_training_state",
        "//tensorflow/python/keras/mixed_precision/experimental:autocast_variable",
        "//tensorflow/python/keras/mixed_precision/experimental:policy",
        "//tensorflow/python/training/tracking:data_structures",
        "//tensorflow/tools/docs:doc_controls",
    ],
)

py_library(
    name = "multi_worker_training_state",
    srcs = [
        "multi_worker_training_state.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow/python/distribute:multi_worker_util",
    ],
)

cuda_py_test(
    name = "multi_worker_training_state_test",
    srcs = ["multi_worker_training_state_test.py"],
    additional_deps = [
        ":multi_worker_testing_utils",
        ":multi_worker_training_state",
        "//tensorflow/python:platform",
        "//tensorflow/python/distribute:combinations",
        "//tensorflow/python/distribute:multi_worker_test_base",
        "//tensorflow/python/keras",
    ],
    shard_count = 4,
    xla_enable_strict_auto_jit = True,
)

py_library(
    name = "distribute_strategy_test_lib",
    srcs = [
        "distribute_strategy_test.py",
    ],
    deps = [
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:training",
        "//tensorflow/python/distribute:combinations",
        "//tensorflow/python/distribute:distribute_lib",
        "//tensorflow/python/distribute:mirrored_strategy",
        "//tensorflow/python/distribute:strategy_combinations",
        "//tensorflow/python/distribute:tpu_strategy",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/eager:test",
        "//tensorflow/python/estimator:estimator_py",
        "//tensorflow/python/keras",
        "//third_party/py/numpy",
        "@absl_py//absl/testing:parameterized",
    ],
)

distribute_py_test(
    name = "keras_premade_models_test",
    srcs = ["keras_premade_models_test.py"],
    full_precision = True,
    main = "keras_premade_models_test.py",
    shard_count = 4,
    tags = [
        "multi_and_single_gpu",
    ],
    deps = [
        ":distribute_strategy_test_lib",
        ":keras_correctness_test_lib",
    ],
)

distribute_py_test(
    name = "distribute_strategy_test",
    srcs = ["distribute_strategy_test.py"],
    full_precision = True,
    main = "distribute_strategy_test.py",
    shard_count = 8,
    tags = [
        "multi_and_single_gpu",
        "no_rocm",  # times out on ROCm
        "no_windows_gpu",
        "notsan",
    ],
    deps = [
        ":distribute_strategy_test_lib",
    ],
)

distribute_py_test(
    name = "distributed_training_utils_test",
    srcs = ["distributed_training_utils_test.py"],
    full_precision = True,
    main = "distributed_training_utils_test.py",
    deps = [
        ":distribute",
        ":distribute_strategy_test_lib",
        "//tensorflow/python:platform",
        "//tensorflow/python:stateless_random_ops",
        "//tensorflow/python/keras:callbacks",
    ],
)

py_library(
    name = "keras_correctness_test_lib",
    srcs = [
        "keras_correctness_test_base.py",
        "keras_dnn_correctness_test.py",
        "keras_embedding_model_correctness_test.py",
        "keras_image_model_correctness_test.py",
        "keras_lstm_model_correctness_test.py",
        "keras_stateful_lstm_model_correctness_test.py",
    ],
    deps = [
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:training",
        "//tensorflow/python/distribute:combinations",
        "//tensorflow/python/distribute:distribute_lib",
        "//tensorflow/python/distribute:mirrored_strategy",
        "//tensorflow/python/distribute:strategy_combinations",
        "//tensorflow/python/distribute:tpu_strategy",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/eager:test",
        "//tensorflow/python/keras",
        "//tensorflow/python/keras:backend",
        "//third_party/py/numpy",
        "@absl_py//absl/testing:parameterized",
    ],
)

distribute_py_test(
    name = "keras_dnn_correctness_test",
    size = "medium",
    srcs = ["keras_dnn_correctness_test.py"],
    full_precision = True,
    main = "keras_dnn_correctness_test.py",
    # Shard count is set to an odd number to distribute tasks across
    # shards more evenly.
    shard_count = 19,
    tags = [
        "multi_and_single_gpu",
        "no_rocm",  # times out on ROCm
        "no_windows_gpu",
        "notsan",
    ],
    deps = [
        ":keras_correctness_test_lib",
    ],
)

distribute_py_test(
    name = "keras_embedding_model_correctness_test",
    size = "medium",
    srcs = ["keras_embedding_model_correctness_test.py"],
    full_precision = True,
    main = "keras_embedding_model_correctness_test.py",
    shard_count = 4,
    tags = [
        "multi_and_single_gpu",
        "no_rocm",  # times out on ROCm
        "no_windows_gpu",
        "notsan",
    ],
    deps = [
        ":keras_correctness_test_lib",
    ],
)

distribute_py_test(
    name = "keras_image_model_correctness_test",
    size = "medium",
    srcs = ["keras_image_model_correctness_test.py"],
    full_precision = True,
    main = "keras_image_model_correctness_test.py",
    shard_count = 8,
    tags = [
        "multi_and_single_gpu",
        "no_rocm",  # times out on ROCm
        "no_windows_gpu",
        "notsan",
    ],
    xla_enable_strict_auto_jit = False,  # Tensorflow also fails.
    deps = [
        ":keras_correctness_test_lib",
    ],
)

distribute_py_test(
    name = "keras_lstm_model_correctness_test",
    size = "medium",
    srcs = ["keras_lstm_model_correctness_test.py"],
    full_precision = True,
    main = "keras_lstm_model_correctness_test.py",
    # Shard count is set to an odd number to distribute tasks across
    # shards more evenly.
    shard_count = 31,
    tags = [
        "multi_and_single_gpu",
        "no_oss",  # b/136660639
        "no_windows_gpu",
        "notsan",
    ],
    deps = [
        ":keras_correctness_test_lib",
    ],
)

distribute_py_test(
    name = "keras_stateful_lstm_model_correctness_test",
    size = "medium",
    srcs = ["keras_stateful_lstm_model_correctness_test.py"],
    full_precision = True,
    main = "keras_stateful_lstm_model_correctness_test.py",
    shard_count = 4,
    tags = [
        "multi_and_single_gpu",
        "no_pip",
        "no_windows_gpu",
        "notsan",
    ],
    deps = [
        ":keras_correctness_test_lib",
    ],
)

distribute_py_test(
    name = "keras_utils_test",
    srcs = ["keras_utils_test.py"],
    full_precision = True,
    main = "keras_utils_test.py",
    shard_count = 4,
    tags = [
        "multi_and_single_gpu",
        "no_windows_gpu",
        "notsan",
    ],
    deps = [
        ":keras_test_lib",
        "//tensorflow/python/keras/distribute:distribute_strategy_test_lib",
    ],
)

py_library(
    name = "keras_test_lib",
    srcs = [
        "keras_utils_test.py",
    ],
    deps = [
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:training",
        "//tensorflow/python/distribute:combinations",
        "//tensorflow/python/distribute:mirrored_strategy",
        "//tensorflow/python/distribute:parameter_server_strategy",
        "//tensorflow/python/distribute:strategy_combinations",
        "//tensorflow/python/distribute:tpu_strategy",
        "//tensorflow/python/eager:test",
        "//tensorflow/python/keras",
        "//third_party/py/numpy",
        "@absl_py//absl/testing:parameterized",
    ],
)

cuda_py_test(
    name = "keras_optimizer_v2_test",
    srcs = ["keras_optimizer_v2_test.py"],
    additional_deps = [
        ":keras_test_lib",
    ],
    shard_count = 4,
    tags = [
        "multi_and_single_gpu",
        "no_oss",  # http://b/119349471
        "tf_integration_test",
    ],
    xla_enable_strict_auto_jit = True,
)

cuda_py_test(
    name = "multi_worker_test",
    srcs = ["multi_worker_test.py"],
    additional_deps = [
        ":multi_worker_testing_utils",
        "@absl_py//absl/testing:parameterized",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:random_ops",
        "//tensorflow/python:util",
        "//tensorflow/python/data/ops:dataset_ops",
        "//tensorflow/python/distribute:collective_all_reduce_strategy",
        "//tensorflow/python/distribute:combinations",
        "//tensorflow/python/distribute:distribute_coordinator",
        "//tensorflow/python/distribute:distribute_coordinator_context",
        "//tensorflow/python/distribute:distribute_lib",
        "//tensorflow/python/distribute:multi_worker_test_base",
        "//tensorflow/python/distribute:parameter_server_strategy",
        "//tensorflow/python/distribute/cluster_resolver:cluster_resolver_lib",
        "//tensorflow/python/keras",
        "//tensorflow/python/keras:backend",
        "//tensorflow/python/keras:callbacks",
        "//tensorflow/python/keras:engine",
        "//tensorflow/python/keras:optimizers",
        "//tensorflow/python/keras/optimizer_v2",
    ],
    shard_count = 32,
    tags = [
        "no_oss",  # TODO(b/130369494): Investigate why it times out on OSS.
        # TODO(b/123307453): Add "multi_and_single_gpu",
    ],
    xla_enable_strict_auto_jit = True,
)

cuda_py_test(
    name = "multi_worker_callback_test",
    srcs = ["multi_worker_callback_test.py"],
    additional_deps = [
        ":distribute",
        "//tensorflow/python:platform",
        "//tensorflow/python/distribute:collective_all_reduce_strategy",
        "//tensorflow/python/distribute:combinations",
        "//tensorflow/python/distribute:multi_worker_test_base",
        "//tensorflow/python/distribute:distribute_config",
        "//tensorflow/python/distribute:distribute_coordinator",
        ":multi_worker_testing_utils",
        "//tensorflow/python/keras",
    ],
    # TODO(b/132384649): Enable for guitar and oss tests.
    shard_count = 24,
    tags = [
        "manual",
        "no_oss",
        "noguitar",
        "notap",
    ],
    xla_enable_strict_auto_jit = True,
)

cuda_py_test(
    name = "multi_worker_fault_tolerance_test",
    srcs = ["multi_worker_fault_tolerance_test.py"],
    additional_deps = [
        ":distribute",
        "//tensorflow/python:platform",
        "//tensorflow/python/distribute:collective_all_reduce_strategy",
        "//tensorflow/python/distribute:combinations",
        "//tensorflow/python/distribute:multi_worker_test_base",
        "//tensorflow/python/distribute:distribute_config",
        "//tensorflow/python/distribute:distribute_coordinator",
        ":multi_worker_testing_utils",
        "//tensorflow/python/keras",
    ],
    shard_count = 14,
    # TODO(b/132384649): Enable once fixed.
    tags = [
        "no_oss",
    ],
    xla_enable_strict_auto_jit = True,
)

cuda_py_test(
    name = "multi_worker_optimizer_comparison_test",
    srcs = ["multi_worker_optimizer_comparison_test.py"],
    additional_deps = [
        "//tensorflow/python/distribute:collective_all_reduce_strategy",
        "//tensorflow/python/distribute:combinations",
        "//tensorflow/python/keras/distribute:multi_worker_test",
        "//tensorflow/python/distribute:multi_worker_test_base",
        "//tensorflow/python:platform",
        "//tensorflow/python/distribute:distribute_coordinator",
        "//tensorflow/python/keras",
    ],
    tags = [
        "no_oss",
        "multi_and_single_gpu",
    ],
    xla_enable_strict_auto_jit = True,
)

py_library(
    name = "multi_worker_testing_utils",
    srcs = [
        "multi_worker_testing_utils.py",
    ],
    deps = [
        "//tensorflow/python:array_ops",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:random_ops",
        "//tensorflow/python/data/ops:dataset_ops",
        "//tensorflow/python/keras",
        "//tensorflow/python/keras/optimizer_v2",
    ],
)
