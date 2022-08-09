load("//mediapipe/framework/port:build_config.bzl", "mediapipe_cc_proto_library")

licenses(["notice"])

package(default_visibility = [
    "//buzz/diffractor/mediapipe:__subpackages__",
    "//mediapipe/examples:__subpackages__",
    "//mediapipe/viz:__subpackages__",
    "//mediapipe/web/solutions:__subpackages__",
])

proto_library(
    name = "scene_cropping_coordinates_calculator_proto",
    srcs = ["scene_cropping_coordinates_calculator.proto"],
    visibility = ["//visibility:public"],
    deps = [
        "//mediapipe/examples/desktop/autoflip/quality:cropping_proto",
        "//mediapipe/framework:calculator_proto",
    ],
)

mediapipe_cc_proto_library(
    name = "scene_cropping_coordinates_calculator_cc_proto",
    srcs = ["scene_cropping_coordinates_calculator.proto"],
    cc_deps = [
        "//mediapipe/examples/desktop/autoflip/quality:cropping_cc_proto",
        "//mediapipe/framework:calculator_cc_proto",
    ],
    visibility = ["//visibility:public"],
    deps = [":scene_cropping_coordinates_calculator_proto"],
)

cc_library(
    name = "scene_cropping_coordinates_calculator",
    srcs = ["scene_cropping_coordinates_calculator.cc"],
    hdrs = ["scene_cropping_coordinates_calculator.h"],
    deps = [
        ":scene_cropping_coordinates_calculator_cc_proto",
        "//mediapipe/examples/desktop/autoflip:autoflip_messages_cc_proto",
        "//mediapipe/examples/desktop/autoflip/quality:cropping_cc_proto",
        "//mediapipe/examples/desktop/autoflip/quality:focus_point_cc_proto",
        "//mediapipe/examples/desktop/autoflip/quality:frame_crop_region_computer",
        "//mediapipe/examples/desktop/autoflip/quality:padding_effect_generator",
        "//mediapipe/examples/desktop/autoflip/quality:piecewise_linear_function",
        "//mediapipe/examples/desktop/autoflip/quality:polynomial_regression_path_solver",
        "//mediapipe/examples/desktop/autoflip/quality:scene_camera_motion_analyzer",
        "//mediapipe/examples/desktop/autoflip/quality:scene_cropper",
        "//mediapipe/examples/desktop/autoflip/quality:scene_cropping_viz",
        "//mediapipe/examples/desktop/autoflip/quality:utils",
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework:timestamp",
        "//mediapipe/framework/formats:image_frame",
        "//mediapipe/framework/formats:image_frame_opencv",
        "//mediapipe/framework/port:opencv_core",
        "//mediapipe/framework/port:opencv_imgproc",
        "//mediapipe/framework/port:parse_text_proto",
        "//mediapipe/framework/port:ret_check",
        "//mediapipe/framework/port:status",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings:str_format",
    ],
    alwayslink = 1,  # buildozer: disable=alwayslink-with-hdrs
)

proto_library(
    name = "cropped_frames_json_calculator_proto",
    srcs = ["cropped_frames_json_calculator.proto"],
    deps = [
        "//mediapipe/examples/desktop/autoflip/quality:cropping_proto",
        "//mediapipe/framework:calculator_proto",
    ],
)

mediapipe_cc_proto_library(
    name = "cropped_frames_json_calculator_cc_proto",
    srcs = ["cropped_frames_json_calculator.proto"],
    cc_deps = [
        "//mediapipe/examples/desktop/autoflip/quality:cropping_cc_proto",
        "//mediapipe/framework:calculator_cc_proto",
    ],
    visibility = ["//mediapipe/examples:__subpackages__"],
    deps = [":cropped_frames_json_calculator_proto"],
)

cc_library(
    name = "cropped_frames_json_calculator",
    srcs = ["cropped_frames_json_calculator.cc"],
    deps = [
        ":cropped_frames_json_calculator_cc_proto",
        "//mediapipe/examples/desktop/autoflip:autoflip_messages_cc_proto",
        "//mediapipe/examples/desktop/autoflip/quality:cropping_cc_proto",
        "//mediapipe/examples/desktop/autoflip/quality:scene_cropping_viz",
        "//mediapipe/examples/desktop/autoflip/quality:utils",
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework/formats:image_frame",
        "//mediapipe/framework/formats:image_frame_opencv",
        "//mediapipe/framework/formats:video_stream_header",
        "//mediapipe/framework:timestamp",
        "//mediapipe/framework/port:ret_check",
        "//mediapipe/framework/port:status",
        "@com_google_absl//absl/strings",
    ],
    alwayslink = 1,
)