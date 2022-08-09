load("//mediapipe/framework/port:build_config.bzl", "mediapipe_cc_proto_library")

licenses(["notice"])

package(default_visibility = ["//mediapipe/examples:__subpackages__"])

proto_library(
    name = "autoflip_messages_proto",
    srcs = ["autoflip_messages.proto"],
    deps = [
        "//mediapipe/framework:calculator_proto",
    ],
)

mediapipe_cc_proto_library(
    name = "autoflip_messages_cc_proto",
    srcs = ["autoflip_messages.proto"],
    cc_deps = ["//mediapipe/framework:calculator_cc_proto"],
    visibility = [
        "//mediapipe/examples:__subpackages__",
    ],
    deps = [":autoflip_messages_proto"],
)

cc_binary(
    name = "run_custom_autoflip",
    deps = [
        "//mediapipe/calculators/core:packet_thinner_calculator",
        "//mediapipe/calculators/image:scale_image_calculator",
        "//mediapipe/calculators/video:opencv_video_decoder_calculator",
        "//mediapipe/calculators/video:video_pre_stream_calculator",
        "//mediapipe/examples/desktop:simple_run_graph_main",
        "//mediapipe/examples/desktop/autoflip/calculators:border_detection_calculator",
        "//mediapipe/examples/desktop/autoflip/calculators:face_to_region_calculator",
        "//mediapipe/examples/desktop/autoflip/calculators:localization_to_region_calculator",
        "//mediapipe/examples/custom/autoflip/calculators:scene_cropping_coordinates_calculator",
        "//mediapipe/examples/desktop/autoflip/calculators:shot_boundary_calculator",
        "//mediapipe/examples/desktop/autoflip/calculators:signal_fusing_calculator",
        "//mediapipe/examples/desktop/autoflip/calculators:video_filtering_calculator",
        "//mediapipe/examples/custom/autoflip/calculators:cropped_frames_json_calculator",
        "//mediapipe/examples/desktop/autoflip/subgraph:autoflip_face_detection_subgraph",
        "//mediapipe/examples/desktop/autoflip/subgraph:autoflip_object_detection_subgraph",
    ],
)