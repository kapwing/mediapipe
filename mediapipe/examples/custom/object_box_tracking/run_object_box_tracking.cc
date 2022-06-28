
// Run object tracking on input video specified by input_video_path
// Output box detections in a JSON file.
// Modified from graph running demo at mediapipe/examples/desktop/demo_run_graph_main.cc

#include <cstdlib>
#include <fstream>

#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/formats/image_frame.h"
#include "mediapipe/framework/formats/image_frame_opencv.h"
#include "mediapipe/framework/port/file_helpers.h"
#include "mediapipe/framework/port/opencv_highgui_inc.h"
#include "mediapipe/framework/port/opencv_imgproc_inc.h"
#include "mediapipe/framework/port/opencv_video_inc.h"
#include "mediapipe/framework/port/parse_text_proto.h"

#include "mediapipe/framework/formats/detection.pb.h"
#include "mediapipe/framework/formats/location_data.pb.h"


#include "mediapipe/framework/port/status.h"

using mediapipe::Detection;
using mediapipe::DetectionList;


constexpr char videoInputStream[] = "input_video";
constexpr char kOutputStream[] = "tracked_detections";
constexpr char kWindowName[] = "MediaPipe";

ABSL_FLAG(std::string, input_video_path, "",
          "Full path of video to load. ");
ABSL_FLAG(std::string, output_json_path, "",
          "Full path of where to save result in json file");

std::string graph_config = R"(
      input_stream: "input_video"
      output_stream: "tracked_detections"

      # Resamples the images by specific frame rate. This calculator is used to
      # control the frequecy of subsequent calculators/subgraphs, e.g. less power
      # consumption for expensive process.
      node {
        calculator: "PacketResamplerCalculator"
        input_stream: "DATA:input_video"
        output_stream: "DATA:throttled_input_video"
        node_options: {
          [type.googleapis.com/mediapipe.PacketResamplerCalculatorOptions] {
            frame_rate: 3
          }
        }
      }

      # Subgraph that detections objects (see object_detection_cpu.pbtxt).
      node {
        calculator: "ObjectDetectionSubgraphCpu"
        input_stream: "IMAGE:throttled_input_video"
        output_stream: "DETECTIONS:output_detections"
      }

      # Subgraph that tracks objects (see object_tracking_cpu.pbtxt).
      node {
        calculator: "ObjectTrackingSubgraphCpu"
        input_stream: "VIDEO:input_video"
        input_stream: "DETECTIONS:output_detections"
        output_stream: "DETECTIONS:tracked_detections"
      }
  )";

absl::Status RunMPPGraph() {

  mediapipe::CalculatorGraphConfig config =
      mediapipe::ParseTextProtoOrDie<mediapipe::CalculatorGraphConfig>(
          graph_config);

  LOG(INFO) << "Initialize the calculator graph.";
  mediapipe::CalculatorGraph graph;
  MP_RETURN_IF_ERROR(graph.Initialize(config));

  LOG(INFO) << "Load the video.";
  cv::VideoCapture capture;
  const bool load_video = !absl::GetFlag(FLAGS_input_video_path).empty();
  if (load_video) {
    capture.open(absl::GetFlag(FLAGS_input_video_path));
  } 

  RET_CHECK(capture.isOpened());

  if(absl::GetFlag(FLAGS_output_json_path).empty()) return absl::OkStatus();

  std::ofstream file(absl::GetFlag(FLAGS_output_json_path));

  file << "[\n";

  LOG(INFO) << "Start running the calculator graph.";
  ASSIGN_OR_RETURN(mediapipe::OutputStreamPoller poller,
                   graph.AddOutputStreamPoller(kOutputStream));
  MP_RETURN_IF_ERROR(graph.StartRun({}));

  LOG(INFO) << "Start grabbing and processing frames.";
  bool grab_frames = true;
  bool firstFrame = true;
  while (grab_frames) {
    // Capture opencv camera or video frame.
    cv::Mat camera_frame_raw;
    capture >> camera_frame_raw;
    if (camera_frame_raw.empty()) {
      LOG(INFO) << "Empty frame, end of video reached.";
      break;
    }
    cv::Mat camera_frame;
    cv::cvtColor(camera_frame_raw, camera_frame, cv::COLOR_BGR2RGB);

    // Wrap Mat into an ImageFrame.
    auto input_frame = absl::make_unique<mediapipe::ImageFrame>(
        mediapipe::ImageFormat::SRGB, camera_frame.cols, camera_frame.rows,
        mediapipe::ImageFrame::kDefaultAlignmentBoundary);
    cv::Mat input_frame_mat = mediapipe::formats::MatView(input_frame.get());
    camera_frame.copyTo(input_frame_mat);

    // Send image packet into the graph.
    size_t frame_timestamp_us =
        (double)cv::getTickCount() / (double)cv::getTickFrequency() * 1e6;
    MP_RETURN_IF_ERROR(graph.AddPacketToInputStream(
        videoInputStream, mediapipe::Adopt(input_frame.release())
                          .At(mediapipe::Timestamp(frame_timestamp_us))));
    

    // Get the graph result packet, or stop if that fails.
    mediapipe::Packet packet;
    if (!poller.Next(&packet)) break;
    auto& detections = packet.Get<std::vector<mediapipe::Detection>>();

    float timestamp_milliseconds = capture.get(cv::CAP_PROP_POS_MSEC);
    // write spacing comma if this is not first frame
    if(firstFrame) firstFrame = false;  
    else {
      file << ",\n";
    }

    file << "{\"timestamp\": " << timestamp_milliseconds / 1000;
    if(detections.size() > 0) {
      file << ", \"detections\": [\n    ";
    }

    for(int i=0; i < detections.size(); i++) {
      mediapipe::Detection detection = detections.at(i);

      // add separating comma for items after the first
      if(i > 0) {
        file << ", ";
      }

      // Write ID to file
      int id = detection.detection_id();
      file << "{\"id\": " << id << ", ";

      // Write list of labels and scores to file
      auto num_labels = std::max(detection.label_size(), detection.label_id_size());
      if(num_labels > 0) {
        file << "\"labels\": [";
        for (int j = 0; j < num_labels; ++j) {
          std::string label = detection.label(j);
          float score = detection.score(j);

          if(j > 0) {
            file << ", ";
          }
          file << "{\"label\": \"" << label << "\", \"score\": " << score << "}";
        }
          
         file << "], ";
      }


      // Write location data to file
      mediapipe::LocationData location_data = detection.location_data();
      auto& bounding_box = location_data.relative_bounding_box();
      float xMin = bounding_box.xmin() * 100;
      float yMin = bounding_box.ymin() * 100;
      float width = bounding_box.width() * 100;
      float height = bounding_box.height() * 100;

      file << "\"xMin\": " << xMin << ", \"yMin\": " << yMin << ", \"width\": " << width << ", \"height\": " << height << "} ";
    }
    if(detections.size() > 0) {
      file << "]";
    }
    file << "}";

  }
   file << "]";

  LOG(INFO) << "Shutting down.";
  file.close();
  MP_RETURN_IF_ERROR(graph.CloseInputStream(videoInputStream));
  return graph.WaitUntilDone();
}

int main(int argc, char** argv) {

  absl::ParseCommandLine(argc, argv);
  absl::Status run_status = RunMPPGraph();
  if (!run_status.ok()) {
    LOG(ERROR) << "Failed to run the graph: " << run_status.message();
    return EXIT_FAILURE;
  } else {
    LOG(INFO) << "Success!";
  }
  return EXIT_SUCCESS;
}
