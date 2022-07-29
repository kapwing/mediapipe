// Cropped Frames Json Calculator
//
// Converts cropped frames to JSON.

#include <algorithm>
#include <memory>
#include <fstream>

#include "absl/memory/memory.h"
#include "absl/strings/str_format.h"
#include "mediapipe/examples/desktop/autoflip/quality/scene_cropping_viz.h"
#include "mediapipe/examples/desktop/autoflip/quality/utils.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/formats/image_frame.h"
#include "mediapipe/framework/formats/image_frame_opencv.h"
#include "mediapipe/framework/formats/video_stream_header.h"
#include "mediapipe/framework/port/file_helpers.h"
#include "mediapipe/framework/port/ret_check.h"
#include "mediapipe/framework/port/status.h"
#include "mediapipe/framework/timestamp.h"

namespace mediapipe {
namespace autoflip {

constexpr char kOutputFilePathTag[] = "OUTPUT_FILE_PATH";
constexpr char kCropBoundariesTag[] = "CROP_BOUNDARIES";
constexpr char kVideoPrestreamTag[] = "VIDEO_PRESTREAM";

// This calculator converts cropped frames to JSON.
// CroppedFramesJsonCalculator::scorer_options.
// Example:
//    calculator: "CroppedFramesJsonCalculator"
//    input_side_packet: "OUTPUT_FILE_PATH:output_json_path"
//    input_stream: "CROPPED_FRAMES:cropped_frames"
//    input_stream: "VIDEO_PRESTREAM:video_header"
//
class CroppedFramesJsonCalculator : public CalculatorBase {
 public:
  CroppedFramesJsonCalculator();
  ~CroppedFramesJsonCalculator() override {}
  CroppedFramesJsonCalculator(const CroppedFramesJsonCalculator&) = delete;
  CroppedFramesJsonCalculator& operator=(const CroppedFramesJsonCalculator&) = delete;

  static absl::Status GetContract(mediapipe::CalculatorContract* cc);
  absl::Status Open(mediapipe::CalculatorContext* cc) override;
  absl::Status Process(mediapipe::CalculatorContext* cc) override;
  absl::Status Close(mediapipe::CalculatorContext* cc) override;

 private:
  double frame_rate_;
  std::ofstream json_file_;
};

REGISTER_CALCULATOR(CroppedFramesJsonCalculator);

CroppedFramesJsonCalculator::CroppedFramesJsonCalculator() {}

absl::Status CroppedFramesJsonCalculator::GetContract(CalculatorContract* cc) {
  RET_CHECK(cc->Inputs().UsesTags());
  RET_CHECK(cc->Inputs().HasTag(kCropBoundariesTag));
  RET_CHECK(cc->Inputs().HasTag(kVideoPrestreamTag));
  RET_CHECK(cc->InputSidePackets().HasTag(kOutputFilePathTag));

  cc->Inputs().Tag(kCropBoundariesTag).Set<LinearSceneCropSummary>();
  cc->Inputs().Tag(kVideoPrestreamTag).Set<VideoHeader>();
  cc->InputSidePackets().Tag(kOutputFilePathTag).Set<std::string>();
  
  return absl::OkStatus();
}

absl::Status CroppedFramesJsonCalculator::Open(CalculatorContext* cc) {
  json_file_.open(cc->InputSidePackets().Tag(kOutputFilePathTag).Get<std::string>());
  json_file_ << "[";

  return absl::OkStatus();
}

absl::Status CroppedFramesJsonCalculator::Process(CalculatorContext* cc) {
  cc->GetCounter("Process")->Increment();
  
  if (cc->InputTimestamp() == Timestamp::PreStream()) {
    RET_CHECK(cc->Inputs().Tag(kCropBoundariesTag).IsEmpty());
    RET_CHECK(!cc->Inputs().Tag(kVideoPrestreamTag).IsEmpty());
    frame_rate_ = cc->Inputs().Tag(kVideoPrestreamTag).Get<VideoHeader>().frame_rate;
    RET_CHECK_NE(frame_rate_, 0.0) << "frame rate should be non-zero";
  } else {
    RET_CHECK(cc->Inputs().Tag(kVideoPrestreamTag).IsEmpty())
        << "Packet on VIDEO_PRESTREAM must come in at Timestamp::PreStream().";
    RET_CHECK(!cc->Inputs().Tag(kCropBoundariesTag).IsEmpty());
    RET_CHECK_NE(frame_rate_, 0.0) << "frame rate should be non-zero";
    LinearSceneCropSummary scene_crop_summary = cc->Inputs().Tag(kCropBoundariesTag)
        .Get<LinearSceneCropSummary>();
    

    // todo (david): convert to json and save to file
    //std:format("{}, {}", header_->width, header_->height), Timestamp::PreStream());
  }
  return absl::OkStatus();
}

absl::Status CroppedFramesJsonCalculator::Close(CalculatorContext* cc) {
  json_file_ << "]";
  json_file_.close();
}

REGISTER_CALCULATOR(CroppedFramesJsonCalculator);

}  // namespace autoflip
}  // namespace mediapipe