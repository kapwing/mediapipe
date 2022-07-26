// Cropped Frames Json Calculator
//
// Converts cropped frames to JSON.

#include <algorithm>
#include <memory>

#include "absl/memory/memory.h"
#include "absl/strings/str_format.h"
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
constexpr char kCroppedFramesTag[] = "CROPPED_FRAMES";
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

 private:
  std::unique_ptr<VideoHeader> header_;
  bool emitted_ = false;
};

REGISTER_CALCULATOR(CroppedFramesJsonCalculator);

CroppedFramesJsonCalculator::CroppedFramesJsonCalculator() {}

absl::Status CroppedFramesJsonCalculator::GetContract(CalculatorContract* cc) {
  if (!cc->Inputs().UsesTags()) {
    cc->Inputs().Index(0).Set<ImageFrame>();
  } else {
    cc->Inputs().Tag(kCroppedFramesTag).Set<ImageFrame>();
    cc->Inputs().Tag(kVideoPrestreamTag).Set<VideoHeader>();
  }

  RET_CHECK(cc->InputSidePackets().HasTag(kOutputFilePathTag));
  
  return absl::OkStatus();
}

absl::Status CroppedFramesJsonCalculator::Open(CalculatorContext* cc) {
  bool frame_rate_in_prestream = cc->Inputs().UsesTags() &&
                             cc->Inputs().HasTag(kCroppedFramesTag) &&
                             cc->Inputs().HasTag(kVideoPrestreamTag);
  RET_CHECK(frame_rate_in_prestream) << "frame rate must be in prestream";
  header_ = absl::make_unique<VideoHeader>();
  return absl::OkStatus();
}

absl::Status CroppedFramesJsonCalculator::Process(CalculatorContext* cc) {
  cc->GetCounter("Process")->Increment();

  if (emitted_) {
    return absl::OkStatus();
  }
  
  if (cc->InputTimestamp() == Timestamp::PreStream()) {
    RET_CHECK(cc->Inputs().Tag(kCroppedFramesTag).IsEmpty());
    RET_CHECK(!cc->Inputs().Tag(kVideoPrestreamTag).IsEmpty());
    *header_ = cc->Inputs().Tag(kVideoPrestreamTag).Get<VideoHeader>();
    RET_CHECK_NE(header_->frame_rate, 0.0) << "frame rate should be non-zero";
  } else {
    RET_CHECK(cc->Inputs().Tag(kVideoPrestreamTag).IsEmpty())
        << "Packet on VIDEO_PRESTREAM must come in at Timestamp::PreStream().";
    RET_CHECK(!cc->Inputs().Tag(kCroppedFramesTag).IsEmpty());
    const auto& frame = cc->Inputs().Tag(kCroppedFramesTag).Get<ImageFrame>();
    //header_->format = frame.Format();
    //header_->width = frame.Width();
    //header_->height = frame.Height();
    RET_CHECK_NE(header_->frame_rate, 0.0) << "frame rate should be non-zero";

    // todo (david): convert to json and save to file
    std::cout << header_->frame_rate << std::endl;
    //std:format("{}, {}", header_->width, header_->height), Timestamp::PreStream());
    emitted_ = true;
  }
  return absl::OkStatus();
}

}  // namespace autoflip
}  // namespace mediapipe