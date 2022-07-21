// Cropped Frames Compression Calculator
//
// Converts cropped frames to JSON.

#include <algorithm>
#include <memory>

#include "absl/memory/memory.h"
#include "absl/strings/str_format.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/formats/image_frame.h"
#include "mediapipe/framework/formats/image_frame_opencv.h"
#include "mediapipe/framework/port/ret_check.h"
#include "mediapipe/framework/port/status.h"
#include "mediapipe/framework/timestamp.h"

namespace mediapipe {
namespace autoflip {

constexpr char kOutputFilePathTag[] = "OUTPUT_FILE_PATH";
constexpr char kCroppedFramesTag[] = "CROPPED_FRAMES";
constexpr char kVideoPrestreamTag[] = "VIDEO_PRESTREAM";
constexpr char kStringTag[] = "STRING";

// This calculator converts cropped frames to JSON.
// CroppedFramesCompressionCalculator::scorer_options.
// Example:
//    calculator: "CroppedFramesCompressionCalculator"
//    input_side_packet: "OUTPUT_FILE_PATH:output_video_path"
//    input_stream: "CROPPED_FRAMES:cropped_frames"
//    input_stream: "VIDEO_PRESTREAM:video_header"
//    output_stream: "STRING:cropped_frames_json"
//
class CroppedFramesCompressionCalculator : public CalculatorBase {
 public:
  CroppedFramesCompressionCalculator();
  ~CroppedFramesCompressionCalculator() override {}
  CroppedFramesCompressionCalculator(const CroppedFramesCompressionCalculator&) = delete;
  CroppedFramesCompressionCalculator& operator=(const CroppedFramesCompressionCalculator&) = delete;

  static absl::Status GetContract(mediapipe::CalculatorContract* cc);
  absl::Status Open(mediapipe::CalculatorContext* cc) override;
  absl::Status Process(mediapipe::CalculatorContext* cc) override;

 private:
  std::unique_ptr<VideoHeader> header_;
  bool emitted_ = false;
};

REGISTER_CALCULATOR(CroppedFramesCompressionCalculator);

CroppedFramesCompressionCalculator::CroppedFramesCompressionCalculator() {}

absl::Status VideoPreStreamCalculator::GetContract(CalculatorContract* cc) {
  if (!cc->Inputs().UsesTags()) {
    cc->Inputs().Index(0).Set<ImageFrame>();
  } else {
    cc->Inputs().Tag(kCroppedFramesTag).Set<ImageFrame>();
    cc->Inputs().Tag(kVideoPrestreamTag).Set<VideoHeader>();
  }
  cc->Outputs().Index(0).Set<VideoHeader>();
  return absl::OkStatus();
}

absl::Status VideoPreStreamCalculator::Open(CalculatorContext* cc) {
  bool frame_rate_in_prestream = cc->Inputs().UsesTags() &&
                             cc->Inputs().HasTag(kCroppedFramesTag) &&
                             cc->Inputs().HasTag(kVideoPrestreamTag);
  RET_CHECK(frame_rate_in_prestream) << "frame rate must be in prestream";
  header_ = absl::make_unique<VideoHeader>();
  return absl::OkStatus();
}

absl::Status VideoPreStreamCalculator::Process(CalculatorContext* cc) {
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
    header_->format = frame.Format();
    header_->width = frame.Width();
    header_->height = frame.Height();
    RET_CHECK_NE(header_->frame_rate, 0.0) << "frame rate should be non-zero";
    cc->Outputs().Index(0).Add(header_.release(), Timestamp::PreStream());
    emitted_ = true;
  }
  return absl::OkStatus();
}

}  // namespace autoflip
}  // namespace mediapipe