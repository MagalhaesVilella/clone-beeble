const { execSync } = require("child_process");
const ffmpeg = require("../backend/node_modules/ffmpeg-static");
const ffmpegCmd = `"${ffmpeg}"`;

execSync(
  `${ffmpegCmd} -y -f lavfi -i color=c=blue:size=540x960:rate=8 -f lavfi -i color=c=red:size=200x400:rate=8 -filter_complex "[0][1]overlay=170:280" -t 3 -pix_fmt yuv420p test_video.mp4`,
  { stdio: "inherit" }
);

execSync(
  `${ffmpegCmd} -y -f lavfi -i color=c=green:size=540x960 -frames:v 1 test_bg.jpg`,
  { stdio: "inherit" }
);

console.log("Ficheiros criados: test_video.mp4 e test_bg.jpg");
