const { execSync } = require("child_process");
const ffmpeg = require("../backend/node_modules/ffmpeg-static");
const ff = `"${ffmpeg}"`;

execSync(
  `${ff} -y -f lavfi -i color=c=blue:size=540x960:rate=8 -f lavfi -i color=c=red:size=200x400:rate=8 -filter_complex "[0][1]overlay=170:280" -t 3 -c:v libx264 -profile:v baseline -level 3.0 -pix_fmt yuv420p -movflags +faststart switchx_test_video_compatible.mp4`,
  { stdio: "inherit" }
);

execSync(
  `${ff} -y -f lavfi -i color=c=green:size=540x960 -frames:v 1 -update 1 switchx_test_bg.jpg`,
  { stdio: "inherit" }
);

console.log("OK media pronta");
