/*******************************************************************************
    Copyright (c) The GsTaichi Authors (2016- ). All Rights Reserved.
    The use of this software is governed by the LICENSE file.
*******************************************************************************/

#include "gstaichi/math/math.h"
#include "gstaichi/math/linalg.h"
#include "gstaichi/util/base64.h"

#define STBI_FAILURE_USERMSG
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
#define STB_TRUETYPE_IMPLEMENTATION
#include "stb_truetype.h"

namespace gstaichi {

std::map<std::string, stbtt_fontinfo> fonts;
std::map<std::string, std::vector<uint8>> font_buffers;

void write_pgm(Array2D<real> img, const std::string &fn) {
  std::ofstream fs(fn, std::ios_base::binary);
  Vector2i res = img.get_res();
  fs << fmt::format("P5\n{} {}\n{}\n", res[0], res[1], 255);
  for (int j = 0; j < res[1]; j++) {
    std::string line;
    for (int i = 0; i < res[0]; i++) {
      uint8_t v = clamp((int)(img[i][res[1] - j - 1] * 255), 0, 255);
      line.push_back(v);
    }
    fs.write(line.c_str(), line.size());
  }
}

}  // namespace gstaichi
