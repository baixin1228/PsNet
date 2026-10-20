# GPU shader 自动修图

Python 负责文件、颜色配置和小图统计，ModernGL / OpenGL 4.3 compute shader 执行全部全分辨率修图。独立于训练代码，不需要 PyTorch 或 CUDA，支持具有兼容驱动的 AMD / NVIDIA / Intel GPU。

## 安装与使用

```bash
python -m pip install -r requirements-shader-retouch.txt
python shader_retouch.py "/path/photo.jpg"
python shader_retouch.py "/path/photos" --style vivid --strength 1.2
python shader_retouch.py "/path/photos" --style hdr --max-size 256 --recursive
python shader_retouch.py "/path/photo.jpg" --style cinematic --exposure 0.3 --sharpness 0.5
```

- 单张：原目录生成 `photo_retouched.jpg` 和同名 JSON 参数记录。
- 文件夹：同级创建 `photos_retouched`，输出 `photo.jpg_retouched.jpg`。保留原扩展名是为了避免 `photo.jpg` 和 `photo.png` 相互覆盖。递归模式保留子目录结构。
- 不覆盖源图。已存在的输出或记录默认跳过，显式加 `--overwrite` 才替换。
- 默认保持原分辨率；`--max-size 256` 保持比例把最长边缩到 256，不会放大小图，也不强制正方形。
- 支持 JPG、PNG、TIFF、WebP、BMP；透明图片、多帧图片会报错并继续处理其他图片。文件夹中自动忽略脚本生成的 `_retouched` 图片。
- `--strength`：0–2，默认 1；0 精确跳过增强（仍可能执行尺寸与色彩配置转换）。
- `--exposure`：额外曝光补偿 -3～3 EV；`--sharpness`：0–2。
- 输出 JPEG（quality=85，4:2:0 色度采样），嵌入 sRGB 配置。JSON 保存算法版本、输入、尺寸、预设、参数、渲染器和总耗时。

## 算法与预设

`vivid`：鲜艳通透；`hdr`：更强暗部提升、亮部压制和局部对比；`cinematic`：冷暖分区调色。

1. 256 像素缩略图在线性 RGB 中估计亮度分位数，计算受限自动曝光。
2. GPU 在最长边 768 的分析图上估计对数亮度局部均值／方差，双线性上采样引导系数到原尺寸，，使用自引导滤波形式的系数进行保边明暗分解。为控制成本，使用固定 25 点可分离高斯采样近似局部统计，不是严格的标准 box guided filter。
3. 暗部提升、亮部压制、受限细节增强、柔和高光肩部和整体对比曲线。
4. OKLab 色度增强，对高色度减少增益、蓝色略压暗；电影预设增加阴影冷色和亮部暖色。超色域颜色通过降低色度映射回 sRGB。
5. 亮度锐化，带边缘阈值、暗部抑制和增益限幅。

目前没有人脸／天空语义分割、专门去雾、RAW 显影和复杂色相分区编辑。预设是可调整的初版，不保证所有场景都美观；应先检查人像、夜景、逆光、风景等代表样本，再批量生产。强度大时仍可能出现光晕、肤色偏移或噪声增强。

## GPU 与性能

启动时打印真实 `GL_RENDERER`，检测到 llvmpipe 等常见软件渲染器默认终止，`--allow-software` 仅供验证。Linux 优先 EGL，其次默认上下文；可用 `--backend default` 或 `--backend egl` 指定。Windows 一般使用默认上下文。需要 OpenGL 4.3 驱动。

v2 输入采用 RGBA8、中间结果采用 RGBA16F，分析缓冲限制在最长边 768。阶段间释放不再需要的纹理，主要全分辨率纹理的峰值约 16 字节／像素，另有分析缓冲、驱动和 CPU 数组开销。24.6MP 的主要全图纹理约 394MB。锐化为单次 3×3 亮度邻域处理。上下文和着色器在整个批次复用，没有异步读回或流水线解码。

每张图一次全图上传和一次最终 RGBA8 读回，另有一张小分析图上传；无逐阶段 CPU 读回。分段计时使用 GPU 完成同步，显示 upload、guide、tone、color、sharp、readback，JSON 额外记录保存耗时。JPEG 使用质量 85，在体积和画质之间折中；这减少磁盘占用，不改变 GPU 中间缓冲大小。检测到异常全黑输出会报错，避免保存明显失败结果。
终端耗时包含读取、处理、编码、保存等主要步骤，不等于纯 GPU kernel 时间。小图可能主要受图片解码影响。调参完成后固定预设、强度、尺寸、脚本版本；改变这些设置重新生成时使用新目录或明确覆盖。

## 用于现有 U-Net

当前 notebook 依赖同 stem 配对；文件夹输出名包含 `_retouched`，不能直接原样配对。需在数据加载时按 JSON 的输入路径关联，或显式移除 `.jpg_retouched` 等输出后缀映射回输入。输入也应使用相同 EXIF 方向、sRGB 转换和尺寸处理。多个风格应分开训练，或给模型增加风格条件。
