批量转化mp4格式
for %a in ("X:\xxx\xx\*.mkv") do ffmpeg -i "%a" -c:v copy -c:a aac "Y:\yyy\yy\%~na.mp4"

软链接
/home/chenxingying/ffmpeg-git-20220108-amd64-static/ffmpeg

步骤如下：
1. 将raw转化成avi格式裁帧制作成gt
2. 将raw.yuv转换成MP4格式，使用X265解码器，再裁帧

yuv转换avi裁帧
/home/chenxingying/ffmpeg-git-20220108-amd64-static/ffmpeg -s 416x240 -i BasketballPass_416x240_50.yuv -vcodec copy output.avi
#### 可以不用/home/chenxingying/ffmpeg-git-20220108-amd64-static/ffmpeg -i output.avi -strict -2 -qp 37 output.mp4

yuv转换mkv裁帧
ffmpeg -pix_fmt yuv420p -s 416x240 -i BasketballPass_416x240_50.yuv output.mkv
ffmpeg -i output.mkv -pix_fmt yuv420p raw/%3d.png

ffmpeg裁帧
ffmpeg -i BasketballPass_416x240_50.mkv png/%3d.png

x265压缩
/home/chenxingying/ffmpeg-git-20220-p108-amd64-static/ffmpeg -video_size 416x240 -i BasketballPass_416x240_50.yuv -c:v libx265 -qp 37 -nal-hrd cbr BasketballPass_416x240_50.mp4
x264
/home/chenxingying/ffmpeg-git-20220108-amd64-static/ffmpeg -video_size 416x240 -i BasketballPass_416x240_50.yuv BasketballPass_416x240_50.mp4

以固定的码率压缩测试集
ffmpeg -pix_fmt yuv420p -s (width)x(height) -r (frame_rate) -i xxx.yuv -c:v libx265 -b:v 200k -x265-params pass=1:log-level=error -f null /dev/null
ffmpeg -pix_fmt yuv420p -s (width)x(height) -r (frame_rate) -i xxx.yuv -c:v libx265 -b:v 200k -maxra te 200k -minrate 200k -bufsize 200k xxx.mkv

以固定码率压缩 CBR模式
/home/chenxingying/ffmpeg-git-20220108-amd64-static/ffmpeg -pix_fmt yuv420p -s 416x240 -r 50 -i BasketballDrill_832x480_50.yuv -c:v libx265 -b:v 1036k -maxrate 1036k -minrate 1036k -bufsize 1036K -nal-hrd cbr BasketballDrill_832x480_50.mkv
低码率和高码率都试一下来压缩
高码率
/home/chenxingying/ffmpeg-git-20220108-amd64-static/ffmpeg -pix_fmt yuv420p -s 416x240 -i BasketballPass_416x240_50.yuv -b:v 260k -maxrate 260k -minrate 260k -bufsize 260k -c:v libx265 cbr_high_gt.mkv
低码率修改

两步策略 cbr x265 800bitrate
ffmpeg -pix_fmt yuv420p -s 416x240 -r 50 -i BasketballPass_416x240_50.yuv -c:v libx265 -b:v 260k -x265-params pass=1:log-level=error -x265-params "nal-hrd=cbr" -b:v 260k -maxrate 260k -minrate 260k -bufsize 260k -f null /dev/null
ffmpeg -pix_fmt yuv420p -s 416x240 -r 50 -i BasketballPass_416x240_50.yuv -c:v libx265 -b:v 260k -x265-params pass=2:log-level=error -x265-params "nal-hrd=cbr" -b:v 260k -maxrate 260k -minrate 260k -bufsize 520k BasketballPass_416x240_50.mkv

老师说了，比较小的误差可以接受

QP37
../HM16.20/bin/TAppEncoderStatic -c ../HM16.20/cfg/encoder_lowdelay_P_main.cfg -c ../HM16.20/cfg/per-sequence/BasketballPass.cfg -i test_18/raw/001_960x536_218.yuv -q 37 -wdt 960 -hgt 536 -f 218 -fr 30 -b 001_qp37.mkv

yuv裁帧
ffmpeg -pix_fmt yuv420p -s 960x536 -i 001_960x536_218.yuv -vframes 218 001/%d.png
