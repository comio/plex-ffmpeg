/*
 * Android MediaCodec NDK encoder
 *
 * Copyright (c) 2015-2016, NVIDIA CORPORATION. All rights reserved.
 *
 * This file is part of FFmpeg.
 *
 * FFmpeg is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 *
 * FFmpeg is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with FFmpeg; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA
 */

#include <media/NdkMediaCodec.h>

#include "libavutil/internal.h"
#include "libavutil/imgutils.h"
#include "libavutil/opt.h"
#include "libavutil/avassert.h"
#include "avcodec.h"
#include "internal.h"
#include "mediacodecndk.h"

typedef struct MediaCodecNDKEncoderContext
{
    AVClass *avclass;
    AMediaCodec *encoder;
    AVFrame  frame;
    bool     saw_output_eos;
    int64_t last_dts;
    int rc_mode;
    int width;
    int height;
    uint8_t *new_extradata;
    int new_extradata_size;
} MediaCodecNDKEncoderContext;

#define LOCAL_BUFFER_FLAG_SYNCFRAME 1
#define LOCAL_BUFFER_FLAG_CODECCONFIG 2

#define TIMEOUT_USEC 10000

#define OFFSET(x) offsetof(MediaCodecNDKEncoderContext, x)
#define VE AV_OPT_FLAG_VIDEO_PARAM | AV_OPT_FLAG_ENCODING_PARAM

#define RC_MODE_CQ  0 // Unimplemented
#define RC_MODE_VBR 1
#define RC_MODE_CBR 2

static const AVOption options[] = {
    { "rc-mode", "The bitrate mode to use", OFFSET(rc_mode), AV_OPT_TYPE_INT, { .i64 = RC_MODE_VBR }, RC_MODE_VBR, RC_MODE_CBR, VE, "rc_mode"},
//    { "cq", "Constant quality", 0, AV_OPT_TYPE_CONST, {.i64 = RC_MODE_CQ}, INT_MIN, INT_MAX, VE, "rc_mode" },
    { "vbr", "Variable bitrate", 0, AV_OPT_TYPE_CONST, {.i64 = RC_MODE_VBR}, INT_MIN, INT_MAX, VE, "rc_mode" },
    { "cbr", "Constant bitrate", 0, AV_OPT_TYPE_CONST, {.i64 = RC_MODE_CBR}, INT_MIN, INT_MAX, VE, "rc_mode" },
    { "mediacodec_output_size", "Temporary hack to support scaling on output", OFFSET(width), AV_OPT_TYPE_IMAGE_SIZE, {.i64 = 0} , 48, 3840, AV_OPT_FLAG_VIDEO_PARAM | AV_OPT_FLAG_ENCODING_PARAM },
    { NULL },
};

static av_cold int mediacodecndk_encode_init(AVCodecContext *avctx)
{
    MediaCodecNDKEncoderContext *ctx = avctx->priv_data;
    AMediaFormat* format = NULL;
    int pixelFormat;
    const char* mime = "video/avc";
    int ret = ff_mediacodecndk_init_binder();

    if (ret < 0)
        return ret;

    pixelFormat = ff_mediacodecndk_get_color_format(avctx->pix_fmt);

    format = AMediaFormat_new();
    AMediaFormat_setString(format, AMEDIAFORMAT_KEY_MIME, mime);
    AMediaFormat_setInt32(format, AMEDIAFORMAT_KEY_HEIGHT, avctx->height);
    AMediaFormat_setInt32(format, AMEDIAFORMAT_KEY_WIDTH, avctx->width);
    AMediaFormat_setInt32(format, AMEDIAFORMAT_KEY_MAX_WIDTH, avctx->width);
    AMediaFormat_setInt32(format, AMEDIAFORMAT_KEY_MAX_HEIGHT, avctx->height);
    AMediaFormat_setInt32(format, AMEDIAFORMAT_KEY_COLOR_FORMAT, pixelFormat);

    AMediaFormat_setInt32(format, "bitrate-mode", ctx->rc_mode);

    if (avctx->rc_max_rate && avctx->rc_buffer_size) {
        AMediaFormat_setInt32(format, "max-bitrate", avctx->rc_max_rate);
        AMediaFormat_setInt32(format, "virtualbuffersize", avctx->rc_buffer_size);
    }
    AMediaFormat_setInt32(format, AMEDIAFORMAT_KEY_BIT_RATE, avctx->bit_rate);

    AMediaFormat_setFloat(format, AMEDIAFORMAT_KEY_FRAME_RATE, av_q2d(avctx->framerate));
    AMediaFormat_setInt32(format, AMEDIAFORMAT_KEY_I_FRAME_INTERVAL, 1);//FIXME
    AMediaFormat_setInt32(format, AMEDIAFORMAT_KEY_STRIDE, avctx->width);
    AMediaFormat_setInt32(format, "priority", 1);

    AMediaFormat_setInt32(format, "profile", 0x08);//High
    AMediaFormat_setInt32(format, "level", 0x200);//Level31

    if (ctx->width && ctx->height) {
        AMediaFormat_setInt32(format, "output_width", ctx->width);
        AMediaFormat_setInt32(format, "output_height", ctx->height);
    }

    ctx->encoder = AMediaCodec_createEncoderByType(mime);

    if (ctx->encoder == NULL)
        return AVERROR_EXTERNAL;

    AMediaCodec_configure(ctx->encoder, format, NULL, 0, AMEDIACODEC_CONFIGURE_FLAG_ENCODE);
    AMediaCodec_start(ctx->encoder);

    ctx->saw_output_eos = false;
    AMediaFormat_delete(format);
    return 0;
}

static int mediacodecndk_encode_frame(AVCodecContext *avctx, AVPacket *pkt,
                                      const AVFrame *frame, int *got_packet)
{
    MediaCodecNDKEncoderContext *ctx = avctx->priv_data;
    ssize_t bufferIndex;
    size_t bufferSize = 0;
    uint8_t *buffer = NULL;
    int encoderStatus = AMEDIACODEC_INFO_TRY_AGAIN_LATER;
    size_t outSize;
    uint8_t *outBuffer = NULL;
    int ret;
    uint32_t flags = 0;

    bufferIndex = AMediaCodec_dequeueInputBuffer(ctx->encoder, TIMEOUT_USEC);
    if (bufferIndex >= 0) {
        if (frame == NULL) {
            AMediaCodec_queueInputBuffer(ctx->encoder, bufferIndex, 0, 0, 0, AMEDIACODEC_BUFFER_FLAG_END_OF_STREAM);
        } else {
            buffer = AMediaCodec_getInputBuffer(ctx->encoder, bufferIndex, &bufferSize);
            if (!buffer) {
                av_log(avctx, AV_LOG_ERROR, "Cannot get input buffer!\n");
                return AVERROR_EXTERNAL;
            }

            av_image_copy_to_buffer(buffer, bufferSize, (const uint8_t **)frame->data,
                                    frame->linesize, frame->format,
                                    frame->width, frame->height, 1);

            if (frame->pict_type == AV_PICTURE_TYPE_I)
                flags |= LOCAL_BUFFER_FLAG_SYNCFRAME;
            AMediaCodec_queueInputBuffer(ctx->encoder, bufferIndex, 0, bufferSize, av_rescale_q(frame->pts, avctx->time_base, AV_TIME_BASE_Q), flags);
        }
    } else {
        av_log(avctx, AV_LOG_DEBUG, "No input buffers available\n");
    }

    while (!ctx->saw_output_eos) {
        AMediaCodecBufferInfo bufferInfo;
        encoderStatus = AMediaCodec_dequeueOutputBuffer(ctx->encoder, &bufferInfo, TIMEOUT_USEC);
        if (encoderStatus == AMEDIACODEC_INFO_TRY_AGAIN_LATER) {
            // no output available yet
            if (frame != NULL)
                return 0;
        } else if (encoderStatus == AMEDIACODEC_INFO_OUTPUT_FORMAT_CHANGED) {
            // should happen before receiving buffers, and should only happen once
            av_log(avctx, AV_LOG_DEBUG, "Mediacodec info output format changed\n");
        } else {
            outBuffer = AMediaCodec_getOutputBuffer(ctx->encoder, encoderStatus, &outSize);
            if (bufferInfo.flags & AMEDIACODEC_BUFFER_FLAG_END_OF_STREAM) {
                av_log(avctx, AV_LOG_DEBUG, "Got EOS at output\n");
                AMediaCodec_releaseOutputBuffer(ctx->encoder, encoderStatus, false);
                ctx->saw_output_eos = true;
                return 0;
            }

            av_assert0(outBuffer);
            if (bufferInfo.flags & LOCAL_BUFFER_FLAG_CODECCONFIG) {
                av_log(avctx, AV_LOG_DEBUG, "Got extradata of size %d\n", bufferInfo.size);
                if (ctx->new_extradata)
                    av_free(ctx->new_extradata);
                ctx->new_extradata = av_mallocz(bufferInfo.size + AV_INPUT_BUFFER_PADDING_SIZE);
                ctx->new_extradata_size = bufferInfo.size;
                if (!ctx->new_extradata) {
                    AMediaCodec_releaseOutputBuffer(ctx->encoder, encoderStatus, false);
                    av_log(avctx, AV_LOG_ERROR, "Failed to allocate extradata");
                    return AVERROR(ENOMEM);
                }
                memcpy(ctx->new_extradata, outBuffer, bufferInfo.size);
                AMediaCodec_releaseOutputBuffer(ctx->encoder, encoderStatus, false);
                continue;
            }

            if ((ret = ff_alloc_packet2(avctx, pkt, bufferInfo.size, bufferInfo.size) < 0)) {
                AMediaCodec_releaseOutputBuffer(ctx->encoder, encoderStatus, false);
                av_log(avctx, AV_LOG_ERROR, "Failed to allocate packet: %i\n", ret);
                return ret;
            }
            memcpy(pkt->data, outBuffer, bufferInfo.size);
            pkt->pts = av_rescale_q(bufferInfo.presentationTimeUs, AV_TIME_BASE_Q, avctx->time_base);
            pkt->dts = pkt->pts;
            if (bufferInfo.flags & LOCAL_BUFFER_FLAG_SYNCFRAME)
                pkt->flags |= AV_PKT_FLAG_KEY;
            *got_packet = 1;

            AMediaCodec_releaseOutputBuffer(ctx->encoder, encoderStatus, false);

            if (ctx->new_extradata) {
                ret = av_packet_add_side_data(pkt, AV_PKT_DATA_NEW_EXTRADATA,
                                              ctx->new_extradata,
                                              ctx->new_extradata_size);
                if (ret < 0) {
                    av_log(avctx, AV_LOG_ERROR, "Failed to add extradata: %i\n", ret);
                    return ret;
                }
                ctx->new_extradata = NULL;
            }

            break;
        }
    }
    return 0;
}

static av_cold int mediacodecndk_encode_close(AVCodecContext *avctx)
{
    MediaCodecNDKEncoderContext *ctx = avctx->priv_data;

    if (ctx->encoder) {
        AMediaCodec_stop(ctx->encoder);
        AMediaCodec_flush(ctx->encoder);
        AMediaCodec_delete(ctx->encoder);
    }

    return 0;
}

static const AVClass mediacodecndk_class = {
    .class_name = "h264_mediacodecndk_class",
    .item_name = av_default_item_name,
    .option = options,
    .version = LIBAVUTIL_VERSION_INT,
};

AVCodec ff_h264_mediacodecndk_encoder = {
    .name = "h264_mediacodecndk",
    .long_name = NULL_IF_CONFIG_SMALL("h264 (MediaCodec NDK)"),
    .type = AVMEDIA_TYPE_VIDEO,
    .id = AV_CODEC_ID_H264,
    .priv_data_size = sizeof(MediaCodecNDKEncoderContext),
    .init = mediacodecndk_encode_init,
    .encode2 = mediacodecndk_encode_frame,
    .close = mediacodecndk_encode_close,
    .capabilities = CODEC_CAP_DELAY,
    .priv_class = &mediacodecndk_class,
    .pix_fmts = (const enum AVPixelFormat[]){
        AV_PIX_FMT_NV12,
        AV_PIX_FMT_YUV420P,
        AV_PIX_FMT_NONE
    },
};
