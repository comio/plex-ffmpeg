/*
 * Android MediaCodec NDK decoder
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

#include "libavutil/opt.h"
#include "libavutil/buffer_internal.h"
#include "libavutil/avassert.h"

#include <media/NdkMediaCodec.h>
#include "avcodec.h"
#include "internal.h"
#include "h264.h"
#include "mediacodecndk.h"

typedef struct
{
    AVClass *avclass;
    AMediaCodec *decoder;
    AVBufferRef *decoder_ref;
    AVBSFContext *bsfc;

    uint32_t stride, plane_height;
    int deint_mode;
    int eos_reached;
} MediaCodecNDKDecoderContext;

#define TIMEOUT 10000

#define OFFSET(x) offsetof(MediaCodecNDKDecoderContext, x)
static const AVOption options[] = {
    { "hwdeint_mode", "Used for setting deinterlace mode in MediaCodecNDKDecoder", OFFSET(deint_mode), AV_OPT_TYPE_INT,{ .i64 = 1 } , 0, 2, AV_OPT_FLAG_VIDEO_PARAM | AV_OPT_FLAG_DECODING_PARAM },
    { NULL },
};

static void mediacodecndk_delete_decoder(void *opaque, uint8_t *data)
{
    AMediaCodec *decoder = opaque;
    AMediaCodec_delete(decoder);
}

static av_cold int mediacodecndk_decode_init(AVCodecContext *avctx)
{
    MediaCodecNDKDecoderContext *ctx = avctx->priv_data;
    AMediaFormat* format;
    const char* mime;
    const char *bsf_name = NULL;
    int ret = ff_mediacodecndk_init_binder();

    if (ret < 0)
        return ret;

    switch (avctx->codec_id) {
    case AV_CODEC_ID_H264:
        mime = "video/avc";
        break;
    case AV_CODEC_ID_HEVC:
        mime = "video/hevc";
        break;
    case AV_CODEC_ID_MPEG2VIDEO:
        mime = "video/mpeg2";
        break;
    default:
        av_assert0(!"Unsupported codec ID");
    }

    av_log(avctx, AV_LOG_DEBUG, "codec mime type %s\n", mime);

    if(avctx->extradata && avctx->extradata[0] == 1) {
        if (avctx->codec_id == AV_CODEC_ID_H264)
            bsf_name = "h264_mp4toannexb";
        else if (avctx->codec_id == AV_CODEC_ID_HEVC)
            bsf_name = "hevc_mp4toannexb";
    }
    if (bsf_name) {
        const AVBitStreamFilter *bsf = av_bsf_get_by_name(bsf_name);
        if(!bsf)
            return AVERROR_BSF_NOT_FOUND;
        if ((ret = av_bsf_alloc(bsf, &ctx->bsfc)))
            return ret;
        if (((ret = avcodec_parameters_from_context(ctx->bsfc->par_in, avctx)) < 0) ||
            ((ret = av_bsf_init(ctx->bsfc)) < 0)) {
            av_bsf_free(&ctx->bsfc);
            return ret;
        }
    }

    format = AMediaFormat_new();
    if (!format)
        return AVERROR(ENOMEM);

    AMediaFormat_setString(format, AMEDIAFORMAT_KEY_MIME, mime);
    AMediaFormat_setInt32(format, AMEDIAFORMAT_KEY_COLOR_FORMAT, COLOR_FormatYUV420SemiPlanar);
    // Set these fields to output dimension when HW scaler in decoder is ready
    AMediaFormat_setInt32(format, AMEDIAFORMAT_KEY_WIDTH, avctx->width);
    AMediaFormat_setInt32(format, AMEDIAFORMAT_KEY_HEIGHT, avctx->height);
    AMediaFormat_setInt32(format, "deinterlace-method", ctx->deint_mode);

    ctx->decoder = AMediaCodec_createDecoderByType(mime);
    if (!ctx->decoder) {
        av_log(avctx, AV_LOG_ERROR, "Decoder could not be created\n");
        AMediaFormat_delete(format);
        return AVERROR_EXTERNAL;
    }

    ctx->decoder_ref = av_buffer_create(NULL, 0, mediacodecndk_delete_decoder,
                                        ctx->decoder, 0);

    if (!ctx->decoder_ref) {
        AMediaFormat_delete(format);
        AMediaCodec_delete(ctx->decoder);
        return AVERROR(ENOMEM);
    }

    AMediaCodec_configure(ctx->decoder, format,
                          0, /* surface */
                          0 /* crypto */,
                          0 /* flags */);

    AMediaCodec_start(ctx->decoder);
    AMediaFormat_delete(format);
    return 0;
}

static int mediacodecndk_queue_input_buffer(AVCodecContext *avctx, AVPacket* avpkt)
{
    MediaCodecNDKDecoderContext *ctx = avctx->priv_data;
    int in_index, ret = 0;
    size_t in_size;
    uint8_t* in_buffer = NULL;
    AVPacket filtered_pkt = {0};

    if (ctx->bsfc && avpkt->data) {
        AVPacket filter_pkt = {0};
        if ((ret = av_packet_ref(&filter_pkt, avpkt)) < 0)
            return ret;

        if ((ret = av_bsf_send_packet(ctx->bsfc, &filter_pkt)) < 0) {
            av_packet_unref(&filter_pkt);
            return ret;
        }

        if ((ret = av_bsf_receive_packet(ctx->bsfc, &filtered_pkt)) < 0)
            return ret;

        avpkt = &filtered_pkt;
    }

    in_index = AMediaCodec_dequeueInputBuffer(ctx->decoder, TIMEOUT * 100);
    if (in_index < 0) {
        av_log(avctx, AV_LOG_ERROR, "Failed to get input buffer! ret = %d\n", in_index);
        ret = AVERROR_EXTERNAL;
        goto fail;
    }

    in_buffer = AMediaCodec_getInputBuffer(ctx->decoder, in_index, &in_size);
    if (!in_buffer) {
        av_log(avctx, AV_LOG_ERROR, "Cannot get input buffer!\n");
        ret = AVERROR_EXTERNAL;
        goto fail;
    }

    if (!avpkt->data) {
        AMediaCodec_queueInputBuffer(ctx->decoder, in_index, 0, 0, 0, BUFFER_FLAG_EOS);
        ctx->eos_reached = 1;
        return 0;
    }

    av_assert0(avpkt->size <= in_size);
    memcpy(in_buffer, avpkt->data, avpkt->size);
    AMediaCodec_queueInputBuffer(ctx->decoder, in_index, 0, avpkt->size, avpkt->pts, 0);

fail:
    av_packet_unref(&filtered_pkt);
    return ret;
}

static void mediacodecndk_free_buffer(void *opaque, uint8_t *data)
{
    AVBufferRef *decoder_ref = opaque;
    AMediaCodec_releaseOutputBuffer(av_buffer_get_opaque(decoder_ref), (int32_t)data, false);
}

static int mediacodecndk_dequeue_output_buffer(AVCodecContext *avctx, AVFrame* frame)
{
    MediaCodecNDKDecoderContext *ctx = avctx->priv_data;
    AMediaCodecBufferInfo bufferInfo;
    size_t out_size;
    uint8_t* out_buffer = NULL;
    int32_t out_index = -2;
    int ret;
    AVBufferRef *ref;

    while (1) {
        out_index = AMediaCodec_dequeueOutputBuffer(ctx->decoder, &bufferInfo, TIMEOUT);
        if (out_index >= 0) {
            if (bufferInfo.flags & AMEDIACODEC_BUFFER_FLAG_END_OF_STREAM)
                return 0;
            break;
        } else if (out_index == AMEDIACODEC_INFO_OUTPUT_BUFFERS_CHANGED) {
            av_log(avctx, AV_LOG_DEBUG, "Mediacodec info output buffers changed\n");
        } else if (out_index == AMEDIACODEC_INFO_OUTPUT_FORMAT_CHANGED) {
            int32_t width, height, plane_height, stride;
            AMediaFormat *format = NULL;
            int color_format = 0;
            enum AVPixelFormat pix_fmt;
            format = AMediaCodec_getOutputFormat(ctx->decoder);

            AMediaFormat_getInt32(format, "crop-width", &width);
            AMediaFormat_getInt32(format, "crop-height", &height);
            AMediaFormat_getInt32(format, AMEDIAFORMAT_KEY_HEIGHT, &plane_height);
            AMediaFormat_getInt32(format, AMEDIAFORMAT_KEY_STRIDE, &stride);
            AMediaFormat_getInt32(format, AMEDIAFORMAT_KEY_COLOR_FORMAT, &color_format);
            AMediaFormat_delete(format);
            pix_fmt = ff_mediacodecndk_get_pix_fmt(color_format);
            if (pix_fmt == AV_PIX_FMT_NONE) {
                av_log(avctx, AV_LOG_ERROR, "Unsupported color format: %i\n", color_format);
                return AVERROR_EXTERNAL;
            }
            avctx->pix_fmt = pix_fmt;
            if (stride)
                ctx->stride = stride;
            if (plane_height)
                ctx->plane_height = plane_height;
            if (width && height)
                ff_set_dimensions(avctx, width, height);
            av_assert0(ctx->plane_height >= avctx->height &&
                       ctx->stride >= avctx->width);
        } else if (out_index == AMEDIACODEC_INFO_TRY_AGAIN_LATER) {
            return AVERROR(EAGAIN);
        } else {
            av_log(avctx, AV_LOG_ERROR, "Unexpected info code: %d", out_index);
            return AVERROR_EXTERNAL;
        }
    }

    out_buffer = AMediaCodec_getOutputBuffer(ctx->decoder, out_index, &out_size);

    if ((ret = ff_decode_frame_props(avctx, frame)) < 0)
        goto fail;

    frame->width = avctx->width;
    frame->height = avctx->height;

    if (!(ref = av_buffer_ref(ctx->decoder_ref))) {
        ret = AVERROR(ENOMEM);
        goto fail;
    }

    frame->buf[0] = av_buffer_create((void*)(uint64_t)out_index, out_size, mediacodecndk_free_buffer,
                                     ref, BUFFER_FLAG_READONLY);
    if (!frame->buf[0]) {
        av_buffer_unref(&ref);
        ret = AVERROR(ENOMEM);
        goto fail;
    }
    frame->data[0] = out_buffer;
    frame->linesize[0] = ctx->stride;
    frame->data[1] = out_buffer + ctx->stride * ctx->plane_height;
    if (avctx->pix_fmt == AV_PIX_FMT_NV12) {
        frame->linesize[1] = ctx->stride;
    } else {
        // FIXME: assuming chroma plane's stride is 1/2 of luma plane's for YV12
        frame->linesize[1] = frame->linesize[2] = ctx->stride / 2;
        frame->data[2] = frame->data[1] + ctx->stride * ctx->plane_height / 4;
    }
    frame->pts = frame->pkt_pts = bufferInfo.presentationTimeUs;
    frame->pkt_dts = AV_NOPTS_VALUE;
    return 1;
fail:
    AMediaCodec_releaseOutputBuffer(ctx->decoder, out_index, false);
    return ret;
}

static int mediacodecndk_decode_frame(AVCodecContext *avctx, void *data,
                                      int *got_frame, AVPacket *avpkt)
{
    MediaCodecNDKDecoderContext *ctx = avctx->priv_data;
    int ret;

    if (!ctx->eos_reached) {
        if ((ret = mediacodecndk_queue_input_buffer(avctx, avpkt)) < 0)
            return ret;
    }

    ret = mediacodecndk_dequeue_output_buffer(avctx, data);
    *got_frame = ret <= 0 ? 0 : 1;

    return ret < 0 ? ret : avpkt->size;
}

static av_cold int mediacodecndk_decode_close(AVCodecContext *avctx)
{
    MediaCodecNDKDecoderContext *ctx = avctx->priv_data;

    if (ctx->decoder) {
        AMediaCodec_flush(ctx->decoder);
        AMediaCodec_stop(ctx->decoder);
    }
    av_buffer_unref(&ctx->decoder_ref);
    av_bsf_free(&ctx->bsfc);
    return 0;
}

static av_cold void mediacodecndk_decode_flush(AVCodecContext *avctx)
{
    MediaCodecNDKDecoderContext *ctx = avctx->priv_data;
    AMediaCodec_flush(ctx->decoder);
}

#define FFMC_DEC_CLASS(NAME) \
    static const AVClass ffmediacodecndk_##NAME##_dec_class = { \
        .class_name = "mediacodecndk_" #NAME "_dec", \
        .version    = LIBAVUTIL_VERSION_INT, \
    };

#define FFMC_DEC(NAME, ID) \
    FFMC_DEC_CLASS(NAME) \
    AVCodec ff_##NAME##_mediacodecndk_decoder = { \
        .name           = #NAME "_mediacodecndk", \
        .long_name      = NULL_IF_CONFIG_SMALL(#NAME " (MediaCodec NDK)"), \
        .type           = AVMEDIA_TYPE_VIDEO, \
        .id             = ID, \
        .priv_data_size = sizeof(MediaCodecNDKDecoderContext), \
        .init           = mediacodecndk_decode_init, \
        .close          = mediacodecndk_decode_close, \
        .decode         = mediacodecndk_decode_frame, \
        .flush          = mediacodecndk_decode_flush, \
        .priv_class     = &ffmediacodecndk_##NAME##_dec_class, \
        .capabilities   = AV_CODEC_CAP_DELAY | FF_CODEC_CAP_INIT_CLEANUP, \
    };

FFMC_DEC(h264, AV_CODEC_ID_H264)
FFMC_DEC(hevc, AV_CODEC_ID_HEVC)
FFMC_DEC(mpeg2, AV_CODEC_ID_MPEG2VIDEO)
