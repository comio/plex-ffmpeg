/*
* Copyright (c) 2017, NVIDIA CORPORATION. All rights reserved.
* Copyright (c) 2019 Rodger Combs
*
* This file is part of FFmpeg.
*
* Permission is hereby granted, free of charge, to any person obtaining a
* copy of this software and associated documentation files (the "Software"),
* to deal in the Software without restriction, including without limitation
* the rights to use, copy, modify, merge, publish, distribute, sublicense,
* and/or sell copies of the Software, and to permit persons to whom the
* Software is furnished to do so, subject to the following conditions:
*
* The above copyright notice and this permission notice shall be included in
* all copies or substantial portions of the Software.
*
* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
* IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
* FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
* THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
* LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
* FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
* DEALINGS IN THE SOFTWARE.
*/

#include <stdio.h>
#include <string.h>

#include "libavutil/avassert.h"
#include "libavutil/avstring.h"
#include "libavutil/common.h"
#include "libavutil/hwcontext.h"
#include "libavutil/hwcontext_cuda_internal.h"
#include "libavutil/cuda_check.h"
#include "libavutil/internal.h"
#include "libavutil/opt.h"
#include "libavutil/pixdesc.h"

#include "avfilter.h"
#include "dither_matrix.h"
#include "formats.h"
#include "internal.h"
#include "scale.h"
#include "video.h"

static const enum AVPixelFormat supported_formats[] = {
    AV_PIX_FMT_YUV420P,
    AV_PIX_FMT_NV12,
    AV_PIX_FMT_YUV444P,
    AV_PIX_FMT_P010,
    AV_PIX_FMT_P016
};

#define DIV_UP(a, b) ( ((a) + (b) - 1) / (b) )
#define ALIGN_UP(a, b) (((a) + (b) - 1) & ~((b) - 1))
#define NUM_BUFFERS 2
#define BLOCKX 32
#define BLOCKY 16

#define CHECK_CU(x) FF_CUDA_CHECK_DL(ctx, s->hwctx->internal->cuda_dl, x)

typedef struct CUDAScaleContext {
    const AVClass *class;

    AVCUDADeviceContext *hwctx;

    enum AVPixelFormat in_fmt;
    enum AVPixelFormat out_fmt;

    struct {
        int width;
        int height;
    } planes_in[3], planes_out[3];

    AVBufferRef *frames_ctx;
    AVFrame     *frame;

    AVFrame *tmp_frame;
    int passthrough;

    /**
     * Output sw format. AV_PIX_FMT_NONE for no conversion.
     */
    enum AVPixelFormat format;
    char *format_str;

    char *w_expr;               ///< width  expression string
    char *h_expr;               ///< height expression string

    CUcontext   cu_ctx;
    CUmodule    cu_module;

#define VARIANT(NAME) \
    CUfunction  cu_func_ ## NAME;
#define VARIANTSET(NAME) \
    VARIANT(NAME) \
    VARIANT(NAME ## _c) \
    VARIANT(NAME ## _p2) \
    VARIANT(NAME ## _2) \
    VARIANT(NAME ## _2_u) \
    VARIANT(NAME ## _2_v) \
    VARIANT(NAME ## _4)

    VARIANTSET(8_8)
    VARIANTSET(16_16)
    VARIANTSET(8_16)
    VARIANTSET(16_8)
#undef VARIANTSET
#undef VARIANT

    CUfunction  cu_func_luma;
    CUfunction  cu_func_chroma_u;
    CUfunction  cu_func_chroma_v;

    CUstream    cu_stream;

    CUdeviceptr srcBuffer;
    CUdeviceptr dstBuffer;
    int         tex_alignment;

    const AVPixFmtDescriptor *in_desc, *out_desc;
    int         in_planes, out_planes;

    CUdeviceptr ditherBuffer;
    CUtexObject ditherTex;
} CUDAScaleContext;

static av_cold int cudascale_init(AVFilterContext *ctx)
{
    CUDAScaleContext *s = ctx->priv;

    if (!strcmp(s->format_str, "same")) {
        s->format = AV_PIX_FMT_NONE;
    } else {
        s->format = av_get_pix_fmt(s->format_str);
        if (s->format == AV_PIX_FMT_NONE) {
            av_log(ctx, AV_LOG_ERROR, "Unrecognized pixel format: %s\n", s->format_str);
            return AVERROR(EINVAL);
        }
    }

    s->frame = av_frame_alloc();
    if (!s->frame)
        return AVERROR(ENOMEM);

    s->tmp_frame = av_frame_alloc();
    if (!s->tmp_frame)
        return AVERROR(ENOMEM);

    return 0;
}

static av_cold void cudascale_uninit(AVFilterContext *ctx)
{
    CUDAScaleContext *s = ctx->priv;

    if (s->hwctx) {
        CudaFunctions *cu = s->hwctx->internal->cuda_dl;
        CUcontext dummy, cuda_ctx = s->hwctx->cuda_ctx;

        CHECK_CU(cu->cuCtxPushCurrent(cuda_ctx));

        if (s->ditherTex) {
            CHECK_CU(cu->cuTexObjectDestroy(s->ditherTex));
            s->ditherTex = 0;
        }

        if (s->ditherBuffer) {
            CHECK_CU(cu->cuMemFree(s->ditherBuffer));
            s->ditherBuffer = 0;
        }

        CHECK_CU(cu->cuCtxPopCurrent(&dummy));
    }

    av_frame_free(&s->frame);
    av_buffer_unref(&s->frames_ctx);
    av_frame_free(&s->tmp_frame);
}

static int cudascale_query_formats(AVFilterContext *ctx)
{
    static const enum AVPixelFormat pixel_formats[] = {
        AV_PIX_FMT_CUDA, AV_PIX_FMT_NONE,
    };
    AVFilterFormats *pix_fmts = ff_make_format_list(pixel_formats);

    return ff_set_common_formats(ctx, pix_fmts);
}

static av_cold int init_stage(CUDAScaleContext *s, AVBufferRef *device_ctx)
{
    AVBufferRef *out_ref = NULL;
    AVHWFramesContext *out_ctx;
    int in_sw, in_sh, out_sw, out_sh;
    int ret, i;

    av_pix_fmt_get_chroma_sub_sample(s->in_fmt,  &in_sw,  &in_sh);
    av_pix_fmt_get_chroma_sub_sample(s->out_fmt, &out_sw, &out_sh);
    if (!s->planes_out[0].width) {
        s->planes_out[0].width  = s->planes_in[0].width;
        s->planes_out[0].height = s->planes_in[0].height;
    }

    for (i = 1; i < FF_ARRAY_ELEMS(s->planes_in); i++) {
        s->planes_in[i].width   = s->planes_in[0].width   >> in_sw;
        s->planes_in[i].height  = s->planes_in[0].height  >> in_sh;
        s->planes_out[i].width  = s->planes_out[0].width  >> out_sw;
        s->planes_out[i].height = s->planes_out[0].height >> out_sh;
    }

    out_ref = av_hwframe_ctx_alloc(device_ctx);
    if (!out_ref)
        return AVERROR(ENOMEM);
    out_ctx = (AVHWFramesContext*)out_ref->data;

    out_ctx->format    = AV_PIX_FMT_CUDA;
    out_ctx->sw_format = s->out_fmt;
    out_ctx->width     = FFALIGN(s->planes_out[0].width,  32);
    out_ctx->height    = FFALIGN(s->planes_out[0].height, 32);

    ret = av_hwframe_ctx_init(out_ref);
    if (ret < 0)
        goto fail;

    av_frame_unref(s->frame);
    ret = av_hwframe_get_buffer(out_ref, s->frame, 0);
    if (ret < 0)
        goto fail;

    s->frame->width  = s->planes_out[0].width;
    s->frame->height = s->planes_out[0].height;

    av_buffer_unref(&s->frames_ctx);
    s->frames_ctx = out_ref;

    return 0;
fail:
    av_buffer_unref(&out_ref);
    return ret;
}

static int format_is_supported(enum AVPixelFormat fmt)
{
    int i;

    for (i = 0; i < FF_ARRAY_ELEMS(supported_formats); i++)
        if (supported_formats[i] == fmt)
            return 1;
    return 0;
}

static av_cold int init_processing_chain(AVFilterContext *ctx, int in_width, int in_height,
                                         int out_width, int out_height)
{
    CUDAScaleContext *s = ctx->priv;

    AVHWFramesContext *in_frames_ctx;

    enum AVPixelFormat in_format;
    enum AVPixelFormat out_format;
    int ret;

    /* check that we have a hw context */
    if (!ctx->inputs[0]->hw_frames_ctx) {
        av_log(ctx, AV_LOG_ERROR, "No hw context provided on input\n");
        return AVERROR(EINVAL);
    }
    in_frames_ctx = (AVHWFramesContext*)ctx->inputs[0]->hw_frames_ctx->data;
    in_format     = in_frames_ctx->sw_format;
    out_format    = (s->format == AV_PIX_FMT_NONE) ? in_format : s->format;

    if (!format_is_supported(in_format)) {
        av_log(ctx, AV_LOG_ERROR, "Unsupported input format: %s\n",
               av_get_pix_fmt_name(in_format));
        return AVERROR(ENOSYS);
    }
    if (!format_is_supported(out_format)) {
        av_log(ctx, AV_LOG_ERROR, "Unsupported output format: %s\n",
               av_get_pix_fmt_name(out_format));
        return AVERROR(ENOSYS);
    }

    if (in_width == out_width && in_height == out_height && in_format == out_format)
        s->passthrough = 1;

    s->in_fmt = in_format;
    s->out_fmt = out_format;

    s->planes_in[0].width   = in_width;
    s->planes_in[0].height  = in_height;
    s->planes_out[0].width  = out_width;
    s->planes_out[0].height = out_height;

    ret = init_stage(s, in_frames_ctx->device_ref);
    if (ret < 0)
        return ret;

    ctx->outputs[0]->hw_frames_ctx = av_buffer_ref(s->frames_ctx);
    if (!ctx->outputs[0]->hw_frames_ctx)
        return AVERROR(ENOMEM);

    return 0;
}

static av_cold int cudascale_setup_dither(AVFilterContext *ctx)
{
    CUDAScaleContext    *s  = ctx->priv;
    AVFilterLink        *inlink = ctx->inputs[0];
    AVHWFramesContext   *frames_ctx = (AVHWFramesContext*)inlink->hw_frames_ctx->data;
    AVCUDADeviceContext *device_hwctx = frames_ctx->device_ctx->hwctx;
    CudaFunctions       *cu = device_hwctx->internal->cuda_dl;
    CUcontext dummy, cuda_ctx = device_hwctx->cuda_ctx;
    int ret = 0;

    CUDA_MEMCPY2D cpy = {
        .srcMemoryType = CU_MEMORYTYPE_HOST,
        .dstMemoryType = CU_MEMORYTYPE_DEVICE,
        .srcHost       = ff_fruit_dither_matrix,
        .dstDevice     = 0,
        .srcPitch      = ff_fruit_dither_size * sizeof(ff_fruit_dither_matrix[0]),
        .dstPitch      = ff_fruit_dither_size * sizeof(ff_fruit_dither_matrix[0]),
        .WidthInBytes  = ff_fruit_dither_size * sizeof(ff_fruit_dither_matrix[0]),
        .Height        = ff_fruit_dither_size,
    };

    CUDA_TEXTURE_DESC tex_desc = {
        .filterMode = CU_TR_FILTER_MODE_POINT,
        .flags = CU_TRSF_READ_AS_INTEGER,
    };

    CUDA_RESOURCE_DESC res_desc = {
        .resType = CU_RESOURCE_TYPE_PITCH2D,
        .res.pitch2D.format = CU_AD_FORMAT_UNSIGNED_INT16,
        .res.pitch2D.numChannels = 1,
        .res.pitch2D.width = ff_fruit_dither_size,
        .res.pitch2D.height = ff_fruit_dither_size,
        .res.pitch2D.pitchInBytes = ff_fruit_dither_size * sizeof(ff_fruit_dither_matrix[0]),
        .res.pitch2D.devPtr = 0,
    };

    av_assert0(sizeof(ff_fruit_dither_matrix) == sizeof(ff_fruit_dither_matrix[0]) * ff_fruit_dither_size * ff_fruit_dither_size);

    if ((ret = CHECK_CU(cu->cuCtxPushCurrent(cuda_ctx))) < 0)
        return ret;

    if ((ret = CHECK_CU(cu->cuMemAlloc(&s->ditherBuffer, sizeof(ff_fruit_dither_matrix)))) < 0)
        goto fail;

    res_desc.res.pitch2D.devPtr = cpy.dstDevice = s->ditherBuffer;

    if ((ret = CHECK_CU(cu->cuMemcpy2D(&cpy))) < 0)
        goto fail;

    if ((ret = CHECK_CU(cu->cuTexObjectCreate(&s->ditherTex, &res_desc, &tex_desc, NULL))) < 0)
        goto fail;

fail:
    CHECK_CU(cu->cuCtxPopCurrent(&dummy));
    return ret;
}

static av_cold int cudascale_config_props(AVFilterLink *outlink)
{
    AVFilterContext *ctx = outlink->src;
    AVFilterLink *inlink = outlink->src->inputs[0];
    CUDAScaleContext *s  = ctx->priv;
    AVHWFramesContext     *frames_ctx = (AVHWFramesContext*)inlink->hw_frames_ctx->data;
    AVCUDADeviceContext *device_hwctx = frames_ctx->device_ctx->hwctx;
    CUcontext dummy, cuda_ctx = device_hwctx->cuda_ctx;
    CudaFunctions *cu = device_hwctx->internal->cuda_dl;
    int w, h;
    int i;
    int ret;

    extern char vf_scale_cuda_ptx[];

    s->hwctx = device_hwctx;
    s->cu_stream = s->hwctx->stream;

    ret = CHECK_CU(cu->cuCtxPushCurrent(cuda_ctx));
    if (ret < 0)
        goto fail;

    ret = CHECK_CU(cu->cuModuleLoadData(&s->cu_module, vf_scale_cuda_ptx));
    if (ret < 0)
        goto fail;

#define VARIANT(NAME) \
    CHECK_CU(cu->cuModuleGetFunction(&s->cu_func_ ## NAME, s->cu_module, "Subsample_Bilinear_" #NAME)); \
    if (ret < 0) \
        goto fail;

#define VARIANTSET(NAME) \
    VARIANT(NAME) \
    VARIANT(NAME ## _c) \
    VARIANT(NAME ## _2) \
    VARIANT(NAME ## _p2) \
    VARIANT(NAME ## _2_u) \
    VARIANT(NAME ## _2_v) \
    VARIANT(NAME ## _4)

    VARIANTSET(8_8)
    VARIANTSET(16_16)
    VARIANTSET(8_16)
    VARIANTSET(16_8)
#undef VARIANTSET
#undef VARIANT

    CHECK_CU(cu->cuCtxPopCurrent(&dummy));

    if ((ret = ff_scale_eval_dimensions(s,
                                        s->w_expr, s->h_expr,
                                        inlink, outlink,
                                        &w, &h)) < 0)
        goto fail;

    if (((int64_t)h * inlink->w) > INT_MAX  ||
        ((int64_t)w * inlink->h) > INT_MAX)
        av_log(ctx, AV_LOG_ERROR, "Rescaled value for width or height is too big.\n");

    outlink->w = w;
    outlink->h = h;

    ret = init_processing_chain(ctx, inlink->w, inlink->h, w, h);
    if (ret < 0)
        return ret;

    s->in_desc  = av_pix_fmt_desc_get(s->in_fmt);
    s->out_desc = av_pix_fmt_desc_get(s->out_fmt);

    for (i = 0; i < s->in_desc->nb_components; i++)
        s->in_planes  = FFMAX(s->in_planes,  s->in_desc ->comp[i].plane + 1);

    for (i = 0; i < s->in_desc->nb_components; i++)
        s->out_planes = FFMAX(s->out_planes, s->out_desc->comp[i].plane + 1);

#define VARIANT(INDEPTH, OUTDEPTH, SUFFIX) s->cu_func_ ## INDEPTH ## _ ## OUTDEPTH ## SUFFIX
#define BITS(n) ((n + 7) & ~7)
#define VARIANTSET(INDEPTH, OUTDEPTH) \
    else if (BITS(s->in_desc->comp[0].depth)  == INDEPTH && \
             BITS(s->out_desc->comp[0].depth) == OUTDEPTH) { \
        s->cu_func_luma = VARIANT(INDEPTH, OUTDEPTH,); \
        if (s->in_planes == 3 && s->out_planes == 3) { \
            s->cu_func_chroma_u = s->cu_func_chroma_v = VARIANT(INDEPTH, OUTDEPTH, _c); \
        } else if (s->in_planes == 3 && s->out_planes == 2) { \
            s->cu_func_chroma_u = s->cu_func_chroma_v = VARIANT(INDEPTH, OUTDEPTH, _p2); \
        } else if (s->in_planes == 2 && s->out_planes == 2) { \
            s->cu_func_chroma_u = VARIANT(INDEPTH, OUTDEPTH, _2); \
        } else if (s->in_planes == 2 && s->out_planes == 3) { \
            s->cu_func_chroma_u = VARIANT(INDEPTH, OUTDEPTH, _2_u); \
            s->cu_func_chroma_v = VARIANT(INDEPTH, OUTDEPTH, _2_v); \
        } else { \
            ret = AVERROR_BUG; \
            goto fail; \
        } \
    }

    if (0) {}
    VARIANTSET(8,  8)
    VARIANTSET(16, 16)
    VARIANTSET(8,  16)
    VARIANTSET(16, 8)
    else {
        ret = AVERROR_BUG;
        goto fail;
    }
#undef VARIANTSET
#undef VARIANT

    if (s->in_desc->comp[0].depth > s->out_desc->comp[0].depth) {
        if ((ret = cudascale_setup_dither(ctx)) < 0)
            goto fail;
    }

    av_log(ctx, AV_LOG_VERBOSE, "w:%d h:%d -> w:%d h:%d\n",
           inlink->w, inlink->h, outlink->w, outlink->h);

    if (inlink->sample_aspect_ratio.num) {
        outlink->sample_aspect_ratio = av_mul_q((AVRational){outlink->h*inlink->w,
                                                             outlink->w*inlink->h},
                                                inlink->sample_aspect_ratio);
    } else {
        outlink->sample_aspect_ratio = inlink->sample_aspect_ratio;
    }

    return 0;

fail:
    return ret;
}

static int call_resize_kernel(AVFilterContext *ctx, CUfunction func, int channels,
                              uint8_t *src_dptr, int src_width, int src_height, int src_pitch,
                              uint8_t *dst_dptr, int dst_width, int dst_height, int dst_pitch,
                              int pixel_size)
{
    CUDAScaleContext *s = ctx->priv;
    CudaFunctions *cu = s->hwctx->internal->cuda_dl;
    CUdeviceptr dst_devptr = (CUdeviceptr)dst_dptr;
    CUtexObject tex = 0;
    void *args_uchar[] = { &tex, &dst_devptr, &dst_width, &dst_height, &dst_pitch, &src_width, &src_height, &s->ditherTex };
    int ret;

    CUDA_TEXTURE_DESC tex_desc = {
        .filterMode = CU_TR_FILTER_MODE_LINEAR,
        .flags = CU_TRSF_READ_AS_INTEGER,
    };

    CUDA_RESOURCE_DESC res_desc = {
        .resType = CU_RESOURCE_TYPE_PITCH2D,
        .res.pitch2D.format = pixel_size == 1 ?
                              CU_AD_FORMAT_UNSIGNED_INT8 :
                              CU_AD_FORMAT_UNSIGNED_INT16,
        .res.pitch2D.numChannels = channels,
        .res.pitch2D.width = src_width,
        .res.pitch2D.height = src_height,
        .res.pitch2D.pitchInBytes = src_pitch,
        .res.pitch2D.devPtr = (CUdeviceptr)src_dptr,
    };

    ret = CHECK_CU(cu->cuTexObjectCreate(&tex, &res_desc, &tex_desc, NULL));
    if (ret < 0)
        goto exit;

    ret = CHECK_CU(cu->cuLaunchKernel(func,
                                      DIV_UP(dst_width, BLOCKX), DIV_UP(dst_height, BLOCKY), 1,
                                      BLOCKX, BLOCKY, 1, 0, s->cu_stream, args_uchar, NULL));

exit:
    if (tex)
        CHECK_CU(cu->cuTexObjectDestroy(tex));

    return ret;
}

static int scalecuda_resize(AVFilterContext *ctx,
                            AVFrame *out, AVFrame *in)
{
    CUDAScaleContext *s = ctx->priv;

#define DEPTH_BYTES(depth) (((depth) + 7) / 8)

    call_resize_kernel(ctx, s->cu_func_luma, 1,
                       in->data[0], in->width, in->height, in->linesize[0],
                       out->data[0], out->width, out->height, out->linesize[0],
                       DEPTH_BYTES(s->in_desc->comp[0].depth));

    call_resize_kernel(ctx, s->cu_func_chroma_u, s->in_planes == 2 ? 2 : 1,
                       in->data[1],
                       AV_CEIL_RSHIFT(in->width,  s->in_desc->log2_chroma_w),
                       AV_CEIL_RSHIFT(in->height, s->in_desc->log2_chroma_h),
                       in->linesize[1],
                       out->data[1],
                       AV_CEIL_RSHIFT(out->width,  s->out_desc->log2_chroma_w),
                       AV_CEIL_RSHIFT(out->height, s->out_desc->log2_chroma_h),
                       out->linesize[1],
                       DEPTH_BYTES(s->in_desc->comp[1].depth));

    if (s->cu_func_chroma_v) {
        call_resize_kernel(ctx, s->cu_func_chroma_v, s->in_planes == 2 ? 2 : 1,
                           in->data[s->in_desc->comp[2].plane],
                           AV_CEIL_RSHIFT(in->width,       s->in_desc->log2_chroma_w),
                           AV_CEIL_RSHIFT(in->height,      s->in_desc->log2_chroma_h),
                           in->linesize[s->in_desc->comp[2].plane],
                           out->data[s->out_desc->comp[2].plane] + s->out_desc->comp[2].offset,
                           AV_CEIL_RSHIFT(out->width,       s->out_desc->log2_chroma_w),
                           AV_CEIL_RSHIFT(out->height,      s->out_desc->log2_chroma_h),
                           out->linesize[s->out_desc->comp[2].plane],
                           DEPTH_BYTES(s->in_desc->comp[2].depth));
    }

    return 0;
}

static int cudascale_scale(AVFilterContext *ctx, AVFrame *out, AVFrame *in)
{
    CUDAScaleContext *s = ctx->priv;
    AVFrame *src = in;
    int ret;

    ret = scalecuda_resize(ctx, s->frame, src);
    if (ret < 0)
        return ret;

    src = s->frame;
    ret = av_hwframe_get_buffer(src->hw_frames_ctx, s->tmp_frame, 0);
    if (ret < 0)
        return ret;

    av_frame_move_ref(out, s->frame);
    av_frame_move_ref(s->frame, s->tmp_frame);

    s->frame->width  = s->planes_out[0].width;
    s->frame->height = s->planes_out[0].height;

    ret = av_frame_copy_props(out, in);
    if (ret < 0)
        return ret;

    return 0;
}

static int cudascale_filter_frame(AVFilterLink *link, AVFrame *in)
{
    AVFilterContext       *ctx = link->dst;
    CUDAScaleContext        *s = ctx->priv;
    AVFilterLink      *outlink = ctx->outputs[0];
    CudaFunctions          *cu = s->hwctx->internal->cuda_dl;

    AVFrame *out = NULL;
    CUcontext dummy;
    int ret = 0;

    out = av_frame_alloc();
    if (!out) {
        ret = AVERROR(ENOMEM);
        goto fail;
    }

    ret = CHECK_CU(cu->cuCtxPushCurrent(s->hwctx->cuda_ctx));
    if (ret < 0)
        goto fail;

    ret = cudascale_scale(ctx, out, in);

    CHECK_CU(cu->cuCtxPopCurrent(&dummy));
    if (ret < 0)
        goto fail;

    av_reduce(&out->sample_aspect_ratio.num, &out->sample_aspect_ratio.den,
              (int64_t)in->sample_aspect_ratio.num * outlink->h * link->w,
              (int64_t)in->sample_aspect_ratio.den * outlink->w * link->h,
              INT_MAX);

    av_frame_free(&in);
    return ff_filter_frame(outlink, out);
fail:
    av_frame_free(&in);
    av_frame_free(&out);
    return ret;
}

#define OFFSET(x) offsetof(CUDAScaleContext, x)
#define FLAGS (AV_OPT_FLAG_FILTERING_PARAM|AV_OPT_FLAG_VIDEO_PARAM)
static const AVOption options[] = {
    { "w",      "Output video width",  OFFSET(w_expr),     AV_OPT_TYPE_STRING, { .str = "iw"   }, .flags = FLAGS },
    { "h",      "Output video height", OFFSET(h_expr),     AV_OPT_TYPE_STRING, { .str = "ih"   }, .flags = FLAGS },
    { "format", "Output format",       OFFSET(format_str), AV_OPT_TYPE_STRING, { .str = "same" }, .flags = FLAGS },
    { NULL },
};

static const AVClass cudascale_class = {
    .class_name = "cudascale",
    .item_name  = av_default_item_name,
    .option     = options,
    .version    = LIBAVUTIL_VERSION_INT,
};

static const AVFilterPad cudascale_inputs[] = {
    {
        .name        = "default",
        .type        = AVMEDIA_TYPE_VIDEO,
        .filter_frame = cudascale_filter_frame,
    },
    { NULL }
};

static const AVFilterPad cudascale_outputs[] = {
    {
        .name         = "default",
        .type         = AVMEDIA_TYPE_VIDEO,
        .config_props = cudascale_config_props,
    },
    { NULL }
};

AVFilter ff_vf_scale_cuda = {
    .name      = "scale_cuda",
    .description = NULL_IF_CONFIG_SMALL("GPU accelerated video resizer"),

    .init          = cudascale_init,
    .uninit        = cudascale_uninit,
    .query_formats = cudascale_query_formats,

    .priv_size = sizeof(CUDAScaleContext),
    .priv_class = &cudascale_class,

    .inputs    = cudascale_inputs,
    .outputs   = cudascale_outputs,

    .flags_internal = FF_FILTER_FLAG_HWFRAME_AWARE,
};
