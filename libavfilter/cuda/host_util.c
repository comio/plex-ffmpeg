/*
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

#include "libavfilter/colorspace.h"
#include "host_util.h"

int ff_make_cuda_frame(FFCUDAFrame *dst, const AVFrame *src)
{
    int i = 0;
    for (i = 0; i < 4; i++) {
        dst->data[i] = src->data[i];
        dst->linesize[i] = src->linesize[i];
    }

    dst->width  = src->width;
    dst->height = src->height;

    dst->peak = ff_determine_signal_peak(src);
    dst->avg  = 0.f; //FIXME

    return 0;
}
