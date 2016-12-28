/*
 * Copyright (c) 2016 Plex, Inc.
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

#include "plex.h"
#include "ffmpeg.h"

#include <sys/types.h>
#include <limits.h>
#include "strings.h"
#include "libavcodec/mpegvideo.h"
#include "libavfilter/vf_inlineass.h"
#include "libavformat/http.h"
#include "libavutil/timestamp.h"
#include "libavformat/internal.h"
#include "libavutil/thread.h"

PlexContext plexContext = {0};

#define LOG_LINE_SIZE 1024

#if HAVE_PTHREADS
static pthread_key_t logging_key, cur_line_key;
static pthread_once_t key_once = PTHREAD_ONCE_INIT;

static void make_keys(void)
{
    pthread_key_create(&logging_key, NULL);
    pthread_key_create(&cur_line_key, NULL);
}
#endif

static int av_log_level_plex = AV_LOG_QUIET;

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
int av_log_get_level_plex(void)
{
    return av_log_level_plex;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void av_log_set_level_plex(int level)
{
    av_log_level_plex = level;
}

char* PMS_IssueHttpRequest(const char* url, const char* verb)
{
    char* reply = NULL;
    AVIOContext *ioctx = NULL;
    AVDictionary *settings = NULL;
    int size = 0;
    int ret;
    char headers[1024];
    const char *token = getenv("X_PLEX_TOKEN");

    if (token && *token)
        snprintf(headers, sizeof(headers), "X-Plex-Token: %s\r\n", token);

    av_dict_set(&settings, "method", verb, 0);
    av_dict_set(&settings, "timeout", "1000000", 0);
    if (token && *token)
        av_dict_set(&settings, "headers", headers, 0);

    ret = avio_open2(&ioctx, url, AVIO_FLAG_READ,
                     NULL,
                     &settings);

    if (ret < 0)
        goto fail;

    size = avio_size(ioctx);
    if (size < 0)
        size = 4096;
    else if (!size)
        goto fail;

    reply = av_malloc(size);

    ret = avio_read(ioctx, reply, size);

    if (ret < 0)
        *reply = 0;

    avio_close(ioctx);
    av_dict_free(&settings);
    return reply;

fail:
    avio_close(ioctx);
    av_dict_free(&settings);
    reply = av_malloc(1);
    *reply = 0;
    return reply;
}

void PMS_Log(LogLevel level, const char* format, ...)
{
    // Format the mesage.
    char msg[2048];
    char tb[256];
    char url[4096];
    va_list va;
    if (av_log_level_plex == AV_LOG_QUIET)
        return;

    va_start(va, format);
    vsnprintf(msg, sizeof(msg), format, va);
    va_end(va);

    // Build the URL.
    snprintf(url, sizeof(url), "http://127.0.0.1:32400/log?level=%d&source=Transcoder&message=", level < 0 ? 0 : level);

    for (int i = 0; i < 256; i++) {
        tb[i] = isalnum(i)||i == '*'||i == '-'||i == '.'||i == '_'
            ? i : (i == ' ') ? '+' : 0;
    }
    for (unsigned i = 0; msg[i] && i < sizeof(msg); i++) {
        if (msg[i] > 0 && tb[(int)msg[i]])
            av_strlcatf(url, sizeof(url), "%c", tb[(int)msg[i]]);
        else
            av_strlcatf(url, sizeof(url), "%%%02X", msg[i]);
    }

    // Issue the request.
    av_free(PMS_IssueHttpRequest(url, "GET"));
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
static void plex_log_callback(void* ptr, int level, const char* fmt, va_list vl)
{
    int print_prefix = 1;
    char line[1024];
    va_list vl2;

#if HAVE_PTHREADS
    char *cur_line;
    pthread_once(&key_once, make_keys);
#else
    static __thread char cur_line[LOG_LINE_SIZE] = {0};
    static __thread int logging = 0;
#endif

    va_copy(vl2, vl);
    av_log_default_callback(ptr, level, fmt, vl2);

    if (level > av_log_level_plex)
        return;

    //Avoid recusive logging
#if HAVE_PTHREADS
    if (pthread_getspecific(logging_key))
        return;
    cur_line = pthread_getspecific(cur_line_key);
    if (!cur_line) {
        cur_line = av_mallocz(LOG_LINE_SIZE);
        if (!cur_line)
            return;
        pthread_setspecific(cur_line_key, cur_line);
    }
    pthread_setspecific(logging_key, (void*)1);
#else
    if (logging)
        return;
    logging = 1;
#endif

    av_log_format_line(ptr, level, fmt, vl, line, sizeof(line), &print_prefix);
    av_strlcat(cur_line, line, LOG_LINE_SIZE);
    if (print_prefix) {
        int len = strlen(cur_line);
        if (len) {
            cur_line[len - 1] = 0;
            PMS_Log((level / 8) - 2, "%s", cur_line);
            cur_line[0] = 0;
        }
    }

#if HAVE_PTHREADS
    pthread_setspecific(logging_key, NULL);
#else
    logging = 0;
#endif
}

void plex_report_stream(const AVStream *st)
{
    if (plexContext.progress_url &&
        (st->codecpar->codec_type == AVMEDIA_TYPE_VIDEO ||
         st->codecpar->codec_type == AVMEDIA_TYPE_AUDIO) &&
        !(st->disposition & AV_DISPOSITION_ATTACHED_PIC)) {
        char url[4096];
        snprintf(url, sizeof(url), "%s?index=%i&id=%i&codec=%s&type=%s",
                 plexContext.progress_url, st->index, st->id,
                 avcodec_get_name(st->codecpar->codec_id),
                 av_get_media_type_string(st->codecpar->codec_type));
        av_free(PMS_IssueHttpRequest(url, "PUT"));
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void plex_init(void)
{
    av_log_set_callback(plex_log_callback);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void plex_prepare_setup_streams_for_input_stream(InputStream* ist)
{
#if CONFIG_INLINEASS_FILTER
    int i;
    for (i = 0; i < plexContext.nb_inlineass_ctxs; i++) {
        InlineAssContext *ctx = &plexContext.inlineass_ctxs[i];
        if (ist->st->index == ctx->stream_index &&
            ist->file_index == ctx->file_index) {
            ist->discard = 0;
            ist->st->discard = AVDISCARD_NONE;
        }
    }
#endif
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void plex_link_subtitles_to_graph(AVFilterGraph* g)
{
#if CONFIG_INLINEASS_FILTER
    int contextId = 0;
    for (int i = 0; i < nb_filtergraphs && contextId < plexContext.nb_inlineass_ctxs; i++) {
        AVFilterGraph* graph = filtergraphs[i]->graph;
        for (int i = 0; i < graph->nb_filters && contextId < plexContext.nb_inlineass_ctxs; i++) {
            const AVFilterContext* filterCtx = graph->filters[i];
            if (strcmp(filterCtx->filter->name, "inlineass") == 0) {
                AVFilterContext *ctx = graph->filters[i];
                InlineAssContext *assCtx = &plexContext.inlineass_ctxs[contextId++];
                assCtx->ctx = ctx;

                if (assCtx->codec)
                    avfilter_inlineass_process_header(ctx, assCtx->codec);

                for (int j = 0; j < nb_input_streams; j++)
                    if (input_streams[j]->st->codecpar->codec_type == AVMEDIA_TYPE_ATTACHMENT)
                        avfilter_inlineass_add_attachment(ctx, input_streams[j]->st);

                avfilter_inlineass_set_fonts(ctx);
            }
        }
    }
#endif
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
int plex_opt_subtitle_stream(void *optctx, const char *opt, const char *arg)
{
#if CONFIG_INLINEASS_FILTER
    InlineAssContext *m = NULL;
    int i, file_idx;
    char *p;
    char *map = av_strdup(arg);

    file_idx = strtol(map, &p, 0);
    if (file_idx >= nb_input_files || file_idx < 0) {
        av_log(NULL, AV_LOG_FATAL, "Invalid subtitle input file index: %d.\n", file_idx);
        goto finish;
    }

    for (i = 0; i < input_files[file_idx]->nb_streams; i++) {
        if (check_stream_specifier(input_files[file_idx]->ctx, input_files[file_idx]->ctx->streams[i],
                    *p == ':' ? p + 1 : p) <= 0)
            continue;
        if (input_files[file_idx]->ctx->streams[i]->codecpar->codec_type != AVMEDIA_TYPE_SUBTITLE) {
            av_log(NULL, AV_LOG_ERROR, "Stream '%s' is not a subtitle stream.\n", arg);
            continue;
        }
        GROW_ARRAY(plexContext.inlineass_ctxs, plexContext.nb_inlineass_ctxs);
        m = &plexContext.inlineass_ctxs[plexContext.nb_inlineass_ctxs - 1];

        m->file_index   = file_idx;
        m->stream_index = i;
        break;
    }

finish:
    if (!m)
        av_log(NULL, AV_LOG_ERROR, "Subtitle stream map '%s' matches no streams.\n", arg);

    av_freep(&map);
#endif
    return 0;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void plex_process_subtitle_header(const InputStream *ist)
{
#if CONFIG_INLINEASS_FILTER
    int i;
    for (i = 0; i < plexContext.nb_inlineass_ctxs; i++) {
        InlineAssContext *ctx = &plexContext.inlineass_ctxs[i];
        if (ist->st->index == ctx->stream_index && ist->file_index == ctx->file_index) {
            ctx->codec = ist->st->codec;
            if (ctx->ctx)
                avfilter_inlineass_process_header(ctx->ctx, ist->st->codec);
        }
    }
#endif
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
int plex_process_subtitles(const InputStream *ist, AVPacket *pkt)
{
#if CONFIG_INLINEASS_FILTER
    int i;
    /* If we're burning subtitles, pass discarded subtitle packets of the
     * appropriate stream  to the subtitle renderer */
    for (i = 0; i < plexContext.nb_inlineass_ctxs; i++) {
        InlineAssContext *ctx = &plexContext.inlineass_ctxs[i];
        if (ist->st->index == ctx->stream_index &&
            ist->file_index == ctx->file_index && ctx->ctx) {
            avfilter_inlineass_append_data(ctx->ctx, ist->st, pkt);
            return 1;
        }
    }
#endif
    return 0;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
int plex_opt_progress_url(void *optctx, const char *opt, const char *arg)
{
    plexContext.progress_url = (char*)arg;
    return 0;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
int plex_opt_loglevel(void *o, const char *opt, const char *arg)
{
    opt_loglevel((void*)&av_log_set_level_plex, opt, arg);
    return 0;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void plex_feedback(const AVFormatContext *ic)
{
    if (plexContext.progress_url) {
        char url[4096];
        double duration = -1;
        if (ic && ic->duration != AV_NOPTS_VALUE)
            duration = ic->duration / (double)AV_TIME_BASE;
        snprintf(url, sizeof(url), "%s?duration=%f", plexContext.progress_url, duration);
        av_free(PMS_IssueHttpRequest(url, "PUT"));
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void plex_link_input_stream(const InputStream *ist)
{
#if CONFIG_INLINEASS_FILTER
    int i;
    if (ist->st->codecpar->codec_type == AVMEDIA_TYPE_VIDEO)
        for (i = 0; i < plexContext.nb_inlineass_ctxs; i++)
            if (plexContext.inlineass_ctxs[i].ctx)
                avfilter_inlineass_set_storage_size(plexContext.inlineass_ctxs[i].ctx, ist->st->codecpar->width, ist->st->codecpar->height);
#endif
}
