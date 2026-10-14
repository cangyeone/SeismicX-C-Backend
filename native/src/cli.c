/* Dependency-free raw-waveform CLI. SAC/miniSEED decoding belongs to the caller. */
#include "pnsn.h"
#include <errno.h>
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
static int number(const char *s,size_t *value) {
    if(!s || !*s || *s=='-')return 0;
    char *end;errno=0;unsigned long long n=strtoull(s,&end,10);
    if(errno || *end || n>SIZE_MAX)return 0;*value=(size_t)n;return 1;
}
int main(int argc,char **argv) {
    const char *model=NULL,*input=NULL,*output=NULL;int fixed=0,kind=PNSN_F64,planar=0;size_t batch=0;
    pnsn_options options=pnsn_default_options();size_t temp;
    for(int i=1;i<argc;i++) {
        if(!strcmp(argv[i],"--predict")){fixed=1;continue;}
        if(!strcmp(argv[i],"--history-planar")){planar=1;continue;}
        if(!strcmp(argv[i],"--difference")){options.difference=1;continue;}
        if(i+1>=argc)goto usage;
        const char *key=argv[i++],*value=argv[i];
        if(!strcmp(key,"--batch")){if(!number(value,&batch) || !batch)goto usage;}
        else if(!strcmp(key,"--model"))model=value;
        else if(!strcmp(key,"--input"))input=value;
        else if(!strcmp(key,"--output"))output=value;
        else if(!strcmp(key,"--dtype")) {
            if(!strcmp(value,"f32"))kind=PNSN_F32;else if(!strcmp(value,"f64"))kind=PNSN_F64;else if(!strcmp(value,"i32"))kind=PNSN_I32;else goto usage;
        } else if(!strcmp(key,"--window")){if(!number(value,&options.window))goto usage;}
        else if(!strcmp(key,"--overlap")){if(!number(value,&options.overlap))goto usage;}
        else if(!strcmp(key,"--max-candidates")){if(!number(value,&options.max_candidates))goto usage;}
        else if(!strcmp(key,"--gap")){if(!number(value,&temp) || temp>INT64_MAX)goto usage;options.min_gap=(int64_t)temp;}
        else if(!strcmp(key,"--threads")){if(!number(value,&temp) || temp>256)goto usage;options.threads=(int)temp;}
        else if(!strcmp(key,"--threshold")){char *end;options.threshold=strtof(value,&end);if(*end)goto usage;}
        else goto usage;
    }
    if(!model || !input || !output || (planar && (fixed || batch)))goto usage;
    FILE *f=fopen(input,"rb");if(!f){perror(input);return 1;}
    size_t used=0,cap=0;unsigned char *data=NULL;
    for(;;) {
        if(cap-used<65536) {
            if(cap>SIZE_MAX-65536){fprintf(stderr,"Input too large\n");free(data);fclose(f);return 1;}
            cap+=65536;void *p=realloc(data,cap);if(!p){free(data);fclose(f);return 1;}data=p;
        }
        size_t n=fread(data+used,1,cap-used,f);used+=n;
        if(!n){if(ferror(f)){free(data);fclose(f);return 1;}break;}
    }
    fclose(f);size_t stride=(kind==PNSN_F64?8:4)*3;
    if(used%stride){fprintf(stderr,"Input byte count must be divisible by 3*dtype size\n");free(data);return 1;}
    size_t samples=used/stride;
    pnsn_context *ctx=pnsn_open(model,&options);if(!ctx){fprintf(stderr,"%s\n",pnsn_last_error());free(data);return 1;}
    int status=0,classes=pnsn_classes(ctx);
    const char *phases[4]={classes==3?"P":"Pg",classes==3?"S":"Sg","Pn","Sn"};
    if(batch) {
        if(planar || batch>SIZE_MAX/options.window || samples!=batch*options.window) {fprintf(stderr,"Batch requires [B,3,window] raw input byte count\n");status=1;}
        else if(fixed) {
            float *prob=malloc(samples*classes*sizeof(float));
            if(!prob)status=1;
            else if(pnsn_predict_batch(ctx,data,(pnsn_dtype)kind,batch,options.window,prob)){fprintf(stderr,"%s\n",pnsn_last_error());status=1;}
            else {f=fopen(output,"wb");if(!f || fwrite(prob,sizeof(float),samples*classes,f)!=samples*classes)status=1;if(f && fclose(f))status=1;}
            free(prob);
        } else {
            size_t per=(classes-1)*(options.window/((uint64_t)options.min_gap+1)+1);
            if(per>options.max_candidates)per=options.max_candidates;
            pnsn_pick *p=batch<=SIZE_MAX/(per*sizeof(*p))?malloc(batch*per*sizeof(*p)):NULL;
            size_t *offsets=batch<SIZE_MAX/sizeof(size_t)?malloc((batch+1)*sizeof(size_t)):NULL,count=0;
            if(!p || !offsets)status=1;
            else if(pnsn_pick_batch(ctx,data,(pnsn_dtype)kind,batch,options.window,p,batch*per,offsets,&count)){fprintf(stderr,"%s\n",pnsn_last_error());status=1;}
            else {
                f=fopen(output,"w");if(!f)status=1;
                else {fprintf(f,"record,phase,sample,confidence\n");
                    for(size_t b=0;b<batch;b++)for(size_t j=offsets[b];j<offsets[b+1];j++)if(fprintf(f,"%zu,%s,%" PRId64 ",%.9g\n",b,phases[p[j].phase],p[j].sample,p[j].confidence)<0)status=1;
                    if(fclose(f))status=1;
                }
            }
            free(p);free(offsets);
        }
    } else if(fixed) {
        if(samples!=options.window){fprintf(stderr,"Fixed input must match --window\n");status=1;}
        else {
            float *prob=malloc(samples*classes*sizeof(float));
            if(!prob)status=1;
            else if(pnsn_predict(ctx,data,(pnsn_dtype)kind,samples,prob)){fprintf(stderr,"%s\n",pnsn_last_error());status=1;}
            else {f=fopen(output,"wb");if(!f || fwrite(prob,sizeof(float),samples*classes,f)!=samples*classes)status=1;if(f && fclose(f))status=1;}
            free(prob);
        }
    } else {
        uint64_t per_phase=(uint64_t)samples/((uint64_t)options.min_gap+1)+1;
        size_t limit=per_phase>options.max_candidates/(classes-1)?options.max_candidates:(size_t)per_phase*(classes-1),count=0;
        pnsn_pick *p=limit<=SIZE_MAX/sizeof(*p)?malloc(limit*sizeof(*p)):NULL;
        if(!p)status=1;
        else if((planar?pnsn_pick_history:pnsn_pick_trace)(ctx,data,(pnsn_dtype)kind,samples,p,limit,&count)){fprintf(stderr,"%s\n",pnsn_last_error());status=1;}
        else {
            f=fopen(output,"w");if(!f)status=1;
            else {if(fprintf(f,"phase,sample,confidence\n")<0)status=1;

                for(size_t i=0;i<count;i++)if(fprintf(f,"%s,%" PRId64 ",%.9g\n",phases[p[i].phase],p[i].sample,p[i].confidence)<0)status=1;
                if(fclose(f))status=1;
            }
        }
        free(p);
    }
    pnsn_close(ctx);free(data);return status;
usage:
    fprintf(stderr,"pnsn_cli --model model.bin --input raw.bin --output picks.csv [--history-planar | --batch B] [--predict] [--dtype f32|f64|i32] [--window 10240] [--overlap 1024] [--threshold 0.1] [--gap 300] [--threads 1] [--max-candidates 2000000] [--difference]\n");return 2;
}
